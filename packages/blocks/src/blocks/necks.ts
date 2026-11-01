import { call, dedent, py } from '../py';
import { defineKerasHelpers, defineTorchHelpers } from '../helpers/registry';
import { defineBlock } from '../registry';
import { formatShape, img, pyr, sameShape } from '../shapes';
import { ShapeError, type ImageShape, type Level, type PyramidShape, type Shape } from '../types';
import { cbaParams, int, select } from './common';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

defineTorchHelpers({
  FPN: {
    code: dedent(`
      class FPN(nn.Module):
          """Feature pyramid network: 1×1 laterals, top-down nearest upsampling, 3×3 output convs.

          Paper: Feature Pyramid Networks for Object Detection, Lin et al. 2017 (arXiv:1612.03144).
          """

          def __init__(self, in_channels: list[int], out_channels: int = 256) -> None:
              super().__init__()
              self.lateral = nn.ModuleList(nn.Conv2d(c, out_channels, 1) for c in in_channels)
              self.output = nn.ModuleList(nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels)

          def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
              laterals = [conv(f) for conv, f in zip(self.lateral, feats)]
              for i in range(len(laterals) - 1, 0, -1):
                  up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
                  laterals[i - 1] = laterals[i - 1] + up
              return [conv(x) for conv, x in zip(self.output, laterals)]
    `),
  },
  PAN: {
    deps: ['C2f', 'ConvBNAct'],
    code: dedent(`
      class PAN(nn.Module):
          """YOLOv8 PAN-FPN neck: top-down then bottom-up paths fused with C2f blocks (P3, P4, P5).

          Papers: PANet, Liu et al. 2018 (arXiv:1803.01534); Ultralytics YOLOv8.
          """

          def __init__(self, in_channels: list[int], n: int = 1) -> None:
              super().__init__()
              c3, c4, c5 = in_channels
              self.td4 = C2f(c5 + c4, c4, n, shortcut=False)
              self.td3 = C2f(c4 + c3, c3, n, shortcut=False)
              self.down3 = ConvBNAct(c3, c3, 3, 2)
              self.bu4 = C2f(c3 + c4, c4, n, shortcut=False)
              self.down4 = ConvBNAct(c4, c4, 3, 2)
              self.bu5 = C2f(c4 + c5, c5, n, shortcut=False)

          @staticmethod
          def up(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
              return F.interpolate(x, size=like.shape[-2:], mode="nearest")

          def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
              p3, p4, p5 = feats
              t4 = self.td4(torch.cat([self.up(p5, p4), p4], dim=1))
              n3 = self.td3(torch.cat([self.up(t4, p3), p3], dim=1))
              n4 = self.bu4(torch.cat([self.down3(n3), t4], dim=1))
              n5 = self.bu5(torch.cat([self.down4(n4), p5], dim=1))
              return [n3, n4, n5]
    `),
  },
  BiFPN: {
    deps: ['ConvBNAct', 'DWSepConv'],
    code: dedent(`
      class BiFPNLayer(nn.Module):
          """One BiFPN layer: top-down + bottom-up paths with fast normalised (learned, ReLU) fusion weights."""

          def __init__(self, c: int, levels: int) -> None:
              super().__init__()
              self.td_w = nn.Parameter(torch.ones(levels - 1, 2))
              self.bu_w = nn.Parameter(torch.ones(levels - 1, 3))
              self.td_conv = nn.ModuleList(DWSepConv(c, c, act="silu") for _ in range(levels - 1))
              self.bu_conv = nn.ModuleList(DWSepConv(c, c, act="silu") for _ in range(levels - 1))

          @staticmethod
          def weights(w: torch.Tensor) -> torch.Tensor:
              w = F.relu(w)
              return w / (w.sum() + 1e-4)

          def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
              td = list(feats)
              for i in range(len(feats) - 2, -1, -1):
                  w = self.weights(self.td_w[i])
                  up = F.interpolate(td[i + 1], size=feats[i].shape[-2:], mode="nearest")
                  td[i] = self.td_conv[i](w[0] * feats[i] + w[1] * up)
              out = [td[0]]
              for i in range(1, len(feats)):
                  w = self.weights(self.bu_w[i - 1])
                  down = F.max_pool2d(out[-1], 3, 2, 1)
                  out.append(self.bu_conv[i - 1](w[0] * feats[i] + w[1] * td[i] + w[2] * down))
              return out


      class BiFPN(nn.Module):
          """Bidirectional FPN repeated n times (EfficientDet, Tan et al. 2020, arXiv:1911.09070)."""

          def __init__(self, in_channels: list[int], out_channels: int = 64, repeats: int = 2) -> None:
              super().__init__()
              self.lateral = nn.ModuleList(ConvBNAct(c, out_channels, 1, act="none") for c in in_channels)
              self.layers = nn.ModuleList(BiFPNLayer(out_channels, len(in_channels)) for _ in range(repeats))

          def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
              feats = [conv(f) for conv, f in zip(self.lateral, feats)]
              for layer in self.layers:
                  feats = layer(feats)
              return feats
    `),
  },
});

defineKerasHelpers({
  fpn: {
    code: dedent(`
      def fpn(feats: list, filters: int = 256, name: str = "fpn") -> list:
          """Feature pyramid network (Lin et al. 2017)."""
          laterals = [layers.Conv2D(filters, 1, name=f"{name}_lat{i}")(f) for i, f in enumerate(feats)]
          for i in range(len(laterals) - 1, 0, -1):
              h, w = laterals[i - 1].shape[1], laterals[i - 1].shape[2]
              up = layers.Resizing(h, w, interpolation="nearest", name=f"{name}_up{i}")(laterals[i])
              laterals[i - 1] = layers.Add(name=f"{name}_add{i}")([laterals[i - 1], up])
          return [layers.Conv2D(filters, 3, padding="same", name=f"{name}_out{i}")(x) for i, x in enumerate(laterals)]
    `),
  },
  pan: {
    deps: ['c2f', 'conv_bn_act'],
    code: dedent(`
      def pan(feats: list, n: int = 1, name: str = "pan") -> list:
          """YOLOv8 PAN-FPN neck (top-down + bottom-up with C2f fusion)."""
          p3, p4, p5 = feats
          c3, c4, c5 = p3.shape[-1], p4.shape[-1], p5.shape[-1]

          def up(x, like, tag):
              return layers.Resizing(like.shape[1], like.shape[2], interpolation="nearest", name=f"{name}_{tag}")(x)

          t4 = c2f(layers.Concatenate(name=f"{name}_cat_t4")([up(p5, p4, "up5"), p4]), c4, n, False, name=f"{name}_td4")
          n3 = c2f(layers.Concatenate(name=f"{name}_cat_n3")([up(t4, p3, "up4"), p3]), c3, n, False, name=f"{name}_td3")
          d3 = conv_bn_act(n3, c3, 3, 2, name=f"{name}_down3")
          n4 = c2f(layers.Concatenate(name=f"{name}_cat_n4")([d3, t4]), c4, n, False, name=f"{name}_bu4")
          d4 = conv_bn_act(n4, c4, 3, 2, name=f"{name}_down4")
          n5 = c2f(layers.Concatenate(name=f"{name}_cat_n5")([d4, p5]), c5, n, False, name=f"{name}_bu5")
          return [n3, n4, n5]
    `),
  },
});

// ---------------------------------------------------------------------------
// Necks
// ---------------------------------------------------------------------------

const c2fParams = (cin: number, cout: number, n: number) => {
  const c = Math.floor(cout * 0.5);
  return cbaParams(cin, 2 * c, 1) + cbaParams((2 + n) * c, cout, 1) + n * 2 * cbaParams(c, c, 3);
};

defineBlock({
  type: 'fpn',
  category: 'Necks/merge',
  label: 'FPN',
  doc: 'Feature pyramid network: lateral 1×1, top-down upsampling, 3×3 smoothing. Equalises channels across levels.',
  paper: 'Lin et al. 2017',
  varName: 'fpn',
  inputs: [{ name: 'pyramid', accepts: ['pyramid'] }],
  params: [int('out', 'Channels', 256, { min: 8, max: 2048 })],
  inferShape: (ins, p) => {
    const s = pyr(ins[0]);
    return { kind: 'pyramid', levels: s.levels.map((l) => ({ ...l, c: p.out })) };
  },
  paramCount: (ins, p) => {
    const s = ins[0] as PyramidShape;
    return s.levels.reduce((a, l) => a + l.c * p.out + p.out + 9 * p.out * p.out + p.out, 0);
  },
  macs: (ins, p) => (ins[0] as PyramidShape).levels.reduce((a, l) => a + l.h * l.w * (l.c * p.out + 9 * p.out * p.out), 0),
  emitTorch: ({ ins, params: p }) => ({ init: call('FPN', [`§[${(ins[0] as PyramidShape).levels.map((l) => l.c).join(', ')}]`], { out_channels: p.out }), helpers: ['FPN'] }),
  emitKeras: ({ params: p, name }) => ({ expr: (a) => call('fpn', [`§${a[0]}`], { filters: p.out, name }), helpers: ['fpn'] }),
});

defineBlock({
  type: 'pan',
  category: 'Necks/merge',
  label: 'PAN (YOLOv8)',
  doc: 'Path-aggregation neck: top-down then bottom-up, fused with C2f. Needs exactly 3 levels (P3–P5); keeps channels.',
  paper: 'Liu et al. 2018 (PANet); Ultralytics YOLOv8',
  varName: 'pan',
  inputs: [{ name: 'pyramid', accepts: ['pyramid'] }],
  params: [int('n', 'C2f depth', 1, { min: 1, max: 8 })],
  inferShape: (ins) => {
    const s = pyr(ins[0]);
    if (s.levels.length !== 3) throw new ShapeError(`PAN needs exactly 3 levels (P3, P4, P5), got ${s.levels.length}.`);
    s.levels.forEach((l, i) => {
      if (i && l.stride !== s.levels[i - 1].stride * 2) throw new ShapeError('PAN levels must have strides doubling (e.g. 8, 16, 32).');
    });
    return s;
  },
  paramCount: (ins, p) => {
    const [c3, c4, c5] = (ins[0] as PyramidShape).levels.map((l) => l.c);
    return c2fParams(c5 + c4, c4, p.n) + c2fParams(c4 + c3, c3, p.n) + cbaParams(c3, c3, 3) + c2fParams(c3 + c4, c4, p.n) + cbaParams(c4, c4, 3) + c2fParams(c4 + c5, c5, p.n);
  },
  macs: (ins, _p, _o, params) => params * (ins[0] as PyramidShape).levels[1].h * (ins[0] as PyramidShape).levels[1].w,
  emitTorch: ({ ins, params: p }) => ({ init: call('PAN', [`§[${(ins[0] as PyramidShape).levels.map((l) => l.c).join(', ')}]`], { n: p.n }), helpers: ['PAN'] }),
  emitKeras: ({ params: p, name }) => ({ expr: (a) => call('pan', [`§${a[0]}`], { n: p.n, name }), helpers: ['pan'] }),
});

defineBlock({
  type: 'bifpn',
  category: 'Necks/merge',
  label: 'BiFPN',
  doc: 'Repeated bidirectional FPN with learned fast-normalised fusion weights (EfficientDet). Equalises channels.',
  paper: 'Tan et al. 2020 (EfficientDet)',
  varName: 'bifpn',
  inputs: [{ name: 'pyramid', accepts: ['pyramid'] }],
  params: [int('out', 'Channels', 64, { min: 8, max: 1024 }), int('repeats', 'Repeats', 2, { min: 1, max: 8 })],
  inferShape: (ins, p) => {
    const s = pyr(ins[0]);
    if (s.levels.length < 2) throw new ShapeError('BiFPN needs at least 2 levels.');
    return { kind: 'pyramid', levels: s.levels.map((l) => ({ ...l, c: p.out })) };
  },
  paramCount: (ins, p) => {
    const s = ins[0] as PyramidShape;
    const L = s.levels.length;
    const dws = cbaParams(p.out, p.out, 3, p.out) + cbaParams(p.out, p.out, 1);
    return s.levels.reduce((a, l) => a + cbaParams(l.c, p.out, 1), 0) + p.repeats * (5 * (L - 1) + 2 * (L - 1) * dws);
  },
  emitTorch: ({ ins, params: p }) => ({
    init: call('BiFPN', [`§[${(ins[0] as PyramidShape).levels.map((l) => l.c).join(', ')}]`], { out_channels: p.out, repeats: p.repeats }),
    helpers: ['BiFPN'],
  }),
  emitKeras: () => ({ unsupported: 'BiFPN is only emitted for PyTorch so far; use FPN or PAN.' }),
});

// ---------------------------------------------------------------------------
// Merges
// ---------------------------------------------------------------------------

defineBlock({
  type: 'make_pyramid',
  category: 'Necks/merge',
  label: 'Make pyramid',
  doc: 'Bundle feature maps (fine → coarse, e.g. strides 8/16/32) into a pyramid for FPN / PAN / detection heads.',
  varName: 'pyramid',
  inputs: [{ name: 'level', accepts: ['image'] }],
  variadic: true,
  params: [],
  inferShape: (ins) => {
    const levels: Level[] = ins.map((s, i) => {
      const m = img(s, `Level ${i + 1}`);
      return { c: m.c, h: m.h, w: m.w, stride: m.stride };
    });
    levels.forEach((l, i) => {
      if (i && l.stride <= levels[i - 1].stride) throw new ShapeError('Connect levels from fine to coarse (increasing stride).', 'Reorder the inputs: highest resolution first.');
    });
    return { kind: 'pyramid', levels };
  },
  paramCount: () => 0,
  macs: () => 0,
  emitTorch: () => ({ forward: (a) => `[${a.join(', ')}]` }),
  emitKeras: () => ({ expr: (a) => `[${a.join(', ')}]` }),
});

defineBlock({
  type: 'select_level',
  category: 'Necks/merge',
  label: 'Select level',
  doc: 'Pick one feature map from a pyramid (0 = finest).',
  varName: 'level',
  inputs: [{ name: 'pyramid', accepts: ['pyramid'] }],
  params: [int('index', 'Level index', 0, { min: 0, max: 5 })],
  inferShape: (ins, p) => {
    const s = pyr(ins[0]);
    const l = s.levels[p.index];
    if (!l) throw new ShapeError(`The pyramid has ${s.levels.length} levels; index ${p.index} is out of range.`);
    return { kind: 'image', ...l };
  },
  paramCount: () => 0,
  macs: () => 0,
  emitTorch: ({ params: p }) => ({ forward: (a) => `${a[0]}[${p.index}]` }),
  emitKeras: ({ params: p }) => ({ expr: (a) => `${a[0]}[${p.index}]` }),
});

function spatialFix(a: ImageShape, b: ImageShape): string | undefined {
  if (a.h === b.h * 2 && a.w === b.w * 2) return 'add Upsample ×2 on the second input';
  if (b.h === a.h * 2 && b.w === a.w * 2) return 'add Upsample ×2 on the first input';
  if (a.h % b.h === 0 && a.h / b.h === a.w / b.w) return `add Upsample ×${a.h / b.h} on the second input`;
  if (b.h % a.h === 0 && b.h / a.h === b.w / a.w) return `add Upsample ×${b.h / a.h} on the first input`;
  return 'use Resize like to match sizes';
}

defineBlock({
  type: 'add',
  category: 'Necks/merge',
  label: 'Add',
  doc: 'Element-wise sum (residual / skip connection). Shapes must match.',
  varName: 'add',
  inputs: [{ name: 'x', accepts: ['image', 'sequence', 'vector'] }],
  variadic: true,
  params: [],
  inferShape: (ins) => {
    for (const s of ins.slice(1)) {
      if (!sameShape(s, ins[0])) throw new ShapeError(`Add needs identical shapes: ${formatShape(ins[0])} vs ${formatShape(s)}.`, 'Insert a 1×1 conv / Linear or resize so both inputs match.');
    }
    return ins[0];
  },
  paramCount: () => 0,
  macs: () => 0,
  emitTorch: () => ({ forward: (a) => a.join(' + ') }),
  emitKeras: ({ name }) => ({ expr: (a) => `layers.Add(name=${py(name)})([${a.join(', ')}])` }),
});

defineBlock({
  type: 'concat',
  category: 'Necks/merge',
  label: 'Concat',
  doc: 'Concatenate along channels (feature maps / vectors) or along features / length (sequences).',
  varName: 'cat',
  inputs: [{ name: 'x', accepts: ['image', 'sequence', 'vector'] }],
  variadic: true,
  params: [select('axis', 'Sequence axis', ['features', 'length'], 'features', { help: 'Only used for sequences.' })],
  inferShape: (ins, p): Shape => {
    const k = ins[0].kind;
    if (ins.some((s) => s.kind !== k)) throw new ShapeError('All Concat inputs must be the same kind.');
    if (k === 'image') {
      const a = ins[0] as ImageShape;
      for (const s of ins.slice(1) as ImageShape[]) {
        if (s.h !== a.h || s.w !== a.w) throw new ShapeError(`Concat needs equal spatial sizes: ${a.h}×${a.w} vs ${s.h}×${s.w}.`, spatialFix(a, s));
      }
      return { ...a, c: (ins as ImageShape[]).reduce((t, s) => t + s.c, 0) };
    }
    if (k === 'vector') return { kind: 'vector', d: ins.reduce((t, s) => t + (s as { d: number }).d, 0) };
    if (k === 'sequence') {
      const s0 = ins[0] as Extract<Shape, { kind: 'sequence' }>;
      const ss = ins as Extract<Shape, { kind: 'sequence' }>[];
      if (p.axis === 'length') {
        if (ss.some((s) => s.d !== s0.d)) throw new ShapeError('Concatenating along length needs equal feature sizes.');
        return { kind: 'sequence', l: ss.reduce((t, s) => t + s.l, 0), d: s0.d };
      }
      if (ss.some((s) => s.l !== s0.l)) throw new ShapeError('Concatenating features needs equal sequence lengths.');
      return { ...s0, d: ss.reduce((t, s) => t + s.d, 0) };
    }
    throw new ShapeError('Concat supports feature maps, sequences and vectors.');
  },
  paramCount: () => 0,
  macs: () => 0,
  emitTorch: ({ ins, params: p }) => {
    const dim = ins[0].kind === 'sequence' ? (p.axis === 'length' ? 1 : 2) : 1;
    return { forward: (a) => `torch.cat([${a.join(', ')}], dim=${dim})` };
  },
  emitKeras: ({ ins, params: p, name }) => {
    const axis = ins[0].kind === 'sequence' && p.axis === 'length' ? 1 : -1;
    return { expr: (a) => `layers.Concatenate(axis=${axis}, name=${py(name)})([${a.join(', ')}])` };
  },
});

defineBlock({
  type: 'multiply',
  category: 'Necks/merge',
  label: 'Multiply',
  doc: 'Element-wise product of equal shapes, or channel gating of a feature map by a C-vector.',
  varName: 'mul',
  inputs: [
    { name: 'x', accepts: ['image', 'sequence', 'vector'] },
    { name: 'gate', accepts: ['image', 'sequence', 'vector'] },
  ],
  params: [],
  inferShape: (ins) => {
    const [a, b] = ins;
    if (a.kind === 'image' && b.kind === 'vector') {
      if (b.d !== a.c) throw new ShapeError(`Channel gate size ${b.d} must equal the feature channels ${a.c}.`);
      return a;
    }
    if (!sameShape(a, b)) throw new ShapeError(`Multiply needs identical shapes (or image × C-vector): ${formatShape(a)} vs ${formatShape(b)}.`);
    return a;
  },
  paramCount: () => 0,
  macs: () => 0,
  emitTorch: ({ ins }) => ({ forward: (a) => (ins[0].kind === 'image' && ins[1].kind === 'vector' ? `${a[0]} * ${a[1]}[:, :, None, None]` : `${a[0]} * ${a[1]}`) }),
  emitKeras: ({ ins, name }) => ({
    expr: (a) =>
      ins[0].kind === 'image' && ins[1].kind === 'vector'
        ? `layers.Multiply(name=${py(name)})([${a[0]}, layers.Reshape((1, 1, ${(ins[0] as ImageShape).c}), name="${name}_gate")(${a[1]})])`
        : `layers.Multiply(name=${py(name)})([${a[0]}, ${a[1]}])`,
  }),
});

defineBlock({
  type: 'resize_like',
  category: 'Layers',
  label: 'Resize like',
  doc: 'Resize the first feature map to the spatial size of the second (handles odd sizes in U-Net / decoders).',
  varName: 'resized',
  inputs: [
    { name: 'x', accepts: ['image'] },
    { name: 'reference', accepts: ['image'] },
  ],
  params: [select('mode', 'Mode', ['bilinear', 'nearest'], 'bilinear')],
  inferShape: (ins) => {
    const a = img(ins[0]);
    const r = img(ins[1], 'Reference');
    return { kind: 'image', c: a.c, h: r.h, w: r.w, stride: r.stride };
  },
  paramCount: () => 0,
  macs: () => 0,
  emitTorch: ({ params: p }) => ({
    forward: (a) => `F.interpolate(${a[0]}, size=${a[1]}.shape[-2:], mode=${py(p.mode)}${p.mode === 'bilinear' ? ', align_corners=False' : ''})`,
  }),
  emitKeras: ({ ins, params: p, name }) => {
    const r = ins[1] as ImageShape;
    return { expr: (a) => `layers.Resizing(${r.h}, ${r.w}, interpolation=${py(p.mode)}, name=${py(name)})(${a[0]})` };
  },
});
