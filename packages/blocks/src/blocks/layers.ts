import { kw, py, pyFloat } from '../py';
import { defineBlock } from '../registry';
import { convOut, convTransposeOut, expectKind, featureDim, img, numel, positions, withFeature } from '../shapes';
import { ShapeError, type ImageShape, type Shape } from '../types';
import { actParam, bool, cbaParams, convParams, float, int, kerasAct, kerasActName, linearParams, select, torchAct, ACTS } from './common';

const outCh = (def = 64) => int('out', 'Output channels', def, { min: 1, max: 8192 });
const kernel = (def = 3) => int('kernel', 'Kernel size', def, { min: 1, max: 15 });
const stride = (def = 1) => int('stride', 'Stride', def, { min: 1, max: 8, help: 'Stride 2 halves the spatial size.' });

function convShape(s: ImageShape, cout: number, k: number, st: number, pad: 'same' | 'valid', d: number, fw: 'pytorch' | 'tensorflow'): Shape {
  const h = convOut(s.h, k, st, pad, d, fw);
  const w = convOut(s.w, k, st, pad, d, fw);
  if (h < 1 || w < 1) throw new ShapeError(`Output would be ${h}×${w}: the input (${s.h}×${s.w}) is too small for this kernel/stride.`, 'Use padding "same" or a smaller kernel.');
  return { kind: 'image', c: cout, h, w, stride: s.stride * st };
}

function checkGroups(cin: number, cout: number, g: number): void {
  if (cin % g || cout % g) throw new ShapeError(`Groups (${g}) must divide both input (${cin}) and output (${cout}) channels.`);
}

defineBlock({
  type: 'conv2d',
  category: 'Layers',
  label: 'Conv2d',
  doc: '2-D convolution.',
  varName: 'conv',
  inputs: [{ name: 'x', accepts: ['image'] }],
  params: [
    outCh(),
    kernel(),
    stride(),
    select('padding', 'Padding', ['same', 'valid'], 'same', {
      help: 'PyTorch "same" uses padding = d·(k−1)//2; Keras "same" pads asymmetrically for even kernels (output = ceil(H/s)).',
    }),
    int('dilation', 'Dilation', 1, { min: 1, max: 24 }),
    int('groups', 'Groups', 1, { min: 1, help: 'groups = in_channels gives a depthwise convolution.' }),
    bool('bias', 'Bias', true),
  ],
  inferShape: (ins, p, ctx) => {
    const s = img(ins[0]);
    checkGroups(s.c, p.out, p.groups);
    return convShape(s, p.out, p.kernel, p.stride, p.padding, p.dilation, ctx.framework);
  },
  paramCount: (ins, p) => convParams(featureDim(ins[0]), p.out, p.kernel, p.groups, p.bias),
  emitTorch: ({ params: p, ins }) => {
    const pad = p.padding === 'same' ? Math.floor((p.dilation * (p.kernel - 1)) / 2) : 0;
    return {
      init: `nn.Conv2d(${featureDim(ins[0])}, ${p.out}, ${kw({ kernel_size: p.kernel, stride: p.stride, padding: pad, dilation: p.dilation === 1 ? undefined : p.dilation, groups: p.groups === 1 ? undefined : p.groups, bias: p.bias })})`,
    };
  },
  emitKeras: ({ params: p, ins, name }) => {
    if (p.stride > 1 && p.dilation > 1) return { unsupported: 'Keras does not allow stride > 1 together with dilation > 1.' };
    const cin = featureDim(ins[0]);
    if (p.groups > 1 && p.groups === cin && p.out === cin) {
      return { expr: (a) => `layers.DepthwiseConv2D(${kw({ kernel_size: p.kernel, strides: p.stride, padding: p.padding, dilation_rate: p.dilation, use_bias: p.bias, name })})(${a[0]})` };
    }
    return {
      expr: (a) =>
        `layers.Conv2D(${p.out}, ${kw({ kernel_size: p.kernel, strides: p.stride, padding: p.padding, dilation_rate: p.dilation, groups: p.groups, use_bias: p.bias, name })})(${a[0]})`,
    };
  },
});

defineBlock({
  type: 'conv_bn_act',
  category: 'Layers',
  label: 'Conv·BN·Act',
  doc: 'Convolution without bias, BatchNorm, activation — the standard CNN unit ("same" padding).',
  varName: 'cba',
  inputs: [{ name: 'x', accepts: ['image'] }],
  params: [outCh(), kernel(), stride(), int('dilation', 'Dilation', 1, { min: 1, max: 24 }), int('groups', 'Groups', 1, { min: 1 }), actParam('silu')],
  inferShape: (ins, p, ctx) => {
    const s = img(ins[0]);
    checkGroups(s.c, p.out, p.groups);
    return convShape(s, p.out, p.kernel, p.stride, 'same', p.dilation, ctx.framework);
  },
  paramCount: (ins, p) => cbaParams(featureDim(ins[0]), p.out, p.kernel, p.groups),
  emitTorch: ({ params: p, ins }) => ({
    init: `ConvBNAct(${featureDim(ins[0])}, ${p.out}, ${kw({ k: p.kernel, s: p.stride, d: p.dilation === 1 ? undefined : p.dilation, g: p.groups === 1 ? undefined : p.groups, act: p.act })})`,
    helpers: ['ConvBNAct'],
  }),
  emitKeras: ({ params: p, name }) =>
    p.stride > 1 && p.dilation > 1
      ? { unsupported: 'Keras does not allow stride > 1 together with dilation > 1.' }
      : {
          expr: (a) => `conv_bn_act(${a[0]}, ${p.out}, ${kw({ k: p.kernel, s: p.stride, d: p.dilation, groups: p.groups, act: kerasActName(p.act), name })})`,
          helpers: ['conv_bn_act'],
        },
});

defineBlock({
  type: 'dwsep_conv',
  category: 'Layers',
  label: 'Depthwise-separable conv',
  doc: 'Depthwise k×k conv then pointwise 1×1 conv, each followed by BN + activation.',
  paper: 'MobileNets, Howard et al. 2017',
  varName: 'dws',
  inputs: [{ name: 'x', accepts: ['image'] }],
  params: [outCh(), kernel(), stride(), actParam('relu')],
  inferShape: (ins, p, ctx) => convShape(img(ins[0]), p.out, p.kernel, p.stride, 'same', 1, ctx.framework),
  paramCount: (ins, p) => {
    const c = featureDim(ins[0]);
    return cbaParams(c, c, p.kernel, c) + cbaParams(c, p.out, 1);
  },
  emitTorch: ({ params: p, ins }) => ({
    init: `DWSepConv(${featureDim(ins[0])}, ${p.out}, ${kw({ k: p.kernel, s: p.stride, act: p.act })})`,
    helpers: ['DWSepConv'],
  }),
  emitKeras: ({ params: p, name }) => ({
    expr: (a) => `dw_sep_conv(${a[0]}, ${p.out}, ${kw({ k: p.kernel, s: p.stride, act: kerasActName(p.act), name })})`,
    helpers: ['dw_sep_conv'],
  }),
});

defineBlock({
  type: 'conv_transpose',
  category: 'Layers',
  label: 'ConvTranspose2d',
  doc: 'Transposed convolution (learned upsampling). out = (h−1)·s − 2p + (k−1) + output_padding + 1.',
  varName: 'deconv',
  inputs: [{ name: 'x', accepts: ['image'] }],
  params: [outCh(), kernel(2), stride(2), int('padding', 'Padding', 0), int('outputPadding', 'Output padding', 0), bool('bias', 'Bias', true)],
  inferShape: (ins, p) => {
    const s = img(ins[0]);
    if (p.outputPadding >= p.stride) throw new ShapeError('Output padding must be smaller than the stride.');
    const h = convTransposeOut(s.h, p.kernel, p.stride, p.padding, p.outputPadding);
    const w = convTransposeOut(s.w, p.kernel, p.stride, p.padding, p.outputPadding);
    if (h < 1 || w < 1) throw new ShapeError('Padding is too large for this kernel.');
    return { kind: 'image', c: p.out, h, w, stride: (s.stride * s.h) / h };
  },
  paramCount: (ins, p) => convParams(featureDim(ins[0]), p.out, p.kernel, 1, p.bias),
  macs: (ins, _p, _o, params) => params * positions(ins[0]),
  emitTorch: ({ params: p, ins }) => ({
    init: `nn.ConvTranspose2d(${featureDim(ins[0])}, ${p.out}, ${kw({ kernel_size: p.kernel, stride: p.stride, padding: p.padding, output_padding: p.outputPadding || undefined, bias: p.bias })})`,
  }),
  emitKeras: ({ params: p, name }) => {
    let padding: string | undefined;
    if (p.padding === 0 && p.outputPadding === 0 && p.kernel >= p.stride) padding = 'valid';
    else if (2 * p.padding === p.kernel - p.stride + p.outputPadding) padding = 'same';
    if (!padding) return { unsupported: 'Keras Conv2DTranspose only supports padding 0 (with kernel ≥ stride) or the "same" case 2p = k − s + output_padding.' };
    return {
      expr: (a) => `layers.Conv2DTranspose(${p.out}, ${kw({ kernel_size: p.kernel, strides: p.stride, padding, use_bias: p.bias, name })})(${a[0]})`,
    };
  },
});

// ---------------------------------------------------------------------------
// Normalisation
// ---------------------------------------------------------------------------

defineBlock({
  type: 'batchnorm',
  category: 'Layers',
  label: 'BatchNorm',
  doc: 'Batch normalisation over channels (feature maps) or features (vectors).',
  paper: 'Ioffe & Szegedy 2015',
  varName: 'bn',
  inputs: [{ name: 'x', accepts: ['image', 'vector'] }],
  params: [],
  inferShape: (ins) => {
    if (ins[0].kind !== 'image' && ins[0].kind !== 'vector') throw new ShapeError('BatchNorm needs a feature map or vector.', 'Use LayerNorm for sequences.');
    return ins[0];
  },
  paramCount: (ins) => 2 * featureDim(ins[0]),
  emitTorch: ({ ins }) => ({ init: `nn.BatchNorm${ins[0].kind === 'image' ? '2d' : '1d'}(${featureDim(ins[0])})` }),
  emitKeras: ({ name }) => ({ expr: (a) => `layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=${py(name)})(${a[0]})` }),
});

defineBlock({
  type: 'groupnorm',
  category: 'Layers',
  label: 'GroupNorm',
  doc: 'Group normalisation; batch-size independent (good for detection/segmentation with small batches).',
  paper: 'Wu & He 2018',
  varName: 'gn',
  inputs: [{ name: 'x', accepts: ['image'] }],
  params: [int('groups', 'Groups', 32, { min: 1 })],
  inferShape: (ins, p) => {
    const s = img(ins[0]);
    if (s.c % p.groups) throw new ShapeError(`Channels (${s.c}) must be divisible by groups (${p.groups}).`);
    return s;
  },
  paramCount: (ins) => 2 * featureDim(ins[0]),
  emitTorch: ({ params: p, ins }) => ({ init: `nn.GroupNorm(${p.groups}, ${featureDim(ins[0])})` }),
  emitKeras: ({ params: p, name }) => ({ expr: (a) => `layers.GroupNormalization(groups=${p.groups}, epsilon=1e-5, name=${py(name)})(${a[0]})` }),
});

defineBlock({
  type: 'layernorm',
  category: 'Layers',
  label: 'LayerNorm',
  doc: 'Layer normalisation over the feature axis (channels for feature maps).',
  paper: 'Ba et al. 2016',
  varName: 'ln',
  inputs: [{ name: 'x', accepts: ['image', 'sequence', 'vector'] }],
  params: [],
  inferShape: (ins) => {
    featureDim(ins[0]);
    return ins[0];
  },
  paramCount: (ins) => 2 * featureDim(ins[0]),
  emitTorch: ({ ins }) =>
    ins[0].kind === 'image'
      ? { init: `LayerNorm2d(${featureDim(ins[0])}, eps=1e-6)`, helpers: ['LayerNorm2d'] }
      : { init: `nn.LayerNorm(${featureDim(ins[0])}, eps=1e-6)` },
  emitKeras: ({ name }) => ({ expr: (a) => `layers.LayerNormalization(epsilon=1e-6, name=${py(name)})(${a[0]})` }),
});

defineBlock({
  type: 'rmsnorm',
  category: 'Layers',
  label: 'RMSNorm',
  doc: 'Root-mean-square normalisation (no mean centering), used by modern LLMs.',
  paper: 'Zhang & Sennrich 2019',
  varName: 'rms',
  inputs: [{ name: 'x', accepts: ['sequence', 'vector'] }],
  params: [],
  inferShape: (ins) => {
    if (ins[0].kind === 'image') throw new ShapeError('RMSNorm expects a sequence or vector.');
    featureDim(ins[0]);
    return ins[0];
  },
  paramCount: (ins) => featureDim(ins[0]),
  emitTorch: ({ ins }) => ({ init: `nn.RMSNorm(${featureDim(ins[0])}, eps=1e-6)` }),
  emitKeras: ({ name }) => ({ expr: (a) => `layers.LayerNormalization(epsilon=1e-6, rms_scaling=True, name=${py(name)})(${a[0]})` }),
});

defineBlock({
  type: 'activation',
  category: 'Layers',
  label: 'Activation',
  doc: 'Element-wise non-linearity.',
  varName: 'act',
  inputs: [{ name: 'x', accepts: ['image', 'sequence', 'vector'] }],
  params: [select('act', 'Function', ACTS.filter((a) => a !== 'none'), 'relu')],
  inferShape: (ins) => ins[0],
  paramCount: () => 0,
  macs: () => 0,
  emitTorch: ({ params: p }) => ({ init: torchAct(p.act) }),
  emitKeras: ({ params: p, name }) => ({ expr: (a) => `${kerasAct(p.act, name)}(${a[0]})` }),
});

// ---------------------------------------------------------------------------
// Pooling / reshaping
// ---------------------------------------------------------------------------

for (const kind of ['max', 'avg'] as const) {
  defineBlock({
    type: `${kind}pool`,
    category: 'Layers',
    label: kind === 'max' ? 'Max pool' : 'Avg pool',
    doc: `${kind === 'max' ? 'Max' : 'Average'} pooling over k×k windows.`,
    varName: `${kind}pool`,
    inputs: [{ name: 'x', accepts: ['image'] }],
    params: [kernel(2), stride(2), int('padding', 'Padding', 0, { help: 'Implicit zero / −∞ padding (must be ≤ k/2).' })],
    inferShape: (ins, p) => {
      const s = img(ins[0]);
      if (p.padding * 2 > p.kernel) throw new ShapeError('Padding must be at most half the kernel size.');
      const f = (x: number) => Math.floor((x + 2 * p.padding - p.kernel) / p.stride) + 1;
      const h = f(s.h);
      const w = f(s.w);
      if (h < 1 || w < 1) throw new ShapeError(`Input ${s.h}×${s.w} is smaller than the pooling window.`);
      return { kind: 'image', c: s.c, h, w, stride: s.stride * p.stride };
    },
    paramCount: () => 0,
    macs: (_i, p, o) => numel(o) * p.kernel * p.kernel,
    emitTorch: ({ params: p }) => ({
      init: `nn.${kind === 'max' ? 'MaxPool2d' : 'AvgPool2d'}(${kw({ kernel_size: p.kernel, stride: p.stride, padding: p.padding || undefined })})`,
    }),
    emitKeras: ({ params: p, name }) => {
      let padding: string | undefined;
      if (p.padding === 0) padding = 'valid';
      else if (p.kernel % 2 === 1 && p.padding === (p.kernel - 1) / 2) padding = 'same';
      if (!padding) return { unsupported: 'Keras pooling supports padding 0 ("valid") or (k−1)/2 with odd k ("same").' };
      const layer = kind === 'max' ? 'MaxPooling2D' : 'AveragePooling2D';
      return { expr: (a) => `layers.${layer}(${kw({ pool_size: p.kernel, strides: p.stride, padding, name })})(${a[0]})` };
    },
  });
}

defineBlock({
  type: 'gap',
  category: 'Layers',
  label: 'Global average pool',
  doc: 'Average over all spatial positions: C×H×W → C.',
  paper: 'Lin et al., Network in Network 2013',
  varName: 'gap',
  inputs: [{ name: 'x', accepts: ['image'] }],
  params: [],
  inferShape: (ins) => ({ kind: 'vector', d: img(ins[0]).c }),
  paramCount: () => 0,
  macs: (ins) => numel(ins[0]),
  emitTorch: () => ({ forward: (a) => `${a[0]}.mean(dim=(2, 3))` }),
  emitKeras: ({ name }) => ({ expr: (a) => `layers.GlobalAveragePooling2D(name=${py(name)})(${a[0]})` }),
});

defineBlock({
  type: 'flatten',
  category: 'Layers',
  label: 'Flatten',
  doc: 'Flatten a feature map to a vector (C·H·W). Note: Keras flattens in H·W·C order.',
  varName: 'flat',
  inputs: [{ name: 'x', accepts: ['image', 'sequence'] }],
  params: [],
  inferShape: (ins) => {
    const s = ins[0];
    if (s.kind !== 'image' && s.kind !== 'sequence') throw new ShapeError('Flatten expects a feature map or sequence.');
    return { kind: 'vector', d: numel(s) };
  },
  paramCount: () => 0,
  macs: () => 0,
  emitTorch: () => ({ forward: (a) => `torch.flatten(${a[0]}, 1)` }),
  emitKeras: ({ name }) => ({ expr: (a) => `layers.Flatten(name=${py(name)})(${a[0]})` }),
});

defineBlock({
  type: 'linear',
  category: 'Layers',
  label: 'Linear',
  doc: 'Fully connected layer on the last axis (vectors or per-token on sequences).',
  varName: 'fc',
  inputs: [{ name: 'x', accepts: ['vector', 'sequence'] }],
  params: [int('out', 'Output features', 256, { min: 1, max: 1000000 }), bool('bias', 'Bias', true)],
  inferShape: (ins, p) => {
    if (ins[0].kind === 'image') throw new ShapeError('Linear expects a vector or sequence, got a feature map.', 'Add Global average pool or Flatten first.');
    return withFeature(ins[0], p.out);
  },
  paramCount: (ins, p) => linearParams(featureDim(ins[0]), p.out, p.bias),
  emitTorch: ({ params: p, ins }) => ({ init: `nn.Linear(${featureDim(ins[0])}, ${p.out}${p.bias ? '' : ', bias=False'})` }),
  emitKeras: ({ params: p, name }) => ({ expr: (a) => `layers.Dense(${p.out}, ${kw({ use_bias: p.bias, name })})(${a[0]})` }),
});

defineBlock({
  type: 'dropout',
  category: 'Layers',
  label: 'Dropout',
  doc: 'Randomly zeroes activations during training. "channel" drops whole feature maps (Dropout2d).',
  paper: 'Srivastava et al. 2014',
  varName: 'drop',
  inputs: [{ name: 'x', accepts: ['image', 'sequence', 'vector'] }],
  params: [float('p', 'Drop probability', 0.1, { min: 0, max: 0.95, step: 0.05 }), select('mode', 'Mode', ['element', 'channel'], 'element')],
  inferShape: (ins) => ins[0],
  paramCount: () => 0,
  macs: () => 0,
  emitTorch: ({ params: p, ins }) => ({
    init: p.mode === 'channel' && ins[0].kind === 'image' ? `nn.Dropout2d(${pyFloat(p.p)})` : `nn.Dropout(${pyFloat(p.p)})`,
  }),
  emitKeras: ({ params: p, ins, name }) => ({
    expr: (a) =>
      p.mode === 'channel' && ins[0].kind === 'image'
        ? `layers.SpatialDropout2D(${pyFloat(p.p)}, name=${py(name)})(${a[0]})`
        : `layers.Dropout(${pyFloat(p.p)}, name=${py(name)})(${a[0]})`,
  }),
});

defineBlock({
  type: 'upsample',
  category: 'Layers',
  label: 'Upsample',
  doc: 'Resize a feature map by an integer factor (nearest or bilinear).',
  varName: 'up',
  inputs: [{ name: 'x', accepts: ['image'] }],
  params: [int('scale', 'Scale factor', 2, { min: 2, max: 32 }), select('mode', 'Mode', ['nearest', 'bilinear'], 'nearest')],
  inferShape: (ins, p) => {
    const s = img(ins[0]);
    return { kind: 'image', c: s.c, h: s.h * p.scale, w: s.w * p.scale, stride: s.stride / p.scale };
  },
  paramCount: () => 0,
  macs: () => 0,
  emitTorch: ({ params: p }) => ({
    init: `nn.Upsample(${kw({ scale_factor: p.scale, mode: p.mode, align_corners: p.mode === 'bilinear' ? false : undefined })})`,
  }),
  emitKeras: ({ params: p, name }) => ({ expr: (a) => `layers.UpSampling2D(${kw({ size: p.scale, interpolation: p.mode, name })})(${a[0]})` }),
});

defineBlock({
  type: 'image_to_seq',
  category: 'Seq & attn',
  label: 'Image → sequence',
  doc: 'Flatten spatial positions into tokens: C×H×W → (H·W)×C. Remembers the grid for Sequence → image.',
  varName: 'tokens',
  inputs: [{ name: 'x', accepts: ['image'] }],
  params: [],
  inferShape: (ins) => {
    const s = img(ins[0]);
    return { kind: 'sequence', l: s.h * s.w, d: s.c, grid: { h: s.h, w: s.w, stride: s.stride }, prefix: 0 };
  },
  paramCount: () => 0,
  macs: () => 0,
  emitTorch: () => ({ forward: (a) => `${a[0]}.flatten(2).transpose(1, 2)` }),
  emitKeras: ({ ins, name }) => {
    const s = ins[0] as ImageShape;
    return { expr: (a) => `layers.Reshape((${s.h * s.w}, ${s.c}), name=${py(name)})(${a[0]})` };
  },
});

defineBlock({
  type: 'seq_to_image',
  category: 'Seq & attn',
  label: 'Sequence → image',
  doc: 'Reshape patch tokens back to a feature map (drops CLS/prefix tokens). Used for ViT dense prediction.',
  varName: 'fmap',
  inputs: [{ name: 'x', accepts: ['sequence'] }],
  params: [],
  inferShape: (ins) => {
    const s = expectKind(ins[0], 'sequence');
    if (!s.grid) throw new ShapeError('This sequence has no spatial grid (it did not come from Patch embed / Image → sequence).');
    return { kind: 'image', c: s.d, h: s.grid.h, w: s.grid.w, stride: s.grid.stride };
  },
  paramCount: () => 0,
  macs: () => 0,
  emitTorch: ({ ins }) => {
    const s = ins[0] as Extract<Shape, { kind: 'sequence' }>;
    return { forward: (a) => `seq_to_image(${a[0]}, ${s.grid!.h}, ${s.grid!.w}, prefix=${s.prefix ?? 0})`, helpers: ['seq_to_image'] };
  },
  emitKeras: ({ ins, name }) => {
    const s = ins[0] as Extract<Shape, { kind: 'sequence' }>;
    return {
      expr: (a) => `layers.Reshape((${s.grid!.h}, ${s.grid!.w}, ${s.d}), name=${py(name)})(${s.prefix ? `${a[0]}[:, ${s.prefix}:]` : a[0]})`,
    };
  },
});
