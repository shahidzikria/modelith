import { call, pyFloat } from '../py';
import { defineBlock } from '../registry';
import { convOut, img } from '../shapes';
import { ShapeError, type BlockDef, type Framework, type ImageShape, type Shape } from '../types';
import { actParam, bool, cbaParams, float, int, intList, kerasActName } from './common';

const outCh = (def: number) => int('out', 'Output channels', def, { min: 1, max: 8192 });
const strideP = (def = 1) => int('stride', 'Stride', def, { min: 1, max: 2, help: 'Stride 2 halves the spatial size (applied in the first block).' });
const repeatP = (def = 1) => int('n', 'Repeat ×N', def, { min: 1, max: 32, help: 'Number of stacked blocks; only the first changes channels / stride.' });

function strided(s: ImageShape, c: number, stride: number, fw: Framework): Shape {
  const h = convOut(s.h, 3, stride, 'same', 1, fw);
  const w = convOut(s.w, 3, stride, 'same', 1, fw);
  return { kind: 'image', c, h, w, stride: s.stride * stride };
}

const seParams = (c: number, hidden: number) => c * hidden + hidden + hidden * c + c;

/** Torch: `Block(c_in, c_out, stride=s, **kw)` or `make_stage(Block, c_in, c_out, n=N, stride=s, **kw)`. */
function torchStage(cls: string, cin: number, cout: number, n: number, stride: number, kw: Record<string, unknown> = {}): { init: string; helpers: string[] } {
  if (n === 1) return { init: call(cls, [cin, cout], { stride: stride === 1 ? undefined : stride, ...kw }), helpers: [cls] };
  return { init: call('make_stage', [`§${cls}`, cin, cout], { n, stride, ...kw }), helpers: [cls, 'make_stage'] };
}

/** Keras: `fn(x, filters, stride, ...)` or `repeat_block(x, fn, n, filters, stride, ...)`. */
function kerasStage(fn: string, cout: number, n: number, stride: number, name: string, kw: Record<string, unknown> = {}): { expr: (a: string[]) => string; helpers: string[] } {
  if (n === 1) return { expr: (a) => call(fn, [`§${a[0]}`, cout, stride], { ...kw, name }), helpers: [fn] };
  return { expr: (a) => call('repeat_block', [`§${a[0]}`, `§${fn}`, n, cout, stride], { name, ...kw }), helpers: [fn, 'repeat_block'] };
}

function residualBlock(o: {
  type: string;
  label: string;
  doc: string;
  paper: string;
  varName: string;
  outDefault: number;
  torch: string;
  keras: string;
  params: BlockDef['params'];
  count: (cin: number, cout: number, stride: number, p: Record<string, any>) => number;
  kw?: (p: Record<string, any>) => { torch: Record<string, unknown>; keras: Record<string, unknown> };
  validate?: (s: ImageShape, p: Record<string, any>) => void;
}): void {
  defineBlock({
    type: o.type,
    category: 'CNN blocks',
    label: o.label,
    doc: o.doc,
    paper: o.paper,
    varName: o.varName,
    inputs: [{ name: 'x', accepts: ['image'] }],
    params: [outCh(o.outDefault), strideP(), repeatP(), ...o.params],
    inferShape: (ins, p, ctx) => {
      const s = img(ins[0]);
      o.validate?.(s, p);
      return strided(s, p.out, p.stride, ctx.framework);
    },
    paramCount: (ins, p) => {
      const s = ins[0] as ImageShape;
      let total = 0;
      for (let i = 0; i < p.n; i++) total += o.count(i === 0 ? s.c : p.out, p.out, i === 0 ? p.stride : 1, p);
      return total;
    },
    emitTorch: ({ ins, params: p }) => torchStage(o.torch, (ins[0] as ImageShape).c, p.out, p.n, p.stride, o.kw?.(p).torch),
    emitKeras: ({ params: p, name }) => kerasStage(o.keras, p.out, p.n, p.stride, name, o.kw?.(p).keras),
  });
}

const projection = (cin: number, cout: number, stride: number) => (stride !== 1 || cin !== cout ? cbaParams(cin, cout, 1) : 0);

residualBlock({
  type: 'resnet_basic',
  label: 'ResNet BasicBlock',
  doc: 'Two 3×3 convs with an identity / projection shortcut (ResNet-18/34).',
  paper: 'He et al. 2016',
  varName: 'layer',
  outDefault: 64,
  torch: 'BasicBlock',
  keras: 'basic_block',
  params: [],
  count: (cin, cout, s) => cbaParams(cin, cout, 3) + cbaParams(cout, cout, 3) + projection(cin, cout, s),
});

residualBlock({
  type: 'resnet_bottleneck',
  label: 'ResNet Bottleneck',
  doc: '1×1 reduce → 3×3 → 1×1 expand (×4) with shortcut (ResNet-50/101).',
  paper: 'He et al. 2016',
  varName: 'layer',
  outDefault: 256,
  torch: 'Bottleneck',
  keras: 'bottleneck_block',
  params: [],
  validate: (_s, p) => {
    if (p.out % 4) throw new ShapeError('Bottleneck output channels must be divisible by 4 (expansion).');
  },
  count: (cin, cout, s) => {
    const w = cout / 4;
    return cbaParams(cin, w, 1) + cbaParams(w, w, 3) + cbaParams(w, cout, 1) + projection(cin, cout, s);
  },
});

residualBlock({
  type: 'mbconv',
  label: 'MBConv (+SE)',
  doc: 'Inverted residual: 1×1 expand → k×k depthwise → squeeze-excite → 1×1 project (EfficientNet).',
  paper: 'Sandler et al. 2018; Tan & Le 2019',
  varName: 'mbconv',
  outDefault: 40,
  torch: 'MBConv',
  keras: 'mbconv',
  params: [
    int('kernel', 'Depthwise kernel', 3, { min: 3, max: 7, step: 2 }),
    int('expand', 'Expansion ratio', 4, { min: 1, max: 8 }),
    float('se', 'SE ratio (0 = off)', 0.25, { min: 0, max: 1, step: 0.05 }),
    actParam('silu'),
  ],
  count: (cin, cout, _s, p) => {
    const mid = cin * p.expand;
    const hidden = Math.max(1, Math.floor(cin * p.se));
    return (p.expand !== 1 ? cbaParams(cin, mid, 1) : 0) + cbaParams(mid, mid, p.kernel, mid) + (p.se > 0 ? seParams(mid, hidden) : 0) + cbaParams(mid, cout, 1);
  },
  kw: (p) => ({
    torch: { k: p.kernel, expand: p.expand, se: p.se, act: p.act },
    keras: { k: p.kernel, expand: p.expand, se: p.se, act: kerasActName(p.act) },
  }),
});

residualBlock({
  type: 'inverted_residual',
  label: 'Inverted residual',
  doc: 'MobileNetV2 / V3 block: expand → depthwise → linear bottleneck, no SE by default.',
  paper: 'Sandler et al. 2018 (MobileNetV2)',
  varName: 'ir',
  outDefault: 24,
  torch: 'MBConv',
  keras: 'mbconv',
  params: [int('kernel', 'Depthwise kernel', 3, { min: 3, max: 7, step: 2 }), int('expand', 'Expansion ratio', 6, { min: 1, max: 8 }), actParam('relu6')],
  count: (cin, cout, _s, p) => {
    const mid = cin * p.expand;
    return (p.expand !== 1 ? cbaParams(cin, mid, 1) : 0) + cbaParams(mid, mid, p.kernel, mid) + cbaParams(mid, cout, 1);
  },
  kw: (p) => ({
    torch: { k: p.kernel, expand: p.expand, se: 0.0, act: p.act },
    keras: { k: p.kernel, expand: p.expand, se: 0.0, act: kerasActName(p.act) },
  }),
});

residualBlock({
  type: 'fused_mbconv',
  label: 'Fused-MBConv',
  doc: 'k×k expand conv (with stride) → 1×1 project; faster than MBConv in early stages (EfficientNetV2).',
  paper: 'Tan & Le 2021',
  varName: 'fused',
  outDefault: 48,
  torch: 'FusedMBConv',
  keras: 'fused_mbconv',
  params: [int('kernel', 'Kernel', 3, { min: 3, max: 7, step: 2 }), int('expand', 'Expansion ratio', 4, { min: 1, max: 8 }), actParam('silu')],
  count: (cin, cout, _s, p) => (p.expand === 1 ? cbaParams(cin, cout, p.kernel) : cbaParams(cin, cin * p.expand, p.kernel) + cbaParams(cin * p.expand, cout, 1)),
  kw: (p) => ({ torch: { k: p.kernel, expand: p.expand, act: p.act }, keras: { k: p.kernel, expand: p.expand, act: kerasActName(p.act) } }),
});

residualBlock({
  type: 'repvgg',
  label: 'RepVGG block',
  doc: '3×3 + 1×1 + identity branches (train time), fusable into one 3×3 conv for inference.',
  paper: 'Ding et al. 2021',
  varName: 'repvgg',
  outDefault: 64,
  torch: 'RepVGGBlock',
  keras: 'repvgg_block',
  params: [],
  count: (cin, cout, s) => cbaParams(cin, cout, 3) + cbaParams(cin, cout, 1) + (cin === cout && s === 1 ? 2 * cin : 0),
});

// ---------------------------------------------------------------------------

defineBlock({
  type: 'double_conv',
  category: 'CNN blocks',
  label: 'U-Net DoubleConv',
  doc: '(3×3 Conv → BN → ReLU) × 2, keeps the spatial size.',
  paper: 'Ronneberger et al. 2015 (U-Net)',
  varName: 'double_conv',
  inputs: [{ name: 'x', accepts: ['image'] }],
  params: [outCh(64), int('mid', 'Middle channels (0 = output)', 0)],
  inferShape: (ins, p) => ({ ...img(ins[0]), c: p.out }),
  paramCount: (ins, p) => {
    const mid = p.mid || p.out;
    return cbaParams((ins[0] as ImageShape).c, mid, 3) + cbaParams(mid, p.out, 3);
  },
  emitTorch: ({ ins, params: p }) => ({ init: call('DoubleConv', [(ins[0] as ImageShape).c, p.out], { c_mid: p.mid || undefined }), helpers: ['DoubleConv'] }),
  emitKeras: ({ params: p, name }) => ({ expr: (a) => call('double_conv', [`§${a[0]}`, p.out], { mid: p.mid || undefined, name }), helpers: ['double_conv'] }),
});

defineBlock({
  type: 'convnext_block',
  category: 'CNN blocks',
  label: 'ConvNeXt block',
  doc: 'DW 7×7 → LayerNorm → MLP ×4 → LayerScale, residual. Keeps channels and size; stack ×N per stage.',
  paper: 'Liu et al. 2022 (ConvNeXt)',
  varName: 'stage',
  inputs: [{ name: 'x', accepts: ['image'] }],
  params: [repeatP(3), float('layerScale', 'LayerScale init', 1e-6, { min: 0, max: 1 }), float('dropPath', 'Drop path', 0.0, { min: 0, max: 0.5, step: 0.05 })],
  inferShape: (ins) => img(ins[0]),
  paramCount: (ins, p) => {
    const d = (ins[0] as ImageShape).c;
    return p.n * (49 * d + d + 2 * d + d * 4 * d + 4 * d + 4 * d * d + d + d);
  },
  emitTorch: ({ ins, params: p }) => {
    const d = (ins[0] as ImageShape).c;
    const kw = { layer_scale: p.layerScale === 1e-6 ? undefined : p.layerScale, drop_path: p.dropPath || undefined };
    const one = call('ConvNeXtBlock', [d], kw);
    return { init: p.n === 1 ? one : `nn.Sequential(*[${one} for _ in range(${p.n})])`, helpers: ['ConvNeXtBlock'] };
  },
  emitKeras: ({ params: p, name }) => ({
    expr: (a) => {
      let x = a[0];
      for (let i = 0; i < p.n; i++) x = `convnext_block(${x}, ${pyFloat(p.layerScale)}, name="${name}_${i}")`;
      return x;
    },
    helpers: ['convnext_block'],
  }),
});

defineBlock({
  type: 'c2f',
  category: 'CNN blocks',
  label: 'C2f (CSP)',
  doc: 'YOLOv8 CSP block: split, N bottlenecks, concatenate all partial outputs, 1×1 fuse.',
  paper: 'Ultralytics YOLOv8 (2023)',
  varName: 'c2f',
  inputs: [{ name: 'x', accepts: ['image'] }],
  params: [outCh(128), repeatP(1), bool('shortcut', 'Residual bottlenecks', true), float('e', 'Hidden ratio', 0.5, { min: 0.25, max: 1, step: 0.25 })],
  inferShape: (ins, p) => ({ ...img(ins[0]), c: p.out }),
  paramCount: (ins, p) => {
    const c = Math.floor(p.out * p.e);
    return cbaParams((ins[0] as ImageShape).c, 2 * c, 1) + cbaParams((2 + p.n) * c, p.out, 1) + p.n * 2 * cbaParams(c, c, 3);
  },
  emitTorch: ({ ins, params: p }) => ({
    init: call('C2f', [(ins[0] as ImageShape).c, p.out], { n: p.n, shortcut: p.shortcut ? undefined : false, e: p.e === 0.5 ? undefined : p.e }),
    helpers: ['C2f'],
  }),
  emitKeras: ({ params: p, name }) => ({ expr: (a) => call('c2f', [`§${a[0]}`, p.out], { n: p.n, shortcut: p.shortcut, e: p.e, name }), helpers: ['c2f'] }),
});

defineBlock({
  type: 'sppf',
  category: 'CNN blocks',
  label: 'SPPF',
  doc: 'Spatial pyramid pooling (fast): three chained 5×5 max-pools concatenated — large receptive field cheaply.',
  paper: 'Ultralytics YOLOv5 (2020)',
  varName: 'sppf',
  inputs: [{ name: 'x', accepts: ['image'] }],
  params: [outCh(512), int('kernel', 'Pool kernel', 5, { min: 3, max: 13, step: 2 })],
  inferShape: (ins, p) => ({ ...img(ins[0]), c: p.out }),
  paramCount: (ins, p) => {
    const c = Math.floor((ins[0] as ImageShape).c / 2);
    return cbaParams((ins[0] as ImageShape).c, c, 1) + cbaParams(4 * c, p.out, 1);
  },
  emitTorch: ({ ins, params: p }) => ({ init: call('SPPF', [(ins[0] as ImageShape).c, p.out], { k: p.kernel === 5 ? undefined : p.kernel }), helpers: ['SPPF'] }),
  emitKeras: ({ params: p, name }) => ({ expr: (a) => call('sppf', [`§${a[0]}`, p.out], { k: p.kernel, name }), helpers: ['sppf'] }),
});

defineBlock({
  type: 'squeeze_excite',
  category: 'CNN blocks',
  label: 'Squeeze-Excite',
  doc: 'Channel attention gate: global pool → bottleneck MLP → sigmoid scale.',
  paper: 'Hu et al. 2018',
  varName: 'se',
  inputs: [{ name: 'x', accepts: ['image'] }],
  params: [int('reduction', 'Reduction', 16, { min: 1, max: 64 })],
  inferShape: (ins) => img(ins[0]),
  paramCount: (ins, p) => {
    const c = (ins[0] as ImageShape).c;
    return seParams(c, Math.max(8, Math.floor(c / p.reduction)));
  },
  emitTorch: ({ ins, params: p }) => {
    const c = (ins[0] as ImageShape).c;
    return { init: call('SqueezeExcite', [c, Math.max(8, Math.floor(c / p.reduction))], { act: 'relu' }), helpers: ['SqueezeExcite'] };
  },
  emitKeras: ({ ins, params: p, name }) => {
    const c = (ins[0] as ImageShape).c;
    return { expr: (a) => call('squeeze_excite', [`§${a[0]}`, Math.max(8, Math.floor(c / p.reduction))], { act: 'relu', name }), helpers: ['squeeze_excite'] };
  },
});

defineBlock({
  type: 'cbam',
  category: 'CNN blocks',
  label: 'CBAM',
  doc: 'Channel attention (avg + max pooled MLP) followed by 7×7 spatial attention.',
  paper: 'Woo et al. 2018',
  varName: 'cbam',
  inputs: [{ name: 'x', accepts: ['image'] }],
  params: [int('reduction', 'Reduction', 16, { min: 1, max: 64 }), int('kernel', 'Spatial kernel', 7, { min: 3, max: 9, step: 2 })],
  inferShape: (ins) => img(ins[0]),
  paramCount: (ins, p) => {
    const c = (ins[0] as ImageShape).c;
    return seParams(c, Math.max(8, Math.floor(c / p.reduction))) + 2 * p.kernel * p.kernel + 1;
  },
  emitTorch: ({ ins, params: p }) => ({ init: call('CBAM', [(ins[0] as ImageShape).c], { reduction: p.reduction, k: p.kernel }), helpers: ['CBAM'] }),
  emitKeras: ({ params: p, name }) => ({ expr: (a) => call('cbam', [`§${a[0]}`], { reduction: p.reduction, k: p.kernel, name }), helpers: ['cbam'] }),
});

defineBlock({
  type: 'aspp',
  category: 'CNN blocks',
  label: 'ASPP',
  doc: 'Atrous spatial pyramid pooling: parallel dilated convs + image pooling (DeepLabV3). Train with batch ≥ 2.',
  paper: 'Chen et al. 2017 (DeepLabV3)',
  varName: 'aspp',
  inputs: [{ name: 'x', accepts: ['image'] }],
  params: [outCh(256), intList('rates', 'Dilation rates', [6, 12, 18], { help: '(6, 12, 18) at output stride 16; double them at stride 8.' }), float('dropout', 'Dropout', 0.1, { min: 0, max: 0.5, step: 0.05 })],
  inferShape: (ins, p) => {
    const s = img(ins[0]);
    if (!p.rates.length) throw new ShapeError('ASPP needs at least one dilation rate.');
    return { ...s, c: p.out };
  },
  paramCount: (ins, p) => {
    const c = (ins[0] as ImageShape).c;
    return 2 * cbaParams(c, p.out, 1) + p.rates.length * cbaParams(c, p.out, 3) + cbaParams(p.out * (p.rates.length + 2), p.out, 1);
  },
  emitTorch: ({ ins, params: p }) => ({ init: call('ASPP', [(ins[0] as ImageShape).c, p.out], { rates: p.rates, dropout: p.dropout }), helpers: ['ASPP'] }),
  emitKeras: ({ params: p, name }) => ({ expr: (a) => call('aspp', [`§${a[0]}`, p.out], { rates: p.rates, dropout: p.dropout, name }), helpers: ['aspp'] }),
});
