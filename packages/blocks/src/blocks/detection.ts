import { call, py } from '../py';
import { defineBlock } from '../registry';
import { pyr } from '../shapes';
import { ShapeError, type BlockDef, type HeadTrainSpec, type ImageShape, type InferCtx, type Params, type PyramidShape, type Shape } from '../types';
import { cbaParams, int, linearParams, transformerLayerParams } from './common';

const numClasses = int('classes', 'Classes', 80, { min: 1, max: 10000, help: 'Object classes, excluding background.' });

function detOut(p: Params): Shape {
  return { kind: 'output', task: 'detection', label: `boxes + ${p.classes} cls`, dims: [p.classes] };
}

function imageSize(ctx: InferCtx): { h: number; w: number } {
  if (!ctx.imageSize) throw new ShapeError('Detection heads need an Image input to know the image size.');
  return ctx.imageSize;
}

const sizes = (s: PyramidShape) => `(${s.levels.map((l) => `(${l.h}, ${l.w})`).join(', ')})`;
const strides = (s: PyramidShape) => `(${s.levels.map((l) => l.stride).join(', ')})`;
const chans = (s: PyramidShape) => `[${s.levels.map((l) => l.c).join(', ')}]`;

const DET_DOC = 'dict with "boxes" float (M, 4) xyxy pixels and "labels" int64 (M,) in [0, C-1]';

function requireEqualChannels(s: PyramidShape, who: string): void {
  if (new Set(s.levels.map((l) => l.c)).size > 1) {
    throw new ShapeError(`${who} needs equal channels on every level (got ${s.levels.map((l) => l.c).join('/')}).`, 'Add an FPN, PAN or BiFPN before the head.');
  }
}

// ---------------------------------------------------------------------------
// YOLO
// ---------------------------------------------------------------------------

defineBlock({
  type: 'yolo_head',
  category: 'Detectors',
  label: 'YOLO head (v8)',
  doc: 'Anchor-free decoupled head with DFL box regression, task-aligned assignment, GIoU + DFL + BCE loss, NMS decode.',
  paper: 'Ultralytics YOLOv8 (2023); Li et al. 2020 (GFL / DFL)',
  varName: 'yolo',
  inputs: [{ name: 'pyramid', accepts: ['pyramid'] }],
  params: [numClasses, int('regMax', 'DFL bins (reg_max)', 16, { min: 4, max: 32 })],
  inferShape: (ins, p, ctx) => {
    pyr(ins[0]);
    imageSize(ctx);
    return detOut(p);
  },
  paramCount: (ins, p) => {
    const s = ins[0] as PyramidShape;
    const c2 = Math.max(16, Math.floor(s.levels[0].c / 4), 4 * p.regMax);
    const c3 = Math.max(s.levels[0].c, Math.min(p.classes, 100));
    return s.levels.reduce(
      (a, l) => a + cbaParams(l.c, c2, 3) + cbaParams(c2, c2, 3) + c2 * 4 * p.regMax + 4 * p.regMax + cbaParams(l.c, c3, 3) + cbaParams(c3, c3, 3) + c3 * p.classes + p.classes,
      0,
    );
  },
  macs: (ins, _p, _o, params) => params * (ins[0] as PyramidShape).levels[0].h * (ins[0] as PyramidShape).levels[0].w * 0.6,
  emitTorch: ({ ins, params: p }) => {
    const s = ins[0] as PyramidShape;
    return { init: call('YOLOHead', [`§${chans(s)}`, p.classes], { strides: s.levels.map((l) => l.stride), reg_max: p.regMax === 16 ? undefined : p.regMax }), helpers: ['YOLOHead'] };
  },
  emitKeras: () => ({ unsupported: 'YOLO training needs a custom train_step (task-aligned assigner); only emitted for PyTorch so far.' }),
  head: (p, ins, _out, ctx): HeadTrainSpec => {
    const s = ins[0] as PyramidShape;
    const { h: H, w: W } = imageSize(ctx);
    return {
      task: 'detection',
      torchLoss: `YOLOLoss(${p.classes}, strides=${strides(s)}, feat_sizes=${sizes(s)}, reg_max=${p.regMax})`,
      torchTarget: `random_detection_target(${H}, ${W}, ${p.classes})`,
      metric: 'map',
      targetDoc: DET_DOC,
      postprocess: 'yolo_postprocess',
      postprocessArgs: `num_classes=${p.classes}, strides=${strides(s)}, feat_sizes=${sizes(s)}, reg_max=${p.regMax}, conf=0.001`,
    };
  },
});

// ---------------------------------------------------------------------------
// SSD
// ---------------------------------------------------------------------------

defineBlock({
  type: 'ssd_head',
  category: 'Detectors',
  label: 'SSD head',
  doc: 'Multibox head over a feature pyramid: 4 default boxes per cell, IoU matching, hard-negative mining, NMS decode.',
  paper: 'Liu et al. 2016 (SSD)',
  varName: 'ssd',
  inputs: [{ name: 'pyramid', accepts: ['pyramid'] }],
  params: [numClasses],
  inferShape: (ins, p, ctx) => {
    pyr(ins[0]);
    imageSize(ctx);
    return detOut(p);
  },
  paramCount: (ins, p) => (ins[0] as PyramidShape).levels.reduce((a, l) => a + 9 * l.c * 4 * (p.classes + 1) + 4 * (p.classes + 1) + 9 * l.c * 16 + 16, 0),
  emitTorch: ({ ins, params: p }) => ({ init: call('SSDHead', [`§${chans(ins[0] as PyramidShape)}`, p.classes]), helpers: ['SSDHead'] }),
  emitKeras: () => ({ unsupported: 'SSD training (matching + hard-negative mining) is only emitted for PyTorch so far.' }),
  head: (p, ins, _out, ctx): HeadTrainSpec => {
    const s = ins[0] as PyramidShape;
    const { h: H, w: W } = imageSize(ctx);
    return {
      task: 'detection',
      torchLoss: `SSDLoss(feat_sizes=${sizes(s)}, image_size=(${H}, ${W}))`,
      torchTarget: `random_detection_target(${H}, ${W}, ${p.classes})`,
      metric: 'map',
      targetDoc: DET_DOC,
      postprocess: 'ssd_postprocess',
      postprocessArgs: `feat_sizes=${sizes(s)}, image_size=(${H}, ${W})`,
    };
  },
});

// ---------------------------------------------------------------------------
// torchvision detectors (wrap the whole upstream graph as the backbone)
// ---------------------------------------------------------------------------

function torchvisionDetector(o: {
  type: string;
  label: string;
  doc: string;
  paper: string;
  varName: string;
  cls: string;
  anchors: (s: PyramidShape) => string;
  extra?: (s: PyramidShape) => Record<string, string>;
  instance?: boolean;
  approxParams: (c: number, p: Params, s: PyramidShape) => number;
}): BlockDef {
  return defineBlock({
    type: o.type,
    category: 'Detectors',
    label: o.label,
    doc: o.doc,
    paper: o.paper,
    varName: o.varName,
    wrapsGraph: true,
    inputs: [{ name: 'pyramid', accepts: ['pyramid'] }],
    params: [numClasses],
    inferShape: (ins, p, ctx) => {
      const s = pyr(ins[0]);
      requireEqualChannels(s, o.label);
      imageSize(ctx);
      return o.instance ? { kind: 'output', task: 'instance', label: `boxes + masks · ${p.classes} cls`, dims: [p.classes] } : detOut(p);
    },
    paramCount: (ins, p) => o.approxParams((ins[0] as PyramidShape).levels[0].c, p, ins[0] as PyramidShape),
    emitTorch: ({ ins, params: p, imageSize: size }) => {
      const s = ins[0] as PyramidShape;
      const { h, w } = size!;
      const names = `[${s.levels.map((_, i) => `"${i}"`).join(', ')}]`;
      const kwargs: Record<string, string> = {
        num_classes: `${p.classes + 1}`,
        ...o.extra?.(s),
        min_size: `${Math.min(h, w)}`,
        max_size: `${Math.max(h, w)}`,
      };
      const body = Object.entries({ [o.cls === 'FasterRCNN' || o.cls === 'MaskRCNN' ? 'rpn_anchor_generator' : 'anchor_generator']: o.anchors(s), ...kwargs })
        .map(([k, v]) => `${k}=${v.replace(/§NAMES/g, names)}`)
        .join(', ');
      return {
        init: `TorchvisionDetector(torchvision.models.detection.${o.cls}(GraphBackbone(), ${body}))`,
        helpers: ['TorchvisionDetector'],
        imports: ['import torchvision', 'collections', 'from torchvision.models.detection.anchor_utils import AnchorGenerator', 'from torchvision.ops import MultiScaleRoIAlign'],
      };
    },
    emitKeras: () => ({ unsupported: `${o.label} is a torchvision model; TensorFlow has no equivalent in keras.applications.` }),
    head: (p, ins, _out, ctx): HeadTrainSpec => {
      const s = ins[0] as PyramidShape;
      const { h: H, w: W } = imageSize(ctx);
      return {
        task: o.instance ? 'instance' : 'detection',
        torchLoss: '',
        torchTarget: `${o.instance ? 'random_instance_target' : 'random_detection_target'}(${H}, ${W}, ${p.classes})`,
        metric: 'map',
        internalLoss: true,
        targetDoc: o.instance ? `${DET_DOC}, plus "masks" uint8 (M, H, W)` : DET_DOC,
      };
    },
  });
}

const tupleOfTuples = (xs: number[][]) => `(${xs.map((x) => `(${x.join(', ')}${x.length === 1 ? ',' : ''})`).join(', ')}${xs.length === 1 ? ',' : ''})`;
const ratios = (s: PyramidShape, r: number[]) => tupleOfTuples(s.levels.map(() => r));
const rcnnAnchors = (s: PyramidShape) => `AnchorGenerator(sizes=${tupleOfTuples(s.levels.map((l) => [4 * l.stride]))}, aspect_ratios=${ratios(s, [0.5, 1.0, 2.0])})`;
const roiPool = (size: number) => `MultiScaleRoIAlign(featmap_names=§NAMES, output_size=${size}, sampling_ratio=2)`;

// FPN-style heads in torchvision: RPN conv + cls/bbox, TwoMLPHead (256·7·7 → 1024 → 1024), predictor.
const rcnnParams = (c: number, p: Params) => 9 * c * c + c + c * 3 + 3 + c * 12 + 12 + linearParams(c * 49, 1024) + linearParams(1024, 1024) + linearParams(1024, p.classes + 1) + linearParams(1024, 4 * (p.classes + 1));

torchvisionDetector({
  type: 'faster_rcnn',
  label: 'Faster R-CNN',
  doc: 'Two-stage detector (torchvision): RPN + RoIAlign box head on top of your graph. Needs an equal-channel pyramid (add FPN).',
  paper: 'Ren et al. 2015; Lin et al. 2017 (FPN)',
  varName: 'detector',
  cls: 'FasterRCNN',
  anchors: rcnnAnchors,
  extra: () => ({ box_roi_pool: roiPool(7) }),
  approxParams: rcnnParams,
});

torchvisionDetector({
  type: 'mask_rcnn',
  label: 'Mask R-CNN',
  doc: 'Faster R-CNN plus a per-RoI mask branch (torchvision). Instance segmentation; targets need masks.',
  paper: 'He et al. 2017',
  varName: 'detector',
  cls: 'MaskRCNN',
  anchors: rcnnAnchors,
  extra: () => ({ box_roi_pool: roiPool(7), mask_roi_pool: roiPool(14) }),
  instance: true,
  approxParams: (c, p) => rcnnParams(c, p) +4 * (9 * c * 256 + 256) + 256 * 256 * 4 + 256 + 256 * (p.classes + 1) + p.classes + 1,
});

torchvisionDetector({
  type: 'retinanet',
  label: 'RetinaNet',
  doc: 'One-stage detector with focal loss and shared subnets (torchvision); 3 scales × 3 ratios anchors per level.',
  paper: 'Lin et al. 2017 (Focal Loss)',
  varName: 'detector',
  cls: 'RetinaNet',
  anchors: (s) =>
    `AnchorGenerator(sizes=${tupleOfTuples(s.levels.map((l) => [0, 1 / 3, 2 / 3].map((e) => Math.round(4 * l.stride * 2 ** e))))}, aspect_ratios=${ratios(s, [0.5, 1.0, 2.0])})`,
  approxParams: (c, p) => 2 * 4 * (9 * c * c + c) + 9 * c * 9 * (p.classes + 1) + 9 * (p.classes + 1) + 9 * c * 36 + 36,
});

torchvisionDetector({
  type: 'fcos',
  label: 'FCOS',
  doc: 'Anchor-free one-stage detector with centerness (torchvision). One point per location per level.',
  paper: 'Tian et al. 2019',
  varName: 'detector',
  cls: 'FCOS',
  anchors: (s) => `AnchorGenerator(sizes=${tupleOfTuples(s.levels.map((l) => [l.stride]))}, aspect_ratios=${ratios(s, [1.0])})`,
  approxParams: (c, p) => 2 * 4 * (9 * c * c + c + 2 * c) + 9 * c * (p.classes + 1) + p.classes + 1 + 9 * c * 5 + 5,
});

// ---------------------------------------------------------------------------
// DETR
// ---------------------------------------------------------------------------

defineBlock({
  type: 'detr_head',
  category: 'Detectors',
  label: 'DETR head',
  doc: 'Transformer encoder–decoder over learned object queries; set prediction with Hungarian matching (scipy).',
  paper: 'Carion et al. 2020 (DETR)',
  varName: 'detr',
  requires: [],
  inputs: [{ name: 'features', accepts: ['image', 'pyramid'] }],
  params: [
    numClasses,
    int('dim', 'Model dim', 256, { min: 32, max: 1024 }),
    int('heads', 'Heads', 8, { min: 1, max: 32 }),
    int('encLayers', 'Encoder layers', 6, { min: 0, max: 12 }),
    int('decLayers', 'Decoder layers', 6, { min: 1, max: 12 }),
    int('queries', 'Object queries', 100, { min: 1, max: 1000 }),
    int('ffn', 'FFN width', 2048, { min: 64, max: 8192 }),
  ],
  inferShape: (ins, p, ctx) => {
    if (ins[0].kind !== 'image' && ins[0].kind !== 'pyramid') throw new ShapeError('DETR needs a feature map (e.g. C5) or a pyramid (uses the coarsest level).');
    if (p.dim % p.heads) throw new ShapeError(`Model dim ${p.dim} must be divisible by ${p.heads} heads.`);
    imageSize(ctx);
    return detOut(p);
  },
  paramCount: (ins, p) => {
    const c = ins[0].kind === 'pyramid' ? ins[0].levels[ins[0].levels.length - 1].c : (ins[0] as ImageShape).c;
    const d = p.dim;
    const enc = p.encLayers * transformerLayerParams(d, p.ffn);
    const dec = p.decLayers * (transformerLayerParams(d, p.ffn) + 4 * d * d + 4 * d + 2 * d);
    return c * d + d + enc + dec + 4 * d + p.queries * d + linearParams(d, p.classes + 1) + 2 * linearParams(d, d) + linearParams(d, 4);
  },
  emitTorch: ({ ins, params: p }) => {
    const c = ins[0].kind === 'pyramid' ? ins[0].levels[ins[0].levels.length - 1].c : (ins[0] as ImageShape).c;
    return {
      init: call('DETRHead', [c, p.classes], { dim: p.dim, heads: p.heads, enc_layers: p.encLayers, dec_layers: p.decLayers, queries: p.queries, ffn: p.ffn }),
      helpers: ['DETRHead'],
    };
  },
  emitKeras: () => ({ unsupported: 'DETR (Hungarian matching loss) is only emitted for PyTorch so far.' }),
  head: (p, _ins, _out, ctx): HeadTrainSpec => {
    const { h: H, w: W } = imageSize(ctx);
    return {
      task: 'detection',
      torchLoss: `DETRLoss(${p.classes}, image_size=(${H}, ${W}))`,
      torchTarget: `random_detection_target(${H}, ${W}, ${p.classes})`,
      metric: 'map',
      targetDoc: DET_DOC,
      postprocess: 'detr_postprocess',
      postprocessArgs: `image_size=(${H}, ${W}), conf=0.0`,
    };
  },
});
