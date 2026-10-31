import { call, py, pyFloat } from "../py";
import { defineBlock } from '../registry';
import { featureDim, img, seq, vec } from '../shapes';
import { ShapeError, type Shape } from '../types';
import { bool, cbaParams, float, int, linearParams, select } from './common';

function poolMode(s: Shape): 'none' | 'spatial' | 'tokens' {
  if (s.kind === 'image') return 'spatial';
  if (s.kind === 'sequence') return 'tokens';
  return 'none';
}

function requireSize(size: { h: number; w: number } | undefined): { h: number; w: number } {
  if (!size) throw new ShapeError('Dense heads upsample to the image input size; add an Image input.');
  return size;
}

const numClasses = (def = 10) => int('classes', 'Classes', def, { min: 1, max: 100000 });

// ---------------------------------------------------------------------------

defineBlock({
  type: 'cls_head',
  category: 'Heads',
  label: 'Classification head',
  doc: 'Class logits. Feature maps are average-pooled and sequences mean-pooled automatically. Multi-label uses sigmoid + BCE.',
  varName: 'classifier',
  inputs: [{ name: 'features', accepts: ['vector', 'image', 'sequence'] }],
  params: [
    numClasses(10),
    bool('multilabel', 'Multi-label (sigmoid)', false),
    int('hidden', 'Hidden MLP width (0 = none)', 0),
    float('dropout', 'Dropout', 0.0, { min: 0, max: 0.9, step: 0.05 }),
    float('labelSmoothing', 'Label smoothing', 0.1, { min: 0, max: 0.5, step: 0.05, help: 'Szegedy et al. 2016 (Inception-v3). Single-label only.' }),
  ],
  inferShape: (ins, p) => {
    featureDim(ins[0]);
    return { kind: 'output', task: p.multilabel ? 'multilabel' : 'classification', label: `${p.classes} ${p.multilabel ? 'labels' : 'classes'}`, dims: [p.classes] };
  },
  paramCount: (ins, p) => {
    const c = featureDim(ins[0]);
    return p.hidden > 0 ? linearParams(c, p.hidden) + linearParams(p.hidden, p.classes) : linearParams(c, p.classes);
  },
  macs: (_i, _p, _o, params) => params,
  emitTorch: ({ params: p, ins }) => ({
    init: call("ClassificationHead", [featureDim(ins[0]), p.classes], { hidden: p.hidden || undefined, dropout: p.dropout || undefined, pool: poolMode(ins[0]) === "none" ? undefined : poolMode(ins[0]) }),
    helpers: ['ClassificationHead'],
  }),
  emitKeras: ({ params: p, ins, name }) => ({
    expr: (a) => {
      let x = a[0];
      const pool = poolMode(ins[0]);
      if (pool === 'spatial') x = `layers.GlobalAveragePooling2D(name="${name}_pool")(${x})`;
      if (pool === 'tokens') x = `layers.GlobalAveragePooling1D(name="${name}_pool")(${x})`;
      if (p.hidden > 0) x = `layers.Dense(${p.hidden}, activation="gelu", name="${name}_hidden")(${x})`;
      if (p.dropout > 0) x = `layers.Dropout(${pyFloat(p.dropout)}, name="${name}_drop")(${x})`;
      return `layers.Dense(${p.classes}, name=${py(name)})(${x})`;
    },
  }),
  head: (p) =>
    p.multilabel
      ? {
          task: 'multilabel',
          torchLoss: 'nn.BCEWithLogitsLoss()',
          torchTarget: `torch.randint(0, 2, (${p.classes},)).float()`,
          kerasLoss: 'keras.losses.BinaryCrossentropy(from_logits=True)',
          kerasTarget: `np.random.randint(0, 2, size=(n, ${p.classes})).astype("float32")`,
          kerasMetrics: '[keras.metrics.BinaryAccuracy(threshold=0.0)]',
          metric: 'multilabel',
          targetDoc: `float tensor of shape (${p.classes},) with 1.0 for every present label`,
        }
      : {
          task: 'classification',
          torchLoss: `nn.CrossEntropyLoss(label_smoothing=${pyFloat(p.labelSmoothing)})`,
          torchTarget: `torch.randint(0, ${p.classes}, ())`,
          kerasLoss: 'keras.losses.SparseCategoricalCrossentropy(from_logits=True)',
          kerasTarget: `np.random.randint(0, ${p.classes}, size=(n,))`,
          kerasMetrics: '["accuracy"]',
          metric: 'accuracy',
          targetDoc: `int64 class index in [0, ${p.classes - 1}]`,
        },
});

defineBlock({
  type: 'token_head',
  category: 'Heads',
  label: 'Token classification head',
  doc: 'Per-token logits for NER / POS tagging, with an optional CRF loss + Viterbi decoding.',
  paper: 'Huang et al. 2015 (BiLSTM-CRF)',
  varName: 'tagger',
  inputs: [{ name: 'tokens', accepts: ['sequence'] }],
  params: [numClasses(9), float('dropout', 'Dropout', 0.1, { min: 0, max: 0.9, step: 0.05 }), bool('crf', 'CRF layer', false)],
  inferShape: (ins, p) => {
    const s = seq(ins[0]);
    return { kind: 'output', task: 'token', label: `${s.l}×${p.classes} tags`, dims: [s.l, p.classes] };
  },
  paramCount: (ins, p) => linearParams(featureDim(ins[0]), p.classes),
  emitTorch: ({ params: p, ins }) => ({
    init: `nn.Sequential(nn.Dropout(${pyFloat(p.dropout)}), nn.Linear(${featureDim(ins[0])}, ${p.classes}))`,
  }),
  emitKeras: ({ params: p, name }) =>
    p.crf
      ? { unsupported: 'The CRF layer is only emitted for PyTorch.' }
      : { expr: (a) => `layers.Dense(${p.classes}, name=${py(name)})(layers.Dropout(${pyFloat(p.dropout)}, name="${name}_drop")(${a[0]}))` },
  head: (p, ins) => {
    const s = ins[0] as Extract<Shape, { kind: 'sequence' }>;
    return {
      task: 'token',
      torchLoss: p.crf ? `CRF(${p.classes})` : 'TokenCrossEntropy()',
      torchTarget: `torch.randint(0, ${p.classes}, (${s.l},))`,
      kerasLoss: 'keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=-100)',
      kerasTarget: `np.random.randint(0, ${p.classes}, size=(n, ${s.l}))`,
      kerasMetrics: '["accuracy"]',
      metric: 'accuracy',
      targetDoc: `int64 tensor (${s.l},) of tag ids in [0, ${p.classes - 1}], -100 for padding / ignored sub-words`,
      postprocess: p.crf ? 'crf_decode' : undefined,
    };
  },
});

defineBlock({
  type: 'seg_head',
  category: 'Heads',
  label: 'Semantic segmentation head',
  doc: 'Per-pixel class logits upsampled to the input resolution. 1 class = binary mask (BCE + Dice).',
  varName: 'seg_head',
  inputs: [{ name: 'features', accepts: ['image'] }],
  params: [numClasses(21), int('hidden', 'Hidden channels (0 = none)', 256), float('diceWeight', 'Dice loss weight', 1.0, { min: 0, max: 10, step: 0.1 })],
  inferShape: (ins, p, ctx) => {
    img(ins[0]);
    const { h, w } = requireSize(ctx.imageSize);
    return { kind: 'output', task: 'segmentation', label: `${p.classes}×${h}×${w} mask`, dims: [p.classes, h, w] };
  },
  paramCount: (ins, p) => {
    const c = featureDim(ins[0]);
    return p.hidden > 0 ? cbaParams(c, p.hidden, 3) + p.hidden * p.classes + p.classes : c * p.classes + p.classes;
  },
  macs: (ins, _p, _o, params) => params * (ins[0].kind === 'image' ? ins[0].h * ins[0].w : 1),
  emitTorch: ({ params: p, ins, out }) => {
    const [, h, w] = (out as Extract<Shape, { kind: 'output' }>).dims;
    return { init: `SegmentationHead(${featureDim(ins[0])}, ${p.classes}, hidden=${p.hidden}, size=(${h}, ${w}))`, helpers: ['SegmentationHead'] };
  },
  emitKeras: ({ params: p, out, name }) => {
    const [, h, w] = (out as Extract<Shape, { kind: 'output' }>).dims;
    return {
      expr: (a) => {
        let x = a[0];
        if (p.hidden > 0) x = `conv_bn_act(${x}, ${p.hidden}, 3, act="relu", name="${name}_conv")`;
        x = `layers.Conv2D(${p.classes}, 1, name="${name}_cls")(${x})`;
        return `layers.Resizing(${h}, ${w}, interpolation="bilinear", name=${py(name)})(${x})`;
      },
      helpers: p.hidden > 0 ? ['conv_bn_act'] : [],
    };
  },
  head: (p, _ins, out) => {
    const [, h, w] = (out as Extract<Shape, { kind: 'output' }>).dims;
    const binary = p.classes === 1;
    return {
      task: 'segmentation',
      torchLoss: `SegmentationLoss(${p.classes}, dice_weight=${pyFloat(p.diceWeight)})`,
      torchTarget: binary ? `torch.randint(0, 2, (1, ${h}, ${w})).float()` : `torch.randint(0, ${p.classes}, (${h}, ${w}))`,
      kerasLoss: binary ? 'keras.losses.BinaryCrossentropy(from_logits=True)' : 'keras.losses.SparseCategoricalCrossentropy(from_logits=True)',
      kerasTarget: binary ? `np.random.randint(0, 2, size=(n, ${h}, ${w}, 1)).astype("float32")` : `np.random.randint(0, ${p.classes}, size=(n, ${h}, ${w}))`,
      kerasMetrics: binary ? '[keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.0)]' : `[keras.metrics.MeanIoU(num_classes=${p.classes}, sparse_y_pred=False)]`,
      metric: 'miou',
      targetDoc: binary ? `float mask (1, ${h}, ${w}) with 1.0 = foreground` : `int64 mask (${h}, ${w}) with class ids in [0, ${p.classes - 1}]`,
    };
  },
});

defineBlock({
  type: 'dense_head',
  category: 'Heads',
  label: 'Dense regression head',
  doc: 'Per-pixel regression at input resolution: depth (SILog), restoration / super-resolution / reconstruction (L1, MSE).',
  varName: 'dense_head',
  inputs: [{ name: 'features', accepts: ['image'] }],
  params: [
    int('out', 'Output channels', 1, { min: 1, max: 64, help: '1 for depth, 3 for RGB restoration.' }),
    int('hidden', 'Hidden channels (0 = none)', 64),
    select('outAct', 'Output activation', ['none', 'sigmoid', 'relu', 'softplus'], 'softplus', { help: 'softplus/relu keep depth positive; sigmoid for images in [0,1].' }),
    select('loss', 'Loss', ['silog', 'l1', 'mse'], 'silog'),
  ],
  inferShape: (ins, p, ctx) => {
    img(ins[0]);
    const { h, w } = requireSize(ctx.imageSize);
    return { kind: 'output', task: 'dense', label: `${p.out}×${h}×${w} map`, dims: [p.out, h, w] };
  },
  paramCount: (ins, p) => {
    const c = featureDim(ins[0]);
    return p.hidden > 0 ? cbaParams(c, p.hidden, 3) + p.hidden * p.out + p.out : c * p.out + p.out;
  },
  macs: (ins, _p, _o, params) => params * (ins[0].kind === 'image' ? ins[0].h * ins[0].w : 1),
  emitTorch: ({ params: p, ins, out }) => {
    const [, h, w] = (out as Extract<Shape, { kind: 'output' }>).dims;
    return { init: `DenseHead(${featureDim(ins[0])}, ${p.out}, hidden=${p.hidden}, size=(${h}, ${w}), out_act=${py(p.outAct)})`, helpers: ['DenseHead'] };
  },
  emitKeras: ({ params: p, out, name }) => {
    const [, h, w] = (out as Extract<Shape, { kind: 'output' }>).dims;
    return {
      expr: (a) => {
        let x = a[0];
        if (p.hidden > 0) x = `conv_bn_act(${x}, ${p.hidden}, 3, act="relu", name="${name}_conv")`;
        x = `layers.Conv2D(${p.out}, 1, name="${name}_proj")(${x})`;
        x = `layers.Resizing(${h}, ${w}, interpolation="bilinear", name="${name}_resize")(${x})`;
        return `layers.Activation(${py(p.outAct === 'none' ? 'linear' : p.outAct)}, name=${py(name)})(${x})`;
      },
      helpers: p.hidden > 0 ? ['conv_bn_act'] : [],
    };
  },
  head: (p, _ins, out) => {
    const [c, h, w] = (out as Extract<Shape, { kind: 'output' }>).dims;
    const depth = p.loss === 'silog';
    return {
      task: 'dense',
      torchLoss: depth ? 'SILogLoss()' : p.loss === 'l1' ? 'nn.L1Loss()' : 'nn.MSELoss()',
      torchTarget: depth ? `torch.rand(${c}, ${h}, ${w}) * 10.0 + 0.1` : `torch.rand(${c}, ${h}, ${w})`,
      kerasLoss: depth ? 'silog_loss' : p.loss === 'l1' ? 'keras.losses.MeanAbsoluteError()' : 'keras.losses.MeanSquaredError()',
      kerasTarget: depth ? `np.random.rand(n, ${h}, ${w}, ${c}).astype("float32") * 10 + 0.1` : `np.random.rand(n, ${h}, ${w}, ${c}).astype("float32")`,
      kerasMetrics: '[keras.metrics.RootMeanSquaredError()]',
      metric: depth ? 'depth' : 'rmse',
      targetDoc: depth ? `float depth map (${c}, ${h}, ${w}) in metres; 0 = invalid pixel` : `float target image (${c}, ${h}, ${w})`,
    };
  },
});

defineBlock({
  type: 'embedding_head',
  category: 'Heads',
  label: 'Embedding head',
  doc: 'L2-normalised embedding trained with ArcFace (metric learning, re-identification, retrieval).',
  paper: 'Deng et al. 2019 (ArcFace)',
  varName: 'embed_head',
  inputs: [{ name: 'features', accepts: ['vector', 'image', 'sequence'] }],
  params: [int('dim', 'Embedding size', 128, { min: 2 }), numClasses(100), float('margin', 'Angular margin', 0.5, { min: 0, max: 1, step: 0.05 }), float('scale', 'Scale s', 30, { min: 1, max: 128 })],
  inferShape: (ins, p) => {
    featureDim(ins[0]);
    return { kind: 'output', task: 'embedding', label: `${p.dim}-d embedding`, dims: [p.dim] };
  },
  paramCount: (ins, p) => featureDim(ins[0]) * p.dim + 2 * p.dim,
  emitTorch: ({ params: p, ins }) => ({
    init: `EmbeddingHead(${featureDim(ins[0])}, ${p.dim}${poolMode(ins[0]) === 'none' ? '' : `, pool=${py(poolMode(ins[0]))}`})`,
    helpers: ['EmbeddingHead'],
  }),
  emitKeras: () => ({ unsupported: 'ArcFace training is only emitted for PyTorch.' }),
  head: (p) => ({
    task: 'embedding',
    torchLoss: `ArcFaceLoss(${p.dim}, ${p.classes}, s=${pyFloat(p.scale)}, m=${pyFloat(p.margin)})`,
    torchTarget: `torch.randint(0, ${p.classes}, ())`,
    metric: 'loss',
    targetDoc: `int64 identity / class index in [0, ${p.classes - 1}]`,
  }),
});

defineBlock({
  type: 'regression_head',
  category: 'Heads',
  label: 'Regression head',
  doc: 'Scalar or multi-output regression (tabular targets, scores, keypoints).',
  varName: 'regressor',
  inputs: [{ name: 'features', accepts: ['vector', 'image', 'sequence'] }],
  params: [int('out', 'Outputs', 1, { min: 1 }), int('hidden', 'Hidden MLP width (0 = none)', 0), select('loss', 'Loss', ['mse', 'huber'], 'mse')],
  inferShape: (ins, p) => {
    featureDim(ins[0]);
    return { kind: 'output', task: 'regression', label: `${p.out} value${p.out > 1 ? 's' : ''}`, dims: [p.out] };
  },
  paramCount: (ins, p) => {
    const c = featureDim(ins[0]);
    return p.hidden > 0 ? linearParams(c, p.hidden) + linearParams(p.hidden, p.out) : linearParams(c, p.out);
  },
  emitTorch: ({ params: p, ins }) => ({
    init: call("ClassificationHead", [featureDim(ins[0]), p.out], { hidden: p.hidden || undefined, pool: poolMode(ins[0]) === "none" ? undefined : poolMode(ins[0]) }),
    helpers: ['ClassificationHead'],
  }),
  emitKeras: ({ params: p, ins, name }) => ({
    expr: (a) => {
      let x = a[0];
      const pool = poolMode(ins[0]);
      if (pool === 'spatial') x = `layers.GlobalAveragePooling2D(name="${name}_pool")(${x})`;
      if (pool === 'tokens') x = `layers.GlobalAveragePooling1D(name="${name}_pool")(${x})`;
      if (p.hidden > 0) x = `layers.Dense(${p.hidden}, activation="gelu", name="${name}_hidden")(${x})`;
      return `layers.Dense(${p.out}, name=${py(name)})(${x})`;
    },
  }),
  head: (p) => ({
    task: 'regression',
    torchLoss: p.loss === 'huber' ? 'nn.HuberLoss()' : 'nn.MSELoss()',
    torchTarget: `torch.randn(${p.out})`,
    kerasLoss: p.loss === 'huber' ? 'keras.losses.Huber()' : 'keras.losses.MeanSquaredError()',
    kerasTarget: `np.random.randn(n, ${p.out}).astype("float32")`,
    kerasMetrics: '[keras.metrics.RootMeanSquaredError()]',
    metric: 'rmse',
    targetDoc: `float tensor (${p.out},) of (standardised) targets`,
  }),
});

defineBlock({
  type: 'contrastive_head',
  category: 'Heads',
  label: 'Contrastive (CLIP) head',
  doc: 'Projects two embeddings (e.g. image and text) into a shared space; symmetric InfoNCE loss over the batch.',
  paper: 'Radford et al. 2021 (CLIP)',
  varName: 'clip_head',
  inputs: [
    { name: 'a', accepts: ['vector'] },
    { name: 'b', accepts: ['vector'] },
  ],
  params: [int('dim', 'Shared embedding size', 256, { min: 2 })],
  inferShape: (ins) => {
    vec(ins[0], 'First input');
    vec(ins[1], 'Second input');
    return { kind: 'output', task: 'contrastive', label: 'B×B similarities', dims: [] };
  },
  paramCount: (ins, p) => (featureDim(ins[0]) + featureDim(ins[1])) * p.dim + 1,
  emitTorch: ({ params: p, ins }) => ({
    init: `ContrastiveHead(${featureDim(ins[0])}, ${featureDim(ins[1])}, ${p.dim})`,
    helpers: ['ContrastiveHead'],
  }),
  emitKeras: () => ({ unsupported: 'The CLIP loss needs a custom train_step; only emitted for PyTorch.' }),
  head: () => ({
    task: 'contrastive',
    torchLoss: 'CLIPLoss()',
    torchTarget: 'torch.zeros((), dtype=torch.long)',
    metric: 'loss',
    targetDoc: 'no target needed: sample i of both inputs is a matching pair (the dataset yields a dummy 0)',
  }),
});
