import type { Project } from '../types';
import { GraphBuilder } from './builder';

export interface Template {
  id: string;
  name: string;
  task: string;
  description: string;
  build(): Project;
}

export const templates: Template[] = [];

export function defineTemplate(t: Template): Template {
  templates.push(t);
  return t;
}

export function getTemplate(id: string): Template | undefined {
  return templates.find((t) => t.id === id);
}

const t = (id: string, name: string, task: string, description: string, build: (g: GraphBuilder) => void) =>
  defineTemplate({
    id,
    name,
    task,
    description,
    build() {
      const g = new GraphBuilder();
      build(g);
      return g.project(name);
    },
  });

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

t('resnet18', 'ResNet-18 classifier', 'Classification', 'ResNet-18 from scratch: 7×7 stem, four BasicBlock stages, global pooling (11.7M params).', (g) => {
  const x = g.add('image_input', { height: 224, width: 224 });
  const f = g.chain(
    x,
    ['conv_bn_act', { out: 64, kernel: 7, stride: 2, act: 'relu' }, 'stem'],
    ['maxpool', { kernel: 3, stride: 2, padding: 1 }],
    ['resnet_basic', { out: 64, n: 2 }, 'layer1'],
    ['resnet_basic', { out: 128, n: 2, stride: 2 }, 'layer2'],
    ['resnet_basic', { out: 256, n: 2, stride: 2 }, 'layer3'],
    ['resnet_basic', { out: 512, n: 2, stride: 2 }, 'layer4'],
    ['gap'],
  );
  g.add('cls_head', { classes: 1000 }, f);
});

t('convnext-finetune', 'ConvNeXt-T fine-tune', 'Classification', 'Pretrained ConvNeXt-Tiny (timm) with a fresh classifier; low LR, label smoothing.', (g) => {
  const x = g.add('image_input', { height: 224, width: 224 });
  const b = g.add('timm_backbone', { model: 'convnext_tiny', output: 'final', pretrained: true }, x);
  g.add('cls_head', { classes: 10, dropout: 0.2 }, b);
});

t('vit-tiny', 'ViT-Tiny/16', 'Classification', '16×16 patch embedding, [CLS] token, learned positions, 12 pre-norm transformer blocks (5.7M params).', (g) => {
  const x = g.add('image_input', { height: 224, width: 224 });
  const f = g.chain(
    x,
    ['patch_embed', { patch: 16, dim: 192, cls: true }],
    ['pos_encoding', { mode: 'learned' }],
    ['transformer_encoder', { depth: 12, heads: 3, dropPath: 0.1 }],
    ['seq_pool', { mode: 'cls' }],
  );
  g.add('cls_head', { classes: 1000 }, f);
});

t('efficientnet-transfer', 'EfficientNet-B0 transfer', 'Classification', 'Pretrained EfficientNet-B0 with frozen early stages and a dropout classifier.', (g) => {
  const x = g.add('image_input', { height: 224, width: 224 });
  const b = g.add('timm_backbone', { model: 'efficientnet_b0', output: 'final', pretrained: true, freeze: 2 }, x);
  g.add('cls_head', { classes: 5, dropout: 0.2 }, b);
});

t('simple-cnn', 'Simple CNN classifier', 'Classification', 'Three Conv·BN·Act stages, global pooling and a linear classifier — the minimal starting point.', (g) => {
  const x = g.add('image_input', { height: 64, width: 64 });
  const f = g.chain(x, ['conv_bn_act', { out: 32 }], ['conv_bn_act', { out: 64, stride: 2 }], ['conv_bn_act', { out: 128, stride: 2 }], ['gap']);
  g.add('cls_head', { classes: 10 }, f);
});

// ---------------------------------------------------------------------------
// Segmentation / dense prediction
// ---------------------------------------------------------------------------

t('unet', 'U-Net', 'Segmentation', 'Classic encoder–decoder with skip connections, DoubleConv blocks and transposed-conv upsampling.', (g) => {
  const x = g.add('image_input', { height: 128, width: 128, norm: '0-1' });
  const e1 = g.add('double_conv', { out: 64 }, x, 'enc1');
  const e2 = g.chain(e1, ['maxpool'], ['double_conv', { out: 128 }, 'enc2']);
  const e3 = g.chain(e2, ['maxpool'], ['double_conv', { out: 256 }, 'enc3']);
  const e4 = g.chain(e3, ['maxpool'], ['double_conv', { out: 512 }, 'enc4']);
  const bott = g.chain(e4, ['maxpool'], ['double_conv', { out: 1024 }, 'bottleneck']);
  let d = bott;
  for (const [skip, c, k] of [[e4, 512, 4], [e3, 256, 3], [e2, 128, 2], [e1, 64, 1]] as const) {
    const up = g.add('conv_transpose', { out: c, kernel: 2, stride: 2 }, d, `up${k}`);
    const cat = g.add('concat', {}, [up, skip]);
    d = g.add('double_conv', { out: c }, cat, `dec${k}`);
  }
  g.add('seg_head', { classes: 2, hidden: 0 }, d);
});

t('deeplabv3', 'DeepLabV3 (ResNet-50 + ASPP)', 'Segmentation', 'Pretrained ResNet-50, atrous spatial pyramid pooling, bilinear upsampling to the input.', (g) => {
  const x = g.add('image_input', { height: 384, width: 384 });
  const f = g.chain(x, ['timm_backbone', { model: 'resnet50', output: 'final', pretrained: true }], ['aspp', { out: 256, rates: [6, 12, 18] }]);
  g.add('seg_head', { classes: 21, hidden: 256 }, f);
});

t('segformer-lite', 'SegFormer-lite', 'Segmentation', 'Hierarchical transformer encoder (4 stages, strides 4–32) with an all-MLP decoder fusing every stage.', (g) => {
  const x = g.add('image_input', { height: 256, width: 256 });
  const dims = [32, 64, 160, 256];
  const depths = [1, 1, 2, 2];
  const heads = [1, 2, 5, 8];
  let cur = x;
  const stages: string[] = [];
  dims.forEach((d, i) => {
    const embed = g.add('conv_bn_act', { out: d, kernel: i === 0 ? 7 : 3, stride: i === 0 ? 4 : 2, act: 'none' }, cur, `patch_merge${i + 1}`);
    const tokens = g.add('image_to_seq', {}, embed);
    const enc = g.add('transformer_encoder', { depth: depths[i], heads: heads[i], dropPath: 0.1 }, tokens, `stage${i + 1}`);
    cur = g.add('seq_to_image', {}, enc);
    stages.push(cur);
  });
  const fused = stages.map((s, i) => {
    const proj = g.add('conv2d', { out: 128, kernel: 1 }, s, `mlp${i + 1}`);
    return i === 0 ? proj : g.add('resize_like', { mode: 'bilinear' }, [proj, stages[0]]);
  });
  const cat = g.add('concat', {}, fused);
  const f = g.add('conv_bn_act', { out: 128, kernel: 1, act: 'relu' }, cat, 'fuse');
  g.add('seg_head', { classes: 19, hidden: 0 }, f);
});

t('depth', 'Monocular depth', 'Depth', 'ConvNeXt-T pyramid, FPN fusion and a dense head trained with the scale-invariant log loss (SILog).', (g) => {
  const x = g.add('image_input', { height: 256, width: 320 });
  const f = g.chain(
    x,
    ['timm_backbone', { model: 'convnext_tiny', output: 'pyramid', pretrained: true }],
    ['fpn', { out: 128 }],
    ['select_level', { index: 0 }],
    ['conv_bn_act', { out: 128, act: 'relu' }],
  );
  g.add('dense_head', { out: 1, hidden: 64, outAct: 'softplus', loss: 'silog' }, f);
});

t('forgery', 'Tampering localisation', 'Segmentation', 'EfficientNet-B0 + FPN with a binary mask head (BCE + Dice) to localise manipulated regions.', (g) => {
  const x = g.add('image_input', { height: 256, width: 256 });
  const f = g.chain(
    x,
    ['timm_backbone', { model: 'efficientnet_b0', output: 'pyramid', pretrained: true }],
    ['fpn', { out: 128 }],
    ['select_level', { index: 0 }],
    ['cbam', {}],
  );
  g.add('seg_head', { classes: 1, hidden: 64 }, f, 'mask');
});

t('multitask', 'Multi-task (cls + seg)', 'Multi-task', 'Shared ResNet-18 pyramid feeding an image classifier and an FPN segmentation head.', (g) => {
  const x = g.add('image_input', { height: 224, width: 224 });
  const p = g.add('timm_backbone', { model: 'resnet18', output: 'pyramid', pretrained: true }, x);
  const c5 = g.add('select_level', { index: 2 }, p);
  g.add('cls_head', { classes: 10 }, c5, 'classifier');
  const s = g.chain(p, ['fpn', { out: 128 }], ['select_level', { index: 0 }]);
  g.add('seg_head', { classes: 5, hidden: 128 }, s, 'segmenter');
});

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

t('yolov8s', 'YOLOv8-s style', 'Detection', 'C2f backbone + SPPF, PAN neck and the anchor-free YOLO head (TAL assigner, DFL + GIoU loss, NMS).', (g) => {
  const x = g.add('image_input', { height: 640, width: 640, norm: '0-1' });
  const p2 = g.chain(x, ['conv_bn_act', { out: 32, stride: 2 }, 'stem'], ['conv_bn_act', { out: 64, stride: 2 }], ['c2f', { out: 64, n: 1 }]);
  const p3 = g.chain(p2, ['conv_bn_act', { out: 128, stride: 2 }], ['c2f', { out: 128, n: 2 }, 'c3']);
  const p4 = g.chain(p3, ['conv_bn_act', { out: 256, stride: 2 }], ['c2f', { out: 256, n: 2 }, 'c4']);
  const p5 = g.chain(p4, ['conv_bn_act', { out: 512, stride: 2 }], ['c2f', { out: 512, n: 1 }], ['sppf', { out: 512 }, 'c5']);
  const neck = g.chain(g.add('make_pyramid', {}, [p3, p4, p5]), ['pan', { n: 1 }]);
  g.add('yolo_head', { classes: 80 }, neck);
});

t('faster-rcnn', 'Faster R-CNN R50-FPN', 'Detection', 'torchvision Faster R-CNN with a pretrained ResNet-50 + FPN graph backbone (RPN + RoIAlign).', (g) => {
  const x = g.add('image_input', { height: 512, width: 512, norm: '0-1' });
  const f = g.chain(x, ['timm_backbone', { model: 'resnet50', output: 'pyramid', pretrained: true }], ['fpn', { out: 256 }]);
  g.add('faster_rcnn', { classes: 20 }, f);
});

t('ssd-mobilenet', 'SSD-MobileNetV3', 'Detection', 'Lightweight SSD multibox head on a pretrained MobileNetV3-Large pyramid.', (g) => {
  const x = g.add('image_input', { height: 320, width: 320, norm: '0-1' });
  const f = g.add('timm_backbone', { model: 'mobilenetv3_large_100', output: 'pyramid', pretrained: true }, x);
  g.add('ssd_head', { classes: 20 }, f);
});

t('retinanet', 'RetinaNet', 'Detection', 'torchvision RetinaNet (focal loss, 9 anchors per location) on ResNet-50 + FPN.', (g) => {
  const x = g.add('image_input', { height: 512, width: 512, norm: '0-1' });
  const f = g.chain(x, ['timm_backbone', { model: 'resnet50', output: 'pyramid', pretrained: true }], ['fpn', { out: 256 }]);
  g.add('retinanet', { classes: 20 }, f);
});

t('detr', 'DETR-R50', 'Detection', 'ResNet-50 C5 features, transformer encoder–decoder with 100 queries, Hungarian-matching loss.', (g) => {
  const x = g.add('image_input', { height: 512, width: 512, norm: '0-1' });
  const f = g.add('timm_backbone', { model: 'resnet50', output: 'final', pretrained: true }, x);
  g.add('detr_head', { classes: 80 }, f);
});

t('mask-rcnn', 'Mask R-CNN', 'Instance seg.', 'torchvision Mask R-CNN (boxes + per-instance masks) on ResNet-50 + FPN.', (g) => {
  const x = g.add('image_input', { height: 512, width: 512, norm: '0-1' });
  const f = g.chain(x, ['timm_backbone', { model: 'resnet50', output: 'pyramid', pretrained: true }], ['fpn', { out: 256 }]);
  g.add('mask_rcnn', { classes: 20 }, f);
});

// ---------------------------------------------------------------------------
// Text / multimodal / tabular
// ---------------------------------------------------------------------------

t('bert-classifier', 'BERT-style text classifier', 'Text', 'Token embedding, learned positions, 4-layer transformer encoder, [CLS] pooling, sentiment head.', (g) => {
  const x = g.add('text_input', { seqLen: 128, vocab: 30522 });
  const f = g.chain(
    x,
    ['embedding', { dim: 256 }],
    ['pos_encoding', { mode: 'learned' }],
    ['dropout', { p: 0.1 }],
    ['transformer_encoder', { depth: 4, heads: 4, dropout: 0.1, dropPath: 0 }],
    ['seq_pool', { mode: 'cls' }],
  );
  g.add('cls_head', { classes: 2, hidden: 256, dropout: 0.1 }, f);
});

t('bilstm-ner', 'BiLSTM-CRF NER tagger', 'Text', 'Embeddings, 2-layer bidirectional LSTM and a CRF token-classification head (BIO tags).', (g) => {
  const x = g.add('text_input', { seqLen: 64, vocab: 20000, tokenizer: 'bert-base-cased' });
  const f = g.chain(x, ['embedding', { dim: 128 }], ['dropout', { p: 0.3 }], ['rnn', { cell: 'lstm', hidden: 128, layers: 2, bidirectional: true }]);
  g.add('token_head', { classes: 9, crf: true, dropout: 0.3 }, f);
});

t('clip', 'CLIP-style dual encoder', 'Multimodal', 'ResNet-50 image tower + transformer text tower projected into a shared space with a contrastive loss.', (g) => {
  const img = g.add('image_input', { height: 224, width: 224 });
  const txt = g.add('text_input', { name: 'text', seqLen: 77, vocab: 49408, tokenizer: 'openai/clip-vit-base-patch32' });
  const iv = g.chain(img, ['timm_backbone', { model: 'resnet50', output: 'final', pretrained: true }], ['gap']);
  const tv = g.chain(txt, ['embedding', { dim: 256 }], ['pos_encoding', { mode: 'learned' }], ['transformer_encoder', { depth: 4, heads: 4, dropPath: 0 }], ['seq_pool', { mode: 'mean' }]);
  g.add('contrastive_head', { dim: 256 }, [iv, tv]);
});

t('tabular-mlp', 'Tabular MLP', 'Regression', 'Two hidden layers with LayerNorm and dropout for tabular regression.', (g) => {
  const x = g.add('tabular_input', { features: 32 });
  const f = g.chain(x, ['linear', { out: 256 }], ['layernorm'], ['activation', { act: 'gelu' }], ['dropout', { p: 0.1 }], ['linear', { out: 128 }], ['activation', { act: 'gelu' }]);
  g.add('regression_head', { out: 1 }, f);
});
