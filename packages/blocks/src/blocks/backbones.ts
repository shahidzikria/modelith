import { call, dedent, py, pyList } from "../py";
import { defineBlock } from '../registry';
import { img } from '../shapes';
import { ShapeError, type ImageShape } from '../types';
import { bool, int, select } from './common';
import { defineKerasHelpers, defineTorchHelpers } from '../helpers/registry';

interface BackboneSpec {
  label: string;
  /** timm feature_info (all taps). */
  channels: number[];
  reductions: number[];
  params: number;
  vit?: { dim: number; patch: number };
  /** Spatial size must be a multiple of this (Swin windows). */
  multiple?: number;
  /** keras.applications class and layer taps for strides 8/16/32 (or final only). */
  keras?: { cls: string; taps?: [string, string, string]; final: string; finalChannels: number };
  kerasWhy?: string;
}

/** Verified against timm 1.0.30 `features_only=True` feature_info. */
export const BACKBONES: Record<string, BackboneSpec> = {
  resnet18: { label: 'ResNet-18', channels: [64, 64, 128, 256, 512], reductions: [2, 4, 8, 16, 32], params: 11176512, kerasWhy: 'ResNet-18 is not in keras.applications.' },
  resnet50: {
    label: 'ResNet-50',
    channels: [64, 256, 512, 1024, 2048],
    reductions: [2, 4, 8, 16, 32],
    params: 23508032,
    keras: { cls: 'ResNet50', taps: ['conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'], final: 'conv5_block3_out', finalChannels: 2048 },
  },
  efficientnet_b0: {
    label: 'EfficientNet-B0',
    channels: [16, 24, 40, 112, 320],
    reductions: [2, 4, 8, 16, 32],
    params: 3595388,
    keras: { cls: 'EfficientNetB0', taps: ['block3b_add', 'block5c_add', 'block7a_project_bn'], final: 'block7a_project_bn', finalChannels: 320 },
  },
  efficientnet_b1: { label: 'EfficientNet-B1', channels: [16, 24, 40, 112, 320], reductions: [2, 4, 8, 16, 32], params: 6101024, kerasWhy: 'Layer taps for EfficientNetB1 are not mapped yet.' },
  efficientnet_b2: { label: 'EfficientNet-B2', channels: [16, 24, 48, 120, 352], reductions: [2, 4, 8, 16, 32], params: 7202562, kerasWhy: 'Layer taps for EfficientNetB2 are not mapped yet.' },
  efficientnet_b3: { label: 'EfficientNet-B3', channels: [24, 32, 48, 136, 384], reductions: [2, 4, 8, 16, 32], params: 10103336, kerasWhy: 'Layer taps for EfficientNetB3 are not mapped yet.' },
  convnext_tiny: {
    label: 'ConvNeXt-T',
    channels: [96, 192, 384, 768],
    reductions: [4, 8, 16, 32],
    params: 27818592,
    keras: { cls: 'ConvNeXtTiny', final: '', finalChannels: 768 },
  },
  convnext_small: {
    label: 'ConvNeXt-S',
    channels: [96, 192, 384, 768],
    reductions: [4, 8, 16, 32],
    params: 49453152,
    keras: { cls: 'ConvNeXtSmall', final: '', finalChannels: 768 },
  },
  mobilenetv3_large_100: {
    label: 'MobileNetV3-L',
    channels: [16, 24, 40, 112, 960],
    reductions: [2, 4, 8, 16, 32],
    params: 2971952,
    keras: { cls: 'MobileNetV3Large', final: '', finalChannels: 960 },
  },
  regnety_016: { label: 'RegNetY-1.6GF', channels: [32, 48, 120, 336, 888], reductions: [2, 4, 8, 16, 32], params: 10313430, kerasWhy: 'RegNetY taps are not mapped for keras.applications yet.' },
  swin_tiny_patch4_window7_224: {
    label: 'Swin-T',
    channels: [96, 192, 384, 768],
    reductions: [4, 8, 16, 32],
    params: 27517818,
    multiple: 224,
    kerasWhy: 'Swin is not in keras.applications.',
  },
  vit_small_patch16_224: { label: 'ViT-S/16', channels: [384], reductions: [16], params: 21665664, vit: { dim: 384, patch: 16 }, kerasWhy: 'ViT is not in keras.applications.' },
  vit_base_patch16_224: { label: 'ViT-B/16', channels: [768], reductions: [16], params: 85798656, vit: { dim: 768, patch: 16 }, kerasWhy: 'ViT is not in keras.applications.' },
  vit_small_patch14_dinov2: { label: 'DINOv2-S/14', channels: [384], reductions: [14], params: 22056192, vit: { dim: 384, patch: 14 }, kerasWhy: 'DINOv2 is only available through timm (PyTorch).' },
};

/** Indices of the stride 8/16/32 taps. */
function pyramidIndices(spec: BackboneSpec): number[] {
  return [8, 16, 32].map((s) => spec.reductions.indexOf(s));
}

defineTorchHelpers({
  TimmBackbone: {
    imports: ['import timm'],
    code: dedent(`
      class TimmBackbone(nn.Module):
          """timm feature extractor returning one feature map or a list of maps (e.g. strides 8 / 16 / 32).

          Pretrained weights expect 3-channel images normalised with ImageNet mean / std.
          freeze_stages=n freezes the stem and the first n feature stages (requires_grad=False).
          """

          def __init__(
              self, name: str, out_indices: tuple[int, ...], pretrained: bool = False, freeze_stages: int = 0, in_chans: int = 3
          ) -> None:
              super().__init__()
              self.body = timm.create_model(
                  name, pretrained=pretrained, features_only=True, out_indices=out_indices, in_chans=in_chans
              )
              self.single = len(out_indices) == 1
              self.freeze(freeze_stages)

          def freeze(self, n: int) -> None:
              infos = self.body.feature_info.info
              if n <= 0:
                  return
              params = list(self.body.named_parameters())
              stop = len(params)
              if n < len(infos):
                  nxt = infos[n]["module"]
                  owners = (f".{nxt}.", f".{nxt.replace('.', '_')}.")
                  stop = next((i for i, (k, _) in enumerate(params) if any(o in f".{k}" for o in owners)), stop)
              for _, p in params[:stop]:
                  p.requires_grad_(False)

          def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
              feats = self.body(x)
              return feats[0] if self.single else feats
    `),
  },
  TimmViT: {
    imports: ['import timm'],
    code: dedent(`
      class TimmViT(nn.Module):
          """timm Vision Transformer returning the patch tokens (B, N, D) without [CLS] / register tokens.

          freeze_blocks=n freezes the patch embedding, position embedding and the first n blocks.
          """

          def __init__(self, name: str, img_size: tuple[int, int], pretrained: bool = False, freeze_blocks: int = 0) -> None:
              super().__init__()
              self.body = timm.create_model(name, pretrained=pretrained, num_classes=0, img_size=img_size)
              if freeze_blocks > 0:
                  for module in [self.body.patch_embed, *self.body.blocks[:freeze_blocks]]:
                      module.requires_grad_(False)
                  for p in (self.body.pos_embed, getattr(self.body, "cls_token", None)):
                      if p is not None:
                          p.requires_grad_(False)

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              return self.body.forward_features(x)[:, self.body.num_prefix_tokens :]
    `),
  },
});

defineKerasHelpers({
  keras_backbone: {
    code: dedent(`
      def keras_backbone(x, arch: str, taps: list[str], pretrained: bool, trainable: bool, name: str):
          """keras.applications model truncated at the given layers (returns a tensor or a list).

          Note: ResNet50 expects caffe-style preprocess_input; EfficientNet / ConvNeXt / MobileNetV3
          include their own rescaling and expect raw 0–255 pixels.
          """
          base = getattr(keras.applications, arch)(
              include_top=False, weights="imagenet" if pretrained else None, input_shape=tuple(x.shape[1:])
          )
          base.trainable = trainable
          outputs = [base.get_layer(t).output for t in taps] if taps else base.output
          return keras.Model(base.input, outputs, name=name)(x)
    `),
  },
});

defineBlock({
  type: 'timm_backbone',
  category: 'Backbones',
  label: 'Pretrained backbone',
  doc: 'ImageNet CNN / transformer backbone (timm in PyTorch, keras.applications in TensorFlow). Final map or a stride 8/16/32 pyramid.',
  varName: 'backbone',
  inputs: [{ name: 'image', accepts: ['image'] }],
  params: [
    select('model', 'Architecture', Object.keys(BACKBONES), 'resnet50', { help: Object.entries(BACKBONES).map(([k, v]) => `${k}: ${v.label}`).join(' · ') }),
    select('output', 'Output', ['final', 'pyramid'], 'pyramid', { help: 'final = last stage (stride 32); pyramid = strides 8/16/32 for necks and detectors.' }),
    bool('pretrained', 'Pretrained (ImageNet)', true),
    int('freeze', 'Frozen stages', 0, { min: 0, max: 12, help: 'Freeze the stem and the first N stages (N blocks for ViTs).' }),
  ],
  pretrained: (p) => !!p.pretrained,
  requires: [],
  inferShape: (ins, p) => {
    const s = img(ins[0]);
    const spec = BACKBONES[p.model];
    if (!spec) throw new ShapeError(`Unknown backbone ${p.model}.`);
    if (p.pretrained && s.c !== 3) throw new ShapeError(`Pretrained weights need 3 input channels, got ${s.c}.`, 'Set the image input to 3 channels or turn off "Pretrained".');
    const H = s.h;
    const W = s.w;
    if (spec.multiple && (H % spec.multiple || W % spec.multiple)) throw new ShapeError(`${spec.label} needs input sizes that are multiples of ${spec.multiple} (window attention).`);
    if (spec.vit) {
      if (p.output === 'pyramid') throw new ShapeError('ViT backbones produce a single-scale token sequence.', 'Use output "final", then Sequence → image and your own upsampling.');
      if (H % spec.vit.patch || W % spec.vit.patch) throw new ShapeError(`Input ${H}×${W} must be divisible by the patch size ${spec.vit.patch}.`);
      const gh = H / spec.vit.patch;
      const gw = W / spec.vit.patch;
      return { kind: 'sequence', l: gh * gw, d: spec.vit.dim, grid: { h: gh, w: gw, stride: s.stride * spec.vit.patch }, prefix: 0 };
    }
    const level = (i: number) => {
      const r = spec.reductions[i];
      return { c: spec.channels[i], h: Math.ceil(H / r), w: Math.ceil(W / r), stride: s.stride * r };
    };
    if (p.output === 'final') return { kind: 'image', ...level(spec.channels.length - 1) };
    return { kind: 'pyramid', levels: pyramidIndices(spec).map(level) };
  },
  paramCount: (_ins, p) => BACKBONES[p.model]?.params ?? 0,
  macs: (ins, _p, _out, params) => {
    const s = ins[0] as ImageShape;
    // Rough: ~0.35 × params × positions at stride 8 (within ~2× across CNN families).
    return params * Math.ceil(s.h / 8) * Math.ceil(s.w / 8) * 0.35;
  },
  emitTorch: ({ params: p, ins }) => {
    const spec = BACKBONES[p.model];
    const s = ins[0] as ImageShape;
    if (spec.vit) {
      return {
        init: call('TimmViT', [p.model, [s.h, s.w]], { pretrained: p.pretrained, freeze_blocks: p.freeze || undefined }),
        helpers: ['TimmViT'],
        imports: ['import timm'],
      };
    }
    const idx = p.output === 'final' ? [spec.channels.length - 1] : pyramidIndices(spec);
    return {
      init: call('TimmBackbone', [p.model], { out_indices: idx, pretrained: p.pretrained, freeze_stages: p.freeze || undefined, in_chans: s.c === 3 ? undefined : s.c }),
      helpers: ['TimmBackbone'],
      imports: ['import timm'],
    };
  },
  emitKeras: ({ params: p, ins, name }) => {
    const spec = BACKBONES[p.model];
    if (!spec.keras) return { unsupported: spec.kerasWhy ?? 'Not available in keras.applications.' };
    if ((ins[0] as ImageShape).c !== 3) return { unsupported: 'keras.applications backbones need 3-channel inputs.' };
    let taps: string[];
    if (p.output === 'pyramid') {
      if (!spec.keras.taps) return { unsupported: `Multi-scale taps for ${spec.label} are not mapped in Keras; use output "final".` };
      taps = spec.keras.taps;
    } else taps = spec.keras.final ? [spec.keras.final] : [];
    return {
      expr: (a) => `keras_backbone(${a[0]}, ${py(spec.keras!.cls)}, ${pyList(taps)}, pretrained=${py(!!p.pretrained)}, trainable=${py(!p.freeze)}, name=${py(name)})`,
      helpers: ['keras_backbone'],
    };
  },
});
