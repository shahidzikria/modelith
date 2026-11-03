import { floats, select } from '../../params';
import { py } from '../../py';
import { defineStep } from '../../registry';
import type { ImageMeta, Params } from '../../types';
import { mapFloat32, pyRaw } from '../util';

/** Mean / std presets in RGB order, in [0, 1] units (as torchvision / timm publish them). */
export const NORM_PRESETS: Record<string, { mean: number[]; std: number[]; label: string }> = {
  imagenet: { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], label: 'ImageNet' },
  clip: { mean: [0.48145466, 0.4578275, 0.40821073], std: [0.26862954, 0.26130258, 0.27577711], label: 'CLIP' },
  minus1_1: { mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5], label: '[-1, 1]' },
};

const inputScale = (m: ImageMeta) => (m.dtype === 'uint8' ? 255 : m.dtype === 'uint16' ? 65535 : 1);

/** Mean / std in the image's channel order (BGR images get reversed RGB presets). */
export function meanStd(m: ImageMeta, p: Params): { mean: number[]; std: number[] } {
  const src = p.preset === 'custom' ? { mean: p.mean as number[], std: p.std as number[] } : NORM_PRESETS[p.preset];
  if (m.c === 1) {
    const avg = (v: number[]) => Math.round((v.reduce((a, b) => a + b, 0) / v.length) * 1e6) / 1e6;
    return { mean: [avg(src.mean)], std: [avg(src.std)] };
  }
  const rev = !m.space.startsWith('RGB');
  return { mean: rev ? [...src.mean].reverse() : [...src.mean], std: rev ? [...src.std].reverse() : [...src.std] };
}

defineStep({
  id: 'standardize',
  category: 'Normalise & tensor',
  label: 'Standardise (mean / std)',
  doc: 'Scale pixels to [0, 1] and subtract the per-channel mean, divide by std, giving float32. Presets: ImageNet, CLIP, [-1, 1], or per-image statistics. Preset values are in RGB order and are reversed automatically for BGR images. Pretrained backbones expect the statistics they were trained with.',
  cv2: '(img.astype(np.float32) / 255.0 - mean) / std',
  keywords: 'normalize normalise imagenet mean std zscore clip',
  params: [
    select('preset', 'Preset', [['imagenet', 'ImageNet'], ['clip', 'CLIP'], ['minus1_1', '[-1, 1]'], ['per_image', 'Per-image mean/std'], ['custom', 'Custom']], 'imagenet'),
    floats('mean', 'Mean (RGB, 0–1 units)', [0.485, 0.456, 0.406], { showIf: { key: 'preset', equals: ['custom'] } }),
    floats('std', 'Std (RGB, 0–1 units)', [0.229, 0.224, 0.225], { showIf: { key: 'preset', equals: ['custom'] } }),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1, 3], dtype: ['uint8', 'uint16', 'float32'] },
  tolerance: 1e-4,
  summary: (p) => (p.preset === 'custom' ? 'custom' : p.preset === 'per_image' ? 'per image' : NORM_PRESETS[p.preset].label),
  outMeta: (m, p) => {
    if (p.preset === 'per_image') return { ...m, dtype: 'float32', range: [-3, 3] };
    const { mean, std } = meanStd(m, p);
    const lo = Math.min(...mean.map((x, i) => (0 - x) / std[i]));
    const hi = Math.max(...mean.map((x, i) => (1 - x) / std[i]));
    return { ...m, dtype: 'float32', range: [Math.round(lo * 100) / 100, Math.round(hi * 100) / 100] };
  },
  check: (m, p) => (m.dtype === 'float32' && m.range[1] > 1.5 && p.preset !== 'per_image' ? [{ level: 'warning' as const, message: `Input is float32 with values up to ${m.range[1]}; the presets assume [0, 1] floats.` }] : []),
  applyBrowser: (src, p, ctx) => {
    const k = inputScale(ctx.in);
    const div = Math.fround(k);
    if (p.preset === 'per_image') {
      const c = src.channels();
      const scaled = mapFloat32(src, ctx, (v) => (k === 1 ? v : v / div));
      const a = scaled.data32F;
      const n = a.length / c;
      const mean = new Array(c).fill(0);
      const sq = new Array(c).fill(0);
      for (let i = 0; i < a.length; i++) mean[i % c] += a[i];
      for (let j = 0; j < c; j++) mean[j] = Math.fround(mean[j] / n);
      for (let i = 0; i < a.length; i++) sq[i % c] += (a[i] - mean[i % c]) ** 2;
      const std = sq.map((s) => Math.fround(Math.sqrt(s / n) + 1e-6));
      return mapFloat32(scaled, ctx, (v, ch) => Math.fround(v - mean[ch]) / std[ch]);
    }
    const { mean, std } = meanStd(ctx.in, p);
    const m32 = mean.map(Math.fround);
    const s32 = std.map(Math.fround);
    return mapFloat32(src, ctx, (v, ch) => {
      const x = k === 1 ? v : Math.fround(v / div);
      return Math.fround(x - m32[ch]) / s32[ch];
    });
  },
  emitPython: (p, ctx) => {
    const k = inputScale(ctx.in);
    const scale = k === 1 ? 'img = img.astype(np.float32)' : `img = img.astype(np.float32) / ${k}.0  # ${ctx.in.dtype} → [0, 1]`;
    if (p.preset === 'per_image')
      return {
        body: `${scale}\nmean = img.mean(axis=(0, 1), dtype=np.float32)\nstd = img.std(axis=(0, 1), dtype=np.float32) + np.float32(1e-6)\nimg = (img - mean) / std  # per-image, per-channel statistics`,
      };
    const { mean, std } = meanStd(ctx.in, p);
    const order = ctx.in.c === 1 ? 'gray: average of the RGB values' : `${ctx.in.space.startsWith('RGB') ? 'RGB' : 'BGR'} order`;
    const name = p.preset === 'custom' ? 'custom' : NORM_PRESETS[p.preset].label;
    const M = ctx.c('mean', pyRaw(`np.array(${py(mean)}, np.float32)`), `${name}, ${order}`);
    const S = ctx.c('std', pyRaw(`np.array(${py(std)}, np.float32)`));
    return { body: `${scale}\nimg = (img - ${M}) / ${S}` };
  },
  emitAlbumentations: (p, ctx) => {
    if (p.preset === 'per_image') return `A.Normalize(normalization="image_per_channel", p=1.0)`;
    const { mean, std } = meanStd(ctx.in, p);
    const k = inputScale(ctx.in);
    return `A.Normalize(mean=${py(mean)}, std=${py(std)}, max_pixel_value=${k}.0, p=1.0)`;
  },
  emitTorchvision: (p, ctx) => {
    if (p.preset === 'per_image' || ctx.in.c !== 3) return undefined;
    const { mean, std } = meanStd(ctx.in, p);
    return `v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=${py(mean)}, std=${py(std)})])`;
  },
  emitTf: (p, ctx) => {
    if (p.preset === 'per_image') return '(tf.cast(img, tf.float32) - tf.reduce_mean(tf.cast(img, tf.float32), axis=(0, 1))) / (tf.math.reduce_std(tf.cast(img, tf.float32), axis=(0, 1)) + 1e-6)';
    const { mean, std } = meanStd(ctx.in, p);
    const k = inputScale(ctx.in);
    return `(tf.cast(img, tf.float32) / ${k}.0 - ${py(mean)}) / ${py(std)}`;
  },
});
