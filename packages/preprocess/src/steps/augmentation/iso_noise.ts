import { range } from '../../params';
import { defineStep } from '../../registry';
import { mapPixels } from '../util';
import { normals } from '../denoising/add_noise';

defineStep({
  id: 'iso_noise',
  category: 'Augmentation',
  label: 'ISO noise',
  doc: 'Camera sensor noise at high ISO: a luminance component shared by all channels (σ = intensity × 255) plus independent colour noise per channel (σ = colour × 255). Draws intensity, colour, then the two normal fields.',
  cv2: 'img + z_lum * σ_l + z_col * σ_c',
  keywords: 'iso noise sensor camera low light grain augmentation',
  params: [range('intensity', 'Luminance σ (fraction)', [0.02, 0.06], { min: 0, max: 0.5, step: 0.01 }), range('color', 'Colour σ (fraction)', [0.01, 0.03], { min: 0, max: 0.5, step: 0.01 })],
  engine: 'both',
  kind: 'random',
  affectsGeometry: false,
  accepts: { c: [3], dtype: ['uint8'] },
  tolerance: 1,
  summary: (p) => `lum ${p.intensity[0]}–${p.intensity[1]} · col ${p.color[0]}–${p.color[1]}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { h, w } = ctx.in;
    const si = ctx.rng.uniform(p.intensity[0], p.intensity[1]) * 255;
    const sc = ctx.rng.uniform(p.color[0], p.color[1]) * 255;
    const zl = normals(ctx, h * w);
    const zc = normals(ctx, h * w * 3);
    return mapPixels(src, ctx, 'uint8', (v, k, i) => v + zl[i / 3] * si + zc[i + k] * sc);
  },
  emitPython: (p, ctx) => {
    ctx.helper('uniform');
    ctx.helper('normal');
    ctx.helper('saturate');
    return {
      body: [
        `s_lum = _uniform(rng, *${ctx.c('intensity', p.intensity)}) * 255`,
        `s_col = _uniform(rng, *${ctx.c('color', p.color)}) * 255`,
        'z_lum = _normal(rng, img.shape[:2])[..., None]  # shared by all channels',
        'z_col = _normal(rng, img.shape)  # independent per channel',
        'img = _saturate(img + z_lum * s_lum + z_col * s_col, np.uint8)',
      ].join('\n'),
    };
  },
  emitAlbumentations: (p) => `A.ISONoise(intensity=(${p.intensity[0] * 10}, ${p.intensity[1] * 10}), color_shift=(${p.color[0]}, ${p.color[1]}), p={p})  # approximate mapping`,
  testCases: [{ seed: 151 }],
});
