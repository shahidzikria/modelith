import { float, int } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { onLuminance, onLuminancePy, pyRaw } from '../util';

defineStep({
  id: 'random_clahe',
  category: 'Augmentation',
  label: 'Random CLAHE',
  doc: 'CLAHE (on L of LAB for colour) with a clip limit drawn from [1, max]: random local-contrast enhancement, as in Albumentations CLAHE.',
  cv2: 'cv2.createCLAHE(clipLimit=U(1, max), tileGridSize).apply',
  keywords: 'random clahe contrast augmentation',
  params: [float('clipMax', 'Max clip limit', 4, { min: 1, max: 40, step: 0.5 }), int('tiles', 'Tile grid', 8, { min: 1, max: 64 })],
  engine: 'both',
  kind: 'random',
  affectsGeometry: false,
  accepts: { c: [1, 3], dtype: ['uint8'] },
  needs: ['CLAHE'],
  summary: (p) => `clip 1–${p.clipMax} · ${p.tiles}×${p.tiles}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const clip = ctx.rng.uniform(1, p.clipMax);
    const clahe = ctx.keep(new ctx.cv.CLAHE(clip, new ctx.cv.Size(p.tiles, p.tiles)));
    return onLuminance(src, ctx, (l, out) => clahe.apply(l, out));
  },
  emitPython: (p, ctx) => {
    ctx.helper('uniform');
    const M = ctx.c('clip_max', pyRaw(pyFloat(p.clipMax)));
    const T = ctx.c('tiles', [p.tiles, p.tiles]);
    return { body: `clahe = cv2.createCLAHE(clipLimit=_uniform(rng, 1.0, ${M}), tileGridSize=${T})\n${onLuminancePy(ctx.in, 'clahe.apply', ctx)}` };
  },
  emitAlbumentations: (p) => `A.CLAHE(clip_limit=(1, ${p.clipMax}), tile_grid_size=(${p.tiles}, ${p.tiles}), p={p})`,
  testCases: [{ seed: 111 }],
});
