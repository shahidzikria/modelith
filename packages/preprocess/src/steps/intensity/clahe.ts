import { float, int } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { onLuminance, onLuminancePy, pyRaw } from '../util';

defineStep({
  id: 'clahe',
  category: 'Intensity',
  label: 'CLAHE',
  doc: 'Contrast-limited adaptive histogram equalisation: equalises contrast in tiles, with a clip limit that stops noise being amplified. Colour images are processed on the L channel of LAB so colours do not shift. Typical: clip 2–4, 8×8 tiles; use larger tiles for smoother results.',
  cv2: 'cv2.createCLAHE(clipLimit, tileGridSize).apply',
  keywords: 'contrast equalize equalise histogram adaptive local low light medical',
  params: [
    float('clip', 'Clip limit', 2, { min: 0.1, max: 40, step: 0.1, help: 'Higher = more contrast (and more noise). 1 ≈ no change.' }),
    int('tiles', 'Tile grid', 8, { min: 1, max: 64, help: 'Number of tiles per side (8 → 8×8 tiles).' }),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1, 3], dtype: ['uint8', 'uint16'] },
  needs: ['CLAHE'],
  summary: (p) => `clip ${pyFloat(p.clip)} · ${p.tiles}×${p.tiles}`,
  outMeta: (m) => m,
  check: (m, p) => {
    if (m.dtype === 'uint16' && m.c === 3)
      return [{ level: 'error' as const, message: 'CLAHE on a 16-bit colour image is not supported (LAB conversion needs uint8 or float32). Convert to GRAY or to uint8 first.' }];
    if (p.tiles > Math.min(m.h, m.w)) return [{ level: 'error' as const, message: `${p.tiles}×${p.tiles} tiles do not fit a ${m.w}×${m.h} image.` }];
    return [];
  },
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const clahe = ctx.keep(new cv.CLAHE(p.clip, new cv.Size(p.tiles, p.tiles)));
    return onLuminance(src, ctx, (l, out) => clahe.apply(l, out));
  },
  emitPython: (p, ctx) => {
    const clip = ctx.c('clip', pyRaw(pyFloat(p.clip)));
    const tiles = ctx.c('tiles', [p.tiles, p.tiles]);
    return {
      body: `clahe = cv2.createCLAHE(clipLimit=${clip}, tileGridSize=${tiles})\n${onLuminancePy(ctx.in, 'clahe.apply', ctx)}`,
      doc: ctx.in.c === 3 ? 'Colour image: equalised on the L channel of LAB.' : undefined,
    };
  },
  emitAlbumentations: (p, ctx) => (ctx.in.dtype === 'uint8' ? `A.CLAHE(clip_limit=(${pyFloat(p.clip)}, ${pyFloat(p.clip)}), tile_grid_size=(${p.tiles}, ${p.tiles}), p=1.0)` : undefined),
});
