import { float, range } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';

defineStep({
  id: 'grid_mask',
  category: 'Augmentation',
  label: 'Grid mask',
  doc: 'GridMask (Chen et al. 2020): black out a regular grid of squares with a random period d and random offset. ratio = fraction of each period that is dropped. Removes information more evenly than Cutout. Draws: d, offset x, offset y.',
  cv2: 'img[mask] = 0 on a periodic grid',
  keywords: 'grid mask gridmask occlusion dropout augmentation',
  params: [range('period', 'Period d (px)', [24, 64], { min: 4, max: 1024, step: 1 }), float('ratio', 'Drop ratio', 0.4, { min: 0.05, max: 0.95, step: 0.05 })],
  engine: 'both',
  kind: 'random',
  affectsGeometry: false,
  summary: (p) => `d ${p.period[0]}–${p.period[1]} · ${p.ratio}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { w, h, c } = ctx.in;
    const d = ctx.rng.randint(p.period[0], p.period[1]);
    const ox = ctx.rng.randint(0, d - 1);
    const oy = ctx.rng.randint(0, d - 1);
    const l = Math.max(1, Math.floor(d * p.ratio + 0.5));
    const dst = ctx.mat();
    src.copyTo(dst);
    const a = ctx.in.dtype === 'uint8' ? dst.data : ctx.in.dtype === 'uint16' ? dst.data16U : dst.data32F;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) if ((((x + ox) % d) + d) % d < l && (((y + oy) % d) + d) % d < l) for (let k = 0; k < c; k++) a[(y * w + x) * c + k] = 0;
    return dst;
  },
  emitPython: (p, ctx) => {
    ctx.helper('randint');
    return {
      body: [
        `d = _randint(rng, *${ctx.c('period', p.period)})`,
        'ox, oy = _randint(rng, 0, d - 1), _randint(rng, 0, d - 1)',
        `l = max(1, int(d * ${ctx.c('ratio', pyRaw(pyFloat(p.ratio)))} + 0.5))  # dropped side length`,
        'h, w = img.shape[:2]',
        'yy, xx = np.mgrid[:h, :w]',
        'drop = ((xx + ox) % d < l) & ((yy + oy) % d < l)',
        'img = img.copy()',
        'img[drop] = 0',
      ].join('\n'),
    };
  },
  emitAlbumentations: (p) => `A.GridDropout(ratio=${p.ratio}, unit_size_range=(${p.period[0]}, ${p.period[1]}), p={p})`,
  testCases: [{ seed: 191 }],
});
