import { range } from '../../params';
import { defineStep } from '../../registry';
import { mapPixels } from '../util';

defineStep({
  id: 'solarize',
  category: 'Augmentation',
  label: 'Solarise',
  doc: 'Invert all values above a threshold drawn from the range (255 − x for x ≥ t). A strong photometric augmentation used in self-supervised learning (BYOL, DINO).',
  cv2: 'np.where(img >= t, 255 - img, img)',
  keywords: 'solarize solarise invert threshold byol dino augmentation',
  params: [range('thresh', 'Threshold', [128, 200], { min: 0, max: 255, step: 1 })],
  engine: 'both',
  kind: 'random',
  affectsGeometry: false,
  accepts: { dtype: ['uint8'] },
  summary: (p) => `t ${p.thresh[0]}–${p.thresh[1]}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const t = ctx.rng.uniform(p.thresh[0], p.thresh[1]);
    return mapPixels(src, ctx, 'uint8', (v) => (v >= t ? 255 - v : v));
  },
  emitPython: (p, ctx) => {
    ctx.helper('uniform');
    return { body: `t = _uniform(rng, *${ctx.c('threshold', p.thresh)})\nimg = np.where(img >= t, 255 - img, img).astype(np.uint8)` };
  },
  emitTorchvision: (p) => `v2.RandomSolarize(threshold=${p.thresh[0]}, p={p})`,
  testCases: [{ seed: 131 }],
});
