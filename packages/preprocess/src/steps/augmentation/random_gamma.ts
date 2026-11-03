import { range } from '../../params';
import { defineStep } from '../../registry';
import { applyLut } from '../util';
import { gammaLut } from '../intensity/gamma';

defineStep({
  id: 'random_gamma',
  category: 'Augmentation',
  label: 'Random gamma',
  doc: 'Gamma correction with γ drawn uniformly from a range (γ > 1 brightens shadows, < 1 darkens): simulates exposure and camera response differences without clipping highlights.',
  cv2: 'cv2.LUT(img, table(γ))',
  keywords: 'random gamma exposure brightness augmentation',
  params: [range('gamma', 'γ range', [0.7, 1.5], { min: 0.1, max: 5, step: 0.05 })],
  engine: 'both',
  kind: 'random',
  affectsGeometry: false,
  accepts: { dtype: ['uint8'] },
  needs: ['LUT'],
  tolerance: 1,
  summary: (p) => `γ ${p.gamma[0]}–${p.gamma[1]}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => applyLut(src, ctx, gammaLut(ctx.rng.uniform(p.gamma[0], p.gamma[1]))),
  emitPython: (p, ctx) => {
    ctx.helper('uniform');
    ctx.helper('saturate');
    return { body: `g = _uniform(rng, *${ctx.c('range', p.gamma)})\ntable = _saturate((np.arange(256) / 255.0) ** (1.0 / g) * 255.0, np.uint8)\nimg = cv2.LUT(img, table)` };
  },
  emitAlbumentations: (p) => `A.RandomGamma(gamma_limit=(${Math.round(100 / p.gamma[1])}, ${Math.round(100 / p.gamma[0])}), p={p})`,
  testCases: [{ seed: 81 }],
});
