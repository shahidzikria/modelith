import { float } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';

defineStep({
  id: 'unsharp_mask',
  category: 'Intensity',
  label: 'Unsharp mask (sharpen)',
  doc: 'Sharpen by adding back the detail removed by a Gaussian blur: out = (1 + amount)·img − amount·blur(img, σ). σ sets which detail size is boosted (1–3 px typical), amount how much (0.5–2). Sharpening also amplifies noise; denoise first.',
  cv2: 'cv2.addWeighted(img, 1 + amount, cv2.GaussianBlur(img, (0, 0), sigma), -amount, 0)',
  keywords: 'sharpen unsharp mask detail crisp blur addWeighted',
  params: [float('sigma', 'σ (px)', 2, { min: 0.3, max: 20, step: 0.1 }), float('amount', 'Amount', 1, { min: 0, max: 5, step: 0.05 })],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  needs: ['GaussianBlur', 'addWeighted'],
  tolerance: 1,
  summary: (p) => `σ ${p.sigma} · ×${p.amount}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const blur = ctx.mat();
    cv.GaussianBlur(src, blur, new cv.Size(0, 0), p.sigma, p.sigma, cv.BORDER_DEFAULT);
    const dst = ctx.mat();
    cv.addWeighted(src, 1 + p.amount, blur, -p.amount, 0, dst, -1);
    return dst;
  },
  emitPython: (p, ctx) => {
    const s = ctx.c('sigma', pyRaw(pyFloat(p.sigma)));
    const a = ctx.c('amount', pyRaw(pyFloat(p.amount)));
    return { body: `blur = cv2.GaussianBlur(img, (0, 0), ${s})\nimg = cv2.addWeighted(img, 1.0 + ${a}, blur, -${a}, 0)` };
  },
  emitAlbumentations: (p) => `A.UnsharpMask(sigma_limit=(${pyFloat(p.sigma)}, ${pyFloat(p.sigma)}), alpha=(1.0, 1.0), p=1.0)  # approximate`,
  testCases: [{}, { input: 'float', params: { sigma: 1, amount: 2 }, tolerance: 1e-4 }],
});
