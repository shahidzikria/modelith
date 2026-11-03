import { float, select } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';

defineStep({
  id: 'dog',
  category: 'Edges',
  label: 'Difference of Gaussians',
  doc: 'Band-pass filter: blur(σ1) − blur(σ2) with σ2 > σ1 keeps structures between the two scales (blob / edge enhancement, a cheap Laplacian-of-Gaussian). σ2 ≈ 1.6·σ1 approximates LoG. Output is signed float32, or min-max scaled to uint8.',
  cv2: 'cv2.GaussianBlur(f, (0, 0), s1) - cv2.GaussianBlur(f, (0, 0), s2)',
  keywords: 'difference of gaussians dog band pass log blob edge enhance',
  params: [float('s1', 'σ1', 1, { min: 0.3, max: 20, step: 0.1 }), float('s2', 'σ2', 2, { min: 0.3, max: 40, step: 0.1 }), select('out', 'Output', [['float32', 'float32 (signed)'], ['uint8', 'uint8 (min-max scaled)']], 'uint8')],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  needs: ['GaussianBlur', 'subtract', 'normalize'],
  tolerance: 1,
  summary: (p) => `σ ${p.s1} − ${p.s2}`,
  outMeta: (m, p) => (p.out === 'uint8' ? { ...m, dtype: 'uint8', range: [0, 255] } : { ...m, dtype: 'float32', range: [-m.range[1], m.range[1]] }),
  check: (_m, p) => (p.s1 >= p.s2 ? [{ level: 'warning' as const, message: 'σ1 should be smaller than σ2 (otherwise the result is inverted).' }] : []),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const f = ctx.mat();
    src.convertTo(f, cv.CV_32F, 1, 0);
    const a = ctx.mat();
    const b = ctx.mat();
    cv.GaussianBlur(f, a, new cv.Size(0, 0), p.s1, p.s1, cv.BORDER_DEFAULT);
    cv.GaussianBlur(f, b, new cv.Size(0, 0), p.s2, p.s2, cv.BORDER_DEFAULT);
    const d = ctx.mat();
    cv.subtract(a, b, d);
    if (p.out === 'float32') return d;
    const dst = ctx.mat();
    cv.normalize(d, dst, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
    return dst;
  },
  emitPython: (p, ctx) => {
    const s1 = ctx.c('sigma1', pyRaw(pyFloat(p.s1)));
    const s2 = ctx.c('sigma2', pyRaw(pyFloat(p.s2)));
    const lines = ['f = img.astype(np.float32)', `img = cv2.GaussianBlur(f, (0, 0), ${s1}) - cv2.GaussianBlur(f, (0, 0), ${s2})`];
    if (p.out === 'uint8') lines.push('img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)');
    return { body: lines.join('\n') };
  },
  testCases: [{}, { params: { out: 'float32' }, input: 'gray', tolerance: 1e-3 }],
});
