import { float, int } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';

defineStep({
  id: 'bilateral',
  category: 'Filtering',
  label: 'Bilateral filter',
  doc: 'Edge-preserving smoothing (cv2.bilateralFilter): neighbours are weighted by distance (sigmaSpace) and by colour similarity (sigmaColor), so flat regions are smoothed and edges are kept. d = 5 for real time, 9 for heavy noise; sigmas 10–150. Slow on large images.',
  cv2: 'cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)',
  keywords: 'bilateral edge preserving smoothing denoise skin cartoon',
  params: [int('d', 'Diameter d', 9, { min: 1, max: 31 }), float('sigmaColor', 'sigmaColor', 75, { min: 1, max: 255, step: 1 }), float('sigmaSpace', 'sigmaSpace', 75, { min: 1, max: 255, step: 1 })],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1, 3], dtype: ['uint8', 'float32'] },
  needs: ['bilateralFilter'],
  tolerance: 1,
  summary: (p) => `d ${p.d} · ${p.sigmaColor}/${p.sigmaSpace}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const dst = ctx.mat();
    ctx.cv.bilateralFilter(src, dst, p.d, p.sigmaColor, p.sigmaSpace, ctx.cv.BORDER_DEFAULT);
    return dst;
  },
  emitPython: (p, ctx) => ({
    body: `img = cv2.bilateralFilter(img, ${ctx.c('d', p.d)}, ${ctx.c('sigma_color', pyRaw(pyFloat(p.sigmaColor)))}, ${ctx.c('sigma_space', pyRaw(pyFloat(p.sigmaSpace)))})`,
  }),
  testCases: [{}, { input: 'gray', params: { d: 5, sigmaColor: 30, sigmaSpace: 30 } }],
});
