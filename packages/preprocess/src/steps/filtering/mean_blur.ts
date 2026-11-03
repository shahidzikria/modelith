import { border, int } from '../../params';
import { defineStep } from '../../registry';
import { borderCv, borderPy, kernelTooBig } from '../util';

defineStep({
  id: 'mean_blur',
  category: 'Filtering',
  label: 'Mean blur',
  doc: 'Average over a k×k window (cv2.blur, normalised box filter). The simplest smoothing; blurs edges as much as noise. Even sizes are allowed but shift the image by half a pixel.',
  cv2: 'cv2.blur(img, (k, k))',
  keywords: 'blur mean average smooth box',
  params: [int('k', 'Kernel size', 5, { min: 1, max: 63 }), border('REFLECT_101')],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  needs: ['blur'],
  summary: (p) => `${p.k}×${p.k}`,
  outMeta: (m) => m,
  check: (m, p) => kernelTooBig(m, p.k),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const dst = ctx.mat();
    cv.blur(src, dst, new cv.Size(p.k, p.k), new cv.Point(-1, -1), borderCv(cv, p.border));
    return dst;
  },
  emitPython: (p, ctx) => ({ body: `img = cv2.blur(img, (${ctx.c('k', p.k)}, ${ctx.c('k', p.k)}), borderType=${ctx.c('border', borderPy(p.border))})` }),
  emitAlbumentations: (p) => (p.k % 2 && p.border === 'REFLECT_101' ? `A.Blur(blur_limit=(${p.k}, ${p.k}), p=1.0)` : undefined),
  testCases: [{}, { params: { k: 4, border: 'REPLICATE' } }, { input: 'float', tolerance: 1e-5 }],
});
