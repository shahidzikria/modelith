import { oddInt, select } from '../../params';
import { defineStep } from '../../registry';
import { grayOf, grayPy } from '../geometric/deskew';

defineStep({
  id: 'noise_variance_map',
  category: 'Domain',
  label: 'Local noise map',
  doc: 'Local standard deviation of the noise residual (gray − 3×3 median) over a k×k window: inconsistent noise levels reveal splicing, AI in-painting and double processing. Output: float32 σ map, or min-max scaled uint8.',
  cv2: 'r = gray − medianBlur(gray, 3); sqrt(boxFilter(r²) − boxFilter(r)²)',
  keywords: 'noise variance map local std forensics splicing inpainting inconsistency',
  params: [oddInt('k', 'Window', 15, { min: 3, max: 101 }), select('out', 'Output', [['uint8', 'uint8 (min-max)'], ['float32', 'float32 σ']], 'uint8')],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { dtype: ['uint8'] },
  needs: ['medianBlur', 'boxFilter', 'normalize'],
  tolerance: 1,
  summary: (p) => `${p.k}×${p.k}`,
  outMeta: (m, p) => ({ ...m, c: 1, space: 'GRAY', dtype: p.out === 'uint8' ? 'uint8' : 'float32', range: p.out === 'uint8' ? [0, 255] : [0, 64] }),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const gray = grayOf(src, ctx);
    const med = ctx.mat();
    cv.medianBlur(gray, med, 3);
    const a = ctx.mat();
    const b = ctx.mat();
    gray.convertTo(a, cv.CV_32F, 1, 0);
    med.convertTo(b, cv.CV_32F, 1, 0);
    const r = ctx.mat();
    cv.subtract(a, b, r);
    const r2 = ctx.mat();
    cv.multiply(r, r, r2, 1, -1);
    const m1 = ctx.mat();
    const m2 = ctx.mat();
    const size = new cv.Size(p.k, p.k);
    cv.boxFilter(r, m1, cv.CV_32F, size, new cv.Point(-1, -1), true, cv.BORDER_DEFAULT);
    cv.boxFilter(r2, m2, cv.CV_32F, size, new cv.Point(-1, -1), true, cv.BORDER_DEFAULT);
    const sd = ctx.keep(new cv.Mat(src.rows, src.cols, cv.CV_32FC1));
    const M1 = m1.data32F, M2 = m2.data32F, S = sd.data32F;
    for (let i = 0; i < S.length; i++) S[i] = Math.fround(Math.sqrt(Math.max(Math.fround(M2[i] - Math.fround(M1[i] * M1[i])), 0)));
    if (p.out === 'float32') return sd;
    const dst = ctx.mat();
    cv.normalize(sd, dst, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
    return dst;
  },
  emitPython: (p, ctx) => {
    const k = ctx.c('window', p.k);
    const lines = [
      grayPy(ctx.in),
      'r = gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)  # noise residual',
      `mean = cv2.boxFilter(r, cv2.CV_32F, (${k}, ${k}))`,
      `img = np.sqrt(np.maximum(cv2.boxFilter(r * r, cv2.CV_32F, (${k}, ${k})) - mean * mean, np.float32(0)))  # local σ`,
    ];
    if (p.out === 'uint8') lines.push('img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)');
    return { body: lines.join('\n') };
  },
  testCases: [{}, { params: { out: 'float32' }, tolerance: 1e-3 }],
});
