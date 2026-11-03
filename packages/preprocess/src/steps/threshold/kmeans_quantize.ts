import { int } from '../../params';
import { defineStep } from '../../registry';
import { copyMat } from '../util';

defineStep({
  id: 'kmeans_quantize',
  category: 'Threshold',
  label: 'k-means colour quantisation',
  doc: 'Cluster pixel colours with cv2.kmeans and replace every pixel by its cluster centre: K = 2–3 gives a quick segmentation, 8–16 a poster-like reduced palette. k-means++ initialisation with a fixed OpenCV RNG seed so results are repeatable. Slow on large images.',
  cv2: 'cv2.kmeans(pixels, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)',
  keywords: 'kmeans k-means quantize quantise colour color palette cluster segmentation posterize',
  params: [int('K', 'Clusters K', 4, { min: 2, max: 32 }), int('attempts', 'Attempts', 3, { min: 1, max: 10 }), int('iters', 'Max iterations', 20, { min: 1, max: 100 })],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1, 3], dtype: ['uint8'] },
  needs: ['kmeans', 'setRNGSeed'],
  tolerance: 2,
  summary: (p) => `K ${p.K}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const c = src.channels();
    const n = src.rows * src.cols;
    const data = ctx.keep(new cv.Mat(n, c, cv.CV_32F));
    const d = src.data as Uint8Array;
    const f = data.data32F as Float32Array;
    for (let i = 0; i < d.length; i++) f[i] = d[i];
    const labels = ctx.mat();
    const centers = ctx.mat();
    cv.setRNGSeed(0);
    const crit = new cv.TermCriteria(cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER, p.iters, 1.0);
    cv.kmeans(data, p.K, labels, crit, p.attempts, cv.KMEANS_PP_CENTERS, centers);
    const dst = copyMat(src, ctx);
    const o = dst.data as Uint8Array;
    const L = labels.data32S as Int32Array;
    const C = centers.data32F as Float32Array;
    for (let i = 0; i < n; i++) for (let k = 0; k < c; k++) o[i * c + k] = Math.min(255, Math.max(0, Math.round(C[L[i] * c + k])));
    return dst;
  },
  emitPython: (p, ctx) => ({
    body: [
      `pixels = img.reshape(-1, ${ctx.in.c}).astype(np.float32)`,
      `criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, ${ctx.c('iterations', p.iters)}, 1.0)`,
      'cv2.setRNGSeed(0)  # k-means++ initialisation is random',
      `_, labels, centers = cv2.kmeans(pixels, ${ctx.c('k', p.K)}, None, criteria, ${ctx.c('attempts', p.attempts)}, cv2.KMEANS_PP_CENTERS)`,
      'centers = np.clip(np.round(centers), 0, 255).astype(np.uint8)',
      'img = centers[labels.ravel()].reshape(img.shape)',
    ].join('\n'),
  }),
  testCases: [{}, { input: 'gray', params: { K: 3 } }],
});
