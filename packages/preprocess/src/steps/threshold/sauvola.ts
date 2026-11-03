import { float, oddInt, select } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { f32, pyRaw } from '../util';

defineStep({
  id: 'sauvola',
  category: 'Threshold',
  label: 'Sauvola / Niblack',
  doc: 'Local thresholds from the window mean m and standard deviation s. Sauvola: T = m·(1 + k·(s / R − 1)) (k ≈ 0.2–0.5, R = 128) is the standard for degraded documents. Niblack: T = m + k·s (k ≈ −0.2). Pixels above T become white. Implemented with box filters, so it is fast and runs in the browser too.',
  cv2: 'mean, sq = cv2.boxFilter(f, …), cv2.boxFilter(f * f, …); std = sqrt(sq − mean²)',
  keywords: 'sauvola niblack local threshold document binarization ocr degraded',
  params: [
    select('method', 'Method', [['sauvola', 'Sauvola'], ['niblack', 'Niblack']], 'sauvola'),
    oddInt('window', 'Window (odd)', 25, { min: 3, max: 255 }),
    float('k', 'k', 0.2, { min: -1, max: 1, step: 0.01 }),
    float('R', 'R (dynamic range of std)', 128, { min: 1, max: 255, step: 1, showIf: { key: 'method', equals: ['sauvola'] } }),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1], dtype: ['uint8'] },
  needs: ['boxFilter'],
  // Box-filter sums differ in the last float bit between native SIMD and OpenCV.js, flipping rare borderline pixels.
  tolerance: 1,
  summary: (p) => `${p.method} ${p.window} · k ${p.k}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const f = ctx.mat();
    src.convertTo(f, cv.CV_32F, 1, 0);
    const sq = ctx.mat();
    cv.multiply(f, f, sq, 1, -1);
    const mean = ctx.mat();
    const msq = ctx.mat();
    const size = new cv.Size(p.window, p.window);
    cv.boxFilter(f, mean, cv.CV_32F, size, new cv.Point(-1, -1), true, cv.BORDER_REFLECT);
    cv.boxFilter(sq, msq, cv.CV_32F, size, new cv.Point(-1, -1), true, cv.BORDER_REFLECT);
    const dst = ctx.keep(new cv.Mat(src.rows, src.cols, cv.CV_8UC1));
    const M = mean.data32F, Q = msq.data32F, F = f.data32F, o = dst.data;
    const k = f32(p.k), R = f32(p.R);
    for (let i = 0; i < F.length; i++) {
      const sd = f32(Math.sqrt(Math.max(f32(Q[i] - f32(M[i] * M[i])), 0)));
      const T = p.method === 'sauvola' ? f32(M[i] * f32(1 + f32(k * f32(f32(sd / R) - 1)))) : f32(M[i] + f32(k * sd));
      o[i] = F[i] > T ? 255 : 0;
    }
    return dst;
  },
  emitPython: (p, ctx) => {
    const W = ctx.c('window', p.window, 'odd');
    const k = ctx.c('k', pyRaw(`np.float32(${pyFloat(p.k)})`));
    const lines = [
      'f = img.astype(np.float32)',
      `mean = cv2.boxFilter(f, cv2.CV_32F, (${W}, ${W}), borderType=cv2.BORDER_REFLECT)`,
      `sq_mean = cv2.boxFilter(f * f, cv2.CV_32F, (${W}, ${W}), borderType=cv2.BORDER_REFLECT)`,
      'std = np.sqrt(np.maximum(sq_mean - mean * mean, np.float32(0)))',
    ];
    if (p.method === 'sauvola') {
      const R = ctx.c('r', pyRaw(`np.float32(${pyFloat(p.R)})`));
      lines.push(`thresh = mean * (1 + ${k} * (std / ${R} - 1))  # Sauvola`);
    } else lines.push(`thresh = mean + ${k} * std  # Niblack`);
    lines.push('img = np.where(f > thresh, 255, 0).astype(np.uint8)');
    return { body: lines.join('\n') };
  },
  testCases: [{ input: 'doc' }, { input: 'doc', params: { method: 'niblack', k: -0.2, window: 15 } }],
});
