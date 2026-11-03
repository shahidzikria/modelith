import { float } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { copyMat, pyRaw } from '../util';
import { grayOf, grayPy } from '../geometric/deskew';

defineStep({
  id: 'border_removal',
  category: 'Domain',
  label: 'Scan border removal',
  doc: 'Remove the dark scanner / photocopy margins around a page: dark regions (Otsu) that touch the image edge and cover at least a minimum fraction of the image are painted white. Text inside the page is untouched.',
  cv2: 'cv2.connectedComponentsWithStats(dark) → paint components touching the border white',
  keywords: 'border removal scan scanner margin black edges document page cleanup',
  params: [float('minFrac', 'Min border area (fraction)', 0.002, { min: 0, max: 0.5, step: 0.001 })],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1, 3], dtype: ['uint8'] },
  needs: ['connectedComponentsWithStats', 'threshold'],
  summary: (p) => `≥ ${p.minFrac * 100}% of image`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const gray = grayOf(src, ctx);
    const dark = ctx.mat();
    cv.threshold(gray, dark, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
    const labels = ctx.mat();
    const stats = ctx.mat();
    const cents = ctx.mat();
    const n = cv.connectedComponentsWithStats(dark, labels, stats, cents, 8, cv.CV_32S);
    const S = stats.data32S as Int32Array;
    const W = src.cols, H = src.rows;
    const paint = new Uint8Array(n);
    for (let i = 1; i < n; i++) {
      const [x, y, w, h, a] = [0, 1, 2, 3, 4].map((k) => S[i * 5 + k]);
      if ((x === 0 || y === 0 || x + w === W || y + h === H) && a >= p.minFrac * W * H) paint[i] = 1;
    }
    const dst = copyMat(src, ctx);
    const d = dst.data as Uint8Array;
    const L = labels.data32S as Int32Array;
    const c = src.channels();
    for (let i = 0; i < L.length; i++) if (paint[L[i]]) for (let k = 0; k < c; k++) d[i * c + k] = 255;
    return dst;
  },
  emitPython: (p, ctx) => ({
    body: [
      grayPy(ctx.in),
      '_, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)',
      'n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)',
      'h, w = gray.shape',
      'x, y, bw, bh, area = stats.T',
      `touches = ((x == 0) | (y == 0) | (x + bw == w) | (y + bh == h)) & (area >= ${ctx.c('min_frac', pyRaw(pyFloat(p.minFrac)))} * w * h)`,
      'touches[0] = False  # label 0 is the page itself',
      'img = img.copy()',
      'img[touches[labels]] = 255',
    ].join('\n'),
  }),
  testCases: [{ input: 'doc' }],
});
