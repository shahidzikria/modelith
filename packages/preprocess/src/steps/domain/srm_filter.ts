import { bool, float } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';
import { grayOf, grayPy } from '../geometric/deskew';

/** The three SRM high-pass kernels used by RGB-N / ManTra-Net style manipulation detectors. */
export const SRM_KERNELS: number[][][] = [
  [
    [0, 0, 0, 0, 0],
    [0, -1, 2, -1, 0],
    [0, 2, -4, 2, 0],
    [0, -1, 2, -1, 0],
    [0, 0, 0, 0, 0],
  ].map((r) => r.map((v) => v / 4)),
  [
    [-1, 2, -2, 2, -1],
    [2, -6, 8, -6, 2],
    [-2, 8, -12, 8, -2],
    [2, -6, 8, -6, 2],
    [-1, 2, -2, 2, -1],
  ].map((r) => r.map((v) => v / 12)),
  [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, -2, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
  ].map((r) => r.map((v) => v / 2)),
];

defineStep({
  id: 'srm_filter',
  category: 'Domain',
  label: 'SRM filter bank',
  doc: 'Forensics: filter the gray image with the 3 standard SRM (spatial rich model) high-pass kernels and stack the residuals as 3 float32 channels (the "noise stream" of RGB-N / ManTra-Net). Truncation to ±T (T = 3) as in the paper keeps the residual range small.',
  cv2: 'np.dstack([cv2.filter2D(gray, cv2.CV_32F, k) for k in SRM_KERNELS])',
  keywords: 'srm rich model filter bank forensics tamper detection noise residual high pass rgbn mantranet',
  params: [bool('truncate', 'Truncate to ±T', true), float('T', 'T', 3, { min: 0.5, max: 50, step: 0.5, showIf: { key: 'truncate', equals: [true] } })],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { dtype: ['uint8', 'float32'] },
  needs: ['filter2D', 'merge'],
  tolerance: 1e-4,
  summary: (p) => (p.truncate ? `±${p.T}` : 'raw'),
  outMeta: (m, p) => ({ ...m, c: 3, dtype: 'float32', space: 'BGR', range: p.truncate ? [-p.T, p.T] : [-255, 255] }),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const gray = grayOf(src, ctx);
    const outs = ctx.keep(new cv.MatVector());
    for (const K of SRM_KERNELS) {
      const k = ctx.keep(cv.matFromArray(5, 5, cv.CV_32F, K.flat()));
      const r = ctx.mat();
      cv.filter2D(gray, r, cv.CV_32F, k, new cv.Point(-1, -1), 0, cv.BORDER_DEFAULT);
      if (p.truncate) {
        const d = r.data32F as Float32Array;
        const T = Math.fround(p.T);
        for (let i = 0; i < d.length; i++) d[i] = Math.min(T, Math.max(-T, d[i]));
      }
      outs.push_back(r);
    }
    const dst = ctx.mat();
    cv.merge(outs, dst);
    return dst;
  },
  emitPython: (p, ctx) => {
    const K = ctx.c('kernels', pyRaw(`[\n    np.array(${JSON.stringify([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]])}, np.float32) / 4,\n    np.array(${JSON.stringify([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]])}, np.float32) / 12,\n    np.array(${JSON.stringify([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])}, np.float32) / 2,\n]`), 'the 3 SRM high-pass kernels');
    const lines = [grayPy(ctx.in), `img = cv2.merge([cv2.filter2D(gray, cv2.CV_32F, k) for k in ${K}])  # 3 residual channels`];
    if (p.truncate) lines.push(`T = np.float32(${ctx.c('t', pyRaw(pyFloat(p.T)))})\nimg = np.clip(img, -T, T)`);
    return { body: lines.join('\n') };
  },
  testCases: [{}, { params: { truncate: false } }],
});
