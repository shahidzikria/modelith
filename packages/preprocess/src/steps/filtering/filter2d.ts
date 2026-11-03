import { border, float, matrix, select } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { borderCv, borderPy, pyRaw } from '../util';

defineStep({
  id: 'filter2d',
  category: 'Filtering',
  label: 'Custom kernel (filter2D)',
  doc: 'Convolve (strictly: correlate) with any kernel you type into the grid, then add delta (cv2.filter2D). Output "same" keeps the dtype (saturated); "float32" keeps negative responses, e.g. for edge kernels.',
  cv2: 'cv2.filter2D(img, ddepth, kernel, delta=delta)',
  keywords: 'filter2d kernel convolution custom correlation emboss',
  params: [
    matrix('kernel', 'Kernel', [
      [-2, -1, 0],
      [-1, 1, 1],
      [0, 1, 2],
    ]),
    float('delta', 'Delta', 0, { step: 1 }),
    select('depth', 'Output', [['same', 'Same dtype (saturated)'], ['float32', 'float32']], 'same'),
    border('REFLECT_101'),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  needs: ['filter2D'],
  tolerance: 1e-3,
  summary: (p) => `${p.kernel.length}×${p.kernel[0]?.length ?? 0} kernel`,
  outMeta: (m, p) => (p.depth === 'float32' ? { ...m, dtype: 'float32', range: [-m.range[1], m.range[1]] } : m),
  check: (_m, p) => (!p.kernel.length || p.kernel.some((r: number[]) => r.length !== p.kernel[0].length) ? [{ level: 'error' as const, message: 'Kernel rows must all have the same length.' }] : []),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const K = p.kernel as number[][];
    const k = ctx.keep(cv.matFromArray(K.length, K[0].length, cv.CV_32F, K.flat()));
    const dst = ctx.mat();
    cv.filter2D(src, dst, p.depth === 'float32' ? cv.CV_32F : -1, k, new cv.Point(-1, -1), p.delta, borderCv(cv, p.border));
    return dst;
  },
  emitPython: (p, ctx) => {
    const K = ctx.c('kernel', pyRaw(`np.array([\n${(p.kernel as number[][]).map((r) => `    [${r.map((v) => pyFloat(v)).join(', ')}],`).join('\n')}\n], np.float32)`));
    const d = ctx.c('delta', pyRaw(pyFloat(p.delta)));
    return { body: `img = cv2.filter2D(img, ${p.depth === 'float32' ? 'cv2.CV_32F' : '-1'}, ${K}, delta=${d}, borderType=${ctx.c('border', borderPy(p.border))})` };
  },
  testCases: [{}, { params: { depth: 'float32' } }, { params: { kernel: [[1, 1, 1, 1, 1].map((v) => v / 5)], delta: 10 } }],
});
