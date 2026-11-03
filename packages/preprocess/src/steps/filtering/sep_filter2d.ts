import { floats, select } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';

defineStep({
  id: 'sep_filter2d',
  category: 'Filtering',
  label: 'Separable filter (sepFilter2D)',
  doc: 'Filter with a row kernel then a column kernel (cv2.sepFilter2D). Equivalent to filter2D with their outer product but much faster for large kernels, e.g. [1, 4, 6, 4, 1]/16 binomial smoothing or [−1, 0, 1] × [1, 2, 1] Sobel.',
  cv2: 'cv2.sepFilter2D(img, ddepth, kernelX, kernelY)',
  keywords: 'separable filter sepFilter2D row column kernel binomial',
  params: [floats('kx', 'Row kernel (x)', [0.0625, 0.25, 0.375, 0.25, 0.0625]), floats('ky', 'Column kernel (y)', [0.0625, 0.25, 0.375, 0.25, 0.0625]), select('depth', 'Output', [['same', 'Same dtype'], ['float32', 'float32']], 'same')],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  needs: ['sepFilter2D'],
  tolerance: 1, // uint8 rounding differs by 1 between native SIMD cv2 and OpenCV.js
  summary: (p) => `${p.kx.length}×${p.ky.length}`,
  outMeta: (m, p) => (p.depth === 'float32' ? { ...m, dtype: 'float32' } : m),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const kx = ctx.keep(cv.matFromArray(1, p.kx.length, cv.CV_32F, p.kx));
    const ky = ctx.keep(cv.matFromArray(p.ky.length, 1, cv.CV_32F, p.ky));
    const dst = ctx.mat();
    cv.sepFilter2D(src, dst, p.depth === 'float32' ? cv.CV_32F : -1, kx, ky, new cv.Point(-1, -1), 0, cv.BORDER_DEFAULT);
    return dst;
  },
  emitPython: (p, ctx) => {
    const kx = ctx.c('kx', pyRaw(`np.array([${(p.kx as number[]).map(pyFloat).join(', ')}], np.float32)`));
    const ky = ctx.c('ky', pyRaw(`np.array([${(p.ky as number[]).map(pyFloat).join(', ')}], np.float32)`));
    return { body: `img = cv2.sepFilter2D(img, ${p.depth === 'float32' ? 'cv2.CV_32F' : '-1'}, ${kx}, ${ky})` };
  },
  testCases: [{}, { params: { kx: [-1, 0, 1], ky: [1, 2, 1], depth: 'float32' } }],
});
