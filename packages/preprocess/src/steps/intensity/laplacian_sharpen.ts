import { float } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';

export const laplacianKernel = (s: number) => [0, -s, 0, -s, 1 + 4 * s, -s, 0, -s, 0];

defineStep({
  id: 'laplacian_sharpen',
  category: 'Intensity',
  label: 'Laplacian sharpen',
  doc: 'Sharpen with the 3×3 kernel [[0,−s,0],[−s,1+4s,−s],[0,−s,0]] (image minus s × Laplacian) using cv2.filter2D. s = 1 is the classic sharpening kernel. Faster and harsher than an unsharp mask.',
  cv2: 'cv2.filter2D(img, -1, kernel)',
  keywords: 'sharpen laplacian kernel filter2D edge enhance',
  params: [float('strength', 'Strength s', 1, { min: 0, max: 4, step: 0.05 })],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  needs: ['filter2D'],
  summary: (p) => `s ${p.strength}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const k = ctx.keep(cv.matFromArray(3, 3, cv.CV_32F, laplacianKernel(p.strength)));
    const dst = ctx.mat();
    cv.filter2D(src, dst, -1, k, new cv.Point(-1, -1), 0, cv.BORDER_DEFAULT);
    return dst;
  },
  emitPython: (p, ctx) => {
    const s = ctx.c('strength', pyRaw(pyFloat(p.strength)));
    return { body: `s = ${s}\nkernel = np.array([[0, -s, 0], [-s, 1 + 4 * s, -s], [0, -s, 0]], np.float32)\nimg = cv2.filter2D(img, -1, kernel)` };
  },
  emitAlbumentations: (p) => `A.Sharpen(alpha=(${Math.min(1, p.strength / 2)}, ${Math.min(1, p.strength / 2)}), p=1.0)  # approximate`,
  testCases: [{}, { input: 'gray', params: { strength: 0.5 } }],
});
