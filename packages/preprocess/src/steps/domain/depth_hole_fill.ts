import { int, select } from '../../params';
import { defineStep } from '../../registry';
import { copyMat } from '../util';

defineStep({
  id: 'depth_hole_fill',
  category: 'Domain',
  label: 'Depth hole filling',
  doc: 'Fill invalid (0) depth pixels with the median of their neighbourhood (cv2.medianBlur), repeating to close larger holes; valid pixels never change. uint16 and float32 support kernels up to 5.',
  cv2: 'filled = cv2.medianBlur(d, k); d = np.where(d == 0, filled, d)',
  keywords: 'depth hole fill inpaint median invalid zero kinect realsense',
  params: [select('k', 'Kernel', [['3', '3'], ['5', '5']], '5'), int('iterations', 'Iterations', 3, { min: 1, max: 20 })],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1], dtype: ['uint8', 'uint16', 'float32'] },
  needs: ['medianBlur'],
  summary: (p) => `median ${p.k} × ${p.iterations}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    let cur = copyMat(src, ctx);
    for (let i = 0; i < p.iterations; i++) {
      const f = ctx.mat();
      cv.medianBlur(cur, f, Number(p.k));
      const next = copyMat(cur, ctx);
      const a = ctx.in.dtype === 'uint8' ? next.data : ctx.in.dtype === 'uint16' ? next.data16U : next.data32F;
      const b = ctx.in.dtype === 'uint8' ? f.data : ctx.in.dtype === 'uint16' ? f.data16U : f.data32F;
      for (let j = 0; j < a.length; j++) if (a[j] === 0) a[j] = b[j];
      cur = next;
    }
    return cur;
  },
  emitPython: (p, ctx) => ({
    body: `for _ in range(${ctx.c('iterations', p.iterations)}):\n    filled = cv2.medianBlur(img, ${ctx.c('ksize', Number(p.k))})\n    img = np.where(img == 0, filled, img)  # only holes change`,
  }),
  testCases: [{ input: 'u16', before: [{ def: 'depth_invalid', params: { min: 9000 } }] }],
});
