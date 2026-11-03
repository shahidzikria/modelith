import { int, select } from '../../params';
import { defineStep } from '../../registry';
import { gradBrowser, gradMeta, gradOut, gradPy } from './_gradient_output';

defineStep({
  id: 'sobel',
  category: 'Edges',
  label: 'Sobel derivative',
  doc: 'First (or higher) image derivative with the Sobel operator: dx = 1 finds vertical edges, dy = 1 horizontal ones. Computed in float32; "uint8" output takes the absolute value (cv2.convertScaleAbs), "float32" keeps the sign for gradient features.',
  cv2: 'cv2.Sobel(img, cv2.CV_32F, dx, dy, ksize=k)',
  keywords: 'sobel derivative gradient edges dx dy',
  params: [int('dx', 'dx', 1, { min: 0, max: 2 }), int('dy', 'dy', 0, { min: 0, max: 2 }), select('ksize', 'Kernel', [['1', '1'], ['3', '3'], ['5', '5'], ['7', '7']], '3'), gradOut()],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  needs: ['Sobel', 'convertScaleAbs'],
  tolerance: 1e-3,
  summary: (p) => `dx ${p.dx} · dy ${p.dy} · k${p.ksize}`,
  outMeta: gradMeta,
  check: (_m, p) => (p.dx + p.dy === 0 ? [{ level: 'error' as const, message: 'dx and dy cannot both be 0.' }] : p.dx >= Number(p.ksize) || p.dy >= Number(p.ksize) ? [{ level: 'error' as const, message: 'Derivative order must be smaller than the kernel size.' }] : []),
  applyBrowser: (src, p, ctx) => gradBrowser(src, p, ctx, (s, d, depth) => ctx.cv.Sobel(s, d, depth, p.dx, p.dy, Number(p.ksize), 1, 0, ctx.cv.BORDER_DEFAULT)),
  emitPython: (p, ctx) => ({ body: gradPy(`cv2.Sobel(img, cv2.CV_32F, ${ctx.c('dx', p.dx)}, ${ctx.c('dy', p.dy)}, ksize=${ctx.c('ksize', Number(p.ksize))})`, p) }),
  testCases: [{}, { params: { dx: 0, dy: 1, ksize: '5', out: 'float32' } }],
});
