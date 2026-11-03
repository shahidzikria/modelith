import { select } from '../../params';
import { defineStep } from '../../registry';
import { gradBrowser, gradMeta, gradOut, gradPy } from './_gradient_output';

defineStep({
  id: 'laplacian',
  category: 'Edges',
  label: 'Laplacian',
  doc: 'Second derivative (cv2.Laplacian): responds to edges in every direction and to fine detail. Its variance is the classic blur score (low variance = blurry). Very noise-sensitive: smooth first.',
  cv2: 'cv2.Laplacian(img, cv2.CV_32F, ksize=k)',
  keywords: 'laplacian second derivative edges blur sharpness focus',
  params: [select('ksize', 'Kernel', [['1', '1'], ['3', '3'], ['5', '5'], ['7', '7']], '3'), gradOut()],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  needs: ['Laplacian', 'convertScaleAbs'],
  tolerance: 1e-3,
  summary: (p) => `k${p.ksize}`,
  outMeta: gradMeta,
  applyBrowser: (src, p, ctx) => gradBrowser(src, p, ctx, (s, d, depth) => ctx.cv.Laplacian(s, d, depth, Number(p.ksize), 1, 0, ctx.cv.BORDER_DEFAULT)),
  emitPython: (p, ctx) => ({ body: gradPy(`cv2.Laplacian(img, cv2.CV_32F, ksize=${ctx.c('ksize', Number(p.ksize))})`, p) }),
  testCases: [{}, { params: { ksize: '1', out: 'float32' }, input: 'gray' }],
});
