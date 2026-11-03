import { select } from '../../params';
import { defineStep } from '../../registry';
import { gradBrowser, gradMeta, gradOut, gradPy } from './_gradient_output';

defineStep({
  id: 'scharr',
  category: 'Edges',
  label: 'Scharr derivative',
  doc: 'First derivative with the 3×3 Scharr kernel: more rotation-accurate than a 3×3 Sobel. Choose the x (vertical edges) or y (horizontal edges) direction.',
  cv2: 'cv2.Scharr(img, cv2.CV_32F, dx, dy)',
  keywords: 'scharr derivative gradient edges',
  params: [select('dir', 'Direction', [['x', 'x (vertical edges)'], ['y', 'y (horizontal edges)']], 'x'), gradOut()],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  needs: ['Scharr', 'convertScaleAbs'],
  tolerance: 1e-3,
  summary: (p) => `${p.dir}`,
  outMeta: gradMeta,
  applyBrowser: (src, p, ctx) => gradBrowser(src, p, ctx, (s, d, depth) => ctx.cv.Scharr(s, d, depth, p.dir === 'x' ? 1 : 0, p.dir === 'y' ? 1 : 0, 1, 0, ctx.cv.BORDER_DEFAULT)),
  emitPython: (p) => ({ body: gradPy(`cv2.Scharr(img, cv2.CV_32F, ${p.dir === 'x' ? '1, 0' : '0, 1'})`, p) }),
  testCases: [{}, { params: { dir: 'y', out: 'float32' }, input: 'gray' }],
});
