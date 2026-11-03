import { select } from '../../params';
import { defineStep } from '../../registry';

defineStep({
  id: 'skeletonize',
  category: 'Morphology',
  label: 'Skeletonise',
  doc: 'Thin binary shapes to 1-pixel-wide centre lines (cv2.ximgproc.thinning, Zhang-Suen or Guo-Hall). Used for vessels, cracks, handwriting and road networks. Needs opencv-contrib, so it runs on the server.',
  cv2: 'cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)',
  keywords: 'skeleton skeletonize thinning medial axis zhang suen guo hall',
  params: [select('method', 'Method', [['ZHANGSUEN', 'Zhang-Suen'], ['GUOHALL', 'Guo-Hall']], 'ZHANGSUEN')],
  engine: 'server',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1], dtype: ['uint8'] },
  summary: (p) => (p.method === 'ZHANGSUEN' ? 'Zhang-Suen' : 'Guo-Hall'),
  outMeta: (m) => m,
  emitPython: (p) => ({ body: `binary = np.where(img > 0, 255, 0).astype(np.uint8)\nimg = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_${p.method})  # needs opencv-contrib-python` }),
  testCases: [{ input: 'doc', before: [{ def: 'threshold', params: { method: 'otsu', type: 'BINARY_INV' } }] }],
});
