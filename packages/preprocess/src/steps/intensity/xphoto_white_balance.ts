import { select } from '../../params';
import { defineStep } from '../../registry';

defineStep({
  id: 'xphoto_white_balance',
  category: 'Intensity',
  label: 'White balance (xphoto)',
  doc: 'OpenCV contrib white balancing: Simple (per-channel stretch), Gray-world (with saturation threshold) or Learning-based (trained colour-constancy model). Needs opencv-contrib-python, so it runs on the Python server.',
  cv2: 'cv2.xphoto.create*WB().balanceWhite(img)',
  keywords: 'white balance xphoto contrib learning based gray world simple color constancy',
  params: [select('method', 'Method', [['simple', 'Simple'], ['grayworld', 'Gray-world'], ['learning', 'Learning-based']], 'learning')],
  engine: 'server',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [3], dtype: ['uint8'] },
  summary: (p) => p.method,
  outMeta: (m) => m,
  emitPython: (p) => {
    const fn = { simple: 'createSimpleWB', grayworld: 'createGrayworldWB', learning: 'createLearningBasedWB' }[p.method as string];
    return { body: `img = cv2.xphoto.${fn}().balanceWhite(img)  # needs opencv-contrib-python` };
  },
  testCases: [{}, { params: { method: 'grayworld' } }],
});
