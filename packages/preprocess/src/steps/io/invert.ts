import { defineStep } from '../../registry';
import { mapPixels } from '../util';

defineStep({
  id: 'invert',
  category: 'I/O & colour',
  label: 'Invert',
  doc: 'Negative image: 255 − x for uint8 (cv2.bitwise_not), 65535 − x for uint16, 1 − x for float images in [0, 1]. Useful for documents (dark text on light → light on dark) and for masks.',
  cv2: 'cv2.bitwise_not(img)',
  keywords: 'negative bitwise not invert',
  params: [],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { dtype: ['uint8', 'uint16', 'float32'] },
  needs: ['bitwise_not'],
  outMeta: (m) => m,
  applyBrowser: (src, _p, ctx) => {
    if (ctx.in.dtype === 'float32') return mapPixels(src, ctx, 'float32', (v) => 1 - v);
    const dst = ctx.mat();
    ctx.cv.bitwise_not(src, dst);
    return dst;
  },
  emitPython: (_p, ctx) => ({ body: ctx.in.dtype === 'float32' ? 'img = np.float32(1.0) - img  # float image in [0, 1]' : 'img = cv2.bitwise_not(img)' }),
  emitAlbumentations: (_p, ctx) => (ctx.in.dtype === 'uint8' ? 'A.InvertImg(p=1.0)' : undefined),
  testCases: [{}, { input: 'u16' }, { input: 'float' }],
});
