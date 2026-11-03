import { oddInt } from '../../params';
import { defineStep } from '../../registry';
import { kernelTooBig, oddCheck } from '../util';

defineStep({
  id: 'median_blur',
  category: 'Filtering',
  label: 'Median blur',
  doc: 'Replace each pixel by the median of its k×k neighbourhood (cv2.medianBlur). The best filter for salt-and-pepper noise and keeps edges sharp. k must be odd; k > 5 needs a uint8 image.',
  cv2: 'cv2.medianBlur(img, k)',
  keywords: 'median blur salt pepper impulse noise denoise',
  params: [oddInt('k', 'Kernel size (odd)', 5, { min: 3, max: 31 })],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1, 3, 4], dtype: ['uint8', 'uint16', 'float32'] },
  needs: ['medianBlur'],
  summary: (p) => `${p.k}×${p.k}`,
  outMeta: (m) => m,
  check: (m, p) => [
    ...oddCheck(p.k),
    ...kernelTooBig(m, p.k),
    ...(p.k > 5 && m.dtype !== 'uint8'
      ? [{ level: 'error' as const, message: `medianBlur with ksize > 5 needs uint8; this image is ${m.dtype}.`, fixes: [{ label: 'Use k = 5', kind: 'setParams' as const, params: { k: 5 } }] }]
      : []),
  ],
  applyBrowser: (src, p, ctx) => {
    const dst = ctx.mat();
    ctx.cv.medianBlur(src, dst, p.k);
    return dst;
  },
  emitPython: (p, ctx) => ({ body: `img = cv2.medianBlur(img, ${ctx.c('ksize', p.k, 'odd; > 5 needs uint8')})` }),
  emitAlbumentations: (p) => `A.MedianBlur(blur_limit=(${p.k}, ${p.k}), p=1.0)`,
  testCases: [{}, { params: { k: 9 } }, { input: 'float', params: { k: 3 } }],
});
