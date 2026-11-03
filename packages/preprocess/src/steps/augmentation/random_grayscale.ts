import { defineStep } from '../../registry';

defineStep({
  id: 'random_grayscale',
  category: 'Augmentation',
  label: 'Random grayscale',
  doc: 'With probability p, convert to grey and back to 3 identical channels, so the shape stays (H, W, 3). Teaches the model not to rely on colour alone.',
  cv2: 'cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)',
  keywords: 'random grayscale greyscale desaturate augmentation',
  params: [],
  engine: 'both',
  kind: 'random',
  affectsGeometry: false,
  accepts: { c: [3], space: ['BGR', 'RGB'] },
  needs: ['cvtColor'],
  outMeta: (m) => m,
  applyBrowser: (src, _p, ctx) => {
    const { cv } = ctx;
    const rgb = ctx.in.space === 'RGB';
    const g = ctx.mat();
    cv.cvtColor(src, g, rgb ? cv.COLOR_RGB2GRAY : cv.COLOR_BGR2GRAY);
    const dst = ctx.mat();
    cv.cvtColor(g, dst, rgb ? cv.COLOR_GRAY2RGB : cv.COLOR_GRAY2BGR);
    return dst;
  },
  emitPython: (_p, ctx) => {
    const s = ctx.in.space === 'RGB' ? 'RGB' : 'BGR';
    return { body: `img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_${s}2GRAY), cv2.COLOR_GRAY2${s})  # still 3 channels` };
  },
  emitAlbumentations: () => 'A.ToGray(num_output_channels=3, p={p})',
  emitTorchvision: () => 'v2.RandomGrayscale(p={p})',
  testCases: [{}],
});
