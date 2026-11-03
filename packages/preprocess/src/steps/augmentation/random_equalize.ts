import { defineStep } from '../../registry';
import { getStep } from '../../registry';

defineStep({
  id: 'random_equalize',
  category: 'Augmentation',
  label: 'Random equalise',
  doc: 'With probability p, histogram-equalise the image (Y channel of YCrCb for colour, so hues are kept). Part of the AutoAugment / RandAugment operation set.',
  cv2: 'cv2.equalizeHist(Y)',
  keywords: 'random equalize equalise histogram autoaugment augmentation',
  params: [],
  engine: 'both',
  kind: 'random',
  affectsGeometry: false,
  accepts: { c: [1, 3], dtype: ['uint8'] },
  needs: ['equalizeHist'],
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => getStep('equalize_hist')!.applyBrowser!(src, p, ctx),
  emitPython: (p, ctx) => getStep('equalize_hist')!.emitPython(p, ctx),
  emitAlbumentations: () => 'A.Equalize(mode="cv", by_channels=False, p={p})',
  emitTorchvision: () => 'v2.RandomEqualize(p={p})',
  testCases: [{}],
});
