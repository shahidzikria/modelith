import { select } from '../../params';
import { defineStep } from '../../registry';

const CODE: Record<string, number> = { h: 1, v: 0 };

defineStep({
  id: 'random_flip',
  category: 'Augmentation',
  label: 'Random flip',
  doc: 'Mirror the image with probability p (horizontal is the classic, almost always safe augmentation; vertical only where up/down has no meaning: aerial, microscopy). Boxes, keypoints and masks are flipped too. Note: left/right keypoint labels (e.g. eyes) are not swapped.',
  cv2: 'cv2.flip(img, 1)',
  keywords: 'random flip horizontal vertical mirror augmentation',
  params: [select('dir', 'Direction', [['h', 'Horizontal'], ['v', 'Vertical']], 'h')],
  engine: 'both',
  kind: 'random',
  affectsGeometry: true,
  needs: ['flip'],
  summary: (p) => (p.dir === 'h' ? 'horizontal' : 'vertical'),
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const dst = ctx.mat();
    ctx.cv.flip(src, dst, CODE[p.dir]);
    ctx.t.flip(CODE[p.dir]);
    return dst;
  },
  emitPython: (p) => ({ body: `img = cv2.flip(img, ${CODE[p.dir]})\nt.flip(${CODE[p.dir]})` }),
  emitAlbumentations: (p) => (p.dir === 'h' ? 'A.HorizontalFlip(p={p})' : 'A.VerticalFlip(p={p})'),
  emitTorchvision: (p) => (p.dir === 'h' ? 'v2.RandomHorizontalFlip(p={p})' : 'v2.RandomVerticalFlip(p={p})'),
  emitTf: (p) => (p.dir === 'h' ? 'tf.image.flip_left_right(img)' : 'tf.image.flip_up_down(img)'),
  testCases: [{ p: 0.5, seed: 1 }, { params: { dir: 'v' } }],
});
