import { select } from '../../params';
import { defineStep } from '../../registry';

const CODES: Record<string, number> = { h: 1, v: 0, both: -1 };

defineStep({
  id: 'flip',
  category: 'Geometric',
  label: 'Flip',
  doc: 'Mirror horizontally, vertically or both (cv2.flip codes 1, 0, −1). Boxes, keypoints and masks are mirrored too. For a random flip during training use "Random flip" in Augmentation.',
  cv2: 'cv2.flip(img, code)',
  keywords: 'mirror flip horizontal vertical',
  params: [select('dir', 'Direction', [['h', 'Horizontal (left ↔ right)'], ['v', 'Vertical (top ↔ bottom)'], ['both', 'Both (= rotate 180°)']], 'h')],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: true,
  needs: ['flip'],
  summary: (p) => ({ h: 'horizontal', v: 'vertical', both: 'both' })[p.dir as string] ?? p.dir,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const dst = ctx.mat();
    ctx.cv.flip(src, dst, CODES[p.dir]);
    ctx.t.flip(CODES[p.dir]);
    return dst;
  },
  emitPython: (p) => ({ body: `img = cv2.flip(img, ${CODES[p.dir]})  # 1 horizontal, 0 vertical, -1 both\nt.flip(${CODES[p.dir]})` }),
  emitAlbumentations: (p) => (p.dir === 'h' ? 'A.HorizontalFlip(p=1.0)' : p.dir === 'v' ? 'A.VerticalFlip(p=1.0)' : 'A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)])'),
  emitTorchvision: (p) => (p.dir === 'h' ? 'v2.RandomHorizontalFlip(p=1.0)' : p.dir === 'v' ? 'v2.RandomVerticalFlip(p=1.0)' : undefined),
  emitTf: (p) => (p.dir === 'h' ? 'tf.image.flip_left_right(img)' : p.dir === 'v' ? 'tf.image.flip_up_down(img)' : 'tf.image.rot90(img, 2)'),
  testCases: [{}, { params: { dir: 'v' } }, { params: { dir: 'both' } }],
});
