import { bool } from '../../params';
import { defineStep } from '../../registry';

const CODES = [undefined, 'ROTATE_90_CLOCKWISE', 'ROTATE_180', 'ROTATE_90_COUNTERCLOCKWISE'];

defineStep({
  id: 'random_rotate90',
  category: 'Augmentation',
  label: 'Random rotate 90°',
  doc: 'Rotate by a random multiple of 90° (0, 90, 180 or 270°): lossless, no border filling. Standard for satellite, aerial, microscopy and pathology images where orientation is arbitrary. With "allow 0°" off, the image always changes.',
  cv2: 'cv2.rotate(img, [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE][k])',
  keywords: 'random rotate 90 quarter turn dihedral satellite aerial augmentation',
  params: [bool('allowZero', 'Allow 0°', true)],
  engine: 'both',
  kind: 'random',
  affectsGeometry: true,
  dynamicSize: true,
  needs: ['rotate'],
  summary: (p) => (p.allowZero ? '0–270°' : '90–270°'),
  outMeta: (m) => (m.h === m.w ? m : { ...m, h: Math.max(m.h, m.w), w: Math.max(m.h, m.w) }),
  check: (m) => (m.h !== m.w ? [{ level: 'info' as const, message: 'Non-square image: 90° turns swap width and height, so the output size varies per image.' }] : []),
  applyBrowser: (src, p, ctx) => {
    const k = ctx.rng.randint(p.allowZero ? 0 : 1, 3);
    if (!k) return src;
    const dst = ctx.mat();
    ctx.cv.rotate(src, dst, ctx.cv[CODES[k]!]);
    ctx.t.rot90(k);
    return dst;
  },
  emitPython: (p, ctx) => {
    ctx.helper('randint');
    return {
      body: `k = _randint(rng, ${p.allowZero ? 0 : 1}, 3)  # quarter turns clockwise\nif k:\n    img = cv2.rotate(img, [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE][k])\n    t.rot90(k)`,
    };
  },
  emitAlbumentations: () => 'A.RandomRotate90(p={p})',
  testCases: [{ seed: 3 }, { seed: 4, params: { allowZero: false } }],
});
