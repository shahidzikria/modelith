import { int } from '../../params';
import { defineStep } from '../../registry';
import { mapPixels } from '../util';

defineStep({
  id: 'rgb_shift',
  category: 'Augmentation',
  label: 'RGB shift',
  doc: 'Add a random integer offset in [−limit, limit] to each colour channel (drawn in the image\'s channel order): simulates white-balance and sensor colour differences.',
  cv2: '_saturate(img + (dc0, dc1, dc2), np.uint8)',
  keywords: 'rgb shift channel offset colour cast white balance augmentation',
  params: [int('limit', 'Limit (±)', 20, { min: 0, max: 255 })],
  engine: 'both',
  kind: 'random',
  affectsGeometry: false,
  accepts: { c: [3], dtype: ['uint8'] },
  summary: (p) => `±${p.limit}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const s = [0, 1, 2].map(() => ctx.rng.randint(-p.limit, p.limit));
    return mapPixels(src, ctx, 'uint8', (v, k) => v + s[k]);
  },
  emitPython: (p, ctx) => {
    ctx.helper('randint');
    ctx.helper('saturate');
    const L = ctx.c('limit', p.limit);
    return { body: `shift = np.array([_randint(rng, -${L}, ${L}) for _ in range(3)])  # per channel, image order\nimg = _saturate(img.astype(np.int16) + shift, np.uint8)` };
  },
  emitAlbumentations: (p) => `A.RGBShift(r_shift_limit=${p.limit}, g_shift_limit=${p.limit}, b_shift_limit=${p.limit}, p={p})`,
  testCases: [{ seed: 91 }],
});
