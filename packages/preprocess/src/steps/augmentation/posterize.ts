import { range } from '../../params';
import { defineStep } from '../../registry';
import { mapPixels } from '../util';

defineStep({
  id: 'posterize',
  category: 'Augmentation',
  label: 'Posterise',
  doc: 'Keep only the top k bits of every uint8 value (k drawn from the range): fewer distinct levels, like heavy quantisation or banding.',
  cv2: 'img & (0xFF << (8 - k))',
  keywords: 'posterize posterise bits quantise quantize levels augmentation',
  params: [range('bits', 'Bits kept', [4, 6], { min: 1, max: 8, step: 1 })],
  engine: 'both',
  kind: 'random',
  affectsGeometry: false,
  accepts: { dtype: ['uint8'] },
  summary: (p) => `${p.bits[0]}–${p.bits[1]} bits`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const k = ctx.rng.randint(p.bits[0], p.bits[1]);
    const mask = (0xff << (8 - k)) & 0xff;
    return mapPixels(src, ctx, 'uint8', (v) => v & mask);
  },
  emitPython: (p, ctx) => {
    ctx.helper('randint');
    return { body: `k = _randint(rng, *${ctx.c('bits', p.bits)})\nimg = img & np.uint8((0xFF << (8 - k)) & 0xFF)` };
  },
  emitAlbumentations: (p) => `A.Posterize(num_bits=(${p.bits[0]}, ${p.bits[1]}), p={p})`,
  emitTorchvision: (p) => `v2.RandomPosterize(bits=${p.bits[0]}, p={p})`,
  testCases: [{ seed: 121 }],
});
