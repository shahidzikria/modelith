import { select } from '../../params';
import { defineStep } from '../../registry';
import { mapPixels } from '../util';

defineStep({
  id: 'cast_float',
  category: 'Normalise & tensor',
  label: 'Cast to float32 / float16',
  doc: 'Change the floating-point precision of the final tensor: float16 halves memory and bandwidth for fp16 inference engines; values keep their scale (no normalisation).',
  cv2: 'img.astype(np.float16)',
  keywords: 'cast float16 half precision float32 dtype fp16',
  params: [select('to', 'To', [['float32', 'float32'], ['float16', 'float16 (half)']], 'float16')],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { layout: ['HWC', 'CHW'] },
  summary: (p) => p.to,
  outMeta: (m, p) => ({ ...m, dtype: p.to }),
  check: (m, p) => (m.dtype === 'uint8' && p.to === 'float16' ? [{ level: 'info' as const, message: 'Values stay 0–255; add "Scale to [0, 1]" or "Standardise" first if the model expects normalised input.' }] : []),
  // The browser keeps float16 data as float32 (rounded to the nearest float16 value for parity).
  applyBrowser: (src, p, ctx) =>
    mapPixels(src, ctx, 'float32', (v) => {
      if (p.to !== 'float16' || !Number.isFinite(v) || v === 0) return v;
      const e = Math.floor(Math.log2(Math.abs(v)));
      const q = 2 ** (Math.max(e, -14) - 10);
      return Math.round(v / q) * q;
    }),
  emitPython: (p) => ({ body: `img = img.astype(np.${p.to})` }),
  emitTorchvision: (p) => `v2.ToDtype(torch.${p.to})`,
  emitTf: (p) => `tf.cast(img, tf.${p.to})`,
  testCases: [{ before: [{ def: 'scale_01' }] }, { before: [{ def: 'scale_01' }], params: { to: 'float32' } }],
});
