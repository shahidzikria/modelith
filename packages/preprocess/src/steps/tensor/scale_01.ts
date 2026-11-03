import { defineStep } from '../../registry';
import { mapPixels } from '../util';

defineStep({
  id: 'scale_01',
  category: 'Normalise & tensor',
  label: 'Scale to [0, 1]',
  doc: 'Divide by the dtype maximum (255 for uint8, 65535 for uint16) to get float32 in [0, 1], without mean / std normalisation. What ToTensor() does; enough for models trained from scratch.',
  cv2: 'img.astype(np.float32) / 255.0',
  keywords: 'scale 0 1 totensor float normalize divide 255',
  params: [],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { dtype: ['uint8', 'uint16'] },
  outMeta: (m) => ({ ...m, dtype: 'float32', range: [0, 1] }),
  applyBrowser: (src, _p, ctx) => {
    const k = ctx.in.dtype === 'uint8' ? 255 : 65535;
    return mapPixels(src, ctx, 'float32', (v) => v / k);
  },
  emitPython: (_p, ctx) => ({ body: `img = img.astype(np.float32) / ${ctx.in.dtype === 'uint8' ? '255.0' : '65535.0'}` }),
  emitAlbumentations: (_p, ctx) => `A.ToFloat(max_value=${ctx.in.dtype === 'uint8' ? 255 : 65535}, p=1.0)`,
  emitTorchvision: () => 'v2.ToDtype(torch.float32, scale=True)',
  emitTf: (_p, ctx) => `tf.cast(img, tf.float32) / ${ctx.in.dtype === 'uint8' ? '255.0' : '65535.0'}`,
  testCases: [{}, { input: 'u16' }],
});
