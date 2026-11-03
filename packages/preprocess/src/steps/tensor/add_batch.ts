import { defineStep } from '../../registry';

defineStep({
  id: 'add_batch',
  category: 'Normalise & tensor',
  label: 'Add batch dimension',
  doc: 'Prepend a batch axis of size 1, e.g. (3, 224, 224) → (1, 3, 224, 224), for single-image inference with ONNX Runtime / TensorRT / TorchScript. Leave it out when a DataLoader builds batches.',
  cv2: 'img[None]',
  keywords: 'batch dimension unsqueeze expand dims inference onnx',
  params: [],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { layout: ['HWC', 'CHW'] },
  defaultMode: 'inference',
  outMeta: (m) => ({ ...m, batch: true }),
  check: (m) => (m.batch ? [{ level: 'error' as const, message: 'The image already has a batch dimension.' }] : []),
  applyBrowser: (src) => src,
  emitPython: () => ({ body: 'img = img[None]  # (1, ...)' }),
  emitTf: () => 'tf.expand_dims(img, 0)',
  testCases: [{ before: [{ def: 'scale_01' }, { def: 'to_chw' }] }],
});
