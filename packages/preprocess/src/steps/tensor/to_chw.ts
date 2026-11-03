import { defineStep } from '../../registry';

defineStep({
  id: 'to_chw',
  category: 'Normalise & tensor',
  label: 'HWC → CHW',
  doc: 'Move channels first, (H, W, C) → (C, H, W), the layout PyTorch models expect. Gray images become (1, H, W). Keep this at the end: OpenCV functions need HWC. TensorFlow / Keras models use HWC, so leave it out for them.',
  cv2: 'np.ascontiguousarray(img.transpose(2, 0, 1))',
  keywords: 'transpose channels first pytorch tensor layout permute',
  params: [],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { layout: ['HWC'] },
  outMeta: (m) => ({ ...m, layout: 'CHW' }),
  // The browser keeps data in HWC; only the layout tag changes.
  applyBrowser: (src) => src,
  emitPython: (_p, ctx) => ({
    body: ctx.in.c === 1 && ctx.in.layout === 'HWC' ? 'img = img[None] if img.ndim == 2 else np.ascontiguousarray(img.transpose(2, 0, 1))' : 'img = np.ascontiguousarray(img.transpose(2, 0, 1))',
  }),
  emitTorchvision: () => 'v2.Identity()  # tensors are already CHW',
  emitTf: () => 'img  # TensorFlow uses HWC; no transpose',
});
