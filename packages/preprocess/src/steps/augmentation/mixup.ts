import { float } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';

defineStep({
  id: 'mixup',
  category: 'Augmentation',
  label: 'MixUp (batch)',
  doc: 'Batch-level: blend each image with another image of the batch, x = λ·x₁ + (1 − λ)·x₂, and mix the (one-hot) labels the same way; λ ~ Beta(α, α). Not part of the per-image preprocess(): dataset.py applies it in the collate function. Preview it with 2+ loaded samples.',
  cv2: 'lam * x1 + (1 - lam) * x2',
  keywords: 'mixup mix up batch blend label smoothing augmentation',
  params: [float('alpha', 'α (Beta)', 0.2, { min: 0.01, max: 4, step: 0.01 })],
  engine: 'browser',
  kind: 'random',
  affectsGeometry: false,
  summary: (p) => `α ${p.alpha}`,
  outMeta: (m) => m,
  batch: {
    emitBatchPython: (p) =>
      `def mixup(images, labels, rng, alpha=${pyFloat(p.alpha)}):
    """MixUp: images (B, ...) float tensor/array, labels (B, C) one-hot / soft. Returns mixed copies."""
    lam = float(rng.beta(alpha, alpha))
    perm = rng.permutation(len(images))
    return lam * images + (1 - lam) * images[perm], lam * labels + (1 - lam) * labels[perm]`,
  },
  applyBrowser: (src) => src,
  emitPython: () => ({ body: '# MixUp is applied per batch in dataset.py' }),
});
