import { float } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';

defineStep({
  id: 'cutmix',
  category: 'Augmentation',
  label: 'CutMix (batch)',
  doc: 'Batch-level: paste a random rectangle from another image of the batch and mix labels by the pasted area (λ ~ Beta(α, α) sets the area). Applied in dataset.py\'s collate function, not in preprocess(). Preview it with 2+ loaded samples.',
  cv2: 'x1[:, y:y+h, x:x+w] = x2[:, y:y+h, x:x+w]',
  keywords: 'cutmix cut mix batch paste label mixing augmentation',
  params: [float('alpha', 'α (Beta)', 1, { min: 0.01, max: 4, step: 0.01 })],
  engine: 'browser',
  kind: 'random',
  affectsGeometry: false,
  summary: (p) => `α ${p.alpha}`,
  outMeta: (m) => m,
  batch: {
    emitBatchPython: (p) =>
      `def cutmix(images, labels, rng, alpha=${pyFloat(p.alpha)}):
    """CutMix on a channels-first batch (B, C, H, W); labels (B, C) one-hot / soft."""
    lam = float(rng.beta(alpha, alpha))
    perm = rng.permutation(len(images))
    h, w = images.shape[-2:]
    ch, cw = int(h * np.sqrt(1 - lam)), int(w * np.sqrt(1 - lam))
    cy, cx = int(rng.integers(h)), int(rng.integers(w))
    y0, y1, x0, x1 = max(cy - ch // 2, 0), min(cy + ch // 2, h), max(cx - cw // 2, 0), min(cx + cw // 2, w)
    images = images.clone() if hasattr(images, "clone") else images.copy()
    images[..., y0:y1, x0:x1] = images[perm][..., y0:y1, x0:x1]
    lam = 1 - (y1 - y0) * (x1 - x0) / (h * w)  # exact pasted area
    return images, lam * labels + (1 - lam) * labels[perm]`,
  },
  applyBrowser: (src) => src,
  emitPython: () => ({ body: '# CutMix is applied per batch in dataset.py' }),
});
