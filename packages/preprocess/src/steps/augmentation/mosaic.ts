import { size } from '../../params';
import { defineStep } from '../../registry';

defineStep({
  id: 'mosaic',
  category: 'Augmentation',
  label: 'Mosaic (batch, YOLO)',
  doc: 'Batch-level (YOLO): tile 4 images into one 2×2 mosaic around a random centre and shift their boxes accordingly, so each training image shows many objects at varied scales. Implemented in dataset.py (it needs 4 images); preview it with 2–4 loaded samples. Usually switched off for the last epochs.',
  cv2: '4 images → one (2H, 2W) canvas around a random centre, boxes shifted',
  keywords: 'mosaic yolo batch four images tile detection augmentation',
  params: [size('size', 'Tile size (W × H)', [640, 640])],
  engine: 'browser',
  kind: 'random',
  affectsGeometry: false,
  summary: (p) => `${p.size[0]}×${p.size[1]}`,
  outMeta: (m) => m,
  batch: {
    emitBatchPython: (p) =>
      `def mosaic(samples, rng, size=(${p.size[0]}, ${p.size[1]})):
    """YOLO mosaic of 4 (image HWC uint8, boxes (N, 5) xyxy+cls) samples → one image of size (2H, 2W)."""
    w, h = size
    cx, cy = int(rng.uniform(w * 0.5, w * 1.5)), int(rng.uniform(h * 0.5, h * 1.5))
    canvas = np.full((2 * h, 2 * w, 3), 114, np.uint8)
    out_boxes = []
    for i, (img, boxes) in enumerate(samples[:4]):
        ih, iw = img.shape[:2]
        if i == 0:
            x1, y1, x2, y2 = max(cx - iw, 0), max(cy - ih, 0), cx, cy
            sx, sy = iw - (x2 - x1), ih - (y2 - y1)
        elif i == 1:
            x1, y1, x2, y2 = cx, max(cy - ih, 0), min(cx + iw, 2 * w), cy
            sx, sy = 0, ih - (y2 - y1)
        elif i == 2:
            x1, y1, x2, y2 = max(cx - iw, 0), cy, cx, min(cy + ih, 2 * h)
            sx, sy = iw - (x2 - x1), 0
        else:
            x1, y1, x2, y2 = cx, cy, min(cx + iw, 2 * w), min(cy + ih, 2 * h)
            sx, sy = 0, 0
        canvas[y1:y2, x1:x2] = img[sy : sy + (y2 - y1), sx : sx + (x2 - x1)]
        b = np.asarray(boxes, np.float64).reshape(-1, 5).copy()
        b[:, [0, 2]] += x1 - sx
        b[:, [1, 3]] += y1 - sy
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, 2 * w)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, 2 * h)
        out_boxes.append(b[(b[:, 2] - b[:, 0] > 2) & (b[:, 3] - b[:, 1] > 2)])
    return canvas, np.concatenate(out_boxes)`,
  },
  applyBrowser: (src) => src,
  emitPython: () => ({ body: '# Mosaic is applied per batch in dataset.py' }),
});
