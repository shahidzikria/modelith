import { defineStep } from '../../registry';

defineStep({
  id: 'fill_holes',
  category: 'Morphology',
  label: 'Fill holes',
  doc: 'Fill background regions completely enclosed by foreground: pad with background, flood-fill from the border, invert, and OR with the mask. Output is binary 0 / 255. Typical after thresholding objects with dark interiors (cells, rings, letters "o").',
  cv2: 'cv2.floodFill(padded, mask, (0, 0), 255) → cv2.bitwise_not → cv2.bitwise_or',
  keywords: 'fill holes flood fill binary mask interior imfill',
  params: [],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1], dtype: ['uint8'] },
  needs: ['floodFill'],
  outMeta: (m) => m,
  applyBrowser: (src, _p, ctx) => {
    const { cv } = ctx;
    const bin = ctx.mat();
    cv.threshold(src, bin, 0, 255, cv.THRESH_BINARY);
    const padded = ctx.mat();
    cv.copyMakeBorder(bin, padded, 1, 1, 1, 1, cv.BORDER_CONSTANT, new cv.Scalar(0));
    const mask = ctx.keep(cv.Mat.zeros(padded.rows + 2, padded.cols + 2, cv.CV_8U));
    cv.floodFill(padded, mask, new cv.Point(0, 0), new cv.Scalar(255), new cv.Rect(), new cv.Scalar(), new cv.Scalar(), 4);
    const inv = ctx.mat();
    cv.bitwise_not(padded, inv);
    const holes = ctx.mat();
    ctx.keep(inv.roi(new cv.Rect(1, 1, src.cols, src.rows))).copyTo(holes);
    const dst = ctx.mat();
    cv.bitwise_or(bin, holes, dst);
    return dst;
  },
  emitPython: () => ({
    body: [
      'binary = np.where(img > 0, 255, 0).astype(np.uint8)',
      'padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)  # border is background',
      'mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)',
      'cv2.floodFill(padded, mask, (0, 0), 255)  # everything reachable from outside',
      'holes = cv2.bitwise_not(padded)[1:-1, 1:-1]',
      'img = cv2.bitwise_or(binary, holes)',
    ].join('\n'),
  }),
  testCases: [{ input: 'color', before: [{ def: 'convert_color', params: { to: 'GRAY' } }, { def: 'threshold', params: { method: 'fixed', thresh: 150, type: 'BINARY_INV' } }] }],
});
