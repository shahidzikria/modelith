import { select } from '../../params';
import { defineStep } from '../../registry';
import { grayOf, grayPy } from '../geometric/deskew';

defineStep({
  id: 'gradient_magnitude',
  category: 'Edges',
  label: 'Gradient magnitude & direction',
  doc: 'Sobel gx, gy → magnitude and angle (cv2.cartToPolar). "Direction as colour" encodes the angle as hue and the magnitude as brightness (HSV → BGR), which makes edge orientation visible; "Magnitude" returns float32 √(gx² + gy²).',
  cv2: 'cv2.cartToPolar(gx, gy, angleInDegrees=True)',
  keywords: 'gradient magnitude direction orientation angle hsv visualise cartToPolar',
  params: [select('out', 'Output', [['hsv', 'Direction as colour (BGR uint8)'], ['magnitude', 'Magnitude (float32)']], 'hsv')],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { dtype: ['uint8', 'float32'], c: [1, 3, 4] },
  needs: ['Sobel', 'cartToPolar'],
  tolerance: 1,
  summary: (p) => (p.out === 'hsv' ? 'direction colour' : 'magnitude'),
  outMeta: (m, p) => (p.out === 'hsv' ? { ...m, c: 3, dtype: 'uint8', space: 'BGR', range: [0, 255] } : { ...m, c: 1, dtype: 'float32', space: 'GRAY', range: [0, 1443] }),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const gray = grayOf(src, ctx);
    const gx = ctx.mat();
    const gy = ctx.mat();
    cv.Sobel(gray, gx, cv.CV_32F, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT);
    cv.Sobel(gray, gy, cv.CV_32F, 0, 1, 3, 1, 0, cv.BORDER_DEFAULT);
    const mag = ctx.mat();
    const ang = ctx.mat();
    cv.cartToPolar(gx, gy, mag, ang, true);
    if (p.out === 'magnitude') return mag;
    const h = ctx.mat();
    ang.convertTo(h, cv.CV_8U, 0.5, 0); // degrees → OpenCV hue (0..179)
    const v = ctx.mat();
    cv.normalize(mag, v, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
    const s = ctx.keep(new cv.Mat(src.rows, src.cols, cv.CV_8UC1, new cv.Scalar(255)));
    const parts = ctx.keep(new cv.MatVector());
    parts.push_back(h);
    parts.push_back(s);
    parts.push_back(v);
    const hsv = ctx.mat();
    cv.merge(parts, hsv);
    const dst = ctx.mat();
    cv.cvtColor(hsv, dst, cv.COLOR_HSV2BGR);
    return dst;
  },
  emitPython: (p, ctx) => {
    const lines = [
      grayPy(ctx.in),
      'gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)',
      'gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)',
      'mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)',
    ];
    if (p.out === 'magnitude') lines.push('img = mag');
    else
      lines.push(
        'h = cv2.convertScaleAbs(ang, alpha=0.5)  # degrees → OpenCV hue 0..179',
        'v = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)',
        'img = cv2.cvtColor(cv2.merge([h, np.full_like(h, 255), v]), cv2.COLOR_HSV2BGR)',
      );
    return { body: lines.join('\n') };
  },
  testCases: [{}, { params: { out: 'magnitude' }, tolerance: 1e-3 }],
});
