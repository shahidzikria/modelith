import { border, bool, float, select } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import type { BrowserCtx, Mat } from '../../types';
import { borderPy, pyRaw } from '../util';
import { rotationMatrix, warpAffineBrowser, warpAffinePy } from './rotate';

/** Gray copy of the input for analysis (no-op for gray images). */
export function grayOf(src: Mat, ctx: BrowserCtx): Mat {
  const { cv } = ctx;
  if (src.channels() === 1) return src;
  const g = ctx.mat();
  const rgb = ctx.in.space.startsWith('RGB');
  cv.cvtColor(src, g, src.channels() === 4 ? (rgb ? cv.COLOR_RGBA2GRAY : cv.COLOR_BGRA2GRAY) : rgb ? cv.COLOR_RGB2GRAY : cv.COLOR_BGR2GRAY);
  return g;
}

export function grayPy(m: { c: number; space: string }, v = 'gray'): string {
  if (m.c === 1) return `${v} = img`;
  const rgb = m.space.startsWith('RGB');
  const code = m.c === 4 ? (rgb ? 'RGBA2GRAY' : 'BGRA2GRAY') : rgb ? 'RGB2GRAY' : 'BGR2GRAY';
  return `${v} = cv2.cvtColor(img, cv2.COLOR_${code})`;
}

/** Non-zero pixel coordinates as a CV_32SC2 Mat (cv2.findNonZero is not in OpenCV.js). */
export function nonZeroPoints(mask: Mat, ctx: BrowserCtx): { mat: Mat; n: number } {
  const { cv } = ctx;
  const d = mask.data as Uint8Array;
  const pts: number[] = [];
  for (let i = 0; i < d.length; i++) if (d[i]) pts.push(i % mask.cols, Math.floor(i / mask.cols));
  return { mat: ctx.keep(cv.matFromArray(pts.length / 2, 1, cv.CV_32SC2, pts)), n: pts.length / 2 };
}

function median(v: number[]): number {
  const s = [...v].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

/** Estimated skew in degrees (counter-clockwise rotation that straightens the text). */
export function estimateSkew(src: Mat, ctx: BrowserCtx, method: string, maxAngle: number): number {
  const { cv } = ctx;
  const gray = grayOf(src, ctx);
  let angle = 0;
  if (method === 'minarearect') {
    const ink = ctx.mat();
    cv.threshold(gray, ink, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
    const { mat, n } = nonZeroPoints(ink, ctx);
    if (n > 10) {
      angle = cv.minAreaRect(mat).angle;
      if (angle > 45) angle -= 90;
    }
  } else {
    const edges = ctx.mat();
    cv.Canny(gray, edges, 50, 150, 3, false);
    const lines = ctx.mat();
    cv.HoughLinesP(edges, lines, 1, Math.PI / 180, 50, Math.min(ctx.in.w, ctx.in.h) / 4, 10);
    const angs: number[] = [];
    const L = lines.data32S as Int32Array;
    for (let i = 0; i < lines.rows; i++) {
      const a = (Math.atan2(L[i * 4 + 3] - L[i * 4 + 1], L[i * 4 + 2] - L[i * 4]) * 180) / Math.PI;
      if (Math.abs(a) < maxAngle) angs.push(a);
    }
    if (angs.length) angle = median(angs);
  }
  return Math.abs(angle) > maxAngle ? 0 : angle;
}

defineStep({
  id: 'deskew',
  category: 'Geometric',
  label: 'Deskew',
  doc: 'Straighten scanned or photographed documents: estimate the text angle (minAreaRect of the Otsu "ink" pixels, or the median angle of Hough lines) and rotate by it. Angles beyond the limit are treated as "not skewed".',
  cv2: 'cv2.minAreaRect(cv2.findNonZero(ink)) → cv2.warpAffine',
  keywords: 'deskew straighten document scan text angle skew rotation ocr',
  params: [
    select('method', 'Angle from', [['minarearect', 'minAreaRect of ink'], ['hough', 'Hough lines (median)']], 'minarearect'),
    float('maxAngle', 'Max angle (°)', 30, { min: 1, max: 45, step: 1 }),
    bool('expand', 'Expand canvas', false),
    border('REPLICATE'),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: true,
  dynamicSize: true,
  needs: ['minAreaRect', 'HoughLinesP', 'warpAffine'],
  tolerance: 1,
  accepts: { dtype: ['uint8'] },
  summary: (p) => (p.method === 'hough' ? 'hough' : 'minAreaRect') + ` · ≤${p.maxAngle}°`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const angle = estimateSkew(src, ctx, p.method, p.maxAngle);
    const r = rotationMatrix(ctx.in, angle, 1, p.expand);
    return warpAffineBrowser(src, r.M, r.w, r.h, ctx, 'LINEAR', p.border, '#ffffff');
  },
  emitPython: (p, ctx) => {
    const mx = ctx.c('max_angle', pyRaw(pyFloat(p.maxAngle)), 'larger estimates are ignored');
    const est =
      p.method === 'minarearect'
        ? [
            '_, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # dark text → white',
            'pts = cv2.findNonZero(ink)',
            'angle = 0.0',
            'if pts is not None and len(pts) > 10:',
            '    angle = cv2.minAreaRect(pts)[2]  # (0, 90] in OpenCV >= 4.5',
            '    if angle > 45:',
            '        angle -= 90',
          ]
        : [
            'edges = cv2.Canny(gray, 50, 150)',
            'lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=min(gray.shape) / 4, maxLineGap=10)',
            'angles = [] if lines is None else [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in lines[:, 0]]',
            `angles = [a for a in angles if abs(a) < ${mx}]`,
            'angle = float(np.median(angles)) if angles else 0.0',
          ];
    const lines = [
      grayPy(ctx.in),
      ...est,
      `if abs(angle) > ${mx}:`,
      '    angle = 0.0',
      'h, w = img.shape[:2]',
      'M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.0)',
    ];
    if (p.expand)
      lines.push(
        'cos, sin = abs(M[0, 0]), abs(M[0, 1])',
        'size = (int(h * sin + w * cos + 0.5), int(h * cos + w * sin + 0.5))',
        'M[0, 2] += (size[0] - 1) / 2 - (w - 1) / 2',
        'M[1, 2] += (size[1] - 1) / 2 - (h - 1) / 2',
      );
    else lines.push('size = (w, h)');
    lines.push(warpAffinePy(ctx, 'LINEAR', p.border, '#ffffff'));
    return { body: lines.join('\n') };
  },
  testCases: [{ input: 'doc' }, { input: 'doc', params: { method: 'hough', expand: true } }],
});
void borderPy;
