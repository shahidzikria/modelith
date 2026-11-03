import { float, interp } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import type { ImageMeta, Params } from '../../types';
import { interpCv, interpPy, pyRaw } from '../util';

/** Camera matrix with "auto" defaults: focal length = longest side, principal point = centre. */
export function cameraMatrix(m: ImageMeta, p: Params): number[] {
  const f = Math.max(m.w, m.h);
  const fx = p.fx > 0 ? p.fx : f;
  const fy = p.fy > 0 ? p.fy : f;
  const cx = p.cx >= 0 ? p.cx : (m.w - 1) / 2;
  const cy = p.cy >= 0 ? p.cy : (m.h - 1) / 2;
  return [fx, 0, cx, 0, fy, cy, 0, 0, 1];
}

defineStep({
  id: 'undistort',
  category: 'Geometric',
  label: 'Lens undistortion',
  doc: 'Remove radial / tangential lens distortion with the camera matrix (fx, fy, cx, cy) and distortion coefficients (k1, k2, p1, p2, k3) from cv2.calibrateCamera. Negative k1 fixes barrel distortion (wide-angle, GoPro). fx/fy ≤ 0 use the longest image side; cx/cy < 0 use the centre. Masks follow via the same remap (not invertible by postprocess_*).',
  cv2: 'cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1) + cv2.remap',
  keywords: 'undistort lens distortion barrel fisheye calibration camera matrix',
  params: [
    float('k1', 'k1', -0.25, { min: -2, max: 2, step: 0.01 }),
    float('k2', 'k2', 0.05, { min: -2, max: 2, step: 0.01 }),
    float('p1', 'p1', 0, { min: -0.2, max: 0.2, step: 0.001 }),
    float('p2', 'p2', 0, { min: -0.2, max: 0.2, step: 0.001 }),
    float('k3', 'k3', 0, { min: -2, max: 2, step: 0.01 }),
    float('fx', 'fx (0 = auto)', 0, { min: 0, step: 1 }),
    float('fy', 'fy (0 = auto)', 0, { min: 0, step: 1 }),
    float('cx', 'cx (−1 = centre)', -1, { step: 1 }),
    float('cy', 'cy (−1 = centre)', -1, { step: 1 }),
    interp('LINEAR'),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: true,
  needs: ['initUndistortRectifyMap', 'remap'],
  tolerance: 1,
  summary: (p) => `k1 ${p.k1} · k2 ${p.k2}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const { w, h } = ctx.in;
    const K = ctx.keep(cv.matFromArray(3, 3, cv.CV_64F, cameraMatrix(ctx.in, p)));
    const D = ctx.keep(cv.matFromArray(1, 5, cv.CV_64F, [p.k1, p.k2, p.p1, p.p2, p.k3]));
    const mx = ctx.mat();
    const my = ctx.mat();
    cv.initUndistortRectifyMap(K, D, ctx.mat(), K, new cv.Size(w, h), cv.CV_32FC1, mx, my);
    const dst = ctx.mat();
    cv.remap(src, dst, mx, my, interpCv(cv, p.interp), cv.BORDER_CONSTANT, new cv.Scalar());
    ctx.t.remap(new Float32Array(mx.data32F), new Float32Array(my.data32F), w, h);
    return dst;
  },
  emitPython: (p, ctx) => {
    const D = ctx.c('dist', [p.k1, p.k2, p.p1, p.p2, p.k3], 'k1, k2, p1, p2, k3');
    const F = ctx.c('focal', [p.fx, p.fy], 'fx, fy in pixels (0 = longest image side)');
    const C = ctx.c('center', [p.cx, p.cy], 'cx, cy in pixels (-1 = image centre)');
    const it = ctx.c('interp', interpPy(p.interp));
    return {
      body: [
        'h, w = img.shape[:2]',
        `fx, fy = (f if f > 0 else max(w, h) for f in ${F})`,
        `cx, cy = ${C}`,
        'cx, cy = (cx if cx >= 0 else (w - 1) / 2), (cy if cy >= 0 else (h - 1) / 2)',
        'K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)',
        `map_x, map_y = cv2.initUndistortRectifyMap(K, np.array(${D}, np.float64), None, K, (w, h), cv2.CV_32FC1)`,
        `img = cv2.remap(img, map_x, map_y, ${it}, borderMode=cv2.BORDER_CONSTANT)`,
        't.remap(map_x, map_y)',
      ].join('\n'),
    };
  },
  testCases: [{ targets: false }, { params: { k1: 0.2, k2: 0, p1: 0.01, fx: 100 }, targets: false }],
});
void pyFloat;
void pyRaw;
