import { float, interp } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { interpCv, interpPy, pyRaw } from '../util';

defineStep({
  id: 'optical_distortion',
  category: 'Augmentation',
  label: 'Optical distortion',
  doc: 'Random lens distortion: a radial coefficient k ∈ [−limit, limit] (negative = barrel, positive = pincushion) and a small random shift of the distortion centre, applied with initUndistortRectifyMap + remap. Makes models robust to different cameras / lenses.',
  cv2: 'cv2.initUndistortRectifyMap(K, (k, k, 0, 0, 0), None, K, (w, h), cv2.CV_32FC1) → cv2.remap',
  keywords: 'optical distortion lens barrel pincushion fisheye augmentation',
  params: [float('limit', 'Distortion limit k', 0.3, { min: 0, max: 2, step: 0.05 }), float('shift', 'Centre shift (fraction)', 0.05, { min: 0, max: 0.5, step: 0.01 }), interp('LINEAR')],
  engine: 'both',
  kind: 'random',
  affectsGeometry: true,
  needs: ['initUndistortRectifyMap', 'remap'],
  tolerance: 1,
  summary: (p) => `k ±${p.limit}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const { w, h } = ctx.in;
    const k = ctx.rng.uniform(-p.limit, p.limit);
    const dx = ctx.rng.uniform(-p.shift, p.shift) * w;
    const dy = ctx.rng.uniform(-p.shift, p.shift) * h;
    const f = Math.max(w, h);
    const K = ctx.keep(cv.matFromArray(3, 3, cv.CV_64F, [f, 0, (w - 1) / 2 + dx, 0, f, (h - 1) / 2 + dy, 0, 0, 1]));
    const D = ctx.keep(cv.matFromArray(1, 5, cv.CV_64F, [k, k, 0, 0, 0]));
    const mx = ctx.mat();
    const my = ctx.mat();
    cv.initUndistortRectifyMap(K, D, ctx.mat(), K, new cv.Size(w, h), cv.CV_32FC1, mx, my);
    const dst = ctx.mat();
    cv.remap(src, dst, mx, my, interpCv(cv, p.interp), cv.BORDER_REFLECT_101, new cv.Scalar());
    ctx.t.remap(new Float32Array(mx.data32F), new Float32Array(my.data32F), w, h);
    return dst;
  },
  emitPython: (p, ctx) => {
    ctx.helper('uniform');
    const L = ctx.c('limit', pyRaw(pyFloat(p.limit)));
    const S = ctx.c('shift', pyRaw(pyFloat(p.shift)), 'fraction of width / height');
    return {
      body: [
        'h, w = img.shape[:2]',
        `k = _uniform(rng, -${L}, ${L})  # < 0 barrel, > 0 pincushion`,
        `dx, dy = _uniform(rng, -${S}, ${S}) * w, _uniform(rng, -${S}, ${S}) * h`,
        'f = max(w, h)',
        'K = np.array([[f, 0, (w - 1) / 2 + dx], [0, f, (h - 1) / 2 + dy], [0, 0, 1]], np.float64)',
        'map_x, map_y = cv2.initUndistortRectifyMap(K, np.array([k, k, 0, 0, 0], np.float64), None, K, (w, h), cv2.CV_32FC1)',
        `img = cv2.remap(img, map_x, map_y, ${ctx.c('interp', interpPy(p.interp))}, borderMode=cv2.BORDER_REFLECT_101)`,
        't.remap(map_x, map_y)',
      ].join('\n'),
    };
  },
  emitAlbumentations: (p) => `A.OpticalDistortion(distort_limit=${p.limit}, interpolation=cv2.INTER_${p.interp}, p={p})`,
  testCases: [{ seed: 61, targets: false }],
});
