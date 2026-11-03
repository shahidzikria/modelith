import { border, float, interp } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { borderCv, borderPy, interpCv, interpPy, pyRaw } from '../util';

defineStep({
  id: 'elastic',
  category: 'Augmentation',
  label: 'Elastic transform',
  doc: 'Simard-style elastic deformation: random displacement fields (uniform noise in [−1, 1]) smoothed by a Gaussian of σ and scaled by α, applied with cv2.remap. Simulates tissue / handwriting deformation; a classic for medical segmentation and digit recognition. Masks follow with nearest-neighbour remap.',
  cv2: 'dx = cv2.GaussianBlur(rng.random((h, w)) * 2 - 1, (0, 0), sigma) * alpha → cv2.remap',
  keywords: 'elastic deformation transform simard warp distortion medical augmentation',
  params: [float('alpha', 'α (strength, px)', 30, { min: 0, max: 300, step: 1 }), float('sigma', 'σ (smoothness, px)', 6, { min: 1, max: 50, step: 0.5 }), interp('LINEAR'), border('REFLECT_101')],
  engine: 'both',
  kind: 'random',
  affectsGeometry: true,
  needs: ['remap', 'GaussianBlur'],
  tolerance: 1,
  summary: (p) => `α ${p.alpha} · σ ${p.sigma}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const { w, h } = ctx.in;
    const field = () => {
      const r = ctx.rng.randomArray(w * h);
      const m = ctx.keep(new cv.Mat(h, w, cv.CV_32FC1));
      const d = m.data32F as Float32Array;
      for (let i = 0; i < d.length; i++) d[i] = r[i] * 2 - 1;
      const b = ctx.mat();
      cv.GaussianBlur(m, b, new cv.Size(0, 0), p.sigma, p.sigma, cv.BORDER_DEFAULT);
      return b.data32F as Float32Array;
    };
    const dx = field();
    const dy = field();
    const a = Math.fround(p.alpha);
    const mx = ctx.keep(new cv.Mat(h, w, cv.CV_32FC1));
    const my = ctx.keep(new cv.Mat(h, w, cv.CV_32FC1));
    const X = mx.data32F as Float32Array, Y = my.data32F as Float32Array;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        X[i] = Math.fround(x + Math.fround(dx[i] * a));
        Y[i] = Math.fround(y + Math.fround(dy[i] * a));
      }
    const dst = ctx.mat();
    cv.remap(src, dst, mx, my, interpCv(cv, p.interp), borderCv(cv, p.border), new cv.Scalar());
    ctx.t.remap(new Float32Array(X), new Float32Array(Y), w, h);
    return dst;
  },
  emitPython: (p, ctx) => {
    const A = ctx.c('alpha', pyRaw(`np.float32(${pyFloat(p.alpha)})`), 'displacement strength in pixels');
    const S = ctx.c('sigma', pyRaw(pyFloat(p.sigma)), 'smoothness');
    return {
      body: [
        'h, w = img.shape[:2]',
        `dx = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1).astype(np.float32), (0, 0), ${S}) * ${A}`,
        `dy = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1).astype(np.float32), (0, 0), ${S}) * ${A}`,
        'xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))',
        'map_x, map_y = xx + dx, yy + dy',
        `img = cv2.remap(img, map_x, map_y, ${ctx.c('interp', interpPy(p.interp))}, borderMode=${ctx.c('border', borderPy(p.border))})`,
        't.remap(map_x, map_y)',
      ].join('\n'),
    };
  },
  emitAlbumentations: (p) => `A.ElasticTransform(alpha=${p.alpha}, sigma=${p.sigma}, interpolation=cv2.INTER_${p.interp}, p={p})`,
  emitTorchvision: (p) => `v2.RandomApply([v2.ElasticTransform(alpha=${pyFloat(p.alpha)}, sigma=${pyFloat(p.sigma)})], p={p})`,
  testCases: [{ seed: 41, targets: false }],
});
