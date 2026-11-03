import { border, float, int, interp } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { borderCv, borderPy, interpCv, interpPy, pyRaw } from '../util';

/** np.interp(x, xp, fp) for increasing xp. */
export function interp1(x: number, xp: number[], fp: number[]): number {
  if (x <= xp[0]) return fp[0];
  if (x >= xp[xp.length - 1]) return fp[fp.length - 1];
  let k = 0;
  while (k < xp.length - 2 && x >= xp[k + 1]) k++;
  return fp[k] + ((x - xp[k]) * (fp[k + 1] - fp[k])) / (xp[k + 1] - xp[k]);
}

/** Source coordinate for every output column / row of a randomly stretched grid. */
function axisMap(n: number, steps: number, deltas: number[]): Float64Array {
  const widths = deltas.map((d) => 1 + d);
  const total = widths.reduce((a, b) => a + b, 0);
  const dst = [0];
  for (const wd of widths) dst.push(dst[dst.length - 1] + (wd / total) * n);
  const src = Array.from({ length: steps + 1 }, (_, k) => (k * n) / steps);
  const out = new Float64Array(n);
  for (let x = 0; x < n; x++) out[x] = interp1(x, dst, src);
  return out;
}

defineStep({
  id: 'grid_distortion',
  category: 'Augmentation',
  label: 'Grid distortion',
  doc: 'Split the image into a grid of N cells per axis and stretch / squeeze each cell by a random factor (1 ± limit), giving a smooth, piecewise-linear warp (cv2.remap). Good for documents, cells and road scenes. Draws N x-factors, then N y-factors.',
  cv2: 'map_x = np.interp(…) per column, map_y per row → cv2.remap',
  keywords: 'grid distortion warp stretch remap augmentation',
  params: [int('steps', 'Cells per axis', 5, { min: 2, max: 20 }), float('limit', 'Distortion limit', 0.3, { min: 0, max: 0.9, step: 0.05 }), interp('LINEAR'), border('REFLECT_101')],
  engine: 'both',
  kind: 'random',
  affectsGeometry: true,
  needs: ['remap'],
  tolerance: 1,
  summary: (p) => `${p.steps} cells · ±${p.limit}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const { w, h } = ctx.in;
    const dxs = Array.from({ length: p.steps }, () => ctx.rng.uniform(-p.limit, p.limit));
    const dys = Array.from({ length: p.steps }, () => ctx.rng.uniform(-p.limit, p.limit));
    const ax = axisMap(w, p.steps, dxs);
    const ay = axisMap(h, p.steps, dys);
    const mx = ctx.keep(new cv.Mat(h, w, cv.CV_32FC1));
    const my = ctx.keep(new cv.Mat(h, w, cv.CV_32FC1));
    const X = mx.data32F as Float32Array, Y = my.data32F as Float32Array;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        X[y * w + x] = ax[x];
        Y[y * w + x] = ay[y];
      }
    const dst = ctx.mat();
    cv.remap(src, dst, mx, my, interpCv(cv, p.interp), borderCv(cv, p.border), new cv.Scalar());
    ctx.t.remap(new Float32Array(X), new Float32Array(Y), w, h);
    return dst;
  },
  emitPython: (p, ctx) => {
    ctx.helper('uniform');
    const N = ctx.c('steps', p.steps);
    const L = ctx.c('limit', pyRaw(pyFloat(p.limit)));
    return {
      body: [
        'h, w = img.shape[:2]',
        `dx = [_uniform(rng, -${L}, ${L}) for _ in range(${N})]`,
        `dy = [_uniform(rng, -${L}, ${L}) for _ in range(${N})]`,
        '',
        'def axis_map(n, deltas):',
        '    widths = 1 + np.array(deltas)',
        '    dst = np.concatenate([[0.0], np.cumsum(widths / widths.sum() * n)])  # stretched cell edges',
        `    src = np.arange(${N} + 1) * n / ${N}  # original cell edges`,
        '    return np.interp(np.arange(n), dst, src).astype(np.float32)',
        '',
        'map_x = np.tile(axis_map(w, dx), (h, 1))',
        'map_y = np.tile(axis_map(h, dy)[:, None], (1, w))',
        `img = cv2.remap(img, map_x, map_y, ${ctx.c('interp', interpPy(p.interp))}, borderMode=${ctx.c('border', borderPy(p.border))})`,
        't.remap(map_x, map_y)',
      ].join('\n'),
    };
  },
  emitAlbumentations: (p) => `A.GridDistortion(num_steps=${p.steps}, distort_limit=${p.limit}, interpolation=cv2.INTER_${p.interp}, p={p})`,
  testCases: [{ seed: 51, targets: false }],
});
