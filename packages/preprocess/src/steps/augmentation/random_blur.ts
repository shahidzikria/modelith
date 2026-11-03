import { range, select } from '../../params';
import { defineStep } from '../../registry';
import type { BrowserCtx, Mat } from '../../types';

/** Normalised k×k line kernel at `angle` degrees (motion blur). Same construction as the emitted Python. */
export function motionKernel(ctx: BrowserCtx, k: number, angle: number): Mat {
  const { cv } = ctx;
  const line = ctx.keep(cv.Mat.zeros(k, k, cv.CV_32FC1));
  const d = line.data32F as Float32Array;
  for (let x = 0; x < k; x++) d[Math.floor(k / 2) * k + x] = 1;
  const M = ctx.keep(cv.getRotationMatrix2D(new cv.Point((k - 1) / 2, (k - 1) / 2), angle, 1));
  const rot = ctx.mat();
  cv.warpAffine(line, rot, M, new cv.Size(k, k), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
  const r = rot.data32F as Float32Array;
  let s = 0;
  for (const v of r) s += v;
  const out = ctx.keep(new cv.Mat(k, k, cv.CV_32FC1));
  const o = out.data32F as Float32Array;
  for (let i = 0; i < r.length; i++) o[i] = Math.fround(r[i] / s);
  return out;
}

defineStep({
  id: 'random_blur',
  category: 'Augmentation',
  label: 'Random blur',
  doc: 'Blur with a random odd kernel size from the range: Gaussian (defocus), median (smudge) or motion (camera shake: a line kernel at a random angle). Draws: kernel size, then the motion angle.',
  cv2: 'cv2.GaussianBlur / cv2.medianBlur / cv2.filter2D(img, -1, line_kernel)',
  keywords: 'random blur gaussian median motion shake defocus augmentation',
  params: [select('kind', 'Blur', [['gaussian', 'Gaussian'], ['median', 'Median'], ['motion', 'Motion']], 'gaussian'), range('k', 'Kernel size (odd)', [3, 7], { min: 3, max: 31, step: 2 })],
  engine: 'both',
  kind: 'random',
  affectsGeometry: false,
  needs: ['GaussianBlur', 'medianBlur', 'filter2D', 'warpAffine'],
  tolerance: 1,
  summary: (p) => `${p.kind} ${p.k[0]}–${p.k[1]}`,
  outMeta: (m) => m,
  check: (m, p) => (p.kind === 'median' && p.k[1] > 5 && m.dtype !== 'uint8' ? [{ level: 'error' as const, message: 'Median blur with kernels > 5 needs uint8.' }] : []),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const k = ctx.rng.randint(Math.floor(p.k[0] / 2), Math.floor(p.k[1] / 2)) * 2 + 1;
    const dst = ctx.mat();
    if (p.kind === 'gaussian') cv.GaussianBlur(src, dst, new cv.Size(k, k), 0, 0, cv.BORDER_DEFAULT);
    else if (p.kind === 'median') cv.medianBlur(src, dst, k);
    else cv.filter2D(src, dst, -1, motionKernel(ctx, k, ctx.rng.uniform(0, 180)), new cv.Point(-1, -1), 0, cv.BORDER_DEFAULT);
    return dst;
  },
  emitPython: (p, ctx) => {
    ctx.helper('randint');
    const K = ctx.c('ksize', p.k, 'odd kernel sizes');
    const lines = [`k = _randint(rng, ${K}[0] // 2, ${K}[1] // 2) * 2 + 1  # odd`];
    if (p.kind === 'gaussian') lines.push('img = cv2.GaussianBlur(img, (k, k), 0)');
    else if (p.kind === 'median') lines.push('img = cv2.medianBlur(img, k)');
    else {
      ctx.helper('uniform');
      lines.push(
        'line = np.zeros((k, k), np.float32)',
        'line[k // 2, :] = 1',
        'M = cv2.getRotationMatrix2D(((k - 1) / 2, (k - 1) / 2), _uniform(rng, 0, 180), 1.0)',
        'kernel = cv2.warpAffine(line, M, (k, k))',
        'img = cv2.filter2D(img, -1, kernel / kernel.sum())  # motion blur along a random direction',
      );
    }
    return { body: lines.join('\n') };
  },
  emitAlbumentations: (p) =>
    p.kind === 'gaussian' ? `A.GaussianBlur(blur_limit=(${p.k[0]}, ${p.k[1]}), p={p})` : p.kind === 'median' ? `A.MedianBlur(blur_limit=(${p.k[0]}, ${p.k[1]}), p={p})` : `A.MotionBlur(blur_limit=(${p.k[0]}, ${p.k[1]}), p={p})`,
  testCases: [{ seed: 141 }, { seed: 142, params: { kind: 'median' } }, { seed: 143, params: { kind: 'motion', k: [5, 9] } }],
});
