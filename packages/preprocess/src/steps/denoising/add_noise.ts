import { float, select } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import type { BrowserCtx, Mat } from '../../types';
import { DTYPE_MAX, mapPixels, pyRaw } from '../util';

/** Standard normals via Box-Muller on two rng.random() arrays — the `_normal()` helper in the Python code. */
export function normals(ctx: BrowserCtx, n: number): Float64Array {
  const u1 = ctx.rng.randomArray(n);
  const u2 = ctx.rng.randomArray(n);
  const z = new Float64Array(n);
  for (let i = 0; i < n; i++) z[i] = Math.sqrt(-2 * Math.log(1 - u1[i])) * Math.cos(2 * Math.PI * u2[i]);
  return z;
}

/** Shared by "Add noise" and the noise augmentations. `amount` is σ in pixel units (gaussian / poisson scale), the fraction for salt & pepper, the variance for speckle. */
export function addNoiseBrowser(src: Mat, ctx: BrowserCtx, kind: string, amount: number): Mat {
  const { h, w, c, dtype } = ctx.in;
  const max = dtype === 'uint8' || dtype === 'uint16' ? DTYPE_MAX[dtype] : 1;
  const out = dtype === 'uint8' || dtype === 'uint16' ? dtype : 'float32';
  const clip = (v: number) => (out === 'float32' ? Math.min(1, Math.max(0, v)) : v);
  if (kind === 'salt_pepper') {
    const r = ctx.rng.randomArray(h * w);
    return mapPixels(src, ctx, out, (v, _k, i) => {
      const x = r[Math.floor(i / c)];
      return x < amount / 2 ? max : x > 1 - amount / 2 ? 0 : v;
    });
  }
  const z = normals(ctx, h * w * c);
  if (kind === 'gaussian') return mapPixels(src, ctx, out, (v, k, i) => clip(v + z[i + k] * amount));
  if (kind === 'speckle') return mapPixels(src, ctx, out, (v, k, i) => clip(v + v * z[i + k] * Math.sqrt(amount)));
  // Poisson (shot) noise, normal approximation: σ = sqrt(x · scale).
  return mapPixels(src, ctx, out, (v, k, i) => clip(v + z[i + k] * Math.sqrt(Math.max(v, 0) * amount)));
}

export function addNoisePy(ctx: { in: { dtype: string; c: number }; helper(n: string): void; c(n: string, v: unknown, cm?: string): string }, kind: string, amount: number): string {
  const max = ctx.in.dtype === 'uint8' ? '255' : ctx.in.dtype === 'uint16' ? '65535' : '1.0';
  const A = ctx.c('amount', pyRaw(pyFloat(amount)), kind === 'gaussian' ? 'sigma in pixel units' : kind === 'salt_pepper' ? 'fraction of pixels' : kind === 'speckle' ? 'variance' : 'scale');
  const isInt = ctx.in.dtype !== 'float32' && ctx.in.dtype !== 'float16';
  const finish = (expr: string) => {
    if (isInt) {
      ctx.helper('saturate');
      return `img = _saturate(${expr}, np.${ctx.in.dtype})`;
    }
    return `img = np.clip(${expr}, 0.0, 1.0).astype(np.float32)`;
  };
  if (kind === 'salt_pepper') {
    return [
      'r = rng.random(img.shape[:2])  # one draw per pixel (all channels together)',
      'img = img.copy()',
      `img[r < ${A} / 2] = ${max}  # salt`,
      `img[r > 1 - ${A} / 2] = 0  # pepper`,
    ].join('\n');
  }
  ctx.helper('normal');
  const z = 'z = _normal(rng, img.shape)';
  if (kind === 'gaussian') return `${z}\n${finish(`img + z * ${A}`)}`;
  if (kind === 'speckle') return `${z}\n${finish(`img + img * z * np.sqrt(${A})`)}`;
  return `${z}  # Poisson (shot) noise, normal approximation\n${finish(`img + z * np.sqrt(np.maximum(img, 0) * ${A})`)}`;
}

defineStep({
  id: 'add_noise',
  category: 'Denoising',
  label: 'Add noise',
  doc: 'Add synthetic noise: Gaussian (σ in pixel units), salt & pepper (fraction of pixels set to white / black), speckle (multiplicative, variance) or Poisson shot noise (normal approximation, σ = √(x·scale)). Use it to test denoisers or as an augmentation; it uses the pipeline seed.',
  cv2: 'img + rng-drawn noise (NumPy)',
  keywords: 'noise gaussian salt pepper speckle poisson shot augmentation synthetic',
  params: [
    select('kind', 'Noise', [['gaussian', 'Gaussian'], ['salt_pepper', 'Salt & pepper'], ['speckle', 'Speckle'], ['poisson', 'Poisson (shot)']], 'gaussian'),
    float('amount', 'Amount', 12, { min: 0, max: 100, step: 0.5, help: 'Gaussian: σ (pixels); salt & pepper: fraction 0–1; speckle: variance; Poisson: scale.' }),
  ],
  engine: 'both',
  kind: 'random',
  defaultMode: 'both',
  affectsGeometry: false,
  accepts: { dtype: ['uint8', 'uint16', 'float32'] },
  tolerance: 1,
  summary: (p) => `${String(p.kind).replace('_', ' & ')} ${p.amount}`,
  outMeta: (m) => m,
  check: (_m, p) => (p.kind === 'salt_pepper' && p.amount > 1 ? [{ level: 'error' as const, message: 'Salt & pepper amount is a fraction of pixels (0–1).', fixes: [{ label: 'Use 0.05', kind: 'setParams' as const, params: { amount: 0.05 } }] }] : []),
  applyBrowser: (src, p, ctx) => addNoiseBrowser(src, ctx, p.kind, p.amount),
  emitPython: (p, ctx) => ({ body: addNoisePy(ctx, p.kind, p.amount) }),
  emitAlbumentations: (p) =>
    p.kind === 'gaussian' ? `A.GaussNoise(std_range=(${(p.amount / 255).toFixed(4)}, ${(p.amount / 255).toFixed(4)}), p=1.0)` : p.kind === 'salt_pepper' ? `A.SaltAndPepper(amount=(${p.amount}, ${p.amount}), p=1.0)` : undefined,
  testCases: [{}, { params: { kind: 'salt_pepper', amount: 0.1 } }, { params: { kind: 'speckle', amount: 0.05 } }, { params: { kind: 'poisson', amount: 2 } }, { input: 'float', params: { amount: 0.05 }, tolerance: 1e-5 }, { input: 'u16', params: { amount: 500 } }],
});
