// Image statistics used by quality metrics (§12.6) and auto-suggest (§12.1). Pure TS on ImageBuf (no OpenCV).
import type { ImageBuf } from '../runtime/buffers';

/** Luma (BT.601) of a BGR / RGB / gray buffer as float64, values in the buffer's own units. */
export function luma(b: ImageBuf): Float64Array {
  const { h, w, c, space } = b.meta;
  const out = new Float64Array(h * w);
  const d = b.data;
  const rgb = space.startsWith('RGB');
  for (let i = 0; i < h * w; i++) {
    if (c === 1) out[i] = d[i];
    else {
      const x = d[i * c], y = d[i * c + 1], z = d[i * c + 2];
      out[i] = rgb ? 0.299 * x + 0.587 * y + 0.114 * z : 0.299 * z + 0.587 * y + 0.114 * x;
    }
  }
  return out;
}

/** Peak value for PSNR / scaling: 255, 65535 or 1 (float). */
export const peakOf = (b: ImageBuf) => (b.meta.dtype === 'uint8' ? 255 : b.meta.dtype === 'uint16' ? 65535 : 1);

export function psnr(a: ImageBuf, b: ImageBuf): number {
  let se = 0;
  for (let i = 0; i < a.data.length; i++) se += (a.data[i] - b.data[i]) ** 2;
  const mse = se / a.data.length;
  return mse === 0 ? Infinity : 10 * Math.log10(peakOf(a) ** 2 / mse);
}

/** Mean SSIM over 8×8 windows (stride 4) on luma (Wang et al. constants). */
export function ssim(a: ImageBuf, b: ImageBuf): number {
  const { h, w } = a.meta;
  const x = luma(a), y = luma(b);
  const L = peakOf(a);
  const c1 = (0.01 * L) ** 2, c2 = (0.03 * L) ** 2;
  let total = 0, n = 0;
  for (let y0 = 0; y0 + 8 <= h; y0 += 4)
    for (let x0 = 0; x0 + 8 <= w; x0 += 4) {
      let mx = 0, my = 0;
      for (let j = 0; j < 8; j++) for (let i = 0; i < 8; i++) (mx += x[(y0 + j) * w + x0 + i]), (my += y[(y0 + j) * w + x0 + i]);
      mx /= 64;
      my /= 64;
      let vx = 0, vy = 0, cxy = 0;
      for (let j = 0; j < 8; j++)
        for (let i = 0; i < 8; i++) {
          const dx = x[(y0 + j) * w + x0 + i] - mx, dy = y[(y0 + j) * w + x0 + i] - my;
          vx += dx * dx;
          vy += dy * dy;
          cxy += dx * dy;
        }
      vx /= 63;
      vy /= 63;
      cxy /= 63;
      total += ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2));
      n++;
    }
  return n ? total / n : 1;
}

/** Shannon entropy (bits) of the luma histogram (256 bins over the value range). */
export function entropy(b: ImageBuf): number {
  const y = luma(b);
  const peak = peakOf(b);
  const hist = new Float64Array(256);
  for (const v of y) hist[Math.max(0, Math.min(255, Math.floor((v / peak) * 255)))]++;
  let e = 0;
  for (const c of hist) if (c) e -= (c / y.length) * Math.log2(c / y.length);
  return e;
}

/** Variance of the 4-neighbour Laplacian of luma (in 0–255 units): the classic sharpness / blur score. */
export function sharpness(b: ImageBuf): number {
  const { h, w } = b.meta;
  const y = luma(b);
  const k = 255 / peakOf(b);
  let s = 0, s2 = 0, n = 0;
  for (let r = 1; r < h - 1; r++)
    for (let c = 1; c < w - 1; c++) {
      const i = r * w + c;
      const l = (y[i - 1] + y[i + 1] + y[i - w] + y[i + w] - 4 * y[i]) * k;
      s += l;
      s2 += l * l;
      n++;
    }
  const m = s / Math.max(1, n);
  return s2 / Math.max(1, n) - m * m;
}

/** Robust noise σ estimate (Immerkær 1996) in 0–255 units. */
export function noiseSigma(b: ImageBuf): number {
  const { h, w } = b.meta;
  const y = luma(b);
  const k = 255 / peakOf(b);
  let s = 0;
  for (let r = 1; r < h - 1; r++)
    for (let c = 1; c < w - 1; c++) {
      const i = r * w + c;
      const v = y[i - w - 1] - 2 * y[i - w] + y[i - w + 1] - 2 * y[i - 1] + 4 * y[i] - 2 * y[i + 1] + y[i + w - 1] - 2 * y[i + w] + y[i + w + 1];
      s += Math.abs(v * k);
    }
  return (Math.sqrt(Math.PI / 2) * s) / (6 * (w - 2) * (h - 2));
}

export interface Quality {
  psnr?: number;
  ssim?: number;
  entropy: number;
  sharpness: number;
}

/** Metrics of `after`, with PSNR / SSIM against `before` when the two are comparable. */
export function quality(before: ImageBuf | undefined, after: ImageBuf): Quality {
  const same = before && before.meta.h === after.meta.h && before.meta.w === after.meta.w && before.meta.c === after.meta.c && before.meta.dtype === after.meta.dtype;
  return { psnr: same ? psnr(before!, after) : undefined, ssim: same ? ssim(before!, after) : undefined, entropy: entropy(after), sharpness: sharpness(after) };
}
