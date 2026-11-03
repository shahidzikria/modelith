// Shared DFT helpers for the frequency-domain steps (browser side mirrors np.fft.fftshift / ifftshift).
import type { BrowserCtx, Mat } from '../../types';

/** np.fft.fftshift (inverse = false) or ifftshift (true) of a 2-channel float32 spectrum. */
export function shift(src: Mat, ctx: BrowserCtx, inverse = false): Mat {
  const { cv } = ctx;
  const h = src.rows, w = src.cols, c = src.channels();
  const out = ctx.keep(new cv.Mat(h, w, src.type()));
  const a = src.data32F as Float32Array;
  const o = out.data32F as Float32Array;
  const sy = inverse ? -Math.floor(h / 2) : Math.floor(h / 2);
  const sx = inverse ? -Math.floor(w / 2) : Math.floor(w / 2);
  for (let y = 0; y < h; y++) {
    const yy = (((y - sy) % h) + h) % h;
    for (let x = 0; x < w; x++) {
      const xx = (((x - sx) % w) + w) % w;
      for (let k = 0; k < c; k++) o[(y * w + x) * c + k] = a[(yy * w + xx) * c + k];
    }
  }
  return out;
}

/** Centred complex spectrum (2-channel float32) of a single-channel image. */
export function spectrum(src: Mat, ctx: BrowserCtx): Mat {
  const { cv } = ctx;
  const f = ctx.mat();
  src.convertTo(f, cv.CV_32F, 1, 0);
  const d = ctx.mat();
  cv.dft(f, d, cv.DFT_COMPLEX_OUTPUT, 0);
  return shift(d, ctx);
}

/** log(1 + |F|) of a centred spectrum, float32 1-channel (cv2.log for identical maths on both sides). */
export function logMagnitude(spec: Mat, ctx: BrowserCtx): Mat {
  const { cv } = ctx;
  const parts = ctx.keep(new cv.MatVector());
  cv.split(spec, parts);
  const mag = ctx.mat();
  cv.magnitude(ctx.keep(parts.get(0)), ctx.keep(parts.get(1)), mag);
  const one = ctx.mat();
  mag.convertTo(one, cv.CV_32F, 1, 1);
  const out = ctx.mat();
  cv.log(one, out);
  return out;
}

/** Multiply a centred spectrum by a real mask (float32, same size), invert, return the real float32 image. */
export function filterAndInvert(spec: Mat, mask: Float32Array, ctx: BrowserCtx): Mat {
  const { cv } = ctx;
  const s = spec.data32F as Float32Array;
  const m = ctx.keep(new cv.Mat(spec.rows, spec.cols, cv.CV_32FC2));
  const o = m.data32F as Float32Array;
  for (let i = 0; i < mask.length; i++) {
    o[2 * i] = Math.fround(s[2 * i] * mask[i]);
    o[2 * i + 1] = Math.fround(s[2 * i + 1] * mask[i]);
  }
  const back = shift(m, ctx, true);
  const img = ctx.mat();
  cv.dft(back, img, cv.DFT_INVERSE | cv.DFT_SCALE | cv.DFT_REAL_OUTPUT, 0);
  return img;
}

/** Distance of every pixel from the spectrum centre (h // 2, w // 2). */
export function centreDistance(h: number, w: number): Float64Array {
  const d = new Float64Array(h * w);
  const cy = Math.floor(h / 2), cx = Math.floor(w / 2);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) d[y * w + x] = Math.sqrt((y - cy) ** 2 + (x - cx) ** 2);
  return d;
}

export const PY_SPECTRUM = [
  'spec = np.fft.fftshift(cv2.dft(img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT), axes=(0, 1))  # centred',
];

export const PY_INVERT = [
  'spec = np.fft.ifftshift(spec * mask[..., None], axes=(0, 1))',
  'img = cv2.dft(spec, flags=cv2.DFT_INVERSE | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)',
];

export const PY_DIST = ['h, w = img.shape[:2]', 'yy, xx = np.mgrid[:h, :w]', 'dist = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2)  # distance from the spectrum centre'];
