// Shared output handling for derivative filters (Sobel, Scharr, Laplacian): |x| as uint8, or signed float32.
import { select } from '../../params';
import type { BrowserCtx, ImageMeta, Mat, Params } from '../../types';

export const gradOut = () => select('out', 'Output', [['abs', 'uint8 |gradient| (convertScaleAbs)'], ['float32', 'float32 (signed)']], 'abs');

export function gradMeta(m: ImageMeta, p: Params): ImageMeta {
  return p.out === 'abs' ? { ...m, dtype: 'uint8', range: [0, 255] } : { ...m, dtype: 'float32', range: [-1020, 1020] };
}

/** Apply `fn(src, dst, depth)` then convertScaleAbs when the output is uint8. */
export function gradBrowser(src: Mat, p: Params, ctx: BrowserCtx, fn: (s: Mat, d: Mat, depth: number) => void): Mat {
  const { cv } = ctx;
  const f = ctx.mat();
  fn(src, f, cv.CV_32F);
  if (p.out !== 'abs') return f;
  const dst = ctx.mat();
  cv.convertScaleAbs(f, dst, 1, 0);
  return dst;
}

export function gradPy(call: string, p: Params): string {
  return p.out === 'abs' ? `img = cv2.convertScaleAbs(${call})  # |gradient| as uint8` : `img = ${call}`;
}
