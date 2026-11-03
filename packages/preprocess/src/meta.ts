import type { ColorSpace, DType, ImageMeta } from './types';

export function channelsOf(space: ColorSpace): number {
  if (space === 'GRAY') return 1;
  if (space === 'BGRA' || space === 'RGBA') return 4;
  return 3;
}

export function dtypeRange(dtype: DType): [number, number] {
  if (dtype === 'uint8') return [0, 255];
  if (dtype === 'uint16') return [0, 65535];
  return [0, 1];
}

export function isFloat(m: ImageMeta | DType): boolean {
  const d = typeof m === 'string' ? m : m.dtype;
  return d === 'float32' || d === 'float16';
}

/** Default meta of a decoded image. */
export function imageMeta(h: number, w: number, c: number, dtype: DType = 'uint8', space?: ColorSpace): ImageMeta {
  const sp: ColorSpace = space ?? (c === 1 ? 'GRAY' : c === 4 ? 'BGRA' : 'BGR');
  return { h, w, c, dtype, space: sp, range: dtypeRange(dtype), layout: 'HWC', batch: false };
}

export function withMeta(m: ImageMeta, patch: Partial<ImageMeta>): ImageMeta {
  return { ...m, ...patch };
}

/** `512×512×3 uint8`, `3×224×224 float32`, `1×3×224×224 float32`. */
export function formatMeta(m: ImageMeta): string {
  const dims = m.layout === 'CHW' ? [m.c, m.h, m.w] : m.c === 1 && m.space === 'GRAY' ? [m.h, m.w] : [m.h, m.w, m.c];
  if (m.batch) dims.unshift(1);
  return `${dims.join('×')} ${m.dtype}`;
}

/** NumPy shape tuple text, e.g. `(H, W, 3)` → `(512, 512, 3)`. */
export function shapeTuple(m: ImageMeta): string {
  const dims = m.layout === 'CHW' ? [m.c, m.h, m.w] : m.c === 1 ? [m.h, m.w] : [m.h, m.w, m.c];
  if (m.batch) dims.unshift(1);
  return `(${dims.join(', ')})`;
}

/** Comment describing the image, e.g. `BGR uint8 (512, 512, 3)`. */
export function describeMeta(m: ImageMeta): string {
  const rng = isFloat(m) ? `, range ~[${fmtNum(m.range[0])}, ${fmtNum(m.range[1])}]` : '';
  return `${m.space} ${m.dtype} ${shapeTuple(m)}${m.layout === 'CHW' ? ' CHW' : ''}${rng}`;
}

export function fmtNum(n: number): string {
  if (Number.isInteger(n)) return String(n);
  return Math.abs(n) >= 100 ? n.toFixed(0) : Math.abs(n) >= 1 ? n.toFixed(2) : n.toFixed(3);
}

export function sameMeta(a: ImageMeta, b: ImageMeta): boolean {
  return a.h === b.h && a.w === b.w && a.c === b.c && a.dtype === b.dtype && a.space === b.space && a.layout === b.layout && a.batch === b.batch;
}

/** The 3-channel colour spaces OpenCV can convert to and from BGR. */
export const COLOR_SPACES: ColorSpace[] = ['BGR', 'RGB', 'GRAY', 'HSV', 'HLS', 'LAB', 'LUV', 'YCrCb', 'XYZ', 'BGRA', 'RGBA'];

/** Value range of each colour space for a dtype (OpenCV conventions). */
export function spaceRange(space: ColorSpace, dtype: DType): [number, number] {
  if (dtype === 'uint8' || dtype === 'uint16') return dtypeRange(dtype);
  if (space === 'HSV' || space === 'HLS') return [0, 360];
  if (space === 'LAB') return [-127, 127];
  if (space === 'LUV') return [-134, 220];
  return [0, 1];
}
