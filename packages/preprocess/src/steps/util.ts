// Shared helpers for step files (browser + Python emission).
import { pyRaw } from '../py';
import { channelsOf, spaceRange } from '../meta';
import { hexToRgb, py } from '../py';
import type { BrowserCtx, ColorSpace, CV, ImageMeta, Mat } from '../types';

export { pyRaw };

export const interpPy = (name: string) => pyRaw(`cv2.INTER_${name}`);
export const borderPy = (name: string) => pyRaw(`cv2.BORDER_${name}`);
export const interpCv = (cv: CV, name: string): number => cv[`INTER_${name}`];
export const borderCv = (cv: CV, name: string): number => cv[`BORDER_${name}`];

/** OpenCV constant spelling of a colour space inside COLOR_X2Y codes. */
const CODE_NAME: Partial<Record<ColorSpace, string>> = { LAB: 'Lab', LUV: 'Luv' };
const codeName = (s: ColorSpace) => CODE_NAME[s] ?? s;

const DIRECT = new Set<string>();
for (const b of ['BGR', 'RGB']) {
  for (const x of ['HSV', 'HLS', 'LAB', 'LUV', 'YCrCb', 'XYZ', 'GRAY', 'BGRA', 'RGBA']) {
    DIRECT.add(`${b}>${x}`);
    DIRECT.add(`${x}>${b}`);
  }
}
for (const p of ['BGR>RGB', 'RGB>BGR', 'BGRA>GRAY', 'RGBA>GRAY', 'GRAY>BGRA', 'GRAY>RGBA', 'BGRA>RGBA', 'RGBA>BGRA']) DIRECT.add(p);

/** cvtColor codes (as `COLOR_X2Y` names) to go from one space to another, via BGR when no direct code exists. */
export function colorPath(from: ColorSpace, to: ColorSpace): string[] {
  if (from === to) return [];
  if (DIRECT.has(`${from}>${to}`)) return [`COLOR_${codeName(from)}2${codeName(to)}`];
  return [...colorPath(from, 'BGR'), ...colorPath('BGR', to)];
}

/** Convert in the browser along colorPath. Returns a tracked Mat (or src when nothing to do). */
export function cvtColorBrowser(src: Mat, from: ColorSpace, to: ColorSpace, ctx: BrowserCtx): Mat {
  let cur = src;
  for (const code of colorPath(from, to)) {
    const dst = ctx.mat();
    ctx.cv.cvtColor(cur, dst, ctx.cv[code]);
    cur = dst;
  }
  return cur;
}

/** Python lines converting `img` along colorPath. */
export function cvtColorPy(from: ColorSpace, to: ColorSpace, v = 'img'): string {
  return colorPath(from, to)
    .map((code) => `${v} = cv2.cvtColor(${v}, cv2.${code})`)
    .join('\n');
}

export function convertedMeta(m: ImageMeta, to: ColorSpace): ImageMeta {
  return { ...m, space: to, c: channelsOf(to), range: spaceRange(to, m.dtype) };
}

/** Colour param ('#rrggbb') as values in the image's channel order (gray → luminance). */
export function colorValues(hex: string, m: ImageMeta): number[] {
  const [r, g, b] = hexToRgb(hex);
  const scale = m.dtype === 'uint16' ? 257 : m.dtype === 'uint8' ? 1 : 1 / 255;
  const v = (x: number) => (m.dtype === 'float32' || m.dtype === 'float16' ? Math.round(x * scale * 1e4) / 1e4 : x * scale);
  if (m.c === 1) return [v(Math.round(0.299 * r + 0.587 * g + 0.114 * b))];
  const rgb = m.space.startsWith('RGB') ? [r, g, b] : [b, g, r];
  return m.c === 4 ? [...rgb.map(v), v(255)] : rgb.map(v);
}

export const colorPy = (hex: string, m: ImageMeta) => py(colorValues(hex, m));
export const colorScalar = (cv: CV, hex: string, m: ImageMeta) => {
  const v = colorValues(hex, m);
  return new cv.Scalar(v[0], v[1] ?? 0, v[2] ?? 0, v[3] ?? 0);
};

/** int(x + 0.5): rounding that is identical in Python and JS (Python's round() is banker's rounding). */
export const roundHalfUp = (x: number) => Math.floor(x + 0.5);

/** Kernel larger than the image check. */
export function kernelTooBig(m: ImageMeta, k: number, label = 'Kernel') {
  return k > Math.min(m.h, m.w) ? [{ level: 'error' as const, message: `${label} ${k} is larger than the ${m.w}×${m.h} image.` }] : [];
}

/** Even-kernel check for params that must be odd. */
export function oddCheck(k: number, label = 'Kernel size') {
  return k % 2 === 1 ? [] : [{ level: 'error' as const, message: `${label} must be odd (3, 5, 7…); got ${k}.`, fixes: [{ label: `Use ${k + 1}`, kind: 'setParams' as const, params: {} }] }];
}

/** Run fn on each channel of a multi-channel Mat (functions that only accept 1 channel). */
export function perChannel(src: Mat, ctx: BrowserCtx, fn: (ch: Mat, out: Mat) => void): Mat {
  const { cv } = ctx;
  if (src.channels() === 1) {
    const out = ctx.mat();
    fn(src, out);
    return out;
  }
  const chans = ctx.keep(new cv.MatVector());
  cv.split(src, chans);
  const outs = ctx.keep(new cv.MatVector());
  for (let i = 0; i < chans.size(); i++) {
    const ch = ctx.keep(chans.get(i));
    const o = ctx.mat();
    fn(ch, o);
    outs.push_back(o);
  }
  const dst = ctx.mat();
  cv.merge(outs, dst);
  return dst;
}

/** Apply fn to the L channel of LAB for colour images (or directly for gray). */
export function onLuminance(src: Mat, ctx: BrowserCtx, fn: (l: Mat, out: Mat) => void): Mat {
  const { cv } = ctx;
  if (src.channels() === 1) {
    const out = ctx.mat();
    fn(src, out);
    return out;
  }
  const rgb = ctx.in.space === 'RGB';
  const lab = ctx.mat();
  cv.cvtColor(src, lab, rgb ? cv.COLOR_RGB2Lab : cv.COLOR_BGR2Lab);
  const chans = ctx.keep(new cv.MatVector());
  cv.split(lab, chans);
  const l = ctx.keep(chans.get(0));
  const l2 = ctx.mat();
  fn(l, l2);
  const merged = ctx.keep(new cv.MatVector());
  merged.push_back(l2);
  merged.push_back(ctx.keep(chans.get(1)));
  merged.push_back(ctx.keep(chans.get(2)));
  const lab2 = ctx.mat();
  cv.merge(merged, lab2);
  const dst = ctx.mat();
  cv.cvtColor(lab2, dst, rgb ? cv.COLOR_Lab2RGB : cv.COLOR_Lab2BGR);
  return dst;
}

/** Python: apply `call` (an expression of `x`) on the luminance channel, matching onLuminance. */
export function onLuminancePy(m: ImageMeta, fnExpr: string, ctx: { helper(n: string): void }): string {
  if (m.c === 1) return `img = ${fnExpr}(img)`;
  ctx.helper('luminance');
  return `img = _on_luminance(img, ${fnExpr}, ${py(m.space === 'RGB' ? 'RGB' : 'BGR')})  # L channel of LAB only, so colours do not shift`;
}

/** Elementwise float32 map over a Mat's data in JS (exactly NumPy's float32 arithmetic via Math.fround). */
export function mapFloat32(src: Mat, ctx: BrowserCtx, fn: (v: number, channel: number) => number): Mat {
  const { cv } = ctx;
  const c = src.channels();
  let f = src;
  if (src.depth() !== cv.CV_32F) {
    f = ctx.mat();
    src.convertTo(f, cv.CV_32F);
  }
  const out = ctx.keep(new cv.Mat(src.rows, src.cols, cv[`CV_32FC${c}`]));
  const a = f.data32F;
  const o = out.data32F;
  for (let i = 0; i < a.length; i++) o[i] = Math.fround(fn(a[i], i % c));
  return out;
}

/** Look-up table (uint8 → uint8) applied like cv2.LUT. */
export function applyLut(src: Mat, ctx: BrowserCtx, lut: Uint8Array): Mat {
  const { cv } = ctx;
  const out = copyMat(src, ctx);
  const d = out.data;
  for (let i = 0; i < d.length; i++) d[i] = lut[d[i]];
  void cv;
  return out;
}

/** Round half to even, like np.rint and OpenCV's saturate_cast (cvRound). */
export function rint(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

export const DTYPE_MAX = { uint8: 255, uint16: 65535 } as const;

/**
 * Per-value map into a new Mat of the given depth. Integer outputs are rounded half-to-even and saturated
 * (what `_saturate()` does in the generated Python); float outputs are rounded to float32.
 */
export function mapPixels(src: Mat, ctx: BrowserCtx, out: 'uint8' | 'uint16' | 'float32', fn: (v: number, channel: number, index: number) => number, channels?: number): Mat {
  const { cv } = ctx;
  const c = src.channels();
  const oc = channels ?? c;
  const depth = src.depth();
  const a: ArrayLike<number> = depth === cv.CV_8U ? src.data : depth === cv.CV_16U ? src.data16U : src.data32F;
  const type = out === 'uint8' ? `CV_8UC${oc}` : out === 'uint16' ? `CV_16UC${oc}` : `CV_32FC${oc}`;
  const m = ctx.keep(new cv.Mat(src.rows, src.cols, cv[type]));
  const o: number[] & ArrayLike<number> = out === 'uint8' ? m.data : out === 'uint16' ? m.data16U : m.data32F;
  const n = src.rows * src.cols;
  const max = out === 'uint8' ? 255 : 65535;
  for (let p = 0; p < n; p++)
    for (let k = 0; k < oc; k++) {
      const v = fn(a[p * c + Math.min(k, c - 1)], k, p * c);
      o[p * oc + k] = out === 'float32' ? Math.fround(v) : Number.isNaN(v) ? 0 : Math.min(max, Math.max(0, rint(v)));
    }
  return m;
}

/** Typed view of a Mat's data by depth. */
export function matData(cv: CV, m: Mat): Uint8Array | Uint16Array | Float32Array {
  const d = m.depth();
  return d === cv.CV_8U ? m.data : d === cv.CV_16U ? m.data16U : m.data32F;
}

export const f32 = Math.fround;

/**
 * Deep copy of a Mat. Use this instead of `m.clone()`: in the OpenCV.js 4.12 build, clone() shares the pixel
 * buffer with the original (and a ROI clone stays non-continuous).
 */
export function copyMat(src: Mat, ctx: BrowserCtx): Mat {
  const dst = ctx.mat();
  src.copyTo(dst);
  return dst;
}
