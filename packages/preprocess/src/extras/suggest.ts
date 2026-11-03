// Auto-suggest (§12.1): analyse a sample and propose steps with a one-line reason.
import type { ImageBuf } from '../runtime/buffers';
import type { Params } from '../types';
import { luma, noiseSigma, peakOf, sharpness } from './quality';

export interface Suggestion {
  def: string;
  params?: Params;
  reason: string;
  /** Where to insert: start of the pipeline (cleanup) or before tensor conversion. */
  at: 'start' | 'beforeTensor';
}

export interface ImageStats {
  mean: number;
  std: number;
  noise: number;
  sharpness: number;
  /** Channel means (B, G, R order of the buffer) in 0–255 units. */
  channelMeans: number[];
  darkFraction: number;
  brightFraction: number;
}

export function imageStats(b: ImageBuf): ImageStats {
  const y = luma(b);
  const k = 255 / peakOf(b);
  let s = 0, s2 = 0, dark = 0, bright = 0;
  for (const v0 of y) {
    const v = v0 * k;
    s += v;
    s2 += v * v;
    if (v < 30) dark++;
    if (v > 235) bright++;
  }
  const n = y.length;
  const mean = s / n;
  const { c } = b.meta;
  const cm = Array.from({ length: Math.min(c, 3) }, (_, ch) => {
    let t = 0;
    for (let i = ch; i < b.data.length; i += c) t += b.data[i];
    return (t / (b.data.length / c)) * k;
  });
  return { mean, std: Math.sqrt(Math.max(0, s2 / n - mean * mean)), noise: noiseSigma(b), sharpness: sharpness(b), channelMeans: cm, darkFraction: dark / n, brightFraction: bright / n };
}

export function suggestSteps(b: ImageBuf): { stats: ImageStats; suggestions: Suggestion[] } {
  const st = imageStats(b);
  const out: Suggestion[] = [];
  const f = (x: number) => x.toFixed(x < 10 ? 1 : 0);
  if (b.meta.dtype === 'uint16') out.push({ def: 'percentile_stretch', reason: '16-bit image: stretch p2–p98 (or window it) before 8-bit steps.', at: 'start' });
  if (st.noise > 6) out.push({ def: st.noise > 15 ? 'nlm_denoise' : 'bilateral', reason: `noisy: estimated σ ≈ ${f(st.noise)} (0–255); denoise before sharpening or thresholding.`, at: 'start' });
  if (st.mean < 70) out.push({ def: 'gamma', params: { gamma: st.mean < 45 ? 2.0 : 1.6 }, reason: `images are dark: mean ${f(st.mean)}/255; add Gamma ${st.mean < 45 ? '2.0' : '1.6'} or CLAHE.`, at: 'start' });
  else if (st.mean > 190) out.push({ def: 'gamma', params: { gamma: 0.7 }, reason: `images are bright: mean ${f(st.mean)}/255; Gamma 0.7 recovers highlights.`, at: 'start' });
  if (st.std < 35 && b.meta.c <= 3) out.push({ def: 'clahe', params: { clip: 2.5 }, reason: `low contrast: std ${f(st.std)}; CLAHE (on L of LAB) spreads the histogram locally.`, at: 'start' });
  if (st.darkFraction > 0.25 && st.brightFraction > 0.05) out.push({ def: 'clahe', reason: `${Math.round(st.darkFraction * 100)}% of pixels are near black while highlights are bright; CLAHE handles both.`, at: 'start' });
  if (st.sharpness < 60 && st.noise < 6) out.push({ def: 'unsharp_mask', reason: `soft / blurry: variance of Laplacian ${f(st.sharpness)} (< 60); a mild unsharp mask helps.`, at: 'start' });
  if (st.channelMeans.length === 3) {
    const m = st.channelMeans;
    const avg = (m[0] + m[1] + m[2]) / 3;
    const dev = Math.max(...m.map((v) => Math.abs(v - avg)));
    if (dev > 18 && avg > 40) out.push({ def: 'white_balance', reason: `colour cast: channel means ${m.map((v) => Math.round(v)).join(' / ')}; gray-world white balance evens them out.`, at: 'start' });
  }
  if (Math.max(b.meta.h, b.meta.w) > 2048) out.push({ def: 'resize', params: { mode: 'longer', side: 1024, interp: 'AREA' }, reason: `large image (${b.meta.w}×${b.meta.h}): resize first so later steps run faster.`, at: 'start' });
  return { stats: st, suggestions: out };
}
