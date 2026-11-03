import { float } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { DTYPE_MAX, mapPixels, matData, pyRaw } from '../util';

/** np.percentile(values, q) with the default 'linear' method. */
export function percentile(sorted: ArrayLike<number>, q: number): number {
  const n = sorted.length;
  const idx = (q / 100) * (n - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(n - 1, lo + 1);
  const f = idx - lo;
  return sorted[lo] + (sorted[hi] - sorted[lo]) * f;
}

defineStep({
  id: 'percentile_stretch',
  category: 'Intensity',
  label: 'Percentile contrast stretch',
  doc: 'Per channel, map the low percentile (e.g. 2 %) to 0 and the high percentile (98 %) to the maximum, clipping the rest. Robust to hot pixels and specular highlights; the standard first step for satellite / aerial imagery and 16-bit sensors.',
  cv2: 'lo, hi = np.percentile(img, (2, 98), axis=(0, 1)); (img - lo) * max / (hi - lo)',
  keywords: 'percentile stretch contrast robust satellite remote sensing clip p2 p98',
  params: [float('pLo', 'Low percentile', 2, { min: 0, max: 50, step: 0.5 }), float('pHi', 'High percentile', 98, { min: 50, max: 100, step: 0.5 })],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { dtype: ['uint8', 'uint16', 'float32'] },
  tolerance: 1,
  summary: (p) => `p${p.pLo}–p${p.pHi}`,
  outMeta: (m) => m,
  check: (_m, p) => (p.pLo >= p.pHi ? [{ level: 'error' as const, message: 'Low percentile must be below the high percentile.' }] : []),
  applyBrowser: (src, p, ctx) => {
    const { c, dtype } = ctx.in;
    const d = matData(ctx.cv, src);
    const lo: number[] = [];
    const hi: number[] = [];
    for (let k = 0; k < c; k++) {
      const vals = new Float64Array(d.length / c);
      for (let i = 0; i < vals.length; i++) vals[i] = d[i * c + k];
      vals.sort();
      lo.push(percentile(vals, p.pLo));
      hi.push(percentile(vals, p.pHi));
    }
    const max = dtype === 'float32' ? 1 : DTYPE_MAX[dtype as 'uint8' | 'uint16'];
    const out = dtype === 'float32' ? 'float32' : dtype;
    return mapPixels(src, ctx, out as 'uint8' | 'uint16' | 'float32', (v, k) => {
      const x = (Math.fround(v) - lo[k]) * (max / Math.max(hi[k] - lo[k], 1e-6));
      return dtype === 'float32' ? Math.min(1, Math.max(0, x)) : x;
    });
  },
  emitPython: (p, ctx) => {
    const P = ctx.c('percentiles', [p.pLo, p.pHi]);
    const max = ctx.in.dtype === 'float32' ? '1.0' : `${DTYPE_MAX[ctx.in.dtype as 'uint8' | 'uint16']}.0`;
    const lines = [`lo, hi = np.percentile(img, ${P}, axis=(0, 1))  # per channel`, `out = (img.astype(np.float32) - lo) * (${max} / np.maximum(hi - lo, 1e-6))`];
    if (ctx.in.dtype === 'float32') lines.push('img = np.clip(out, 0.0, 1.0).astype(np.float32)');
    else {
      ctx.helper('saturate');
      lines.push(`img = _saturate(out, np.${ctx.in.dtype})`);
    }
    return { body: lines.join('\n') };
  },
  testCases: [{}, { input: 'u16', params: { pLo: 1, pHi: 99 } }, { input: 'float', tolerance: 1e-5 }],
});
void pyFloat;
void pyRaw;
