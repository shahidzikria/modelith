import { float, int, select } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import type { Params } from '../../types';
import { pyRaw } from '../util';
import { centreDistance, filterAndInvert, PY_DIST, PY_INVERT, PY_SPECTRUM, spectrum } from './_fft';

/** Low-pass response at distance d for the chosen shape. */
function lowpass(shape: string, d: number, d0: number, n: number): number {
  if (shape === 'ideal') return d <= d0 ? 1 : 0;
  if (shape === 'butterworth') return 1 / (1 + (d / Math.max(d0, 1e-6)) ** (2 * n));
  return Math.exp(-(d * d) / (2 * d0 * d0));
}

export function freqMask(h: number, w: number, p: Params): Float32Array {
  const dist = centreDistance(h, w);
  const m = new Float32Array(h * w);
  for (let i = 0; i < m.length; i++) {
    const lo = lowpass(p.shape, dist[i], p.d0, p.order);
    m[i] = p.type === 'low' ? lo : p.type === 'high' ? 1 - lo : (1 - lo) * lowpass(p.shape, dist[i], p.d1, p.order);
  }
  return m;
}

defineStep({
  id: 'freq_filter',
  category: 'Frequency',
  label: 'Frequency filter (low / high / band-pass)',
  doc: 'Filter in the Fourier domain: multiply the centred spectrum by a mask and invert (cv2.dft / DFT_INVERSE). Low-pass smooths, high-pass keeps edges and texture, band-pass keeps frequencies between D0 and D1 (cycles per image). Ideal masks ring; Butterworth (order n) and Gaussian do not.',
  cv2: 'cv2.dft → × mask → cv2.dft(..., cv2.DFT_INVERSE | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)',
  keywords: 'frequency filter low pass high pass band pass butterworth gaussian ideal fourier dft',
  params: [
    select('type', 'Type', [['low', 'Low-pass'], ['high', 'High-pass'], ['band', 'Band-pass']], 'low'),
    select('shape', 'Shape', [['gaussian', 'Gaussian'], ['butterworth', 'Butterworth'], ['ideal', 'Ideal']], 'gaussian'),
    float('d0', 'Cut-off D0', 20, { min: 1, max: 1000, step: 1 }),
    float('d1', 'Upper cut-off D1', 60, { min: 1, max: 2000, step: 1, showIf: { key: 'type', equals: ['band'] } }),
    int('order', 'Butterworth order n', 2, { min: 1, max: 10, showIf: { key: 'shape', equals: ['butterworth'] } }),
    select('out', 'Output', [['uint8', 'uint8 (clipped)'], ['float32', 'float32']], 'uint8'),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1], dtype: ['uint8', 'uint16', 'float32'] },
  needs: ['dft'],
  tolerance: 1,
  summary: (p) => `${p.type} ${p.shape} D0 ${p.d0}${p.type === 'band' ? `–${p.d1}` : ''}`,
  outMeta: (m, p) => (p.out === 'float32' ? { ...m, dtype: 'float32' } : { ...m, dtype: 'uint8', range: [0, 255] }),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const img = filterAndInvert(spectrum(src, ctx), freqMask(src.rows, src.cols, p), ctx);
    if (p.out === 'float32') return img;
    const dst = ctx.mat();
    img.convertTo(dst, cv.CV_8U, 1, 0);
    return dst;
  },
  emitPython: (p, ctx) => {
    const d0 = ctx.c('d0', pyRaw(pyFloat(p.d0)), 'cycles per image');
    const n = ctx.c('order', p.order);
    const lp = (d: string) =>
      p.shape === 'ideal' ? `(dist <= ${d}).astype(np.float64)` : p.shape === 'butterworth' ? `1 / (1 + (dist / ${d}) ** (2 * ${n}))` : `np.exp(-(dist**2) / (2 * ${d} ** 2))`;
    let mask: string;
    if (p.type === 'low') mask = lp(d0);
    else if (p.type === 'high') mask = `1 - ${lp(d0)}`;
    else mask = `(1 - ${lp(d0)}) * ${lp(ctx.c('d1', pyRaw(pyFloat(p.d1))))}`;
    const lines = [...PY_DIST, `mask = (${mask}).astype(np.float32)  # ${p.shape} ${p.type}-pass`, ...PY_SPECTRUM, ...PY_INVERT];
    if (p.out === 'uint8') {
      ctx.helper('saturate');
      lines.push('img = _saturate(img, np.uint8)');
    }
    return { body: lines.join('\n') };
  },
  testCases: [{ input: 'gray' }, { input: 'gray', params: { type: 'high', shape: 'butterworth', d0: 10 } }, { input: 'gray', params: { type: 'band', shape: 'ideal', out: 'float32' }, tolerance: 1e-2 }],
});
