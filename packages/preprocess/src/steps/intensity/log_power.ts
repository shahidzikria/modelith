import { float, select } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { applyLut, pyRaw, rint } from '../util';

export function logPowerLut(mode: string, exponent: number): Uint8Array {
  const t = new Uint8Array(256);
  const c = 255 / Math.log(256);
  for (let i = 0; i < 256; i++) {
    const v = mode === 'log' ? c * Math.log(1 + i) : 255 * (i / 255) ** exponent;
    t[i] = Math.min(255, Math.max(0, rint(v)));
  }
  return t;
}

defineStep({
  id: 'log_power',
  category: 'Intensity',
  label: 'Log / power-law transform',
  doc: 'Log: s = c·log(1 + r) with c = 255 / log(256) expands dark values and compresses bright ones (e.g. spectra, high-dynamic-range scenes). Power-law: s = 255·(r/255)^p (p < 1 brightens, p > 1 darkens). Both run as a uint8 look-up table.',
  cv2: 'cv2.LUT(img, table)',
  keywords: 'log transform power law lut dynamic range',
  params: [select('mode', 'Transform', [['log', 'Log'], ['power', 'Power-law']], 'log'), float('exponent', 'Exponent p', 0.6, { min: 0.05, max: 5, step: 0.05, showIf: { key: 'mode', equals: ['power'] } })],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { dtype: ['uint8'] },
  needs: ['LUT'],
  tolerance: 1,
  summary: (p) => (p.mode === 'log' ? 'log' : `power ${p.exponent}`),
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => applyLut(src, ctx, logPowerLut(p.mode, p.exponent)),
  emitPython: (p, ctx) => {
    ctx.helper('saturate');
    if (p.mode === 'log') return { body: 'table = _saturate(255.0 / np.log(256.0) * np.log1p(np.arange(256)), np.uint8)  # s = c * log(1 + r)\nimg = cv2.LUT(img, table)' };
    const e = ctx.c('exponent', pyRaw(pyFloat(p.exponent)));
    return { body: `table = _saturate(255.0 * (np.arange(256) / 255.0) ** ${e}, np.uint8)\nimg = cv2.LUT(img, table)` };
  },
  testCases: [{}, { params: { mode: 'power', exponent: 2.2 } }],
});
