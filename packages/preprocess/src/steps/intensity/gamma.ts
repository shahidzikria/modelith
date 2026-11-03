import { float } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { applyLut, mapPixels, pyRaw, rint } from '../util';

/** uint8 gamma table, identical to the emitted NumPy expression. */
export function gammaLut(gamma: number): Uint8Array {
  const t = new Uint8Array(256);
  for (let i = 0; i < 256; i++) t[i] = Math.min(255, Math.max(0, rint((i / 255) ** (1 / gamma) * 255)));
  return t;
}

defineStep({
  id: 'gamma',
  category: 'Intensity',
  label: 'Gamma correction',
  doc: 'out = (x / max)^(1/γ) · max via a 256-entry look-up table (cv2.LUT) for uint8. γ > 1 brightens shadows (dark, low-light images: 1.5–2.2), γ < 1 darkens. Highlights are preserved, unlike adding brightness.',
  cv2: 'cv2.LUT(img, table)',
  keywords: 'gamma lut exposure shadows dark low light power',
  params: [float('gamma', 'γ', 1.8, { min: 0.1, max: 5, step: 0.05 })],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { dtype: ['uint8', 'uint16', 'float32'] },
  needs: ['LUT'],
  tolerance: 1,
  summary: (p) => `γ ${p.gamma}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    if (ctx.in.dtype === 'uint8') return applyLut(src, ctx, gammaLut(p.gamma));
    if (ctx.in.dtype === 'uint16') return mapPixels(src, ctx, 'uint16', (v) => (v / 65535) ** (1 / p.gamma) * 65535);
    return mapPixels(src, ctx, 'float32', (v) => Math.max(v, 0) ** Math.fround(1 / p.gamma));
  },
  emitPython: (p, ctx) => {
    const g = ctx.c('gamma', pyRaw(pyFloat(p.gamma)), '> 1 brightens');
    if (ctx.in.dtype === 'uint8') {
      ctx.helper('saturate');
      return { body: `table = _saturate((np.arange(256) / 255.0) ** (1.0 / ${g}) * 255.0, np.uint8)\nimg = cv2.LUT(img, table)` };
    }
    if (ctx.in.dtype === 'uint16') {
      ctx.helper('saturate');
      return { body: `img = _saturate((img / 65535.0) ** (1.0 / ${g}) * 65535.0, np.uint16)` };
    }
    return { body: `img = np.power(np.maximum(img, 0), np.float32(1.0 / ${g}))  # float image in [0, 1]` };
  },
  emitAlbumentations: (p, ctx) => (ctx.in.dtype === 'uint8' ? `A.RandomGamma(gamma_limit=(${Math.round(100 / p.gamma)}, ${Math.round(100 / p.gamma)}), p=1.0)` : undefined),
  testCases: [{}, { params: { gamma: 0.5 } }, { input: 'u16' }, { input: 'float', tolerance: 1e-5 }],
});
