import { float } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';

defineStep({
  id: 'detail_enhance',
  category: 'Intensity',
  label: 'Detail enhance',
  doc: 'Edge-preserving detail boost (cv2.detailEnhance, Gastal & Oliveira domain transform): local texture is amplified while edges do not halo. sigma_s (spatial, 0–200) sets the neighbourhood, sigma_r (range, 0–1) how different colours must be to count as an edge. Not in OpenCV.js, so it runs on the server.',
  cv2: 'cv2.detailEnhance(img, sigma_s, sigma_r)',
  keywords: 'detail enhance sharpen edge preserving photo domain transform',
  params: [float('sigmaS', 'sigma_s', 10, { min: 0, max: 200, step: 1 }), float('sigmaR', 'sigma_r', 0.15, { min: 0, max: 1, step: 0.01 })],
  engine: 'server',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [3], dtype: ['uint8'] },
  summary: (p) => `σs ${p.sigmaS} · σr ${p.sigmaR}`,
  outMeta: (m) => m,
  emitPython: (p, ctx) => ({ body: `img = cv2.detailEnhance(img, sigma_s=${ctx.c('sigma_s', pyRaw(pyFloat(p.sigmaS)))}, sigma_r=${ctx.c('sigma_r', pyRaw(pyFloat(p.sigmaR)))})` }),
});
