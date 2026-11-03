import { float, select } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';

defineStep({
  id: 'edge_preserving',
  category: 'Filtering',
  label: 'Edge-preserving filter',
  doc: 'Domain-transform edge-preserving smoothing (cv2.edgePreservingFilter): recursive (faster) or normalised convolution. sigma_s 0–200 sets the spatial extent, sigma_r 0–1 the colour difference that counts as an edge. Not in OpenCV.js; runs on the server.',
  cv2: 'cv2.edgePreservingFilter(img, flags, sigma_s, sigma_r)',
  keywords: 'edge preserving smoothing domain transform photo stylize',
  params: [select('flags', 'Method', [['RECURS_FILTER', 'Recursive'], ['NORMCONV_FILTER', 'Normalised convolution']], 'RECURS_FILTER'), float('sigmaS', 'sigma_s', 60, { min: 0, max: 200, step: 1 }), float('sigmaR', 'sigma_r', 0.4, { min: 0, max: 1, step: 0.01 })],
  engine: 'server',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [3], dtype: ['uint8'] },
  summary: (p) => `${p.flags === 'RECURS_FILTER' ? 'recursive' : 'normconv'} · σs ${p.sigmaS}`,
  outMeta: (m) => m,
  emitPython: (p, ctx) => ({
    body: `img = cv2.edgePreservingFilter(img, flags=cv2.${p.flags}, sigma_s=${ctx.c('sigma_s', pyRaw(pyFloat(p.sigmaS)))}, sigma_r=${ctx.c('sigma_r', pyRaw(pyFloat(p.sigmaR)))})`,
  }),
});
