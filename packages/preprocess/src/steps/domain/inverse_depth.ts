import { float } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { mapPixels, pyRaw } from '../util';

defineStep({
  id: 'inverse_depth',
  category: 'Domain',
  label: 'Inverse depth',
  doc: 'Convert depth to inverse depth (disparity-like) k / d as float32, with invalid (≤ 0) pixels kept at 0. Monocular depth networks (MiDaS, DPT, Depth Anything) are usually trained on inverse depth; it gives near objects more resolution.',
  cv2: 'np.where(d > 0, k / d, 0).astype(np.float32)',
  keywords: 'inverse depth disparity midas monocular depth float',
  params: [float('k', 'Scale k', 1000, { min: 0.001, step: 1, help: 'e.g. 1000 for depth in mm → inverse metres.' })],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1], dtype: ['uint8', 'uint16', 'float32'] },
  tolerance: 1e-5,
  summary: (p) => `${p.k} / d`,
  outMeta: (m) => ({ ...m, dtype: 'float32', range: [0, 1] }),
  applyBrowser: (src, p, ctx) => mapPixels(src, ctx, 'float32', (v) => (v > 0 ? p.k / v : 0)),
  emitPython: (p, ctx) => {
    const k = ctx.c('k', pyRaw(pyFloat(p.k)));
    return { body: `d = img.astype(np.float64)\nimg = np.where(d > 0, ${k} / np.maximum(d, 1e-12), 0).astype(np.float32)` };
  },
  testCases: [{ input: 'u16' }],
});
