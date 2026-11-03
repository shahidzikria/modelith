import { float, select } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { mapPixels, pyRaw } from '../util';

defineStep({
  id: 'depth_invalid',
  category: 'Domain',
  label: 'Depth invalid-pixel mask',
  doc: 'Depth sensors return 0, NaN or out-of-range values where they failed (glass, sky, shadows of the projector). Either output the validity mask (255 = valid) to weight the loss, or set invalid pixels to 0 so later steps and the loss can ignore them.',
  cv2: 'valid = np.isfinite(d) & (d > min_depth) & (d < max_depth)',
  keywords: 'depth invalid mask holes nan zero range sensor kinect realsense lidar',
  params: [
    float('min', 'Min valid depth', 0, { step: 1, help: 'Values ≤ min are invalid (0 is always invalid).' }),
    float('max', 'Max valid depth', 65535, { step: 1 }),
    select('out', 'Output', [['zero', 'Depth with invalid set to 0'], ['mask', 'Validity mask (uint8)']], 'zero'),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1], dtype: ['uint8', 'uint16', 'float32'] },
  summary: (p) => `${p.min} < d < ${p.max} · ${p.out}`,
  outMeta: (m, p) => (p.out === 'mask' ? { ...m, dtype: 'uint8', range: [0, 255] } : m),
  applyBrowser: (src, p, ctx) => {
    const valid = (v: number) => Number.isFinite(v) && v > Math.max(p.min, 0) && v < p.max;
    if (p.out === 'mask') return mapPixels(src, ctx, 'uint8', (v) => (valid(v) ? 255 : 0));
    const out = ctx.in.dtype === 'float16' ? 'float32' : (ctx.in.dtype as 'uint8' | 'uint16' | 'float32');
    return mapPixels(src, ctx, out, (v) => (valid(v) ? v : 0));
  },
  emitPython: (p, ctx) => {
    const lines = [`valid = np.isfinite(img) & (img > max(${ctx.c('min', pyRaw(pyFloat(p.min)))}, 0)) & (img < ${ctx.c('max', pyRaw(pyFloat(p.max)))})`];
    lines.push(p.out === 'mask' ? 'img = np.where(valid, 255, 0).astype(np.uint8)' : 'img = np.where(valid, img, 0).astype(img.dtype)');
    return { body: lines.join('\n') };
  },
  testCases: [{ input: 'u16', params: { min: 8000, max: 40000 } }, { input: 'u16', params: { min: 8000, out: 'mask' } }],
});
