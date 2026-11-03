import { select } from '../../params';
import { defineStep } from '../../registry';
import type { DType, ImageMeta, Params } from '../../types';
import { DTYPE_MAX, f32, mapPixels, rint } from '../util';

const maxOf = (d: DType) => (d === 'uint8' || d === 'uint16' ? DTYPE_MAX[d] : 1);

/** Multiplier applied in 'shift' mode (uint8 ↔ uint16 ↔ float [0,1]). */
function shiftFactor(from: ImageMeta, to: DType): number {
  return maxOf(to) / maxOf(from.dtype);
}

function outRange(m: ImageMeta, p: Params): [number, number] {
  if (p.to === 'float32') return p.scale === 'shift' || p.scale === 'minmax' ? [0, 1] : m.range;
  return [0, maxOf(p.to)];
}

defineStep({
  id: 'convert_dtype',
  category: 'I/O & colour',
  label: 'Change dtype / bit depth',
  doc: 'Convert between uint8, uint16 and float32 with an explicit scaling rule. Scale by range: 16-bit → 8-bit divides by 257, uint8 → float divides by 255. Min-max: stretch the actual min..max to the full output range (cv2.normalize). Clip: cast and saturate without scaling. Abs: cv2.convertScaleAbs (|x| saturated to uint8, for gradients).',
  cv2: 'img.astype(...) / cv2.normalize(..., NORM_MINMAX) / cv2.convertScaleAbs',
  keywords: 'astype dtype bit depth 16 bit uint8 float convert normalize convertScaleAbs',
  params: [
    select('to', 'Convert to', [['uint8', 'uint8 (0–255)'], ['uint16', 'uint16 (0–65535)'], ['float32', 'float32']], 'float32'),
    select('scale', 'Scaling', [['shift', 'Scale by range (255 ↔ 65535 ↔ 1)'], ['minmax', 'Min-max stretch'], ['clip', 'Clip only (no scaling)'], ['abs', 'Absolute value (convertScaleAbs)']], 'shift'),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1, 2, 3, 4] },
  needs: ['normalize'],
  tolerance: 1e-6,
  summary: (p) => `→ ${p.to} (${p.scale})`,
  check: (m, p) => {
    if (p.scale === 'abs' && p.to !== 'uint8') return [{ level: 'error' as const, message: 'convertScaleAbs always produces uint8; choose uint8 or another scaling.' }];
    if (m.dtype === p.to && p.scale !== 'minmax') return [{ level: 'info' as const, message: `Already ${m.dtype}; this step does nothing.`, fixes: [{ label: 'Remove it', kind: 'remove' as const }] }];
    return [];
  },
  outMeta: (m, p) => ({ ...m, dtype: p.to, range: outRange(m, p) }),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    if (p.scale === 'minmax') {
      const dst = ctx.mat();
      const depth = p.to === 'uint8' ? cv.CV_8U : p.to === 'uint16' ? cv.CV_16U : cv.CV_32F;
      cv.normalize(src, dst, 0, maxOf(p.to), cv.NORM_MINMAX, depth);
      return dst;
    }
    if (p.scale === 'abs') {
      const dst = ctx.mat();
      cv.convertScaleAbs(src, dst, 1, 0);
      return dst;
    }
    const k = p.scale === 'shift' ? shiftFactor(ctx.in, p.to) : 1;
    const fromFloat = ctx.in.dtype === 'float32' || ctx.in.dtype === 'float16';
    if (p.to === 'float32') {
      // NumPy: img.astype(np.float32) / 255.0 — one float32 division (mapPixels rounds to float32).
      const div = Math.round(1 / k);
      return mapPixels(src, ctx, 'float32', (v) => (k === 1 ? v : k < 1 ? v / div : f32(v) * k));
    }
    // To an integer type: _saturate(img.astype(float32) * k) — or exact integer arithmetic for uint16 → uint8.
    if (!fromFloat && k < 1) return mapPixels(src, ctx, p.to, (v) => rint(v / Math.round(1 / k)));
    return mapPixels(src, ctx, p.to, (v) => (k === 1 ? v : f32(f32(v) * f32(k))));
  },
  emitPython: (p, ctx) => {
    const to = `np.${p.to}`;
    if (p.scale === 'minmax') return { body: `img = cv2.normalize(img, None, 0, ${maxOf(p.to)}, cv2.NORM_MINMAX, dtype=cv2.CV_${p.to === 'uint8' ? '8U' : p.to === 'uint16' ? '16U' : '32F'})` };
    if (p.scale === 'abs') return { body: 'img = cv2.convertScaleAbs(img)  # |x|, saturated to uint8' };
    const k = p.scale === 'shift' ? shiftFactor(ctx.in, p.to) : 1;
    const fromFloat = ctx.in.dtype === 'float32' || ctx.in.dtype === 'float16';
    if (p.to === 'float32') {
      if (k === 1) return { body: 'img = img.astype(np.float32)' };
      if (k < 1) return { body: `img = img.astype(np.float32) / ${Math.round(1 / k)}.0  # ${ctx.in.dtype} → [0, 1]` };
      return { body: `img = img.astype(np.float32) * ${k}` };
    }
    ctx.helper('saturate');
    if (k === 1) return { body: `img = _saturate(img, ${to})  # clip to the ${p.to} range, no scaling` };
    if (!fromFloat && k < 1) return { body: `img = _saturate(img / ${Math.round(1 / k)}.0, ${to})  # ${ctx.in.dtype} → ${p.to}` };
    return { body: `img = _saturate(img.astype(np.float32) * np.float32(${k}), ${to})` };
  },
  testCases: [
    {},
    { params: { to: 'uint16' } },
    { input: 'u16', params: { to: 'uint8', scale: 'shift' } },
    { input: 'u16', params: { to: 'uint8', scale: 'minmax' } },
    { input: 'float', params: { to: 'uint8', scale: 'shift' } },
    { input: 'float', params: { to: 'uint8', scale: 'abs' } },
    { params: { to: 'float32', scale: 'minmax' } },
  ],
});
