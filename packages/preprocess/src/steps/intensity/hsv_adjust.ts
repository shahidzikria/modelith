import { float } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { mapPixels, pyRaw, rint } from '../util';

defineStep({
  id: 'hsv_adjust',
  category: 'Intensity',
  label: 'HSV adjust',
  doc: 'Shift hue and scale saturation and value in HSV space, then convert back. In OpenCV\'s uint8 HSV, H runs 0–179 (degrees ÷ 2), so a 20° shift adds 10. Saturation 0 gives grey, > 1 makes colours more vivid.',
  cv2: 'cv2.cvtColor(img, cv2.COLOR_BGR2HSV) → shift / scale → cv2.COLOR_HSV2BGR',
  keywords: 'hue saturation value vibrance hsv colour adjust',
  params: [
    float('hue', 'Hue shift (°)', 0, { min: -180, max: 180, step: 2 }),
    float('sat', 'Saturation ×', 1.3, { min: 0, max: 4, step: 0.05 }),
    float('val', 'Value ×', 1, { min: 0, max: 4, step: 0.05 }),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [3], dtype: ['uint8'], space: ['BGR', 'RGB'] },
  needs: ['cvtColor'],
  tolerance: 1, // HSV → BGR rounds differently in native SIMD cv2 vs OpenCV.js
  summary: (p) => `h ${p.hue}° · s ×${p.sat} · v ×${p.val}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const rgb = ctx.in.space === 'RGB';
    const hsv = ctx.mat();
    cv.cvtColor(src, hsv, rgb ? cv.COLOR_RGB2HSV : cv.COLOR_BGR2HSV);
    const shift = rint(p.hue / 2);
    const adj = mapPixels(hsv, ctx, 'uint8', (v, k) => (k === 0 ? (((v + shift) % 180) + 180) % 180 : v * (k === 1 ? p.sat : p.val)));
    const dst = ctx.mat();
    cv.cvtColor(adj, dst, rgb ? cv.COLOR_HSV2RGB : cv.COLOR_HSV2BGR);
    return dst;
  },
  emitPython: (p, ctx) => {
    ctx.helper('saturate');
    const s = ctx.in.space === 'RGB' ? 'RGB' : 'BGR';
    const shift = ctx.c('hue_shift', rint(p.hue / 2), 'OpenCV hue units (degrees / 2)');
    const sat = ctx.c('sat', pyRaw(pyFloat(p.sat)));
    const val = ctx.c('val', pyRaw(pyFloat(p.val)));
    return {
      body: [
        `h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_${s}2HSV))  # H in 0..179`,
        `h = ((h.astype(np.int16) + ${shift}) % 180).astype(np.uint8)`,
        `s = _saturate(s * ${sat}, np.uint8)`,
        `v = _saturate(v * ${val}, np.uint8)`,
        `img = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2${s})`,
      ].join('\n'),
    };
  },
  testCases: [{}, { params: { hue: 40, sat: 0.5, val: 1.2 } }, { params: { hue: -100 } }],
});
