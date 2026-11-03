import { bool, float } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { perChannel, pyRaw } from '../util';

defineStep({
  id: 'minmax_normalize',
  category: 'Intensity',
  label: 'Min-max normalise',
  doc: 'Stretch the image so its minimum maps to "low" and its maximum to "high" (cv2.normalize with NORM_MINMAX), keeping the dtype. Per channel stretches each channel separately (changes colour balance). Sensitive to single outlier pixels: see percentile stretch.',
  cv2: 'cv2.normalize(img, None, low, high, cv2.NORM_MINMAX)',
  keywords: 'normalize normalise minmax stretch range contrast',
  params: [float('lo', 'Low', 0, { step: 1 }), float('hi', 'High', 255, { step: 1 }), bool('perChannel', 'Per channel', false)],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  needs: ['normalize'],
  tolerance: 1e-5,
  summary: (p) => `${p.lo}…${p.hi}${p.perChannel ? ' per channel' : ''}`,
  outMeta: (m, p) => ({ ...m, range: [p.lo, p.hi] }),
  check: (m, p) => (m.dtype === 'uint8' && (p.hi > 255 || p.lo < 0) ? [{ level: 'warning' as const, message: 'Values outside 0–255 are clipped in a uint8 image.' }] : m.dtype.startsWith('float') && p.hi > 1.5 ? [{ level: 'info' as const, message: 'Float image normalised to 0–255; set High = 1 to keep [0, 1].' }] : []),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const norm = (s: any, d: any) => cv.normalize(s, d, p.lo, p.hi, cv.NORM_MINMAX, -1);
    if (p.perChannel) return perChannel(src, ctx, norm);
    const dst = ctx.mat();
    norm(src, dst);
    return dst;
  },
  emitPython: (p, ctx) => {
    const lo = ctx.c('low', pyRaw(pyFloat(p.lo)));
    const hi = ctx.c('high', pyRaw(pyFloat(p.hi)));
    if (p.perChannel && ctx.in.c > 1) return { body: `img = cv2.merge([cv2.normalize(ch, None, ${lo}, ${hi}, cv2.NORM_MINMAX) for ch in cv2.split(img)])` };
    return { body: `img = cv2.normalize(img, None, ${lo}, ${hi}, cv2.NORM_MINMAX)` };
  },
  testCases: [{ params: { lo: 20, hi: 200 } }, { params: { perChannel: true } }, { input: 'float', params: { lo: 0, hi: 1 } }, { input: 'u16', params: { lo: 0, hi: 65535 } }],
});
