import { bool, float, select } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { matData, pyRaw } from '../util';

defineStep({
  id: 'canny',
  category: 'Edges',
  label: 'Canny edges',
  doc: 'Canny edge detector (cv2.Canny): gradient, non-maximum suppression and hysteresis between the low and high thresholds (ratio 1:2–1:3). "Auto" sets them from the median intensity m: low = (1 − σ)·m, high = (1 + σ)·m with σ = 0.33. Blur first on noisy images.',
  cv2: 'cv2.Canny(img, low, high, apertureSize=3, L2gradient=False)',
  keywords: 'canny edges edge detection hysteresis auto threshold',
  params: [
    bool('auto', 'Auto thresholds (median)', false),
    float('sigma', 'Auto σ', 0.33, { min: 0.05, max: 1, step: 0.01, showIf: { key: 'auto', equals: [true] } }),
    float('low', 'Low threshold', 50, { min: 0, max: 1000, step: 1, showIf: { key: 'auto', equals: [false] } }),
    float('high', 'High threshold', 150, { min: 0, max: 1000, step: 1, showIf: { key: 'auto', equals: [false] } }),
    select('aperture', 'Aperture', [['3', '3'], ['5', '5'], ['7', '7']], '3'),
    bool('l2', 'L2 gradient', false),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1, 3], dtype: ['uint8'] },
  needs: ['Canny'],
  summary: (p) => (p.auto ? `auto σ ${p.sigma}` : `${p.low} / ${p.high}`),
  outMeta: (m) => ({ ...m, c: 1, space: 'GRAY', range: [0, 255] }),
  check: (_m, p) => (!p.auto && p.low > p.high ? [{ level: 'error' as const, message: `Canny low threshold (${p.low}) is above the high threshold (${p.high}).`, fixes: [{ label: 'Swap them', kind: 'setParams' as const, params: { low: p.high, high: p.low } }] }] : []),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    let lo = p.low, hi = p.high;
    if (p.auto) {
      const d = Float64Array.from(matData(cv, src)).sort();
      const n = d.length;
      const med = n % 2 ? d[(n - 1) / 2] : (d[n / 2 - 1] + d[n / 2]) / 2;
      lo = Math.trunc(Math.max(0, (1 - p.sigma) * med));
      hi = Math.trunc(Math.min(255, (1 + p.sigma) * med));
    }
    const dst = ctx.mat();
    cv.Canny(src, dst, lo, hi, Number(p.aperture), p.l2);
    return dst;
  },
  emitPython: (p, ctx) => {
    const ap = ctx.c('aperture', Number(p.aperture));
    const tail = `apertureSize=${ap}, L2gradient=${p.l2 ? 'True' : 'False'}`;
    if (p.auto) {
      const s = ctx.c('sigma', pyRaw(pyFloat(p.sigma)));
      return { body: `m = float(np.median(img))\nlow, high = int(max(0, (1 - ${s}) * m)), int(min(255, (1 + ${s}) * m))  # auto thresholds\nimg = cv2.Canny(img, low, high, ${tail})` };
    }
    return { body: `img = cv2.Canny(img, ${ctx.c('low', pyRaw(pyFloat(p.low)))}, ${ctx.c('high', pyRaw(pyFloat(p.high)))}, ${tail})` };
  },
  testCases: [{}, { params: { auto: true } }, { input: 'gray', params: { aperture: '5', l2: true, low: 100, high: 300 } }],
});
