import { bool, int } from '../../params';
import { defineStep } from '../../registry';
import { kernelTooBig } from '../util';

defineStep({
  id: 'box_filter',
  category: 'Filtering',
  label: 'Box filter',
  doc: 'cv2.boxFilter: sum (normalize = off) or mean (on) over a W×H window. Unnormalised sums are the building block of local statistics (e.g. local mean / variance maps); output depth is float32 when not normalised so sums do not overflow.',
  cv2: 'cv2.boxFilter(img, ddepth, (kw, kh), normalize=...)',
  keywords: 'box filter sum window local mean',
  params: [int('kw', 'Width', 5, { min: 1, max: 127 }), int('kh', 'Height', 5, { min: 1, max: 127 }), bool('normalize', 'Normalise (mean)', true)],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  needs: ['boxFilter'],
  tolerance: 1e-3,
  summary: (p) => `${p.kw}×${p.kh}${p.normalize ? '' : ' sum'}`,
  outMeta: (m, p) => (p.normalize ? m : { ...m, dtype: 'float32', range: [0, m.range[1] * p.kw * p.kh] }),
  check: (m, p) => kernelTooBig(m, Math.max(p.kw, p.kh)),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const dst = ctx.mat();
    cv.boxFilter(src, dst, p.normalize ? -1 : cv.CV_32F, new cv.Size(p.kw, p.kh), new cv.Point(-1, -1), p.normalize, cv.BORDER_DEFAULT);
    return dst;
  },
  emitPython: (p, ctx) => ({
    body: `img = cv2.boxFilter(img, ${p.normalize ? '-1' : 'cv2.CV_32F'}, (${ctx.c('kw', p.kw)}, ${ctx.c('kh', p.kh)}), normalize=${p.normalize ? 'True' : 'False'})`,
  }),
  testCases: [{ params: { kw: 7, kh: 3 } }, { params: { normalize: false } }],
});
