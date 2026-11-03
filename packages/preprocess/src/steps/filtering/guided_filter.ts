import { float, int } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';

defineStep({
  id: 'guided_filter',
  category: 'Filtering',
  label: 'Guided filter',
  doc: 'Edge-preserving smoothing guided by the image itself (He et al.; cv2.ximgproc.guidedFilter): like a bilateral filter but linear-time and without gradient reversal. radius 4–16; eps (regularisation, in squared intensity units) controls how much is smoothed: 0.01·255² ≈ 650 is moderate. Needs opencv-contrib, so it runs on the server.',
  cv2: 'cv2.ximgproc.guidedFilter(guide=img, src=img, radius, eps)',
  keywords: 'guided filter edge preserving smoothing ximgproc he haze',
  params: [int('radius', 'Radius', 8, { min: 1, max: 64 }), float('eps', 'eps', 650, { min: 0.001, max: 65025, step: 10 })],
  engine: 'server',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1, 3], dtype: ['uint8', 'float32'] },
  summary: (p) => `r ${p.radius} · eps ${p.eps}`,
  outMeta: (m) => m,
  emitPython: (p, ctx) => ({
    body: `img = cv2.ximgproc.guidedFilter(guide=img, src=img, radius=${ctx.c('radius', p.radius)}, eps=${ctx.c('eps', pyRaw(pyFloat(p.eps)))})  # needs opencv-contrib-python`,
  }),
});
