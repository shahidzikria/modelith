import { float } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';

defineStep({
  id: 'brightness_contrast',
  category: 'Intensity',
  label: 'Brightness / contrast',
  doc: 'Linear adjustment out = |α·x + β| saturated to uint8 (cv2.convertScaleAbs). α > 1 adds contrast, β > 0 brightens. Simple and fast; use CLAHE or gamma when only dark regions need lifting.',
  cv2: 'cv2.convertScaleAbs(img, alpha=α, beta=β)',
  keywords: 'brightness contrast alpha beta linear convertScaleAbs exposure',
  params: [float('alpha', 'Contrast α', 1.2, { min: 0, max: 4, step: 0.05 }), float('beta', 'Brightness β', 10, { min: -255, max: 255, step: 1 })],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { dtype: ['uint8'] },
  needs: ['convertScaleAbs'],
  summary: (p) => `α ${p.alpha} · β ${p.beta}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const dst = ctx.mat();
    ctx.cv.convertScaleAbs(src, dst, p.alpha, p.beta);
    return dst;
  },
  emitPython: (p, ctx) => {
    const a = ctx.c('alpha', pyRaw(pyFloat(p.alpha)), 'contrast');
    const b = ctx.c('beta', pyRaw(pyFloat(p.beta)), 'brightness');
    return { body: `img = cv2.convertScaleAbs(img, alpha=${a}, beta=${b})` };
  },
  testCases: [{}, { params: { alpha: 0.6, beta: -40 } }, { input: 'gray', params: { alpha: 2, beta: 0 } }],
});
