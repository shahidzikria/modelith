import { float, oddInt } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';

defineStep({
  id: 'nlm_denoise',
  category: 'Denoising',
  label: 'Non-local means',
  doc: 'Non-local means (cv2.fastNlMeansDenoising / …Colored): averages similar patches from the whole search window, removing Gaussian-like noise while keeping texture. h ≈ noise σ (3–15); hColor for the colour channels; template 7, search 21. Slow, and not in OpenCV.js, so it runs on the server.',
  cv2: 'cv2.fastNlMeansDenoisingColored(img, None, h, hColor, templateWindowSize, searchWindowSize)',
  keywords: 'non local means nlm denoise noise fastNlMeansDenoising photo',
  params: [float('h', 'h (luminance)', 10, { min: 1, max: 50, step: 0.5 }), float('hColor', 'hColor', 10, { min: 1, max: 50, step: 0.5 }), oddInt('template', 'Template window', 7, { min: 3, max: 21 }), oddInt('search', 'Search window', 21, { min: 5, max: 51 })],
  engine: 'server',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1, 3], dtype: ['uint8'] },
  summary: (p) => `h ${p.h}${''}`,
  outMeta: (m) => m,
  emitPython: (p, ctx) => {
    const h = ctx.c('h', pyRaw(pyFloat(p.h)));
    const t = ctx.c('template', p.template);
    const s = ctx.c('search', p.search);
    if (ctx.in.c === 1) return { body: `img = cv2.fastNlMeansDenoising(img, None, h=${h}, templateWindowSize=${t}, searchWindowSize=${s})` };
    const hc = ctx.c('h_color', pyRaw(pyFloat(p.hColor)));
    return { body: `img = cv2.fastNlMeansDenoisingColored(img, None, h=${h}, hColor=${hc}, templateWindowSize=${t}, searchWindowSize=${s})` };
  },
  testCases: [{}, { input: 'gray' }],
});
