import { float, select } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';

defineStep({
  id: 'noise_residual',
  category: 'Domain',
  label: 'Noise residual',
  doc: 'Image minus a denoised copy (3×3 median): what remains is sensor noise and fine texture. Spliced or retouched regions often have a different noise level. Output: signed float32 residual, or |residual| × gain as uint8 for viewing.',
  cv2: 'img.astype(np.float32) - cv2.medianBlur(img, 3)',
  keywords: 'noise residual forensics tamper prnu sensor noise splicing high pass',
  params: [select('out', 'Output', [['abs', 'uint8 |residual| × gain'], ['float32', 'float32 (signed)']], 'abs'), float('gain', 'Gain', 8, { min: 1, max: 64, step: 1, showIf: { key: 'out', equals: ['abs'] } })],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { dtype: ['uint8'] },
  needs: ['medianBlur', 'subtract', 'convertScaleAbs'],
  summary: (p) => (p.out === 'abs' ? `|r| × ${p.gain}` : 'signed'),
  outMeta: (m, p) => (p.out === 'abs' ? m : { ...m, dtype: 'float32', range: [-255, 255] }),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const den = ctx.mat();
    cv.medianBlur(src, den, 3);
    const a = ctx.mat();
    const b = ctx.mat();
    src.convertTo(a, cv.CV_32F, 1, 0);
    den.convertTo(b, cv.CV_32F, 1, 0);
    const r = ctx.mat();
    cv.subtract(a, b, r);
    if (p.out === 'float32') return r;
    const dst = ctx.mat();
    cv.convertScaleAbs(r, dst, p.gain, 0);
    return dst;
  },
  emitPython: (p, ctx) => {
    const lines = ['residual = img.astype(np.float32) - cv2.medianBlur(img, 3).astype(np.float32)'];
    lines.push(p.out === 'abs' ? `img = cv2.convertScaleAbs(residual, alpha=${ctx.c('gain', pyRaw(pyFloat(p.gain)))})` : 'img = residual');
    return { body: lines.join('\n') };
  },
  testCases: [{}, { params: { out: 'float32' } }],
});
