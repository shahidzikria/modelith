import { select } from '../../params';
import { defineStep } from '../../registry';

const PRESETS: Record<string, { label: string; py: string }> = {
  sp: { label: 'Salt & pepper (median 3)', py: 'img = cv2.medianBlur(img, 3)' },
  sp_strong: { label: 'Salt & pepper, heavy (median 5)', py: 'img = cv2.medianBlur(img, 5)' },
  gaussian: { label: 'Gaussian noise (bilateral 9 / 75 / 75)', py: 'img = cv2.bilateralFilter(img, 9, 75, 75)' },
  gaussian_light: { label: 'Gaussian noise, light (Gaussian 5×5)', py: 'img = cv2.GaussianBlur(img, (5, 5), 0)' },
};

defineStep({
  id: 'denoise_preset',
  category: 'Denoising',
  label: 'Denoise preset',
  doc: 'Quick, well-known choices: a median filter for salt-and-pepper (impulse) noise, a bilateral or Gaussian filter for Gaussian sensor noise. Switch to the individual filters to tune parameters, or to non-local means for the best quality.',
  cv2: 'cv2.medianBlur / cv2.bilateralFilter / cv2.GaussianBlur',
  keywords: 'denoise noise preset median bilateral salt pepper gaussian',
  params: [select('preset', 'Noise type', Object.entries(PRESETS).map(([k, v]) => [k, v.label]), 'sp')],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1, 3], dtype: ['uint8'] },
  needs: ['medianBlur', 'bilateralFilter', 'GaussianBlur'],
  tolerance: 1,
  summary: (p) => PRESETS[p.preset]?.label ?? p.preset,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const dst = ctx.mat();
    if (p.preset === 'sp') cv.medianBlur(src, dst, 3);
    else if (p.preset === 'sp_strong') cv.medianBlur(src, dst, 5);
    else if (p.preset === 'gaussian') cv.bilateralFilter(src, dst, 9, 75, 75, cv.BORDER_DEFAULT);
    else cv.GaussianBlur(src, dst, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
    return dst;
  },
  emitPython: (p) => ({ body: `${PRESETS[p.preset].py}  # ${PRESETS[p.preset].label}` }),
  testCases: [{}, { params: { preset: 'gaussian' } }, { params: { preset: 'gaussian_light' }, input: 'gray' }],
});
