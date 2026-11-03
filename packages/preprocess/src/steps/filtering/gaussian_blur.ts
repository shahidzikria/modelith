import { border, float, oddInt } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { borderCv, borderPy, kernelTooBig, oddCheck, pyRaw } from '../util';

defineStep({
  id: 'gaussian_blur',
  category: 'Filtering',
  label: 'Gaussian blur',
  doc: 'Weighted average with a Gaussian kernel (cv2.GaussianBlur). The kernel size must be odd; σ = 0 derives σ from the size (0.3·((k−1)/2 − 1) + 0.8). The standard pre-smoothing before edges, thresholds and downsampling.',
  cv2: 'cv2.GaussianBlur(img, (k, k), sigmaX, sigmaY=sigmaY)',
  keywords: 'gaussian blur smooth denoise sigma',
  params: [oddInt('k', 'Kernel size (odd)', 5, { max: 63 }), float('sigmaX', 'σX (0 = from size)', 0, { min: 0, max: 30, step: 0.1 }), float('sigmaY', 'σY (0 = σX)', 0, { min: 0, max: 30, step: 0.1 }), border('REFLECT_101')],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  needs: ['GaussianBlur'],
  summary: (p) => `${p.k}×${p.k}${p.sigmaX ? ` σ ${p.sigmaX}` : ''}`,
  outMeta: (m) => m,
  check: (m, p) => [...oddCheck(p.k), ...kernelTooBig(m, p.k)],
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const dst = ctx.mat();
    cv.GaussianBlur(src, dst, new cv.Size(p.k, p.k), p.sigmaX, p.sigmaY, borderCv(cv, p.border));
    return dst;
  },
  emitPython: (p, ctx) => {
    const k = ctx.c('ksize', p.k, 'must be odd');
    const sx = ctx.c('sigma_x', pyRaw(pyFloat(p.sigmaX)));
    const sy = ctx.c('sigma_y', pyRaw(pyFloat(p.sigmaY)));
    return { body: `img = cv2.GaussianBlur(img, (${k}, ${k}), ${sx}, sigmaY=${sy}, borderType=${ctx.c('border', borderPy(p.border))})` };
  },
  emitAlbumentations: (p) => `A.GaussianBlur(blur_limit=(${p.k}, ${p.k}), sigma_limit=(${pyFloat(p.sigmaX)}, ${pyFloat(p.sigmaX)}), p=1.0)`,
  emitTorchvision: (p) => `v2.GaussianBlur(kernel_size=${p.k}, sigma=${p.sigmaX > 0 ? pyFloat(p.sigmaX) : pyFloat(Math.round((0.3 * ((p.k - 1) / 2 - 1) + 0.8) * 1000) / 1000)})`,
  testCases: [{}, { params: { k: 9, sigmaX: 2.5, sigmaY: 1 } }, { input: 'u16', params: { k: 3 } }, { input: 'float', params: { k: 7 }, tolerance: 1e-5 }],
});
