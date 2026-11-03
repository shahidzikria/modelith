import { select, str } from '../../params';
import { defineStep } from '../../registry';

defineStep({
  id: 'wavelet_denoise',
  category: 'Denoising',
  label: 'Wavelet denoise',
  doc: 'Wavelet shrinkage (scikit-image denoise_wavelet with BayesShrink or VisuShrink): the noise level is estimated from the finest wavelet coefficients, so no parameters are needed. Good on Gaussian noise in natural and medical images. Runs on the server (needs scikit-image).',
  cv2: 'skimage.restoration.denoise_wavelet(img, method="BayesShrink", rescale_sigma=True)',
  keywords: 'wavelet denoise bayesshrink visushrink skimage noise',
  params: [select('method', 'Method', [['BayesShrink', 'BayesShrink'], ['VisuShrink', 'VisuShrink']], 'BayesShrink'), str('wavelet', 'Wavelet', 'db1')],
  engine: 'server',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1, 3], dtype: ['uint8', 'float32'] },
  summary: (p) => `${p.method} · ${p.wavelet}`,
  outMeta: (m) => m,
  emitPython: (p, ctx) => {
    const lines = [
      'from skimage.restoration import denoise_wavelet  # optional dependency: scikit-image',
      '',
      `f = img.astype(np.float64)${ctx.in.dtype === 'uint8' ? ' / 255.0' : ''}`,
      `f = denoise_wavelet(f, method=${JSON.stringify(p.method)}, mode="soft", wavelet=${JSON.stringify(p.wavelet)}, rescale_sigma=True, channel_axis=${ctx.in.c === 3 ? '-1' : 'None'})`,
    ];
    if (ctx.in.dtype === 'uint8') {
      ctx.helper('saturate');
      lines.push('img = _saturate(f * 255.0, np.uint8)');
    } else lines.push('img = f.astype(np.float32)');
    return { body: lines.join('\n') };
  },
});
