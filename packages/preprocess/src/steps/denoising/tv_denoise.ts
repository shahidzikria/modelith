import { float } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';

defineStep({
  id: 'tv_denoise',
  category: 'Denoising',
  label: 'Total-variation denoise',
  doc: 'Chambolle total-variation denoising (scikit-image denoise_tv_chambolle): flattens noise into piecewise-constant regions while keeping sharp edges; higher weight = smoother ("cartoon" look). Weight 0.05–0.2 for images in [0, 1]. Runs on the server.',
  cv2: 'skimage.restoration.denoise_tv_chambolle(img, weight)',
  keywords: 'total variation tv chambolle denoise skimage edges',
  params: [float('weight', 'Weight', 0.1, { min: 0.005, max: 1, step: 0.005 })],
  engine: 'server',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1, 3], dtype: ['uint8', 'float32'] },
  summary: (p) => `weight ${p.weight}`,
  outMeta: (m) => m,
  emitPython: (p, ctx) => {
    const w = ctx.c('weight', pyRaw(pyFloat(p.weight)));
    const lines = [
      'from skimage.restoration import denoise_tv_chambolle  # optional dependency: scikit-image',
      '',
      `f = img.astype(np.float64)${ctx.in.dtype === 'uint8' ? ' / 255.0' : ''}`,
      `f = denoise_tv_chambolle(f, weight=${w}, channel_axis=${ctx.in.c === 3 ? '-1' : 'None'})`,
    ];
    if (ctx.in.dtype === 'uint8') {
      ctx.helper('saturate');
      lines.push('img = _saturate(f * 255.0, np.uint8)');
    } else lines.push('img = f.astype(np.float32)');
    return { body: lines.join('\n') };
  },
});
