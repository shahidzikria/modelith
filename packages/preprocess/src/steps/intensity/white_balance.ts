import { float, select } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { mapPixels, matData, pyRaw } from '../util';
import { percentile } from './percentile_stretch';

defineStep({
  id: 'white_balance',
  category: 'Intensity',
  label: 'White balance',
  doc: 'Remove a colour cast. Gray-world: scale each channel so its mean equals the overall mean (works for varied scenes). White-patch: scale each channel so its brightest pixels (high percentile) become white (works when the scene has something white). For OpenCV\'s learning-based method use "White balance (xphoto)".',
  cv2: 'img * (means.mean() / means)',
  keywords: 'white balance colour cast gray world grey world white patch color constancy',
  params: [
    select('method', 'Method', [['gray_world', 'Gray-world'], ['white_patch', 'White-patch']], 'gray_world'),
    float('percentile', 'White percentile', 99, { min: 90, max: 100, step: 0.1, showIf: { key: 'method', equals: ['white_patch'] } }),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [3], dtype: ['uint8'] },
  tolerance: 1,
  summary: (p) => (p.method === 'gray_world' ? 'gray-world' : `white-patch p${p.percentile}`),
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const d = matData(ctx.cv, src);
    const n = d.length / 3;
    const ref: number[] = [];
    for (let k = 0; k < 3; k++) {
      if (p.method === 'gray_world') {
        let s = 0;
        for (let i = 0; i < n; i++) s += d[i * 3 + k];
        ref.push(s / n);
      } else {
        const v = new Float64Array(n);
        for (let i = 0; i < n; i++) v[i] = d[i * 3 + k];
        v.sort();
        ref.push(percentile(v, p.percentile));
      }
    }
    const target = p.method === 'gray_world' ? (ref[0] + ref[1] + ref[2]) / 3 : 255;
    const gain = ref.map((r) => target / Math.max(r, 1e-6));
    return mapPixels(src, ctx, 'uint8', (v, k) => v * gain[k]);
  },
  emitPython: (p, ctx) => {
    ctx.helper('saturate');
    if (p.method === 'gray_world')
      return { body: 'means = img.reshape(-1, 3).mean(axis=0)  # per channel\ngain = means.mean() / np.maximum(means, 1e-6)\nimg = _saturate(img * gain, np.uint8)' };
    const q = ctx.c('percentile', pyRaw(pyFloat(p.percentile)));
    return { body: `white = np.percentile(img.reshape(-1, 3), ${q}, axis=0)  # per channel\nimg = _saturate(img * (255.0 / np.maximum(white, 1e-6)), np.uint8)` };
  },
  testCases: [{}, { params: { method: 'white_patch' } }],
});
