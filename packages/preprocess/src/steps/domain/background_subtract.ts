import { float, int, select } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';
import { structuringElement } from '../morphology/morphology';

defineStep({
  id: 'background_subtract',
  category: 'Domain',
  label: 'Background subtraction',
  doc: 'Remove uneven illumination (microscopy, X-ray, gel images): estimate the background with a rolling-ball approximation (grey opening with a disk of the given radius, larger than the objects) or a large Gaussian blur, then subtract it. "Dark objects" mode uses a closing and subtracts the image from the background.',
  cv2: 'bg = cv2.morphologyEx(img, cv2.MORPH_OPEN, disk(r)); img = cv2.subtract(img, bg)',
  keywords: 'background subtraction rolling ball uneven illumination flat field microscopy top hat',
  params: [
    select('method', 'Background from', [['ball', 'Rolling ball (opening)'], ['gaussian', 'Large Gaussian blur']], 'ball'),
    int('radius', 'Ball radius (px)', 25, { min: 1, max: 500, showIf: { key: 'method', equals: ['ball'] } }),
    float('sigma', 'Gaussian σ', 30, { min: 1, max: 500, step: 1, showIf: { key: 'method', equals: ['gaussian'] } }),
    select('objects', 'Objects are', [['bright', 'Bright on dark'], ['dark', 'Dark on light']], 'bright'),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { dtype: ['uint8', 'uint16', 'float32'] },
  needs: ['morphologyEx', 'GaussianBlur', 'subtract'],
  tolerance: 1,
  summary: (p) => `${p.method === 'ball' ? `ball r ${p.radius}` : `gauss σ ${p.sigma}`} · ${p.objects}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const bg = ctx.mat();
    if (p.method === 'ball') {
      const k = structuringElement(ctx, 'ELLIPSE', 2 * p.radius + 1);
      cv.morphologyEx(src, bg, p.objects === 'bright' ? cv.MORPH_OPEN : cv.MORPH_CLOSE, k, new cv.Point(-1, -1), 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
    } else cv.GaussianBlur(src, bg, new cv.Size(0, 0), p.sigma, p.sigma, cv.BORDER_DEFAULT);
    const dst = ctx.mat();
    if (p.objects === 'bright') cv.subtract(src, bg, dst);
    else cv.subtract(bg, src, dst);
    return dst;
  },
  emitPython: (p, ctx) => {
    let bg: string;
    if (p.method === 'ball') {
      const r = ctx.c('radius', p.radius, 'larger than the objects');
      bg = `disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ${r} + 1, 2 * ${r} + 1))\nbg = cv2.morphologyEx(img, cv2.MORPH_${p.objects === 'bright' ? 'OPEN' : 'CLOSE'}, disk)  # rolling-ball approximation`;
    } else bg = `bg = cv2.GaussianBlur(img, (0, 0), ${ctx.c('sigma', pyRaw(pyFloat(p.sigma)))})`;
    return { body: `${bg}\nimg = cv2.subtract(${p.objects === 'bright' ? 'img, bg' : 'bg, img'})  # saturating` };
  },
  testCases: [{ input: 'gray', params: { radius: 10 } }, { input: 'doc', params: { method: 'gaussian', objects: 'dark', sigma: 15 } }],
});
