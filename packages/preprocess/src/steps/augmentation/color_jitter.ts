import { float } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { mapPixels, pyRaw, rint } from '../util';

defineStep({
  id: 'color_jitter',
  category: 'Augmentation',
  label: 'Colour jitter',
  doc: 'Random brightness, contrast, saturation and hue (torchvision ColorJitter semantics, fixed order): factors are drawn from [1 − x, 1 + x] (hue: a shift of ±x of the colour circle). Brightness scales the pixels, contrast blends with the mean grey, saturation blends with the grey image, hue rotates H in HSV.',
  cv2: 'img * b → (img − mean) * c + mean → img * s + gray * (1 − s) → HSV hue shift',
  keywords: 'color colour jitter brightness contrast saturation hue photometric augmentation',
  params: [
    float('brightness', 'Brightness ±', 0.2, { min: 0, max: 1, step: 0.05 }),
    float('contrast', 'Contrast ±', 0.2, { min: 0, max: 1, step: 0.05 }),
    float('saturation', 'Saturation ±', 0.2, { min: 0, max: 1, step: 0.05 }),
    float('hue', 'Hue ± (fraction of circle)', 0.02, { min: 0, max: 0.5, step: 0.01 }),
  ],
  engine: 'both',
  kind: 'random',
  affectsGeometry: false,
  accepts: { c: [3], dtype: ['uint8'], space: ['BGR', 'RGB'] },
  needs: ['cvtColor'],
  tolerance: 1,
  summary: (p) => `b ${p.brightness} · c ${p.contrast} · s ${p.saturation} · h ${p.hue}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const rgb = ctx.in.space === 'RGB';
    const b = ctx.rng.uniform(1 - p.brightness, 1 + p.brightness);
    const c = ctx.rng.uniform(1 - p.contrast, 1 + p.contrast);
    const s = ctx.rng.uniform(1 - p.saturation, 1 + p.saturation);
    const h = ctx.rng.uniform(-p.hue, p.hue);
    const grayOf = (m: any) => {
      const g = ctx.mat();
      cv.cvtColor(m, g, rgb ? cv.COLOR_RGB2GRAY : cv.COLOR_BGR2GRAY);
      return g.data as Uint8Array;
    };
    let img = mapPixels(src, ctx, 'uint8', (v) => v * b);
    const g1 = grayOf(img);
    let sum = 0;
    for (const v of g1) sum += v;
    const mean = sum / g1.length;
    img = mapPixels(img, ctx, 'uint8', (v) => (v - mean) * c + mean);
    const g2 = grayOf(img);
    img = mapPixels(img, ctx, 'uint8', (v, _k, i) => v * s + g2[i / 3] * (1 - s));
    const shift = rint(h * 180);
    if (shift) {
      const hsv = ctx.mat();
      cv.cvtColor(img, hsv, rgb ? cv.COLOR_RGB2HSV : cv.COLOR_BGR2HSV);
      const d = hsv.data as Uint8Array;
      for (let i = 0; i < d.length; i += 3) d[i] = (((d[i] + shift) % 180) + 180) % 180;
      const out = ctx.mat();
      cv.cvtColor(hsv, out, rgb ? cv.COLOR_HSV2RGB : cv.COLOR_HSV2BGR);
      img = out;
    }
    return img;
  },
  emitPython: (p, ctx) => {
    ctx.helper('uniform');
    ctx.helper('saturate');
    const s = ctx.in.space === 'RGB' ? 'RGB' : 'BGR';
    const B = ctx.c('brightness', pyRaw(pyFloat(p.brightness)));
    const C = ctx.c('contrast', pyRaw(pyFloat(p.contrast)));
    const S = ctx.c('saturation', pyRaw(pyFloat(p.saturation)));
    const H = ctx.c('hue', pyRaw(pyFloat(p.hue)), 'fraction of the colour circle');
    return {
      body: [
        `b = _uniform(rng, 1 - ${B}, 1 + ${B})`,
        `c = _uniform(rng, 1 - ${C}, 1 + ${C})`,
        `s = _uniform(rng, 1 - ${S}, 1 + ${S})`,
        `hue = _uniform(rng, -${H}, ${H})`,
        'img = _saturate(img * b, np.uint8)  # brightness',
        `mean = cv2.cvtColor(img, cv2.COLOR_${s}2GRAY).mean()`,
        'img = _saturate((img - mean) * c + mean, np.uint8)  # contrast',
        `gray = cv2.cvtColor(img, cv2.COLOR_${s}2GRAY)[..., None]`,
        'img = _saturate(img * s + gray * (1 - s), np.uint8)  # saturation',
        'shift = int(np.rint(hue * 180))  # OpenCV hue units (0..179)',
        'if shift:',
        `    hsv = cv2.cvtColor(img, cv2.COLOR_${s}2HSV)`,
        '    hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + shift) % 180).astype(np.uint8)',
        `    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2${s})`,
      ].join('\n'),
    };
  },
  emitAlbumentations: (p) => `A.ColorJitter(brightness=${p.brightness}, contrast=${p.contrast}, saturation=${p.saturation}, hue=${p.hue}, p={p})`,
  emitTorchvision: (p) => `v2.RandomApply([v2.ColorJitter(brightness=${p.brightness}, contrast=${p.contrast}, saturation=${p.saturation}, hue=${p.hue})], p={p})`,
  testCases: [{ seed: 71 }, { seed: 72, params: { hue: 0.2 } }],
});
