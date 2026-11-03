import { range } from '../../params';
import { defineStep } from '../../registry';
import { roundHalfUp } from '../util';

defineStep({
  id: 'downscale',
  category: 'Augmentation',
  label: 'Downscale → upscale',
  doc: 'Shrink by a random factor (INTER_AREA) and enlarge back to the original size (INTER_LINEAR): the image keeps its size but loses detail, like low-resolution cameras, digital zoom or thumbnails.',
  cv2: 'cv2.resize(cv2.resize(img, small, interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)',
  keywords: 'downscale upscale low resolution quality degrade augmentation',
  params: [range('scale', 'Scale', [0.25, 0.75], { min: 0.05, max: 1, step: 0.05 })],
  engine: 'both',
  kind: 'random',
  affectsGeometry: false,
  needs: ['resize'],
  tolerance: 1,
  summary: (p) => `×${p.scale[0]}–${p.scale[1]}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const s = ctx.rng.uniform(p.scale[0], p.scale[1]);
    const small = ctx.mat();
    cv.resize(src, small, new cv.Size(Math.max(1, roundHalfUp(ctx.in.w * s)), Math.max(1, roundHalfUp(ctx.in.h * s))), 0, 0, cv.INTER_AREA);
    const dst = ctx.mat();
    cv.resize(small, dst, new cv.Size(ctx.in.w, ctx.in.h), 0, 0, cv.INTER_LINEAR);
    return dst;
  },
  emitPython: (p, ctx) => {
    ctx.helper('uniform');
    return {
      body: [
        `s = _uniform(rng, *${ctx.c('scale', p.scale)})`,
        'h, w = img.shape[:2]',
        'small = cv2.resize(img, (max(1, int(w * s + 0.5)), max(1, int(h * s + 0.5))), interpolation=cv2.INTER_AREA)',
        'img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)',
      ].join('\n'),
    };
  },
  emitAlbumentations: (p) => `A.Downscale(scale_range=(${p.scale[0]}, ${p.scale[1]}), p={p})`,
  testCases: [{ seed: 171 }],
});
