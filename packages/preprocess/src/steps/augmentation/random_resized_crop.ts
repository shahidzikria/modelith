import { interp, range, size } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import type { BrowserCtx, Params } from '../../types';
import { interpCv, interpPy, pyRaw } from '../util';

/** torchvision's RandomResizedCrop box sampling, drawing only from rng.random(). */
export function sampleCropBox(ctx: BrowserCtx, w: number, h: number, scale: number[], ratio: number[]): [number, number, number, number] {
  const area = w * h;
  const lr = [Math.log(ratio[0]), Math.log(ratio[1])];
  for (let i = 0; i < 10; i++) {
    const target = area * ctx.rng.uniform(scale[0], scale[1]);
    const ar = Math.exp(ctx.rng.uniform(lr[0], lr[1]));
    const cw = Math.floor(Math.sqrt(target * ar) + 0.5);
    const ch = Math.floor(Math.sqrt(target / ar) + 0.5);
    if (cw > 0 && cw <= w && ch > 0 && ch <= h) return [ctx.rng.randint(0, w - cw), ctx.rng.randint(0, h - ch), cw, ch];
  }
  // Fallback: centre crop at the closest allowed aspect ratio.
  const r = w / h;
  let cw = w, ch = h;
  if (r < ratio[0]) ch = Math.floor(w / ratio[0] + 0.5);
  else if (r > ratio[1]) cw = Math.floor(h * ratio[1] + 0.5);
  return [Math.floor((w - cw) / 2), Math.floor((h - ch) / 2), cw, ch];
}

export const PY_SAMPLE_CROP = (S: string, R: string) =>
  [
    'h, w = img.shape[:2]',
    `log_r = (np.log(${R}[0]), np.log(${R}[1]))`,
    'for _ in range(10):',
    `    target = w * h * _uniform(rng, *${S})`,
    '    ar = np.exp(_uniform(rng, *log_r))',
    '    cw, ch = int(np.sqrt(target * ar) + 0.5), int(np.sqrt(target / ar) + 0.5)',
    '    if 0 < cw <= w and 0 < ch <= h:',
    '        x, y = _randint(rng, 0, w - cw), _randint(rng, 0, h - ch)',
    '        break',
    'else:  # fallback: centre crop at the nearest allowed aspect ratio',
    '    cw, ch = w, h',
    `    if w / h < ${R}[0]:`,
    `        ch = int(w / ${R}[0] + 0.5)`,
    `    elif w / h > ${R}[1]:`,
    `        cw = int(h * ${R}[1] + 0.5)`,
    '    x, y = (w - cw) // 2, (h - ch) // 2',
  ].join('\n');

defineStep({
  id: 'random_resized_crop',
  category: 'Augmentation',
  label: 'Random resized crop',
  doc: 'The standard ImageNet training augmentation (torchvision RandomResizedCrop): crop a random region covering a fraction of the image (scale, default 8–100 %) with a random aspect ratio (3/4–4/3), then resize it to the output size. Forces the model to recognise objects from parts and at different scales.',
  cv2: 'img[y:y + ch, x:x + cw] → cv2.resize(…, (W, H))',
  keywords: 'random resized crop rrc scale aspect imagenet augmentation',
  params: [size('size', 'Output (W × H)', [224, 224]), range('scale', 'Area fraction', [0.08, 1], { min: 0.01, max: 1, step: 0.01 }), range('ratio', 'Aspect ratio', [0.75, 1.3333], { min: 0.1, max: 10, step: 0.01 }), interp('LINEAR')],
  engine: 'both',
  kind: 'random',
  defaultMode: 'train',
  affectsGeometry: true,
  needs: ['resize'],
  tolerance: 1,
  summary: (p) => `${p.size[0]}×${p.size[1]} · ${p.scale[0]}–${p.scale[1]}`,
  outMeta: (m, p) => ({ ...m, w: p.size[0], h: p.size[1] }),
  applyBrowser: (src, p: Params, ctx) => {
    const { cv } = ctx;
    const [x, y, cw, ch] = sampleCropBox(ctx, ctx.in.w, ctx.in.h, p.scale, p.ratio);
    const crop = ctx.mat();
    ctx.keep(src.roi(new cv.Rect(x, y, cw, ch))).copyTo(crop);
    ctx.t.crop(x, y, cw, ch);
    const dst = ctx.mat();
    cv.resize(crop, dst, new cv.Size(p.size[0], p.size[1]), 0, 0, interpCv(cv, p.interp));
    ctx.t.resize(p.size[0], p.size[1]);
    return dst;
  },
  emitPython: (p, ctx) => {
    ctx.helper('uniform');
    ctx.helper('randint');
    const S = ctx.c('scale', p.scale, 'area fraction');
    const R = ctx.c('ratio', p.ratio, 'aspect ratio w / h');
    const Z = ctx.c('size', p.size, '(width, height)');
    return {
      body: [PY_SAMPLE_CROP(S, R), 'img = img[y : y + ch, x : x + cw]', 't.crop(x, y, cw, ch)', `img = cv2.resize(img, ${Z}, interpolation=${ctx.c('interp', interpPy(p.interp))})`, `t.resize(${Z})`].join('\n'),
    };
  },
  emitAlbumentations: (p) => `A.RandomResizedCrop(size=(${p.size[1]}, ${p.size[0]}), scale=(${p.scale[0]}, ${p.scale[1]}), ratio=(${p.ratio[0]}, ${p.ratio[1]}), interpolation=cv2.INTER_${p.interp}, p={p})`,
  emitTorchvision: (p) => `v2.RandomResizedCrop((${p.size[1]}, ${p.size[0]}), scale=(${p.scale[0]}, ${p.scale[1]}), ratio=(${p.ratio[0]}, ${p.ratio[1]}), antialias=True)`,
  testCases: [{ seed: 5, params: { size: [64, 64] } }, { seed: 6, params: { size: [80, 48], scale: [0.5, 1] } }],
});
void pyFloat;
void pyRaw;
