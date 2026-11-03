import { range, select } from '../../params';
import { defineStep } from '../../registry';
import { DTYPE_MAX } from '../util';
import { PY_SAMPLE_CROP, sampleCropBox } from './random_resized_crop';

defineStep({
  id: 'random_erasing',
  category: 'Augmentation',
  label: 'Random erasing',
  doc: 'Random Erasing (Zhong et al.; torchvision RandomErasing): one rectangle with a random area fraction and aspect ratio is replaced by random noise (or black). Usually applied after normalisation in torchvision; here it runs on the uint8 / float image wherever you place it.',
  cv2: 'img[y:y + ch, x:x + cw] = rng.random(...) * max',
  keywords: 'random erasing occlusion noise rectangle augmentation reid',
  params: [range('scale', 'Area fraction', [0.02, 0.33], { min: 0.001, max: 1, step: 0.01 }), range('ratio', 'Aspect ratio', [0.3, 3.3], { min: 0.05, max: 20, step: 0.1 }), select('value', 'Fill', [['random', 'Random noise'], ['zero', 'Black']], 'random')],
  engine: 'both',
  kind: 'random',
  affectsGeometry: false,
  summary: (p) => `${p.scale[0]}–${p.scale[1]} · ${p.value}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { w, h, c, dtype } = ctx.in;
    const [x, y, cw, ch] = sampleCropBox(ctx, w, h, p.scale, p.ratio);
    const noise = p.value === 'random' ? ctx.rng.randomArray(ch * cw * c) : undefined;
    const max = dtype === 'uint8' || dtype === 'uint16' ? DTYPE_MAX[dtype] : 1;
    const isInt = dtype === 'uint8' || dtype === 'uint16';
    const dst = ctx.mat();
    src.copyTo(dst);
    const a = dtype === 'uint8' ? dst.data : dtype === 'uint16' ? dst.data16U : dst.data32F;
    for (let yy = 0; yy < ch; yy++)
      for (let xx = 0; xx < cw; xx++)
        for (let k = 0; k < c; k++) {
          const v = noise ? noise[(yy * cw + xx) * c + k] * (isInt ? max + 1 : 1) : 0;
          a[((y + yy) * w + x + xx) * c + k] = isInt ? Math.floor(v) : v;
        }
    return dst;
  },
  emitPython: (p, ctx) => {
    ctx.helper('uniform');
    ctx.helper('randint');
    const S = ctx.c('scale', p.scale, 'area fraction');
    const R = ctx.c('ratio', p.ratio, 'aspect ratio w / h');
    const isInt = ctx.in.dtype === 'uint8' || ctx.in.dtype === 'uint16';
    const fill =
      p.value === 'random'
        ? isInt
          ? `(rng.random((ch, cw) + img.shape[2:]) * ${DTYPE_MAX[ctx.in.dtype as 'uint8' | 'uint16'] + 1}).astype(img.dtype)  # uniform integers`
          : 'rng.random((ch, cw) + img.shape[2:]).astype(img.dtype)'
        : '0';
    return { body: [PY_SAMPLE_CROP(S, R), 'img = img.copy()', `img[y : y + ch, x : x + cw] = ${fill}`].join('\n') };
  },
  emitTorchvision: (p) => `v2.RandomErasing(p={p}, scale=(${p.scale[0]}, ${p.scale[1]}), ratio=(${p.ratio[0]}, ${p.ratio[1]}), value=${p.value === 'random' ? '"random"' : '0'})`,
  testCases: [{ seed: 201 }, { seed: 202, params: { value: 'zero' } }],
});
