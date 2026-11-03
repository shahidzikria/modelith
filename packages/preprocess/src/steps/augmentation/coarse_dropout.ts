import { bool, color, range } from '../../params';
import { defineStep } from '../../registry';
import { colorPy, colorValues } from '../util';

defineStep({
  id: 'coarse_dropout',
  category: 'Augmentation',
  label: 'Cutout / coarse dropout',
  doc: 'Fill a random number of random rectangles with a constant colour (Cutout, Albumentations CoarseDropout). Forces the model to use the whole object instead of one discriminative part. Per hole the draws are: height, width, y, x. Optionally clear the mask under the holes too.',
  cv2: 'img[y:y + hh, x:x + ww] = fill',
  keywords: 'cutout coarse dropout occlusion holes rectangles augmentation',
  params: [range('holes', 'Number of holes', [1, 4], { min: 1, max: 50, step: 1 }), range('hh', 'Hole height (px)', [8, 32], { min: 1, max: 1024, step: 1 }), range('ww', 'Hole width (px)', [8, 32], { min: 1, max: 1024, step: 1 }), color('fill', 'Fill', '#000000'), bool('maskToo', 'Also clear the mask', false)],
  engine: 'both',
  kind: 'random',
  affectsGeometry: false,
  usesTargets: true,
  summary: (p) => `${p.holes[0]}–${p.holes[1]} holes`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const { w, h, c } = ctx.in;
    const fill = colorValues(p.fill, ctx.in);
    const dst = ctx.mat();
    src.copyTo(dst);
    const d = ctx.in.dtype === 'uint8' ? dst.data : ctx.in.dtype === 'uint16' ? dst.data16U : dst.data32F;
    const mask = p.maskToo && ctx.t.mask ? ctx.t.mask : undefined;
    const md = mask ? new (mask.data.constructor as any)(mask.data) : undefined;
    const n = ctx.rng.randint(p.holes[0], p.holes[1]);
    for (let i = 0; i < n; i++) {
      const hh = Math.min(h, ctx.rng.randint(p.hh[0], p.hh[1]));
      const ww = Math.min(w, ctx.rng.randint(p.ww[0], p.ww[1]));
      const y = ctx.rng.randint(0, h - hh);
      const x = ctx.rng.randint(0, w - ww);
      for (let yy = y; yy < y + hh; yy++)
        for (let xx = x; xx < x + ww; xx++) {
          for (let k = 0; k < c; k++) d[(yy * w + xx) * c + k] = fill[k] ?? fill[0];
          if (md) for (let k = 0; k < mask!.meta.c; k++) md[(yy * w + xx) * mask!.meta.c + k] = 0;
        }
    }
    if (mask && md) ctx.t.mask = { meta: mask.meta, data: md };
    void cv;
    return dst;
  },
  emitPython: (p, ctx) => {
    ctx.helper('randint');
    const lines = [
      'h, w = img.shape[:2]',
      'img = img.copy()',
      ...(p.maskToo ? ['if t.mask is not None:', '    t.mask = t.mask.copy()'] : []),
      `for _ in range(_randint(rng, *${ctx.c('holes', p.holes)})):`,
      `    hh = min(h, _randint(rng, *${ctx.c('hole_h', p.hh)}))`,
      `    ww = min(w, _randint(rng, *${ctx.c('hole_w', p.ww)}))`,
      '    y, x = _randint(rng, 0, h - hh), _randint(rng, 0, w - ww)',
      `    img[y : y + hh, x : x + ww] = ${ctx.c('fill', { __raw: colorPy(p.fill, ctx.in) })}`,
    ];
    if (p.maskToo) lines.push('    if t.mask is not None:', '        t.mask[y : y + hh, x : x + ww] = 0');
    return { body: lines.join('\n') };
  },
  emitAlbumentations: (p) => `A.CoarseDropout(num_holes_range=(${p.holes[0]}, ${p.holes[1]}), hole_height_range=(${p.hh[0]}, ${p.hh[1]}), hole_width_range=(${p.ww[0]}, ${p.ww[1]}), p={p})`,
  testCases: [{ seed: 181 }, { seed: 182, params: { maskToo: true, fill: '#808080' }, targets: true }],
});
