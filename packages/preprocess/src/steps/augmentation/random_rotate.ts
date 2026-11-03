import { border, color, float, interp } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';
import { rotationMatrix, warpAffineBrowser, warpAffinePy } from '../geometric/rotate';

defineStep({
  id: 'random_rotate',
  category: 'Augmentation',
  label: 'Random rotate',
  doc: 'Rotate by an angle drawn uniformly from [−limit, +limit] about the centre (same size; corners filled by the border mode). Boxes become the enclosing box of their rotated corners, which grows them a little: keep the limit small (≤ 15°) for detection.',
  cv2: 'cv2.warpAffine(img, cv2.getRotationMatrix2D(center, angle, 1.0), (w, h))',
  keywords: 'random rotate rotation angle augmentation',
  params: [float('limit', 'Limit (±°)', 15, { min: 0, max: 180, step: 1 }), interp('LINEAR'), border('REFLECT_101'), color('color', 'Fill colour', '#000000', { showIf: { key: 'border', equals: ['CONSTANT'] } })],
  engine: 'both',
  kind: 'random',
  affectsGeometry: true,
  needs: ['warpAffine'],
  tolerance: 1,
  summary: (p) => `±${p.limit}°`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const angle = ctx.rng.uniform(-p.limit, p.limit);
    const r = rotationMatrix(ctx.in, angle, 1, false);
    return warpAffineBrowser(src, r.M, r.w, r.h, ctx, p.interp, p.border, p.color);
  },
  emitPython: (p, ctx) => {
    ctx.helper('uniform');
    const lim = ctx.c('limit', pyRaw(pyFloat(p.limit)), 'degrees');
    return {
      body: [`angle = _uniform(rng, -${lim}, ${lim})`, 'h, w = img.shape[:2]', 'M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.0)', 'size = (w, h)', warpAffinePy(ctx, p.interp, p.border, p.color)].join('\n'),
    };
  },
  emitAlbumentations: (p) => `A.Rotate(limit=(${-p.limit}, ${p.limit}), interpolation=cv2.INTER_${p.interp}, border_mode=cv2.BORDER_${p.border}, p={p})`,
  emitTorchvision: (p) => `v2.RandomApply([v2.RandomRotation(${p.limit})], p={p})`,
  testCases: [{ seed: 11 }, { seed: 12, params: { limit: 90, border: 'CONSTANT', color: '#808080' } }],
});
