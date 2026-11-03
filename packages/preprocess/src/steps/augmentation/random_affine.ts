import { border, color, float, interp, range } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import type { ImageMeta } from '../../types';
import { pyRaw } from '../util';
import { warpAffineBrowser, warpAffinePy } from '../geometric/rotate';

/** Rotation + scale + shear about the centre, then translation (same formula as the emitted NumPy). */
export function randomAffineMatrix(m: ImageMeta, angle: number, scale: number, tx: number, ty: number, shear: number): number[] {
  const a = (angle * Math.PI) / 180;
  const sh = Math.tan((shear * Math.PI) / 180);
  const R = [Math.cos(a) * scale, Math.sin(a) * scale, -Math.sin(a) * scale, Math.cos(a) * scale];
  const A = [R[0], R[0] * sh + R[1], R[2], R[2] * sh + R[3]];
  const cx = (m.w - 1) / 2, cy = (m.h - 1) / 2;
  return [A[0], A[1], cx + tx - (A[0] * cx + A[1] * cy), A[2], A[3], cy + ty - (A[2] * cx + A[3] * cy)];
}

defineStep({
  id: 'random_affine',
  category: 'Augmentation',
  label: 'Random affine',
  doc: 'Random rotation, scale, translation (fraction of the size) and shear in one warp (like Albumentations Affine / YOLO "degrees, scale, translate, shear"). Values are drawn in that order. Keep detection augmentations mild: large rotations and shears inflate boxes.',
  cv2: 'cv2.warpAffine(img, M, (w, h))',
  keywords: 'random affine shift scale rotate shear translate augmentation yolo',
  params: [
    float('degrees', 'Rotation (±°)', 10, { min: 0, max: 180, step: 1 }),
    range('scale', 'Scale', [0.9, 1.1], { min: 0.1, max: 4, step: 0.05 }),
    float('translate', 'Translate (± fraction)', 0.1, { min: 0, max: 0.5, step: 0.01 }),
    float('shear', 'Shear (±°)', 5, { min: 0, max: 45, step: 1 }),
    interp('LINEAR'),
    border('CONSTANT'),
    color('color', 'Fill colour', '#727272', { showIf: { key: 'border', equals: ['CONSTANT'] } }),
  ],
  engine: 'both',
  kind: 'random',
  affectsGeometry: true,
  needs: ['warpAffine'],
  tolerance: 1,
  summary: (p) => `±${p.degrees}° · ×${p.scale[0]}–${p.scale[1]} · ±${p.translate} · shear ${p.shear}°`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { w, h } = ctx.in;
    const angle = ctx.rng.uniform(-p.degrees, p.degrees);
    const s = ctx.rng.uniform(p.scale[0], p.scale[1]);
    const tx = ctx.rng.uniform(-p.translate, p.translate) * w;
    const ty = ctx.rng.uniform(-p.translate, p.translate) * h;
    const sh = ctx.rng.uniform(-p.shear, p.shear);
    return warpAffineBrowser(src, randomAffineMatrix(ctx.in, angle, s, tx, ty, sh), w, h, ctx, p.interp, p.border, p.color);
  },
  emitPython: (p, ctx) => {
    ctx.helper('uniform');
    const D = ctx.c('degrees', pyRaw(pyFloat(p.degrees)));
    const S = ctx.c('scale', p.scale);
    const T = ctx.c('translate', pyRaw(pyFloat(p.translate)), 'fraction of width / height');
    const H = ctx.c('shear', pyRaw(pyFloat(p.shear)), 'degrees');
    return {
      body: [
        'h, w = img.shape[:2]',
        `angle = np.deg2rad(_uniform(rng, -${D}, ${D}))`,
        `s = _uniform(rng, *${S})`,
        `tx, ty = _uniform(rng, -${T}, ${T}) * w, _uniform(rng, -${T}, ${T}) * h`,
        `sh = np.tan(np.deg2rad(_uniform(rng, -${H}, ${H})))`,
        'R = s * np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])',
        'A = R @ np.array([[1.0, sh], [0.0, 1.0]])  # rotate·scale after shear',
        'c = np.array([(w - 1) / 2, (h - 1) / 2])',
        'M = np.hstack([A, (c + [tx, ty] - A @ c)[:, None]])',
        'size = (w, h)',
        warpAffinePy(ctx, p.interp, p.border, p.color),
      ].join('\n'),
    };
  },
  emitAlbumentations: (p) =>
    `A.Affine(rotate=(${-p.degrees}, ${p.degrees}), scale=(${p.scale[0]}, ${p.scale[1]}), translate_percent=(${-p.translate}, ${p.translate}), shear=(${-p.shear}, ${p.shear}), interpolation=cv2.INTER_${p.interp}, border_mode=cv2.BORDER_${p.border}, p={p})`,
  emitTorchvision: (p) => `v2.RandomApply([v2.RandomAffine(degrees=${p.degrees}, translate=(${p.translate}, ${p.translate}), scale=(${p.scale[0]}, ${p.scale[1]}), shear=${p.shear})], p={p})`,
  testCases: [{ seed: 21 }, { seed: 22, params: { degrees: 30, translate: 0.2, shear: 15, border: 'REFLECT_101' } }],
});
