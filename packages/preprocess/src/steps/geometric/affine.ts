import { border, color, float, interp } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import type { ImageMeta, Params } from '../../types';
import { pyRaw } from '../util';
import { warpAffineBrowser, warpAffinePy } from './rotate';

/** Scale and shear about the image centre, then translate. Same formula as the emitted NumPy code. */
export function affineMatrix(m: ImageMeta, p: Params): number[] {
  const cx = (m.w - 1) / 2;
  const cy = (m.h - 1) / 2;
  const shx = Math.tan((p.shearX * Math.PI) / 180);
  const shy = Math.tan((p.shearY * Math.PI) / 180);
  const a = [p.scale, p.scale * shx, p.scale * shy, p.scale];
  return [a[0], a[1], cx + p.tx - (a[0] * cx + a[1] * cy), a[2], a[3], cy + p.ty - (a[2] * cx + a[3] * cy)];
}

defineStep({
  id: 'affine',
  category: 'Geometric',
  label: 'Affine (scale · shear · translate)',
  doc: 'General affine warp with cv2.warpAffine: scale and shear about the centre, then shift by (tx, ty) pixels. Parallel lines stay parallel. For random affine augmentation use "Random affine".',
  cv2: 'cv2.warpAffine(img, M, (w, h))',
  keywords: 'affine shear translate shift scale warpAffine',
  params: [
    float('scale', 'Scale', 1, { min: 0.1, max: 4, step: 0.05 }),
    float('shearX', 'Shear X (°)', 10, { min: -60, max: 60, step: 0.5 }),
    float('shearY', 'Shear Y (°)', 0, { min: -60, max: 60, step: 0.5 }),
    float('tx', 'Translate X (px)', 0, { min: -2048, max: 2048, step: 1 }),
    float('ty', 'Translate Y (px)', 0, { min: -2048, max: 2048, step: 1 }),
    interp('LINEAR'),
    border('CONSTANT'),
    color('color', 'Fill colour', '#000000', { showIf: { key: 'border', equals: ['CONSTANT'] } }),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: true,
  needs: ['warpAffine'],
  tolerance: 1,
  summary: (p) => `×${p.scale} · shear ${p.shearX}°/${p.shearY}° · (${p.tx}, ${p.ty})`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => warpAffineBrowser(src, affineMatrix(ctx.in, p), ctx.in.w, ctx.in.h, ctx, p.interp, p.border, p.color),
  emitPython: (p, ctx) => {
    const sc = ctx.c('scale', pyRaw(pyFloat(p.scale)));
    const sx = ctx.c('shear_x', pyRaw(pyFloat(p.shearX)), 'degrees');
    const sy = ctx.c('shear_y', pyRaw(pyFloat(p.shearY)), 'degrees');
    const tr = ctx.c('translate', [p.tx, p.ty], 'pixels (x, y)');
    return {
      body: [
        'h, w = img.shape[:2]',
        'c = np.array([(w - 1) / 2, (h - 1) / 2])',
        `A = ${sc} * np.array([[1.0, np.tan(np.deg2rad(${sx}))], [np.tan(np.deg2rad(${sy})), 1.0]])`,
        `M = np.hstack([A, (c + np.array(${tr}) - A @ c)[:, None]])  # scale/shear about the centre, then shift`,
        'size = (w, h)',
        warpAffinePy(ctx, p.interp, p.border, p.color),
      ].join('\n'),
    };
  },
  testCases: [{}, { params: { scale: 0.8, shearX: -15, shearY: 8, tx: 12, ty: -6, border: 'REPLICATE' } }],
});
