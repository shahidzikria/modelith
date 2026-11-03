import { border, bool, color, float, interp } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import type { BrowserCtx, ImageMeta, Mat } from '../../types';
import { borderCv, borderPy, colorPy, colorScalar, interpCv, interpPy, pyRaw, roundHalfUp } from '../util';

/** cv2.getRotationMatrix2D about the image centre, optionally enlarged so nothing is cut off. */
export function rotationMatrix(m: ImageMeta, angle: number, scale: number, expand: boolean): { M: number[]; w: number; h: number } {
  const a = (angle * Math.PI) / 180;
  const alpha = Math.cos(a) * scale;
  const beta = Math.sin(a) * scale;
  const cx = (m.w - 1) / 2;
  const cy = (m.h - 1) / 2;
  const M = [alpha, beta, (1 - alpha) * cx - beta * cy, -beta, alpha, beta * cx + (1 - alpha) * cy];
  if (!expand) return { M, w: m.w, h: m.h };
  const c = Math.abs(alpha);
  const s = Math.abs(beta);
  const w = roundHalfUp(m.h * s + m.w * c);
  const h = roundHalfUp(m.h * c + m.w * s);
  M[2] += (w - 1) / 2 - cx;
  M[5] += (h - 1) / 2 - cy;
  return { M, w, h };
}

/** warpAffine in the browser with the same flags as the Python code. */
export function warpAffineBrowser(src: Mat, M: number[], w: number, h: number, ctx: BrowserCtx, interpName: string, borderName: string, colorHex: string): Mat {
  const { cv } = ctx;
  const dst = ctx.mat();
  cv.warpAffine(src, dst, ctx.keep(cv.matFromArray(2, 3, cv.CV_64F, M)), new cv.Size(w, h), interpCv(cv, interpName), borderCv(cv, borderName), colorScalar(cv, colorHex, ctx.in));
  ctx.t.warp(M, w, h);
  return dst;
}

/** Python lines for warpAffine + targets. Expects `M`, `size` defined. */
export function warpAffinePy(ctx: { c(n: string, v: unknown, comment?: string): string; in: ImageMeta }, interpName: string, borderName: string, colorHex: string): string {
  const it = ctx.c('interp', interpPy(interpName));
  const bd = ctx.c('border', borderPy(borderName));
  const col = borderName === 'CONSTANT' ? `, borderValue=${ctx.c('color', pyRaw(colorPy(colorHex, ctx.in)))}` : '';
  return `img = cv2.warpAffine(img, M, size, flags=${it}, borderMode=${bd}${col})\nt.warp(M, size)`;
}

defineStep({
  id: 'rotate',
  category: 'Geometric',
  label: 'Rotate (any angle)',
  doc: 'Rotate about the centre by any angle (counter-clockwise, degrees) with getRotationMatrix2D + warpAffine. "Expand canvas" enlarges the output so corners are not cut off. Boxes become the enclosing box of their rotated corners.',
  cv2: 'cv2.warpAffine(img, cv2.getRotationMatrix2D(center, angle, scale), size)',
  keywords: 'rotation angle warpAffine getRotationMatrix2D tilt',
  params: [
    float('angle', 'Angle (° counter-clockwise)', 15, { min: -180, max: 180, step: 0.5 }),
    float('scale', 'Scale', 1, { min: 0.1, max: 4, step: 0.05 }),
    bool('expand', 'Expand canvas', false),
    interp('LINEAR'),
    border('CONSTANT'),
    color('color', 'Fill colour', '#000000', { showIf: { key: 'border', equals: ['CONSTANT'] } }),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: true,
  needs: ['warpAffine'],
  tolerance: 1,
  summary: (p) => `${p.angle}°${p.expand ? ' · expand' : ''}`,
  outMeta: (m, p) => {
    const r = rotationMatrix(m, p.angle, p.scale, p.expand);
    return { ...m, w: r.w, h: r.h };
  },
  applyBrowser: (src, p, ctx) => {
    const r = rotationMatrix(ctx.in, p.angle, p.scale, p.expand);
    return warpAffineBrowser(src, r.M, r.w, r.h, ctx, p.interp, p.border, p.color);
  },
  emitPython: (p, ctx) => {
    const ang = ctx.c('angle', pyRaw(pyFloat(p.angle)), 'degrees, counter-clockwise');
    const sc = ctx.c('scale', pyRaw(pyFloat(p.scale)));
    const lines = ['h, w = img.shape[:2]', `M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), ${ang}, ${sc})`];
    if (p.expand)
      lines.push(
        'cos, sin = abs(M[0, 0]), abs(M[0, 1])',
        'size = (int(h * sin + w * cos + 0.5), int(h * cos + w * sin + 0.5))  # expanded (width, height)',
        'M[0, 2] += (size[0] - 1) / 2 - (w - 1) / 2',
        'M[1, 2] += (size[1] - 1) / 2 - (h - 1) / 2',
      );
    else lines.push('size = (w, h)');
    lines.push(warpAffinePy(ctx, p.interp, p.border, p.color));
    return { body: lines.join('\n') };
  },
  emitAlbumentations: (p) => (p.scale === 1 && !p.expand ? `A.Rotate(limit=(${p.angle}, ${p.angle}), interpolation=cv2.INTER_${p.interp}, border_mode=cv2.BORDER_${p.border}, p=1.0)` : undefined),
  testCases: [{}, { params: { angle: -37.5, expand: true, border: 'REFLECT' } }, { params: { angle: 90, scale: 0.7, color: '#808080' } }],
});
