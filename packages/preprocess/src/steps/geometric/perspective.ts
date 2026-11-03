import { border, color, interp, points, select, size } from '../../params';
import { py } from '../../py';
import { defineStep } from '../../registry';
import type { ImageMeta, Params } from '../../types';
import { borderCv, borderPy, colorPy, colorScalar, interpCv, interpPy, pyRaw } from '../util';

/** Source corners (TL, TR, BR, BL); missing points fall back to the image corners. */
export function srcCorners(m: ImageMeta, p: Params): number[][] {
  const def = [
    [0, 0],
    [m.w, 0],
    [m.w, m.h],
    [0, m.h],
  ];
  const pts = (p.corners as number[][]) ?? [];
  return def.map((d, i) => pts[i] ?? d);
}

export function outSize(m: ImageMeta, p: Params): [number, number] {
  if (p.out === 'size') return [p.size[0], p.size[1]];
  if (p.out === 'fit') {
    const c = srcCorners(m, p);
    const d = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1]);
    return [Math.max(1, Math.round(Math.max(d(c[0], c[1]), d(c[3], c[2])))), Math.max(1, Math.round(Math.max(d(c[0], c[3]), d(c[1], c[2]))))];
  }
  return [m.w, m.h];
}

defineStep({
  id: 'perspective',
  category: 'Geometric',
  label: 'Perspective warp',
  doc: 'Map four source corners (drag them on the viewer: top-left, top-right, bottom-right, bottom-left) onto a rectangle with getPerspectiveTransform + warpPerspective. Use it to straighten a photographed page, screen or sign.',
  cv2: 'cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), size)',
  keywords: 'perspective homography keystone warpPerspective corners four point transform',
  params: [
    points('corners', 'Corners (TL, TR, BR, BL)', [], { count: 4, help: 'Click up to 4 points on the image; drag to move. Missing points use the image corners.' }),
    select('out', 'Output size', [['same', 'Same as input'], ['fit', 'From corner distances'], ['size', 'Fixed']], 'fit'),
    size('size', 'Size (W × H)', [512, 512], { showIf: { key: 'out', equals: ['size'] } }),
    interp('LINEAR'),
    border('CONSTANT'),
    color('color', 'Fill colour', '#000000', { showIf: { key: 'border', equals: ['CONSTANT'] } }),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: true,
  needs: ['warpPerspective', 'getPerspectiveTransform'],
  tolerance: 1,
  summary: (p) => `${(p.corners as unknown[]).length}/4 corners · ${p.out}`,
  outMeta: (m, p) => {
    const [w, h] = outSize(m, p);
    return { ...m, w, h };
  },
  check: (_m, p) => ((p.corners as unknown[]).length < 4 ? [{ level: 'info' as const, message: 'Fewer than 4 corners set: missing ones use the image corners. Use “edit on image” to place them.' }] : []),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const [w, h] = outSize(ctx.in, p);
    const s = srcCorners(ctx.in, p).flat();
    const d = [0, 0, w, 0, w, h, 0, h];
    const M = ctx.keep(cv.getPerspectiveTransform(ctx.keep(cv.matFromArray(4, 1, cv.CV_32FC2, s)), ctx.keep(cv.matFromArray(4, 1, cv.CV_32FC2, d))));
    const dst = ctx.mat();
    cv.warpPerspective(src, dst, M, new cv.Size(w, h), interpCv(cv, p.interp), borderCv(cv, p.border), colorScalar(cv, p.color, ctx.in));
    ctx.t.warp(Array.from(M.data64F as Float64Array), w, h);
    return dst;
  },
  emitPython: (p, ctx) => {
    const [w, h] = outSize(ctx.in, p);
    const S = ctx.c('src', srcCorners(ctx.in, p).map((q) => q.map((v) => Math.round(v * 100) / 100)), 'TL, TR, BR, BL in input pixels');
    const Z = ctx.c('size', [w, h], '(width, height)');
    const it = ctx.c('interp', interpPy(p.interp));
    const bd = ctx.c('border', borderPy(p.border));
    const col = p.border === 'CONSTANT' ? `, borderValue=${ctx.c('color', pyRaw(colorPy(p.color, ctx.in)))}` : '';
    return {
      body: [
        `w, h = ${Z}`,
        `dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])`,
        `M = cv2.getPerspectiveTransform(np.float32(${S}), dst)`,
        `img = cv2.warpPerspective(img, M, (w, h), flags=${it}, borderMode=${bd}${col})`,
        't.warp(M, (w, h))',
      ].join('\n'),
    };
  },
  testCases: [{ params: { corners: [[10, 8], [118, 14], [124, 90], [4, 84]] } }, { params: { corners: [[20, 10], [100, 0]], out: 'size', size: [80, 60] } }],
});
void py;
