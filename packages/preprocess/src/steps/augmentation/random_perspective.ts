import { border, color, float, interp } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { borderCv, borderPy, colorPy, colorScalar, interpCv, interpPy, pyRaw } from '../util';

defineStep({
  id: 'random_perspective',
  category: 'Augmentation',
  label: 'Random perspective',
  doc: 'Move each image corner by a random offset of up to "distortion" × min(w, h) and warp (getPerspectiveTransform + warpPerspective): simulates viewing the scene from a different angle. Offsets are drawn x then y for TL, TR, BR, BL.',
  cv2: 'cv2.warpPerspective(img, cv2.getPerspectiveTransform(corners, jittered), (w, h))',
  keywords: 'random perspective warp viewpoint keystone augmentation',
  params: [float('distortion', 'Distortion (fraction)', 0.1, { min: 0, max: 0.5, step: 0.01 }), interp('LINEAR'), border('CONSTANT'), color('color', 'Fill colour', '#000000', { showIf: { key: 'border', equals: ['CONSTANT'] } })],
  engine: 'both',
  kind: 'random',
  affectsGeometry: true,
  needs: ['warpPerspective', 'getPerspectiveTransform'],
  tolerance: 1,
  summary: (p) => `±${p.distortion}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const { w, h } = ctx.in;
    const d = p.distortion * Math.min(w, h);
    const corners = [0, 0, w, 0, w, h, 0, h];
    const jit = corners.map((v) => v + ctx.rng.uniform(-d, d));
    const M = ctx.keep(cv.getPerspectiveTransform(ctx.keep(cv.matFromArray(4, 1, cv.CV_32FC2, corners)), ctx.keep(cv.matFromArray(4, 1, cv.CV_32FC2, jit))));
    const dst = ctx.mat();
    cv.warpPerspective(src, dst, M, new cv.Size(w, h), interpCv(cv, p.interp), borderCv(cv, p.border), colorScalar(cv, p.color, ctx.in));
    ctx.t.warp(Array.from(M.data64F as Float64Array), w, h);
    return dst;
  },
  emitPython: (p, ctx) => {
    ctx.helper('uniform');
    const D = ctx.c('distortion', pyRaw(pyFloat(p.distortion)), 'fraction of min(w, h)');
    const col = p.border === 'CONSTANT' ? `, borderValue=${ctx.c('color', pyRaw(colorPy(p.color, ctx.in)))}` : '';
    return {
      body: [
        'h, w = img.shape[:2]',
        `d = ${D} * min(w, h)`,
        'corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])',
        'jittered = np.float32([[x + _uniform(rng, -d, d), y + _uniform(rng, -d, d)] for x, y in corners])',
        'M = cv2.getPerspectiveTransform(corners, jittered)',
        `img = cv2.warpPerspective(img, M, (w, h), flags=${ctx.c('interp', interpPy(p.interp))}, borderMode=${ctx.c('border', borderPy(p.border))}${col})`,
        't.warp(M, (w, h))',
      ].join('\n'),
    };
  },
  emitAlbumentations: (p) => `A.Perspective(scale=(0.0, ${p.distortion}), interpolation=cv2.INTER_${p.interp}, border_mode=cv2.BORDER_${p.border}, p={p})`,
  emitTorchvision: (p) => `v2.RandomPerspective(distortion_scale=${p.distortion * 2}, p={p})`,
  testCases: [{ seed: 31 }],
});
