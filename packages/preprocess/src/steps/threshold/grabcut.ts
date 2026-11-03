import { int, rect, select } from '../../params';
import { defineStep } from '../../registry';
import { cropRect } from '../geometric/crop';

defineStep({
  id: 'grabcut',
  category: 'Threshold',
  label: 'GrabCut (foreground)',
  doc: 'Interactive foreground extraction (cv2.grabCut): draw a rectangle around the object on the viewer; colour models of inside / outside are iterated with graph cuts. Output: the image with the background set to black, or the foreground mask. The OpenCV RNG is seeded so results are repeatable.',
  cv2: 'cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_RECT)',
  keywords: 'grabcut foreground extraction background removal cutout segment object rectangle',
  params: [rect('rect', 'Rectangle (x, y, w, h)', [10, 10, 100, 70]), int('iterations', 'Iterations', 5, { min: 1, max: 20 }), select('out', 'Output', [['image', 'Image, background black'], ['mask', 'Foreground mask']], 'image')],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [3], dtype: ['uint8'] },
  needs: ['grabCut', 'setRNGSeed'],
  tolerance: 1,
  summary: (p) => `${p.rect[2]}×${p.rect[3]} · ${p.iterations} it`,
  outMeta: (m, p) => (p.out === 'mask' ? { ...m, c: 1, space: 'GRAY' } : m),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const [x, y, w, h] = cropRect(ctx.in, { mode: 'rect', rect: p.rect });
    const mask = ctx.keep(cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC1));
    const bgd = ctx.mat();
    const fgd = ctx.mat();
    cv.setRNGSeed(0);
    cv.grabCut(src, mask, new cv.Rect(x, y, w, h), bgd, fgd, p.iterations, cv.GC_INIT_WITH_RECT);
    const fg = ctx.keep(new cv.Mat(src.rows, src.cols, cv.CV_8UC1));
    const m = mask.data as Uint8Array;
    const o = fg.data as Uint8Array;
    for (let i = 0; i < m.length; i++) o[i] = m[i] === cv.GC_FGD || m[i] === cv.GC_PR_FGD ? 255 : 0;
    if (p.out === 'mask') return fg;
    const dst = ctx.mat();
    cv.bitwise_and(src, src, dst, fg);
    return dst;
  },
  emitPython: (p, ctx) => {
    const [x, y, w, h] = cropRect(ctx.in, { mode: 'rect', rect: p.rect });
    const R = ctx.c('rect', [x, y, w, h], 'x, y, width, height around the object');
    const lines = [
      'mask = np.zeros(img.shape[:2], np.uint8)',
      'cv2.setRNGSeed(0)  # GrabCut initialises its colour models randomly',
      `cv2.grabCut(img, mask, ${R}, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), ${ctx.c('iterations', p.iterations)}, cv2.GC_INIT_WITH_RECT)`,
      'fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)',
      p.out === 'mask' ? 'img = fg' : 'img = cv2.bitwise_and(img, img, mask=fg)',
    ];
    return { body: lines.join('\n') };
  },
  testCases: [{ params: { rect: [70, 15, 50, 60] } }],
});
