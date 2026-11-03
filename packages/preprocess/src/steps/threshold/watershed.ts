import { float, select } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';
import { grayOf, grayPy } from '../geometric/deskew';

defineStep({
  id: 'watershed',
  category: 'Threshold',
  label: 'Watershed (touching objects)',
  doc: 'Split touching objects (coins, cells, grains): Otsu threshold → opening → sure background (dilate) and sure foreground (distance transform > fraction × max) → markers from connected components → cv2.watershed. Output: a label image (uint16, 0 = background, 1…n objects) or the boundaries as a binary mask.',
  cv2: 'cv2.distanceTransform → cv2.connectedComponents → cv2.watershed(img, markers)',
  keywords: 'watershed segmentation touching objects split cells coins markers distance transform',
  params: [float('fg', 'Foreground fraction', 0.5, { min: 0.05, max: 0.95, step: 0.05, help: 'Distance-transform level (× max) that counts as surely inside an object.' }), select('out', 'Output', [['labels', 'Labels (uint16)'], ['boundaries', 'Boundaries (uint8 mask)']], 'labels')],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [3], dtype: ['uint8'] },
  needs: ['watershed', 'distanceTransform', 'connectedComponents'],
  summary: (p) => `fg ${p.fg} · ${p.out}`,
  outMeta: (m, p) => (p.out === 'labels' ? { ...m, c: 1, dtype: 'uint16', space: 'GRAY', range: [0, 255] } : { ...m, c: 1, space: 'GRAY' }),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const gray = grayOf(src, ctx);
    const bin = ctx.mat();
    cv.threshold(gray, bin, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
    const k = ctx.keep(cv.Mat.ones(3, 3, cv.CV_8U));
    const opened = ctx.mat();
    cv.morphologyEx(bin, opened, cv.MORPH_OPEN, k, new cv.Point(-1, -1), 2, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
    const bg = ctx.mat();
    cv.dilate(opened, bg, k, new cv.Point(-1, -1), 3, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
    const dist = ctx.mat();
    cv.distanceTransform(opened, dist, cv.DIST_L2, 5);
    let mx = 0;
    for (const v of dist.data32F as Float32Array) if (v > mx) mx = v;
    const fg = ctx.mat();
    cv.threshold(dist, fg, Math.fround(p.fg * mx), 255, cv.THRESH_BINARY);
    const fg8 = ctx.mat();
    fg.convertTo(fg8, cv.CV_8U, 1, 0);
    const markers = ctx.mat();
    cv.connectedComponents(fg8, markers, 8, cv.CV_32S);
    const M = markers.data32S as Int32Array;
    const B = bg.data as Uint8Array;
    const F = fg8.data as Uint8Array;
    for (let i = 0; i < M.length; i++) {
      M[i] += 1; // background = 1
      if (B[i] && !F[i]) M[i] = 0; // unknown region
    }
    cv.watershed(src, markers);
    const dst = ctx.keep(new cv.Mat(src.rows, src.cols, p.out === 'labels' ? cv.CV_16UC1 : cv.CV_8UC1));
    const o = p.out === 'labels' ? dst.data16U : dst.data;
    for (let i = 0; i < M.length; i++) o[i] = p.out === 'labels' ? (M[i] > 1 ? M[i] - 1 : 0) : M[i] === -1 ? 255 : 0;
    return dst;
  },
  emitPython: (p, ctx) => {
    const fg = ctx.c('fg_fraction', pyRaw(pyFloat(p.fg)));
    const lines = [
      grayPy(ctx.in),
      '_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)',
      'kernel = np.ones((3, 3), np.uint8)',
      'opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)',
      'sure_bg = cv2.dilate(opened, kernel, iterations=3)',
      'dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)',
      `_, sure_fg = cv2.threshold(dist, np.float32(${fg} * dist.max()), 255, cv2.THRESH_BINARY)`,
      'sure_fg = sure_fg.astype(np.uint8)',
      '_, markers = cv2.connectedComponents(sure_fg)',
      'markers = markers + 1  # background = 1',
      'markers[(sure_bg > 0) & (sure_fg == 0)] = 0  # unknown region, decided by the watershed',
      'markers = cv2.watershed(img, markers.astype(np.int32))',
    ];
    lines.push(p.out === 'labels' ? 'img = np.where(markers > 1, markers - 1, 0).astype(np.uint16)  # 0 = background, 1..n objects' : 'img = np.where(markers == -1, 255, 0).astype(np.uint8)  # boundaries');
    return { body: lines.join('\n') };
  },
  testCases: [{}, { params: { out: 'boundaries', fg: 0.3 } }],
});
