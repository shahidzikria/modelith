import { float } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';
import { grayOf, grayPy } from './deskew';

defineStep({
  id: 'doc_rectify',
  category: 'Geometric',
  label: 'Document rectification',
  doc: 'Find the page: blur → Canny → dilate → contours, take the largest contour that simplifies to 4 corners (approxPolyDP) and covers enough of the image, then warp it to a flat rectangle. If no page is found the image is left unchanged.',
  cv2: 'cv2.findContours → cv2.approxPolyDP → cv2.getPerspectiveTransform → cv2.warpPerspective',
  keywords: 'document scanner page detect four point perspective rectify flatten receipt',
  params: [
    float('minArea', 'Min page area (fraction)', 0.1, { min: 0.01, max: 0.95, step: 0.01 }),
    float('epsilon', 'Polygon tolerance', 0.02, { min: 0.005, max: 0.1, step: 0.005, help: 'approxPolyDP epsilon as a fraction of the contour perimeter.' }),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: true,
  dynamicSize: true,
  needs: ['findContours', 'approxPolyDP', 'arcLength', 'getPerspectiveTransform', 'warpPerspective'],
  accepts: { dtype: ['uint8'] },
  tolerance: 1,
  summary: (p) => `area ≥ ${Math.round(p.minArea * 100)}%`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const gray = grayOf(src, ctx);
    const blur = ctx.mat();
    cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
    const edges = ctx.mat();
    cv.Canny(blur, edges, 75, 200, 3, false);
    const k = ctx.keep(cv.Mat.ones(3, 3, cv.CV_8U));
    const dil = ctx.mat();
    cv.dilate(edges, dil, k, new cv.Point(-1, -1), 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
    const contours = ctx.keep(new cv.MatVector());
    const hier = ctx.mat();
    cv.findContours(dil, contours, hier, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
    const items = Array.from({ length: contours.size() }, (_, i) => {
      const c = ctx.keep(contours.get(i));
      return { c, area: cv.contourArea(c, false) };
    }).sort((a, b) => b.area - a.area);
    let quad: number[][] | undefined;
    for (const { c } of items.slice(0, 5)) {
      const approx = ctx.mat();
      cv.approxPolyDP(c, approx, p.epsilon * cv.arcLength(c, true), true);
      if (approx.rows === 4 && cv.contourArea(approx, false) > p.minArea * ctx.in.w * ctx.in.h) {
        const d = approx.data32S as Int32Array;
        quad = [0, 1, 2, 3].map((i) => [d[i * 2], d[i * 2 + 1]]);
        break;
      }
    }
    if (!quad) return src;
    const pick = (fn: (q: number[]) => number, max: boolean) => {
      let best = 0;
      for (let i = 1; i < 4; i++) if (max ? fn(quad![i]) > fn(quad![best]) : fn(quad![i]) < fn(quad![best])) best = i;
      return quad![best];
    };
    const tl = pick((q) => q[0] + q[1], false), br = pick((q) => q[0] + q[1], true);
    const tr = pick((q) => q[1] - q[0], false), bl = pick((q) => q[1] - q[0], true);
    const dist = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1]);
    const W = Math.floor(Math.max(dist(tl, tr), dist(bl, br)) + 0.5);
    const H = Math.floor(Math.max(dist(tl, bl), dist(tr, br)) + 0.5);
    const M = ctx.keep(
      cv.getPerspectiveTransform(ctx.keep(cv.matFromArray(4, 1, cv.CV_32FC2, [...tl, ...tr, ...br, ...bl])), ctx.keep(cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, W - 1, 0, W - 1, H - 1, 0, H - 1]))),
    );
    const dst = ctx.mat();
    cv.warpPerspective(src, dst, M, new cv.Size(W, H), cv.INTER_LINEAR, cv.BORDER_REPLICATE, new cv.Scalar());
    ctx.t.warp(Array.from(M.data64F as Float64Array), W, H);
    return dst;
  },
  emitPython: (p, ctx) => {
    const ma = ctx.c('min_area', pyRaw(pyFloat(p.minArea)), 'fraction of the image the page must cover');
    const ep = ctx.c('epsilon', pyRaw(pyFloat(p.epsilon)));
    return {
      body: [
        grayPy(ctx.in),
        'edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 75, 200)',
        'edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))',
        'contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)',
        'h, w = img.shape[:2]',
        'quad = None',
        'for c in sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)[:5]:',
        `    approx = cv2.approxPolyDP(c, ${ep} * cv2.arcLength(c, True), True)`,
        `    if len(approx) == 4 and cv2.contourArea(approx) > ${ma} * w * h:`,
        '        quad = approx.reshape(4, 2).astype(np.float64)',
        '        break',
        'if quad is None:',
        '    return img  # no page found: unchanged',
        's, d = quad.sum(1), quad[:, 1] - quad[:, 0]',
        'tl, br, tr, bl = quad[np.argmin(s)], quad[np.argmax(s)], quad[np.argmin(d)], quad[np.argmax(d)]',
        'W = int(max(np.hypot(*(tl - tr)), np.hypot(*(bl - br))) + 0.5)',
        'H = int(max(np.hypot(*(tl - bl)), np.hypot(*(tr - br))) + 0.5)',
        'dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])',
        'M = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), dst)',
        'img = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)',
        't.warp(M, (W, H))',
      ].join('\n'),
    };
  },
  testCases: [{ input: 'color' }, { input: 'doc' }],
});
