import { int, matrix, oddInt, select } from '../../params';
import { defineStep } from '../../registry';
import type { BrowserCtx, Mat } from '../../types';

export const SHAPES: [string, string][] = [
  ['RECT', 'Rectangle'],
  ['ELLIPSE', 'Ellipse'],
  ['CROSS', 'Cross'],
];

const OPS: [string, string][] = [
  ['ERODE', 'Erode (shrink bright)'],
  ['DILATE', 'Dilate (grow bright)'],
  ['OPEN', 'Open (remove specks)'],
  ['CLOSE', 'Close (fill gaps)'],
  ['GRADIENT', 'Gradient (outline)'],
  ['TOPHAT', 'Top-hat (small bright details)'],
  ['BLACKHAT', 'Black-hat (small dark details)'],
  ['HITMISS', 'Hit-or-miss (pattern)'],
];

export function structuringElement(ctx: BrowserCtx, shape: string, k: number): Mat {
  const { cv } = ctx;
  return ctx.keep(cv.getStructuringElement(cv[`MORPH_${shape}`], new cv.Size(k, k), new cv.Point(-1, -1)));
}

defineStep({
  id: 'morphology',
  category: 'Morphology',
  label: 'Morphology',
  doc: 'Erode, dilate, open (erode → dilate: removes small bright specks), close (dilate → erode: fills small dark holes and gaps), gradient (outline), top-hat / black-hat (small bright / dark details on an uneven background) and hit-or-miss (find a binary pattern; kernel of 1 = foreground, −1 = background, 0 = don\'t care). Structuring element: rectangle, ellipse or cross.',
  cv2: 'cv2.morphologyEx(img, cv2.MORPH_*, cv2.getStructuringElement(shape, (k, k)), iterations=n)',
  keywords: 'erode dilate open close opening closing gradient tophat top-hat blackhat black-hat hit miss structuring element morphologyEx',
  params: [
    select('op', 'Operation', OPS, 'OPEN'),
    select('shape', 'Element', SHAPES, 'ELLIPSE', { showIf: { key: 'op', equals: OPS.map(([v]) => v).filter((v) => v !== 'HITMISS') } }),
    oddInt('k', 'Size', 5, { min: 1, max: 51, showIf: { key: 'op', equals: OPS.map(([v]) => v).filter((v) => v !== 'HITMISS') } }),
    int('iterations', 'Iterations', 1, { min: 1, max: 20 }),
    matrix(
      'pattern',
      'Hit-or-miss kernel',
      [
        [0, 1, 0],
        [1, -1, 1],
        [0, 1, 0],
      ],
      { showIf: { key: 'op', equals: ['HITMISS'] } },
    ),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  needs: ['morphologyEx', 'getStructuringElement'],
  summary: (p) => (p.op === 'HITMISS' ? 'hit-or-miss' : `${String(p.op).toLowerCase()} ${String(p.shape).toLowerCase()} ${p.k}×${p.k}${p.iterations > 1 ? ` ×${p.iterations}` : ''}`),
  outMeta: (m) => m,
  check: (m, p) => (p.op === 'HITMISS' && (m.c !== 1 || m.dtype !== 'uint8') ? [{ level: 'error' as const, message: 'Hit-or-miss needs a binary uint8 single-channel image (threshold first).', fixes: [{ label: "Add 'Threshold (Otsu)' before it", kind: 'insertBefore' as const, def: 'threshold', params: { method: 'otsu' } }] }] : []),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const K = p.op === 'HITMISS' ? ctx.keep(cv.matFromArray(p.pattern.length, p.pattern[0].length, cv.CV_32S, (p.pattern as number[][]).flat())) : structuringElement(ctx, p.shape, p.k);
    const dst = ctx.mat();
    const bv = cv.morphologyDefaultBorderValue();
    const anchor = new cv.Point(-1, -1);
    if (p.op === 'ERODE') cv.erode(src, dst, K, anchor, p.iterations, cv.BORDER_CONSTANT, bv);
    else if (p.op === 'DILATE') cv.dilate(src, dst, K, anchor, p.iterations, cv.BORDER_CONSTANT, bv);
    else cv.morphologyEx(src, dst, cv[`MORPH_${p.op}`], K, anchor, p.iterations, cv.BORDER_CONSTANT, bv);
    return dst;
  },
  emitPython: (p, ctx) => {
    const it = ctx.c('iterations', p.iterations);
    if (p.op === 'HITMISS') {
      const K = ctx.c('pattern', { __raw: `np.array(${JSON.stringify(p.pattern)}, np.int32)` }, '1 = foreground, -1 = background, 0 = any');
      return { body: `img = cv2.morphologyEx(img, cv2.MORPH_HITMISS, ${K}, iterations=${it})` };
    }
    const k = ctx.c('ksize', p.k);
    const kernel = `kernel = cv2.getStructuringElement(cv2.MORPH_${p.shape}, (${k}, ${k}))`;
    if (p.op === 'ERODE' || p.op === 'DILATE') return { body: `${kernel}\nimg = cv2.${p.op.toLowerCase()}(img, kernel, iterations=${it})` };
    return { body: `${kernel}\nimg = cv2.morphologyEx(img, cv2.MORPH_${p.op}, kernel, iterations=${it})` };
  },
  testCases: [{}, { params: { op: 'ERODE', shape: 'RECT', k: 3, iterations: 2 } }, { params: { op: 'GRADIENT', shape: 'CROSS' } }, { params: { op: 'TOPHAT', k: 15 }, input: 'gray' }, { params: { op: 'HITMISS' }, input: 'doc', before: [{ def: 'threshold', params: { method: 'otsu' } }] }],
});
