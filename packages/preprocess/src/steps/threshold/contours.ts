import { int, select } from '../../params';
import { defineStep } from '../../registry';

defineStep({
  id: 'contours',
  category: 'Threshold',
  label: 'Contours',
  doc: 'Find contours of a binary image (cv2.findContours) and redraw them (cv2.drawContours): filled to get clean solid masks, or as outlines. Retrieval: EXTERNAL (outermost only) or LIST (all, including holes); chain: SIMPLE (corners) or NONE (every point). Contours smaller than the minimum area are dropped.',
  cv2: 'cv2.findContours(binary, cv2.RETR_*, cv2.CHAIN_APPROX_*) → cv2.drawContours',
  keywords: 'contours findContours drawContours outline fill mask shapes objects',
  params: [
    select('mode', 'Retrieval', [['EXTERNAL', 'External only'], ['LIST', 'All (list)'], ['CCOMP', 'Two-level (CCOMP)'], ['TREE', 'Full hierarchy (TREE)']], 'EXTERNAL'),
    select('chain', 'Chain', [['SIMPLE', 'Simple (corners)'], ['NONE', 'None (all points)']], 'SIMPLE'),
    select('draw', 'Draw as', [['filled', 'Filled mask'], ['outline', 'Outlines']], 'filled'),
    int('thickness', 'Outline thickness', 2, { min: 1, max: 20, showIf: { key: 'draw', equals: ['outline'] } }),
    int('minArea', 'Min area (px)', 0, { min: 0, max: 1000000 }),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1], dtype: ['uint8'] },
  needs: ['findContours', 'drawContours', 'contourArea'],
  summary: (p) => `${String(p.mode).toLowerCase()} · ${p.draw}${p.minArea ? ` ≥ ${p.minArea}` : ''}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const contours = ctx.keep(new cv.MatVector());
    const hier = ctx.mat();
    cv.findContours(src, contours, hier, cv[`RETR_${p.mode}`], cv[`CHAIN_APPROX_${p.chain}`]);
    const keep = ctx.keep(new cv.MatVector());
    for (let i = 0; i < contours.size(); i++) {
      const c = ctx.keep(contours.get(i));
      if (cv.contourArea(c, false) >= p.minArea) keep.push_back(c);
    }
    const dst = ctx.keep(cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC1));
    cv.drawContours(dst, keep, -1, new cv.Scalar(255), p.draw === 'filled' ? -1 : p.thickness, cv.LINE_8);
    return dst;
  },
  emitPython: (p, ctx) => ({
    body: [
      `contours, _ = cv2.findContours(img, cv2.RETR_${p.mode}, cv2.CHAIN_APPROX_${p.chain})`,
      `contours = [c for c in contours if cv2.contourArea(c) >= ${ctx.c('min_area', p.minArea)}]`,
      'out = np.zeros_like(img)',
      `cv2.drawContours(out, contours, -1, 255, ${p.draw === 'filled' ? '-1)  # thickness -1 = filled' : `${ctx.c('thickness', p.thickness)})`}`,
      'img = out',
    ].join('\n'),
  }),
  testCases: [{ input: 'gray', before: [{ def: 'threshold', params: { method: 'otsu' } }] }, { input: 'doc', before: [{ def: 'threshold', params: { method: 'otsu', type: 'BINARY_INV' } }], params: { mode: 'LIST', draw: 'outline', minArea: 10 } }],
});
