import { int, select } from '../../params';
import { defineStep } from '../../registry';
import type { ImageMeta, Params } from '../../types';

function pyrSize(m: ImageMeta, p: Params): [number, number] {
  let w = m.w, h = m.h;
  for (let i = 0; i < p.levels; i++) {
    if (p.op === 'down') (w = Math.floor((w + 1) / 2)), (h = Math.floor((h + 1) / 2));
    else (w *= 2), (h *= 2);
  }
  return [w, h];
}

defineStep({
  id: 'pyramid',
  category: 'Geometric',
  label: 'Image pyramid (pyrDown / pyrUp)',
  doc: 'Gaussian pyramid step: pyrDown blurs with a 5×5 Gaussian and halves each side ((w+1)/2), pyrUp doubles each side. Several levels give fast, alias-free 1/2, 1/4, 1/8 downscaling.',
  cv2: 'cv2.pyrDown(img) / cv2.pyrUp(img)',
  keywords: 'pyramid pyrdown pyrup gaussian half double scale',
  params: [select('op', 'Direction', [['down', 'pyrDown (½)'], ['up', 'pyrUp (×2)']], 'down'), int('levels', 'Levels', 1, { min: 1, max: 5 })],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: true,
  needs: ['pyrDown', 'pyrUp'],
  summary: (p) => `${p.op} ×${p.levels}`,
  outMeta: (m, p) => {
    const [w, h] = pyrSize(m, p);
    return { ...m, w, h };
  },
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    let cur = src;
    for (let i = 0; i < p.levels; i++) {
      const d = ctx.mat();
      if (p.op === 'down') cv.pyrDown(cur, d, new cv.Size(0, 0), cv.BORDER_DEFAULT);
      else cv.pyrUp(cur, d, new cv.Size(0, 0), cv.BORDER_DEFAULT);
      ctx.t.resize(d.cols, d.rows);
      cur = d;
    }
    return cur;
  },
  emitPython: (p, ctx) => {
    const n = ctx.c('levels', p.levels);
    const fn = p.op === 'down' ? 'pyrDown' : 'pyrUp';
    return { body: `for _ in range(${n}):\n    img = cv2.${fn}(img)\n    t.resize((img.shape[1], img.shape[0]))` };
  },
  testCases: [{}, { params: { op: 'up', levels: 1 } }, { params: { levels: 2 } }],
});
