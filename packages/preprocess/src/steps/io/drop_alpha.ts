import { defineStep } from '../../registry';
import { cvtColorBrowser, cvtColorPy } from '../util';

defineStep({
  id: 'drop_alpha',
  category: 'I/O & colour',
  label: 'Drop alpha',
  doc: 'Remove the alpha channel (BGRA → BGR, RGBA → RGB) and keep the colour as stored. Use "Alpha-composite on background" instead when transparent areas should become a solid colour.',
  cv2: 'cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)',
  keywords: 'alpha transparency rgba bgra png',
  params: [],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [4] },
  needs: ['cvtColor'],
  outMeta: (m) => ({ ...m, c: 3, space: m.space === 'RGBA' ? 'RGB' : 'BGR' }),
  applyBrowser: (src, _p, ctx) => cvtColorBrowser(src, ctx.in.space, ctx.out.space, ctx),
  emitPython: (_p, ctx) => ({ body: cvtColorPy(ctx.in.space, ctx.out.space) }),
  testCases: [{ input: 'rgba' }],
});
