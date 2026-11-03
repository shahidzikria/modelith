import { str } from '../../params';
import { defineStep } from '../../registry';
import type { ColorSpace } from '../../types';

export const parseOrder = (s: string): number[] =>
  String(s)
    .split(/[\s,]+/)
    .filter(Boolean)
    .map(Number);

const SWAP: Partial<Record<ColorSpace, ColorSpace>> = { BGR: 'RGB', RGB: 'BGR', BGRA: 'RGBA', RGBA: 'BGRA' };

defineStep({
  id: 'reorder_channels',
  category: 'I/O & colour',
  label: 'Split & merge channels',
  doc: 'Split the channels and merge them back in a new order (cv2.split + cv2.merge), e.g. "2,1,0" swaps BGR ↔ RGB and "0,0,0" repeats the first channel. Reversing the first three channels also flips the colour-space label.',
  cv2: 'cv2.merge([cv2.split(img)[i] for i in ORDER])',
  keywords: 'merge split swap reorder channels permute bgr rgb',
  params: [str('order', 'New order (indices)', '2,1,0', { help: 'Comma-separated channel indices of the input, e.g. 2,1,0.' })],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1, 2, 3, 4] },
  needs: ['split', 'merge'],
  summary: (p) => `[${p.order}]`,
  check: (m, p) => {
    const o = parseOrder(p.order);
    if (!o.length || o.length > 4 || o.some((i) => !Number.isInteger(i) || i < 0 || i >= m.c))
      return [{ level: 'error' as const, message: `Order must list 1–4 channel indices between 0 and ${m.c - 1}; got "${p.order}".` }];
    return [];
  },
  outMeta: (m, p) => {
    const o = parseOrder(p.order);
    const swapped = o.length >= 3 && o[0] === 2 && o[1] === 1 && o[2] === 0 && (o.length === 3 || o[3] === 3);
    const space: ColorSpace = o.length === 1 ? 'GRAY' : swapped && SWAP[m.space] ? SWAP[m.space]! : o.length === 3 ? (m.space === 'GRAY' ? 'BGR' : m.space.replace(/A$/, '') as ColorSpace) : m.space;
    return { ...m, c: o.length, space: o.length === 4 && !space.endsWith('A') ? 'BGRA' : space };
  },
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const chans = ctx.keep(new cv.MatVector());
    cv.split(src, chans);
    const parts = chans.size();
    const got = Array.from({ length: parts }, (_, i) => ctx.keep(chans.get(i)));
    const v = ctx.keep(new cv.MatVector());
    for (const i of parseOrder(p.order)) v.push_back(got[i]);
    const dst = ctx.mat();
    cv.merge(v, dst);
    return dst;
  },
  emitPython: (p, ctx) => {
    const o = ctx.c('order', parseOrder(p.order));
    return { body: `ch = cv2.split(img)\nimg = cv2.merge([ch[i] for i in ${o}])` };
  },
  testCases: [{}, { params: { order: '1,1,0' } }],
});
