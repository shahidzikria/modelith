import { float, oddInt, select } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';

defineStep({
  id: 'adaptive_threshold',
  category: 'Threshold',
  label: 'Adaptive threshold',
  doc: 'Local threshold per pixel (cv2.adaptiveThreshold): the mean or Gaussian-weighted mean of the blockSize × blockSize neighbourhood minus C. Handles shadows and uneven lighting in documents and whiteboards. blockSize must be odd and ≥ 3; needs uint8 single-channel input.',
  cv2: 'cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C)',
  keywords: 'adaptive threshold local document binarize binarise uneven lighting',
  params: [
    select('method', 'Method', [['GAUSSIAN_C', 'Gaussian-weighted mean'], ['MEAN_C', 'Mean']], 'GAUSSIAN_C'),
    select('type', 'Type', [['BINARY', 'Binary'], ['BINARY_INV', 'Binary inverted']], 'BINARY'),
    oddInt('block', 'Block size (odd ≥ 3)', 25, { min: 3, max: 255 }),
    float('C', 'C (subtracted)', 10, { min: -50, max: 50, step: 1 }),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1], dtype: ['uint8'] },
  needs: ['adaptiveThreshold'],
  summary: (p) => `${p.method === 'MEAN_C' ? 'mean' : 'gauss'} ${p.block} · C ${p.C}`,
  outMeta: (m) => m,
  check: (m, p) => [
    ...(p.block % 2 === 0 || p.block < 3 ? [{ level: 'error' as const, message: `blockSize must be odd and ≥ 3; got ${p.block}.`, fixes: [{ label: `Use ${Math.max(3, p.block + 1)}`, kind: 'setParams' as const, params: { block: Math.max(3, p.block + (p.block % 2 === 0 ? 1 : 0)) } }] }] : []),
    ...(p.block > Math.min(m.h, m.w) ? [{ level: 'warning' as const, message: `Block size ${p.block} is larger than the image.` }] : []),
  ],
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const dst = ctx.mat();
    cv.adaptiveThreshold(src, dst, 255, cv[`ADAPTIVE_THRESH_${p.method}`], cv[`THRESH_${p.type}`], p.block, p.C);
    return dst;
  },
  emitPython: (p, ctx) => ({
    body: `img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_${p.method}, cv2.THRESH_${p.type}, ${ctx.c('block', p.block, 'odd, >= 3')}, ${ctx.c('c', pyRaw(pyFloat(p.C)))})`,
  }),
  testCases: [{ input: 'doc' }, { input: 'doc', params: { method: 'MEAN_C', type: 'BINARY_INV', block: 11, C: 5 } }],
});
