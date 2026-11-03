import { float, select } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';

const TYPES: [string, string][] = [
  ['BINARY', 'Binary (> t → max)'],
  ['BINARY_INV', 'Binary inverted'],
  ['TRUNC', 'Truncate (> t → t)'],
  ['TOZERO', 'To zero (≤ t → 0)'],
  ['TOZERO_INV', 'To zero inverted'],
];

defineStep({
  id: 'threshold',
  category: 'Threshold',
  label: 'Threshold',
  doc: 'Global threshold (cv2.threshold). Fixed: you choose t. Otsu: t minimises the within-class variance (bimodal histograms). Triangle: good when one peak dominates (small objects on a big background). Otsu and triangle need a uint8 single-channel image. Use binary-inverted for dark text on light paper.',
  cv2: 'cv2.threshold(img, t, maxval, cv2.THRESH_BINARY [+ cv2.THRESH_OTSU])',
  keywords: 'threshold binarize binarise otsu triangle binary inverse truncate tozero',
  params: [
    select('method', 'Threshold', [['fixed', 'Fixed value'], ['otsu', 'Otsu (automatic)'], ['triangle', 'Triangle (automatic)']], 'otsu'),
    float('thresh', 'Threshold t', 127, { min: 0, max: 65535, step: 1, showIf: { key: 'method', equals: ['fixed'] } }),
    float('maxval', 'Max value', 255, { min: 0, max: 65535, step: 1 }),
    select('type', 'Type', TYPES, 'BINARY'),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { dtype: ['uint8', 'uint16', 'float32'] },
  needs: ['threshold'],
  summary: (p) => `${p.method === 'fixed' ? `t ${p.thresh}` : p.method} · ${String(p.type).toLowerCase()}`,
  outMeta: (m) => m,
  check: (m, p) =>
    p.method !== 'fixed' && (m.c !== 1 || m.dtype !== 'uint8')
      ? [
          {
            level: 'error' as const,
            message: `${p.method === 'otsu' ? 'Otsu' : 'Triangle'} thresholding needs a uint8 single-channel image; this one is ${m.c}-channel ${m.dtype}.`,
            fixes: m.c !== 1 ? [{ label: "Add 'Convert colour → GRAY' before it", kind: 'insertBefore' as const, def: 'convert_color', params: { to: 'GRAY' } }] : [{ label: 'Use a fixed threshold', kind: 'setParams' as const, params: { method: 'fixed' } }],
          },
        ]
      : [],
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const dst = ctx.mat();
    const flag = cv[`THRESH_${p.type}`] + (p.method === 'otsu' ? cv.THRESH_OTSU : p.method === 'triangle' ? cv.THRESH_TRIANGLE : 0);
    cv.threshold(src, dst, p.method === 'fixed' ? p.thresh : 0, p.maxval, flag);
    return dst;
  },
  emitPython: (p, ctx) => {
    const mv = ctx.c('maxval', pyRaw(pyFloat(p.maxval)));
    const flag = `cv2.THRESH_${p.type}${p.method === 'otsu' ? ' + cv2.THRESH_OTSU' : p.method === 'triangle' ? ' + cv2.THRESH_TRIANGLE' : ''}`;
    const t = p.method === 'fixed' ? ctx.c('thresh', pyRaw(pyFloat(p.thresh))) : '0';
    return { body: `t, img = cv2.threshold(img, ${t}, ${mv}, ${flag})${p.method === 'fixed' ? '' : '  # t: the automatically chosen threshold'}` };
  },
  testCases: [{ input: 'gray' }, { input: 'gray', params: { method: 'triangle', type: 'BINARY_INV' } }, { params: { method: 'fixed', thresh: 100, type: 'TRUNC' } }, { input: 'float', params: { method: 'fixed', thresh: 0.5, maxval: 1, type: 'TOZERO' } }],
});
