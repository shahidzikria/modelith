import { COLOR_SPACES } from '../../meta';
import { select } from '../../params';
import { defineStep } from '../../registry';
import type { ColorSpace } from '../../types';
import { colorPath, convertedMeta, cvtColorBrowser, cvtColorPy } from '../util';

defineStep({
  id: 'convert_color',
  category: 'I/O & colour',
  label: 'Convert colour',
  doc: 'Convert between colour spaces with cv2.cvtColor (BGR↔RGB, GRAY, HSV, HLS, LAB, LUV, YCrCb, XYZ, BGRA). Converting GRAY to BGR gives 3 identical channels. Pairs without a direct code go through BGR. For uint8, OpenCV stores H in 0–179 and S, V in 0–255; for float32, H is 0–360 and S, V are 0–1.',
  cv2: 'cv2.cvtColor(img, cv2.COLOR_<FROM>2<TO>)',
  keywords: 'cvtColor rgb bgr gray grey hsv lab ycrcb channels',
  params: [select('to', 'Convert to', COLOR_SPACES.map((s) => [s, s === 'BGR' ? 'BGR (OpenCV default)' : s === 'GRAY' ? 'GRAY (1 channel)' : s]), 'RGB')],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1, 3, 4], dtype: ['uint8', 'uint16', 'float32'] },
  needs: ['cvtColor'],
  // HSV/HLS → BGR (and BGR → HLS) round differently in native SIMD cv2 vs OpenCV.js: ±1 on some pixels.
  tolerance: 1,
  summary: (p) => `→ ${p.to}`,
  outMeta: (m, p) => convertedMeta(m, p.to as ColorSpace),
  check: (m, p) => {
    const issues = [];
    if (m.space === p.to) issues.push({ level: 'info' as const, message: `The image is already ${m.space}; this step does nothing.`, fixes: [{ label: 'Remove it', kind: 'remove' as const }] });
    if (m.dtype === 'uint16' && ['HSV', 'HLS', 'LAB', 'LUV'].includes(p.to))
      issues.push({ level: 'error' as const, message: `OpenCV converts to ${p.to} only from uint8 or float32 images; this one is uint16.`, fixes: [{ label: "Add 'Change dtype → uint8' before it", kind: 'insertBefore' as const, def: 'convert_dtype', params: { to: 'uint8', scale: 'shift' } }] });
    return issues;
  },
  applyBrowser: (src, p, ctx) => cvtColorBrowser(src, ctx.in.space, p.to, ctx),
  emitPython: (p, ctx) => {
    const path = colorPath(ctx.in.space, p.to);
    return { body: path.length ? cvtColorPy(ctx.in.space, p.to) : '# already in the target colour space', doc: `${ctx.in.space} → ${p.to}` };
  },
  testCases: [
    { tolerance: 0 },
    { params: { to: 'GRAY' }, tolerance: 0 },
    { params: { to: 'HSV' }, tolerance: 0 },
    { params: { to: 'HLS' } },
    { params: { to: 'LAB' }, tolerance: 0 },
    { params: { to: 'LUV' }, tolerance: 0 },
    { params: { to: 'YCrCb' }, tolerance: 0 },
    { params: { to: 'XYZ' }, tolerance: 0 },
    { params: { to: 'BGRA' }, tolerance: 0 },
    { input: 'gray', params: { to: 'BGR' }, tolerance: 0 },
    { input: 'float', params: { to: 'HSV' }, tolerance: 1e-3 },
    { input: 'float', params: { to: 'LAB' }, tolerance: 1e-3 },
    { before: [{ def: 'convert_color', params: { to: 'HSV' } }], params: { to: 'BGR' } },
  ],
});
