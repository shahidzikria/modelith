import { range, select } from '../../params';
import { defineStep } from '../../registry';
import type { BrowserCtx, ImageMeta, Mat, Params } from '../../types';
import { cvtColorBrowser, cvtColorPy } from '../util';

/** Lower / upper bounds per channel from the three range params. */
const bounds = (p: Params): [number[], number[]] => [
  [p.c0[0], p.c1[0], p.c2[0]],
  [p.c0[1], p.c1[1], p.c2[1]],
];

/** HSV (uint8) of a BGR / RGB pixel, like cv2.cvtColor on a 1×1 image. */
function hsvOf(px: number[], rgb: boolean): number[] {
  const [b, g, r] = (rgb ? [px[2], px[1], px[0]] : px).map((v) => v / 255);
  const v = Math.max(r, g, b);
  const d = v - Math.min(r, g, b);
  const s = v ? d / v : 0;
  let h = 0;
  if (d) h = v === r ? (60 * (g - b)) / d : v === g ? 120 + (60 * (b - r)) / d : 240 + (60 * (r - g)) / d;
  if (h < 0) h += 360;
  return [Math.round(h / 2), Math.round(s * 255), Math.round(v * 255)];
}

defineStep({
  id: 'in_range',
  category: 'Threshold',
  label: 'Colour range mask (inRange)',
  doc: 'Keep pixels whose channels lie within [low, high] (cv2.inRange), usually in HSV so a colour can be picked independent of brightness. Click 💧 and then a pixel to set a range around it. Red wraps around the hue circle: a low hue above the high hue selects H ≥ low OR H ≤ high. Output: the mask, or the image with everything else black.',
  cv2: 'cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), low, high)',
  keywords: 'inrange colour color range mask hsv segment pick eyedropper chroma key green screen',
  params: [
    select('space', 'Compare in', [['HSV', 'HSV (H 0–179)'], ['image', 'Image channels as they are']], 'HSV'),
    range('c0', 'Channel 0 (H)', [35, 85], { min: 0, max: 255, tool: 'eyedropper' }),
    range('c1', 'Channel 1 (S)', [40, 255], { min: 0, max: 255 }),
    range('c2', 'Channel 2 (V)', [40, 255], { min: 0, max: 255 }),
    select('out', 'Output', [['mask', 'Mask (uint8 0 / 255)'], ['masked', 'Image with mask applied']], 'mask'),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [3], dtype: ['uint8'] },
  needs: ['inRange', 'cvtColor', 'bitwise_and'],
  summary: (p) => `${p.space} H ${p.c0[0]}–${p.c0[1]} · S ${p.c1[0]}–${p.c1[1]} · V ${p.c2[0]}–${p.c2[1]}`,
  outMeta: (m: ImageMeta, p) => (p.out === 'mask' ? { ...m, c: 1, space: 'GRAY' } : m),
  onPick: (px, m) => {
    const [h, s, v] = hsvOf(px, m.space === 'RGB');
    const lo = (h - 10 + 180) % 180;
    const hi = (h + 10) % 180;
    return { space: 'HSV', c0: [lo, hi], c1: [Math.max(0, s - 60), Math.min(255, s + 60)], c2: [Math.max(0, v - 60), Math.min(255, v + 60)] };
  },
  applyBrowser: (src, p, ctx: BrowserCtx) => {
    const { cv } = ctx;
    const conv: Mat = p.space === 'HSV' ? cvtColorBrowser(src, ctx.in.space, 'HSV', ctx) : src;
    const [lo, hi] = bounds(p);
    const full = (v: number[]) => ctx.keep(new cv.Mat(src.rows, src.cols, cv.CV_8UC3, new cv.Scalar(v[0], v[1], v[2], 0)));
    const mask = ctx.mat();
    if (p.space === 'HSV' && lo[0] > hi[0]) {
      const a = ctx.mat();
      const b = ctx.mat();
      cv.inRange(conv, full([lo[0], lo[1], lo[2]]), full([179, hi[1], hi[2]]), a);
      cv.inRange(conv, full([0, lo[1], lo[2]]), full([hi[0], hi[1], hi[2]]), b);
      cv.bitwise_or(a, b, mask);
    } else cv.inRange(conv, full(lo), full(hi), mask);
    if (p.out === 'mask') return mask;
    const dst = ctx.mat();
    cv.bitwise_and(src, src, dst, mask);
    return dst;
  },
  emitPython: (p, ctx) => {
    const [lo, hi] = bounds(p);
    const L = ctx.c('low', lo);
    const H = ctx.c('high', hi);
    const lines = [p.space === 'HSV' ? cvtColorPy(ctx.in.space, 'HSV', 'conv').replace('cv2.cvtColor(conv', 'cv2.cvtColor(img') + '  # H 0..179, S/V 0..255' : 'conv = img'];
    if (p.space === 'HSV' && lo[0] > hi[0])
      lines.push(`# hue wraps around (reds): H >= low OR H <= high\nmask = cv2.inRange(conv, np.array(${L}), np.array((179,) + ${H}[1:])) | cv2.inRange(conv, np.array((0,) + ${L}[1:]), np.array(${H}))`);
    else lines.push(`mask = cv2.inRange(conv, np.array(${L}), np.array(${H}))`);
    lines.push(p.out === 'mask' ? 'img = mask' : 'img = cv2.bitwise_and(img, img, mask=mask)');
    return { body: lines.join('\n') };
  },
  testCases: [{}, { params: { c0: [170, 15], c1: [50, 255], c2: [50, 255], out: 'masked' } }, { params: { space: 'image', c0: [0, 120], c1: [0, 255], c2: [100, 255] } }],
});
