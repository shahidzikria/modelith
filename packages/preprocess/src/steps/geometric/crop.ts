import { int, rect, select, size } from '../../params';
import { defineStep } from '../../registry';
import type { ImageMeta, Params } from '../../types';
import { matData } from '../util';

/** Crop rectangle (x, y, w, h) for fixed / centre modes, clipped to the image. */
export function cropRect(m: ImageMeta, p: Params): [number, number, number, number] {
  if (p.mode === 'center') {
    const w = Math.min(p.size[0], m.w);
    const h = Math.min(p.size[1], m.h);
    return [Math.floor((m.w - w) / 2), Math.floor((m.h - h) / 2), w, h];
  }
  const [x, y, w, h] = p.rect as number[];
  const x0 = Math.max(0, Math.min(m.w - 1, Math.round(x)));
  const y0 = Math.max(0, Math.min(m.h - 1, Math.round(y)));
  return [x0, y0, Math.max(1, Math.min(m.w - x0, Math.round(w))), Math.max(1, Math.min(m.h - y0, Math.round(h)))];
}

defineStep({
  id: 'crop',
  category: 'Geometric',
  label: 'Crop',
  doc: 'Crop a fixed rectangle (draw it on the viewer), a centred W×H region, or the bounding box of the content (pixels brighter than a threshold, plus a margin) — handy for scans and medical images with black borders.',
  cv2: 'img[y:y + h, x:x + w] / cv2.boundingRect(img > thresh)',
  keywords: 'crop roi rectangle center centre content trim border',
  params: [
    select('mode', 'Mode', [['rect', 'Rectangle'], ['center', 'Centre crop'], ['content', 'Crop to content']], 'center'),
    rect('rect', 'Rectangle (x, y, w, h)', [0, 0, 256, 256], { showIf: { key: 'mode', equals: ['rect'] } }),
    size('size', 'Size (W × H)', [224, 224], { showIf: { key: 'mode', equals: ['center'] } }),
    int('thresh', 'Content threshold', 10, { min: 0, max: 254, showIf: { key: 'mode', equals: ['content'] }, help: 'Pixels whose brightest channel is above this count as content.' }),
    int('margin', 'Margin (px)', 0, { min: 0, max: 512, showIf: { key: 'mode', equals: ['content'] } }),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: true,
  dynamicSize: true, // 'content' mode: the size depends on the pixels
  summary: (p) => (p.mode === 'rect' ? `${p.rect[2]}×${p.rect[3]} @ ${p.rect[0]},${p.rect[1]}` : p.mode === 'center' ? `centre ${p.size[0]}×${p.size[1]}` : `content > ${p.thresh}`),
  outMeta: (m, p) => {
    if (p.mode === 'content') return m; // size depends on the pixels (see dynamicSize below)
    const [, , w, h] = cropRect(m, p);
    return { ...m, w, h };
  },
  check: (m, p) =>
    p.mode === 'center' && (p.size[0] > m.w || p.size[1] > m.h)
      ? [{ level: 'warning' as const, message: `Centre crop ${p.size[0]}×${p.size[1]} is larger than the ${m.w}×${m.h} image; it is clipped. Resize first (e.g. shorter side ${Math.max(...p.size)}).`, fixes: [{ label: `Insert Resize (shorter side ${Math.max(...p.size)}) before`, kind: 'insertBefore' as const, def: 'resize', params: { mode: 'shorter', side: Math.max(...p.size) } }] }]
      : [],
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    let r: number[];
    if (p.mode === 'content') {
      const { w, h, c } = ctx.in;
      const d = matData(cv, src);
      let x0 = w, y0 = h, x1 = -1, y1 = -1;
      for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
          let mx = -Infinity;
          for (let k = 0; k < c; k++) mx = Math.max(mx, d[(y * w + x) * c + k]);
          if (mx > p.thresh) {
            if (x < x0) x0 = x;
            if (x > x1) x1 = x;
            if (y < y0) y0 = y;
            if (y > y1) y1 = y;
          }
        }
      if (x1 < 0) r = [0, 0, w, h];
      else {
        const m = p.margin;
        const a = Math.max(0, x0 - m), b = Math.max(0, y0 - m);
        r = [a, b, Math.min(w, x1 + 1 + m) - a, Math.min(h, y1 + 1 + m) - b];
      }
    } else r = cropRect(ctx.in, p);
    const roi = ctx.keep(src.roi(new cv.Rect(r[0], r[1], r[2], r[3])));
    const dst = ctx.mat();
    roi.copyTo(dst); // clone() of a ROI is not continuous in OpenCV.js
    ctx.t.crop(r[0], r[1], r[2], r[3]);
    return dst;
  },
  emitPython: (p, ctx) => {
    if (p.mode === 'content') {
      const th = ctx.c('thresh', p.thresh);
      const mg = ctx.c('margin', p.margin);
      return {
        body: [
          `content = (img.max(axis=2) if img.ndim == 3 else img) > ${th}`,
          'x, y, w, h = cv2.boundingRect(content.astype(np.uint8))',
          'if w == 0:  # nothing above the threshold: keep the whole image',
          '    x, y, w, h = 0, 0, img.shape[1], img.shape[0]',
          'else:',
          `    x0, y0 = max(0, x - ${mg}), max(0, y - ${mg})`,
          `    x, y, w, h = x0, y0, min(img.shape[1], x + w + ${mg}) - x0, min(img.shape[0], y + h + ${mg}) - y0`,
          'img = img[y : y + h, x : x + w].copy()',
          't.crop(x, y, w, h)',
        ].join('\n'),
      };
    }
    if (p.mode === 'center') {
      const W = ctx.c('w', p.size[0]);
      const H = ctx.c('h', p.size[1]);
      return { body: `h, w = img.shape[:2]\ncw, ch = min(${W}, w), min(${H}, h)\nx, y = (w - cw) // 2, (h - ch) // 2\nimg = img[y : y + ch, x : x + cw].copy()\nt.crop(x, y, cw, ch)` };
    }
    const [x, y, w, h] = cropRect(ctx.in, p);
    const R = ctx.c('rect', [x, y, w, h], 'x, y, width, height (clipped to the sample image)');
    return { body: `x, y, w, h = ${R}\nimg = img[y : y + h, x : x + w].copy()\nt.crop(x, y, w, h)` };
  },
  emitAlbumentations: (p) => (p.mode === 'center' ? `A.CenterCrop(height=${p.size[1]}, width=${p.size[0]})` : p.mode === 'rect' ? `A.Crop(x_min=${p.rect[0]}, y_min=${p.rect[1]}, x_max=${p.rect[0] + p.rect[2]}, y_max=${p.rect[1] + p.rect[3]})` : undefined),
  emitTorchvision: (p) => (p.mode === 'center' ? `v2.CenterCrop((${p.size[1]}, ${p.size[0]}))` : undefined),
  testCases: [{ params: { size: [100, 64] } }, { params: { mode: 'rect', rect: [10, 5, 60, 70] } }, { params: { mode: 'content', thresh: 200, margin: 2 } }],
});
