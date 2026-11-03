import { bool, color, interp, size } from '../../params';
import { defineStep } from '../../registry';
import type { ImageMeta, Params } from '../../types';
import { colorPy, colorScalar, interpCv, interpPy, roundHalfUp } from '../util';

export function letterboxGeom(m: ImageMeta, p: Params) {
  const [W, H] = p.size as [number, number];
  let r = Math.min(W / m.w, H / m.h);
  if (!p.scaleUp) r = Math.min(r, 1);
  const nw = roundHalfUp(m.w * r);
  const nh = roundHalfUp(m.h * r);
  const top = p.center ? Math.floor((H - nh) / 2) : 0;
  const left = p.center ? Math.floor((W - nw) / 2) : 0;
  return { r, nw, nh, top, left, bottom: H - nh - top, right: W - nw - left };
}

defineStep({
  id: 'letterbox',
  category: 'Geometric',
  label: 'Letterbox',
  doc: 'Resize keeping the aspect ratio, then pad to the target size with a constant colour (YOLO style; 114 grey by default). The scale and padding are recorded so postprocess_boxes() maps predictions back: x_orig = (x − pad_left) / scale.',
  cv2: 'cv2.resize + cv2.copyMakeBorder(..., cv2.BORDER_CONSTANT)',
  keywords: 'yolo pad aspect ratio resize keep letterbox',
  params: [
    size('size', 'Target (W × H)', [640, 640]),
    color('color', 'Padding colour', '#727272'),
    interp('LINEAR'),
    bool('scaleUp', 'Allow enlarging', true, { help: 'Off: images smaller than the target are only padded (YOLO "scaleup=False").' }),
    bool('center', 'Centre the image', true, { help: 'Off: image at the top-left, padding right / bottom.' }),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: true,
  needs: ['resize', 'copyMakeBorder'],
  tolerance: 1,
  summary: (p) => `${p.size[0]}×${p.size[1]}`,
  outMeta: (m, p) => ({ ...m, w: p.size[0], h: p.size[1] }),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const g = letterboxGeom(ctx.in, p);
    let cur = src;
    if (g.nw !== ctx.in.w || g.nh !== ctx.in.h) {
      cur = ctx.mat();
      cv.resize(src, cur, new cv.Size(g.nw, g.nh), 0, 0, interpCv(cv, p.interp));
      ctx.t.resize(g.nw, g.nh);
    }
    const dst = ctx.mat();
    cv.copyMakeBorder(cur, dst, g.top, g.bottom, g.left, g.right, cv.BORDER_CONSTANT, colorScalar(cv, p.color, ctx.in));
    ctx.t.pad(g.top, g.bottom, g.left, g.right);
    return dst;
  },
  emitPython: (p, ctx) => {
    const W = ctx.c('w', p.size[0]);
    const H = ctx.c('h', p.size[1]);
    const col = ctx.c('color', { __raw: colorPy(p.color, ctx.in) }, 'padding value in the image channel order');
    const it = ctx.c('interp', interpPy(p.interp));
    const lines = [
      'h, w = img.shape[:2]',
      `scale = min(${W} / w, ${H} / h)${p.scaleUp ? '' : '\nscale = min(scale, 1.0)  # never enlarge'}`,
      'nw, nh = int(w * scale + 0.5), int(h * scale + 0.5)',
      'if (nw, nh) != (w, h):',
      `    img = cv2.resize(img, (nw, nh), interpolation=${it})  # (width, height)`,
      '    t.resize((nw, nh))',
      p.center ? `top, left = (${H} - nh) // 2, (${W} - nw) // 2` : 'top, left = 0, 0',
      `bottom, right = ${H} - nh - top, ${W} - nw - left`,
      `img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=${col})`,
      't.pad(top, bottom, left, right)  # postprocess_boxes: x_orig = (x - left) / scale',
    ];
    return { body: lines.join('\n') };
  },
  emitAlbumentations: (p, ctx) =>
    p.center && p.scaleUp
      ? `A.Compose([A.LongestMaxSize(max_size=${Math.max(...p.size)}, interpolation=cv2.INTER_${p.interp}), A.PadIfNeeded(min_height=${p.size[1]}, min_width=${p.size[0]}, border_mode=cv2.BORDER_CONSTANT, fill=${colorPy(p.color, ctx.in)})])`
      : undefined,
  testCases: [{ params: { size: [160, 160] } }, { params: { size: [64, 80], center: false } }, { params: { size: [256, 200], scaleUp: false } }],
});
