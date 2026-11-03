import { bool, float } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { copyMat, cvtColorBrowser, cvtColorPy, pyRaw } from '../util';

const PALETTE = [
  [75, 75, 230],
  [75, 180, 60],
  [230, 130, 70],
  [40, 190, 240],
  [220, 90, 170],
  [210, 210, 70],
];

defineStep({
  id: 'draw_overlay',
  category: 'Domain',
  label: 'Draw boxes / mask overlay',
  doc: 'Burn the current targets into the image: the mask as a translucent colour per label, boxes as rectangles and keypoints as dots (cv2.addWeighted, cv2.rectangle, cv2.circle). Useful for exporting debug images from preprocess.py --save-steps. Visualisation only: left out of the exported code unless ticked.',
  cv2: 'cv2.addWeighted(img, 1 - a, colour_mask, a, 0); cv2.rectangle; cv2.circle',
  keywords: 'draw overlay boxes mask keypoints visualise debug annotate',
  params: [bool('mask', 'Mask', true), float('alpha', 'Mask opacity', 0.4, { min: 0, max: 1, step: 0.05 }), bool('boxes', 'Boxes', true), bool('keypoints', 'Keypoints', true)],
  engine: 'both',
  kind: 'deterministic',
  visualOnly: true,
  affectsGeometry: false,
  usesTargets: true,
  accepts: { c: [1, 3], dtype: ['uint8'] },
  needs: ['addWeighted', 'rectangle', 'circle'],
  tolerance: 1,
  summary: (p) => [p.mask && 'mask', p.boxes && 'boxes', p.keypoints && 'points'].filter(Boolean).join(' · '),
  outMeta: (m) => ({ ...m, c: 3, space: m.space === 'RGB' ? 'RGB' : 'BGR' }),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const rgb = ctx.in.space === 'RGB';
    let img = src.channels() === 1 ? cvtColorBrowser(src, 'GRAY', 'BGR', ctx) : copyMat(src, ctx);
    const col = (k: number) => {
      const [b, g, r] = PALETTE[(Math.max(1, k) - 1) % PALETTE.length];
      return rgb ? new cv.Scalar(r, g, b, 255) : new cv.Scalar(b, g, r, 255);
    };
    const mk = ctx.t.mask;
    if (p.mask && mk && mk.meta.w === src.cols && mk.meta.h === src.rows) {
      const layer = copyMat(img, ctx);
      const L = layer.data as Uint8Array;
      for (let i = 0; i < mk.meta.w * mk.meta.h; i++) {
        const k = mk.data[i * mk.meta.c];
        if (!k) continue;
        const c = col(k);
        L[i * 3] = c[0];
        L[i * 3 + 1] = c[1];
        L[i * 3 + 2] = c[2];
      }
      const out = ctx.mat();
      cv.addWeighted(img, 1 - p.alpha, layer, p.alpha, 0, out, -1);
      img = out;
    }
    if (p.boxes)
      for (const b of ctx.t.boxes) cv.rectangle(img, new cv.Point(Math.floor(b[0] + 0.5), Math.floor(b[1] + 0.5)), new cv.Point(Math.floor(b[2] + 0.5), Math.floor(b[3] + 0.5)), col(Math.floor((b[4] ?? 1) + 0.5)), 2, cv.LINE_8, 0);
    if (p.keypoints) for (const k of ctx.t.keypoints) cv.circle(img, new cv.Point(Math.floor(k[0] + 0.5), Math.floor(k[1] + 0.5)), 3, rgb ? new cv.Scalar(255, 235, 59, 255) : new cv.Scalar(59, 235, 255, 255), -1, cv.LINE_8, 0);
    return img;
  },
  emitPython: (p, ctx) => {
    const rgb = ctx.in.space === 'RGB';
    const lines = [
      ctx.in.c === 1 ? cvtColorPy('GRAY', 'BGR') : 'img = img.copy()',
      `PALETTE = ${JSON.stringify(PALETTE.map((c) => (rgb ? [...c].reverse() : c))).replace(/\[/g, '(').replace(/\]/g, ')')}  # ${rgb ? 'RGB' : 'BGR'}`,
      'colour = lambda k: PALETTE[(max(1, int(k)) - 1) % len(PALETTE)]',
    ];
    if (p.mask)
      lines.push(
        'if t.mask is not None:',
        '    m = t.mask if t.mask.ndim == 2 else t.mask[..., 0]',
        '    layer = img.copy()',
        '    for k in np.unique(m[m > 0]):',
        '        layer[m == k] = colour(k)',
        `    a = ${ctx.c('alpha', pyRaw(pyFloat(p.alpha)))}`,
        '    img = cv2.addWeighted(img, 1 - a, layer, a, 0)',
      );
    if (p.boxes) lines.push('for b in t.boxes:', '    cls = b[4] if len(b) > 4 else 1', '    cv2.rectangle(img, (int(b[0] + 0.5), int(b[1] + 0.5)), (int(b[2] + 0.5), int(b[3] + 0.5)), colour(int(cls + 0.5)), 2)');
    if (p.keypoints) lines.push('for x, y in t.keypoints[:, :2]:', `    cv2.circle(img, (int(x + 0.5), int(y + 0.5)), 3, ${rgb ? '(255, 235, 59)' : '(59, 235, 255)'}, -1)`);
    return { body: lines.join('\n') };
  },
  testCases: [{ targets: true }],
});
