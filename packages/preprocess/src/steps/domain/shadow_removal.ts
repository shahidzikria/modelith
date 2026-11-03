import { oddInt } from '../../params';
import { defineStep } from '../../registry';

defineStep({
  id: 'shadow_removal',
  category: 'Domain',
  label: 'Document shadow removal',
  doc: 'Flatten shadows and uneven lighting on documents and receipts: per channel, estimate the paper background (dilate to erase the text, then a large median blur), take 255 − |channel − background| and stretch to 0–255. Text stays dark, the page becomes evenly white.',
  cv2: 'bg = cv2.medianBlur(cv2.dilate(ch, ones(7, 7)), 21); cv2.normalize(255 - cv2.absdiff(ch, bg))',
  keywords: 'shadow removal document receipt uneven lighting background paper ocr flatten',
  params: [oddInt('dilate', 'Dilate size (> stroke width)', 7, { min: 3, max: 31 }), oddInt('median', 'Background median size', 21, { min: 3, max: 255 })],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1, 3], dtype: ['uint8'] },
  needs: ['dilate', 'medianBlur', 'absdiff', 'normalize'],
  summary: (p) => `dilate ${p.dilate} · median ${p.median}`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const parts = ctx.keep(new cv.MatVector());
    cv.split(src, parts);
    const outs = ctx.keep(new cv.MatVector());
    const k = ctx.keep(cv.Mat.ones(p.dilate, p.dilate, cv.CV_8U));
    for (let i = 0; i < parts.size(); i++) {
      const ch = ctx.keep(parts.get(i));
      const dil = ctx.mat();
      cv.dilate(ch, dil, k, new cv.Point(-1, -1), 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
      const bg = ctx.mat();
      cv.medianBlur(dil, bg, p.median);
      const diff = ctx.mat();
      cv.absdiff(ch, bg, diff);
      const inv = ctx.mat();
      cv.bitwise_not(diff, inv); // 255 - diff
      const norm = ctx.mat();
      cv.normalize(inv, norm, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
      outs.push_back(norm);
    }
    const dst = ctx.mat();
    cv.merge(outs, dst);
    return dst;
  },
  emitPython: (p, ctx) => ({
    body: [
      'out = []',
      'for ch in cv2.split(img):',
      `    bg = cv2.medianBlur(cv2.dilate(ch, np.ones((${ctx.c('dilate', p.dilate)}, ${ctx.c('dilate', p.dilate)}), np.uint8)), ${ctx.c('median', p.median)})  # paper only`,
      '    flat = 255 - cv2.absdiff(ch, bg)',
      '    out.append(cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))',
      'img = cv2.merge(out)',
    ].join('\n'),
  }),
  testCases: [{ input: 'doc' }, {}],
});
