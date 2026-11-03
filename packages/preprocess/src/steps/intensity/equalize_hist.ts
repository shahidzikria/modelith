import { defineStep } from '../../registry';

defineStep({
  id: 'equalize_hist',
  category: 'Intensity',
  label: 'Histogram equalisation',
  doc: 'Global histogram equalisation (cv2.equalizeHist) spreads intensities over the full range. Colour images are equalised on the Y channel of YCrCb, not per BGR channel (that would shift colours). Prefer CLAHE when the image has very dark and very bright regions.',
  cv2: 'cv2.equalizeHist(img)  # Y of YCrCb for colour',
  keywords: 'equalize equalise histogram contrast global',
  params: [],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1, 3], dtype: ['uint8'] },
  needs: ['equalizeHist'],
  outMeta: (m) => m,
  applyBrowser: (src, _p, ctx) => {
    const { cv } = ctx;
    const dst = ctx.mat();
    if (src.channels() === 1) {
      cv.equalizeHist(src, dst);
      return dst;
    }
    const rgb = ctx.in.space === 'RGB';
    const ycc = ctx.mat();
    cv.cvtColor(src, ycc, rgb ? cv.COLOR_RGB2YCrCb : cv.COLOR_BGR2YCrCb);
    const ch = ctx.keep(new cv.MatVector());
    cv.split(ycc, ch);
    const y = ctx.mat();
    cv.equalizeHist(ctx.keep(ch.get(0)), y);
    const v = ctx.keep(new cv.MatVector());
    v.push_back(y);
    v.push_back(ctx.keep(ch.get(1)));
    v.push_back(ctx.keep(ch.get(2)));
    const merged = ctx.mat();
    cv.merge(v, merged);
    cv.cvtColor(merged, dst, rgb ? cv.COLOR_YCrCb2RGB : cv.COLOR_YCrCb2BGR);
    return dst;
  },
  emitPython: (_p, ctx) => {
    if (ctx.in.c === 1) return { body: 'img = cv2.equalizeHist(img)' };
    const s = ctx.in.space === 'RGB' ? 'RGB' : 'BGR';
    return {
      body: `ycc = cv2.cvtColor(img, cv2.COLOR_${s}2YCrCb)\nycc[..., 0] = cv2.equalizeHist(np.ascontiguousarray(ycc[..., 0]))  # luminance only, colours keep their hue\nimg = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2${s})`,
      doc: 'Colour image: equalised on the Y channel of YCrCb.',
    };
  },
  emitAlbumentations: () => 'A.Equalize(mode="cv", by_channels=False, p=1.0)',
  testCases: [{}, { input: 'gray' }],
});
