import { floats, select } from '../../params';
import { py } from '../../py';
import { defineStep } from '../../registry';

defineStep({
  id: 'retinex',
  category: 'Intensity',
  label: 'Retinex (SSR / MSR)',
  doc: 'Retinex low-light enhancement: log(I + 1) − log(Gaussian(I, σ) + 1) removes slowly varying illumination and keeps reflectance; multi-scale averages several σ (15, 80, 250). The result is min-max stretched back to uint8 per channel. Strong on dark, unevenly lit scenes; can wash out colours.',
  cv2: 'cv2.log(img + 1) - cv2.log(cv2.GaussianBlur(img, (0, 0), sigma) + 1)',
  keywords: 'retinex low light msr ssr illumination enhance night dark',
  params: [select('mode', 'Mode', [['ssr', 'Single-scale'], ['msr', 'Multi-scale']], 'msr'), floats('sigmas', 'σ values', [15, 80, 250])],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { dtype: ['uint8'] },
  needs: ['GaussianBlur', 'log', 'normalize'],
  tolerance: 1,
  summary: (p) => (p.mode === 'ssr' ? `SSR σ ${p.sigmas[0]}` : `MSR σ ${(p.sigmas as number[]).join('/')}`),
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const sig = (p.mode === 'ssr' ? [p.sigmas[0]] : p.sigmas) as number[];
    const f = ctx.mat();
    src.convertTo(f, cv.CV_32F, 1, 1); // img + 1
    const logI = ctx.mat();
    cv.log(f, logI);
    let acc: any;
    for (const s of sig) {
      const b = ctx.mat();
      cv.GaussianBlur(f, b, new cv.Size(0, 0), s, s, cv.BORDER_DEFAULT);
      const lb = ctx.mat();
      cv.log(b, lb);
      const d = ctx.mat();
      cv.subtract(logI, lb, d);
      if (!acc) acc = d;
      else {
        const sum = ctx.mat();
        cv.add(acc, d, sum);
        acc = sum;
      }
    }
    const avg = ctx.mat();
    acc.convertTo(avg, cv.CV_32F, 1 / sig.length, 0);
    const parts = ctx.keep(new cv.MatVector());
    cv.split(avg, parts);
    const outs = ctx.keep(new cv.MatVector());
    for (let i = 0; i < parts.size(); i++) {
      const o = ctx.mat();
      cv.normalize(ctx.keep(parts.get(i)), o, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
      outs.push_back(o);
    }
    const dst = ctx.mat();
    cv.merge(outs, dst);
    return dst;
  },
  emitPython: (p, ctx) => {
    const S = ctx.c('sigmas', p.mode === 'ssr' ? [p.sigmas[0]] : p.sigmas);
    return {
      body: [
        'f = img.astype(np.float32) + 1.0',
        'log_i = cv2.log(f)',
        `r = sum(log_i - cv2.log(cv2.GaussianBlur(f, (0, 0), s)) for s in ${S})`,
        `r = r * np.float32(1.0 / len(${S}))  # average over scales`,
        'chans = cv2.split(r) if r.ndim == 3 else [r]',
        'img = cv2.merge([cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) for c in chans])  # per channel',
      ].join('\n'),
    };
  },
  testCases: [{}, { params: { mode: 'ssr', sigmas: [30] } }, { input: 'gray' }],
});
void py;
