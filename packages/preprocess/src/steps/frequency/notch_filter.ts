import { float, points, select } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';
import { filterAndInvert, logMagnitude, PY_INVERT, PY_SPECTRUM, spectrum } from './_fft';

/** Gaussian notch-reject mask for the points and their mirror images about the centre. */
export function notchMask(h: number, w: number, pts: number[][], r: number): Float32Array {
  const cy = Math.floor(h / 2), cx = Math.floor(w / 2);
  const m = new Float64Array(h * w).fill(1);
  for (const [px, py] of pts)
    for (const [ux, uy] of [
      [px, py],
      [2 * cx - px, 2 * cy - py],
    ])
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) m[y * w + x] *= 1 - Math.exp(-((x - ux) ** 2 + (y - uy) ** 2) / (2 * r * r));
  return Float32Array.from(m);
}

defineStep({
  id: 'notch_filter',
  category: 'Frequency',
  label: 'Notch filter (periodic noise)',
  doc: 'Remove periodic noise (scan lines, halftone, moiré): set "Show" to spectrum, click the bright off-centre spikes (✥ edit on image), then switch back to image. Each spike and its mirror get a Gaussian notch of radius r in the spectrum.',
  cv2: 'cv2.dft → × Gaussian notches → inverse DFT',
  keywords: 'notch filter periodic noise moire scan lines halftone spectrum fft spikes',
  params: [
    select('show', 'Show', [['image', 'Filtered image'], ['spectrum', 'Spectrum (to pick spikes)']], 'image'),
    points('notches', 'Spikes (spectrum pixels)', [], { tool: 'spectrum-points', help: 'Click spikes on the spectrum; mirrored points are added automatically.' }),
    float('radius', 'Notch radius', 4, { min: 1, max: 50, step: 0.5 }),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1], dtype: ['uint8', 'uint16', 'float32'] },
  needs: ['dft', 'magnitude', 'log'],
  tolerance: 1,
  summary: (p) => `${(p.notches as unknown[]).length} notch${(p.notches as unknown[]).length === 1 ? '' : 'es'} · r ${p.radius}${p.show === 'spectrum' ? ' · spectrum' : ''}`,
  outMeta: (m, p) => (p.show === 'spectrum' ? { ...m, dtype: 'float32', range: [0, 20] } : { ...m, dtype: 'uint8', range: [0, 255] }),
  check: (_m, p) => (!(p.notches as unknown[]).length ? [{ level: 'info' as const, message: 'No notches yet: set Show → spectrum and click the bright spikes.', fixes: [{ label: 'Show spectrum', kind: 'setParams' as const, params: { show: 'spectrum' } }] }] : []),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const spec = spectrum(src, ctx);
    const mask = notchMask(src.rows, src.cols, p.notches, p.radius);
    if (p.show === 'spectrum') {
      const s = spec.data32F as Float32Array;
      for (let i = 0; i < mask.length; i++) {
        s[2 * i] = Math.fround(s[2 * i] * mask[i]);
        s[2 * i + 1] = Math.fround(s[2 * i + 1] * mask[i]);
      }
      return logMagnitude(spec, ctx);
    }
    const img = filterAndInvert(spec, mask, ctx);
    const dst = ctx.mat();
    img.convertTo(dst, cv.CV_8U, 1, 0);
    return dst;
  },
  emitPython: (p, ctx) => {
    const P = ctx.c('points', (p.notches as number[][]).map((q) => q.map(Math.round)), 'spike centres in the centred spectrum (x, y)');
    const r = ctx.c('radius', pyRaw(pyFloat(p.radius)));
    const lines = [
      'h, w = img.shape[:2]',
      'yy, xx = np.mgrid[:h, :w]',
      'mask = np.ones((h, w))',
      `for x, y in ${P}:`,
      '    for ux, uy in ((x, y), (2 * (w // 2) - x, 2 * (h // 2) - y)):  # the spike and its mirror',
      `        mask *= 1 - np.exp(-((xx - ux) ** 2 + (yy - uy) ** 2) / (2 * ${r} ** 2))`,
      'mask = mask.astype(np.float32)',
      ...PY_SPECTRUM,
    ];
    if (p.show === 'spectrum') lines.push('spec = spec * mask[..., None]', 'img = cv2.log(cv2.magnitude(spec[..., 0], spec[..., 1]) + np.float32(1))  # log spectrum with the notches');
    else {
      ctx.helper('saturate');
      lines.push(...PY_INVERT, 'img = _saturate(img, np.uint8)');
    }
    return { body: lines.join('\n') };
  },
  testCases: [{ input: 'gray', params: { notches: [[80, 40], [70, 55]] } }, { input: 'gray', params: { notches: [[80, 40]], show: 'spectrum' }, tolerance: 1e-2 }],
});
