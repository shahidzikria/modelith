import { select } from '../../params';
import { defineStep } from '../../registry';
import { logMagnitude, PY_SPECTRUM, spectrum } from './_fft';

defineStep({
  id: 'fft_view',
  category: 'Frequency',
  label: 'FFT magnitude / phase',
  doc: 'Centred 2-D Fourier spectrum (cv2.dft + fftshift): log(1 + |F|) shows how much of each spatial frequency is present (centre = low frequencies). Periodic noise (scan lines, moiré, screen patterns) appears as bright isolated spikes. Phase = angle of F in radians. Output is float32 for viewing or as a feature.',
  cv2: 'np.fft.fftshift(cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)) → cv2.magnitude / phase',
  keywords: 'fft fourier spectrum dft frequency magnitude phase periodic noise',
  params: [select('out', 'Show', [['magnitude', 'log magnitude'], ['phase', 'Phase']], 'magnitude')],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1], dtype: ['uint8', 'uint16', 'float32'] },
  needs: ['dft', 'magnitude', 'cartToPolar', 'log'],
  tolerance: 1e-2,
  summary: (p) => p.out,
  outMeta: (m) => ({ ...m, dtype: 'float32', range: [0, 20] }),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const spec = spectrum(src, ctx);
    if (p.out === 'magnitude') return logMagnitude(spec, ctx);
    const parts = ctx.keep(new cv.MatVector());
    cv.split(spec, parts);
    const mag = ctx.mat();
    const ang = ctx.mat();
    cv.cartToPolar(ctx.keep(parts.get(0)), ctx.keep(parts.get(1)), mag, ang, false);
    return ang;
  },
  emitPython: (p) => ({
    body: [...PY_SPECTRUM, p.out === 'magnitude' ? 'img = cv2.log(cv2.magnitude(spec[..., 0], spec[..., 1]) + np.float32(1))  # log(1 + |F|)' : 'img = cv2.phase(spec[..., 0], spec[..., 1])  # radians, 0..2π'].join('\n'),
  }),
  testCases: [{ input: 'gray' }, { input: 'gray', params: { out: 'phase' }, tolerance: 0.05 }],
});
