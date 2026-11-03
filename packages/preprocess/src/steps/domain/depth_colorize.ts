import { bool, float, select } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { mapPixels, matData, pyRaw } from '../util';

export const COLORMAPS: [string, string][] = [
  ['TURBO', 'Turbo'],
  ['INFERNO', 'Inferno'],
  ['VIRIDIS', 'Viridis'],
  ['MAGMA', 'Magma'],
  ['JET', 'Jet'],
  ['BONE', 'Bone'],
];

defineStep({
  id: 'depth_colorize',
  category: 'Domain',
  label: 'Depth colourisation',
  doc: 'Visualise a depth / disparity map: scale to 0–255 (min-max over valid pixels, or a fixed near–far range), optionally invert so near = warm, then cv2.applyColorMap. Invalid pixels (0) are drawn black. Visualisation only: left out of the exported code unless ticked.',
  cv2: 'cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)',
  keywords: 'depth colorize colourise disparity colormap turbo visualise stereo',
  params: [
    select('cmap', 'Colormap', COLORMAPS, 'TURBO'),
    select('scale', 'Range', [['auto', 'Min-max of valid pixels'], ['fixed', 'Fixed near … far']], 'auto'),
    float('near', 'Near', 0, { step: 1, showIf: { key: 'scale', equals: ['fixed'] } }),
    float('far', 'Far', 10000, { step: 1, showIf: { key: 'scale', equals: ['fixed'] } }),
    bool('invert', 'Near = warm (invert)', true),
  ],
  engine: 'both',
  kind: 'deterministic',
  visualOnly: true,
  affectsGeometry: false,
  accepts: { c: [1], dtype: ['uint8', 'uint16', 'float32'] },
  needs: ['applyColorMap'],
  tolerance: 1,
  summary: (p) => `${String(p.cmap).toLowerCase()}${p.scale === 'fixed' ? ` ${p.near}–${p.far}` : ''}`,
  outMeta: (m) => ({ ...m, c: 3, dtype: 'uint8', space: 'BGR', range: [0, 255] }),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const d = matData(cv, src);
    let lo = p.near, hi = p.far;
    if (p.scale === 'auto') {
      lo = Infinity;
      hi = -Infinity;
      for (const v of d) if (v > 0 && Number.isFinite(v)) (lo = Math.min(lo, v)), (hi = Math.max(hi, v));
      if (!Number.isFinite(lo)) (lo = 0), (hi = 1);
    }
    const k = 255 / Math.max(hi - lo, 1e-6);
    const u8 = mapPixels(src, ctx, 'uint8', (v) => {
      const x = Math.min(255, Math.max(0, (v - lo) * k));
      return p.invert ? 255 - x : x;
    });
    const col = ctx.mat();
    cv.applyColorMap(u8, col, cv[`COLORMAP_${p.cmap}`]);
    const o = col.data as Uint8Array;
    for (let i = 0; i < d.length; i++) if (!(d[i] > 0)) o[i * 3] = o[i * 3 + 1] = o[i * 3 + 2] = 0;
    return col;
  },
  emitPython: (p, ctx) => {
    ctx.helper('saturate');
    const lines = ['valid = np.isfinite(img) & (img > 0)'];
    if (p.scale === 'auto') lines.push('lo, hi = (img[valid].min(), img[valid].max()) if valid.any() else (0, 1)');
    else lines.push(`lo, hi = ${ctx.c('near', pyRaw(pyFloat(p.near)))}, ${ctx.c('far', pyRaw(pyFloat(p.far)))}`);
    lines.push(
      'x = np.clip((img.astype(np.float64) - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255)',
      `u8 = _saturate(${p.invert ? '255 - x' : 'x'}, np.uint8)${p.invert ? '  # near = warm' : ''}`,
      `img = cv2.applyColorMap(u8, cv2.COLORMAP_${p.cmap})`,
      'img[~valid] = 0  # invalid depth in black',
    );
    return { body: lines.join('\n') };
  },
  testCases: [{ input: 'u16' }, { input: 'gray', params: { cmap: 'INFERNO', scale: 'fixed', near: 50, far: 200, invert: false } }],
});
