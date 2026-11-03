import { select } from '../../params';
import { defineStep } from '../../registry';
import { COLORMAPS } from './depth_colorize';

defineStep({
  id: 'apply_colormap',
  category: 'Domain',
  label: 'Apply colormap',
  doc: 'False-colour a single-channel uint8 image (heatmaps, residuals, thermal, masks) with cv2.applyColorMap. Visualisation only: left out of the exported code unless ticked.',
  cv2: 'cv2.applyColorMap(img, cv2.COLORMAP_*)',
  keywords: 'colormap colourmap heatmap false colour jet turbo viridis visualise',
  params: [select('cmap', 'Colormap', [...COLORMAPS, ['HOT', 'Hot'], ['HSV', 'HSV']], 'TURBO')],
  engine: 'both',
  kind: 'deterministic',
  visualOnly: true,
  affectsGeometry: false,
  accepts: { c: [1], dtype: ['uint8'] },
  needs: ['applyColorMap'],
  summary: (p) => String(p.cmap).toLowerCase(),
  outMeta: (m) => ({ ...m, c: 3, space: 'BGR' }),
  applyBrowser: (src, p, ctx) => {
    const dst = ctx.mat();
    ctx.cv.applyColorMap(src, dst, ctx.cv[`COLORMAP_${p.cmap}`]);
    return dst;
  },
  emitPython: (p) => ({ body: `img = cv2.applyColorMap(img, cv2.COLORMAP_${p.cmap})` }),
  testCases: [{ input: 'gray' }],
});
