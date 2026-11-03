import { float, select } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { mapPixels, pyRaw } from '../util';

export const CT_PRESETS: Record<string, { level: number; width: number; label: string }> = {
  brain: { level: 40, width: 80, label: 'Brain (40 / 80)' },
  subdural: { level: 75, width: 215, label: 'Subdural (75 / 215)' },
  lung: { level: -600, width: 1500, label: 'Lung (−600 / 1500)' },
  abdomen: { level: 40, width: 400, label: 'Abdomen / soft tissue (40 / 400)' },
  bone: { level: 400, width: 1800, label: 'Bone (400 / 1800)' },
};

defineStep({
  id: 'ct_window',
  category: 'Domain',
  label: 'Windowing (level / width)',
  doc: 'Map a window of a high-bit-depth image to 0–255: values below level − width/2 become black, above level + width/2 white. For CT, raw values are converted to Hounsfield units first with the rescale slope / intercept (DICOM 0028,1053 / 0028,1052; often 1 and −1024 for 16-bit PNG exports). Presets: brain, subdural, lung, abdomen, bone.',
  cv2: 'np.clip((hu - (L - W/2)) / W, 0, 1) * 255',
  keywords: 'window level width ct hounsfield hu dicom medical x-ray 16 bit brain lung bone',
  params: [
    select('preset', 'Preset', [...Object.entries(CT_PRESETS).map(([k, v]) => [k, v.label] as [string, string]), ['custom', 'Custom']], 'abdomen'),
    float('level', 'Level L', 40, { step: 1, showIf: { key: 'preset', equals: ['custom'] } }),
    float('width', 'Width W', 400, { min: 1, step: 1, showIf: { key: 'preset', equals: ['custom'] } }),
    float('slope', 'Rescale slope', 1, { step: 0.01 }),
    float('intercept', 'Rescale intercept', -1024, { step: 1 }),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1], dtype: ['uint8', 'uint16', 'float32'] },
  tolerance: 1,
  summary: (p) => (p.preset === 'custom' ? `L ${p.level} / W ${p.width}` : CT_PRESETS[p.preset].label),
  outMeta: (m) => ({ ...m, dtype: 'uint8', range: [0, 255] }),
  applyBrowser: (src, p, ctx) => {
    const { level, width } = p.preset === 'custom' ? p : CT_PRESETS[p.preset];
    const lo = level - width / 2;
    return mapPixels(src, ctx, 'uint8', (v) => Math.min(1, Math.max(0, (v * p.slope + p.intercept - lo) / width)) * 255);
  },
  emitPython: (p, ctx) => {
    ctx.helper('saturate');
    const { level, width } = p.preset === 'custom' ? p : CT_PRESETS[p.preset];
    const L = ctx.c('level', pyRaw(pyFloat(level)));
    const W = ctx.c('width', pyRaw(pyFloat(width)));
    const s = ctx.c('slope', pyRaw(pyFloat(p.slope)));
    const b = ctx.c('intercept', pyRaw(pyFloat(p.intercept)));
    return {
      body: [`hu = img.astype(np.float64) * ${s} + ${b}  # raw → Hounsfield units`, `x = np.clip((hu - (${L} - ${W} / 2)) / ${W}, 0.0, 1.0)`, 'img = _saturate(x * 255.0, np.uint8)'].join('\n'),
    };
  },
  testCases: [{ input: 'u16', params: { preset: 'custom', level: 10000, width: 20000, intercept: 0 } }, { input: 'u16', params: { preset: 'lung', intercept: -10000 } }],
});
