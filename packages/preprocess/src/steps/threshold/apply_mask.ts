import { bool, int, select } from '../../params';
import { defineStep } from '../../registry';

defineStep({
  id: 'apply_mask',
  category: 'Threshold',
  label: 'Apply mask',
  doc: 'Black out everything outside the loaded mask (cv2.bitwise_and with mask > 0), or keep only one label value. The mask follows all earlier geometric steps, so it stays aligned. Typical for "image × ROI" inputs in medical and industrial inspection.',
  cv2: 'cv2.bitwise_and(img, img, mask=(t.mask > 0).astype(np.uint8))',
  keywords: 'mask apply roi bitwise and region of interest label',
  params: [select('which', 'Keep', [['any', 'Any non-zero label'], ['label', 'One label value']], 'any'), int('label', 'Label', 1, { min: 0, max: 65535, showIf: { key: 'which', equals: ['label'] } }), bool('invert', 'Invert (keep outside)', false)],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  usesTargets: true,
  needs: ['bitwise_and'],
  summary: (p) => `${p.which === 'label' ? `label ${p.label}` : 'mask > 0'}${p.invert ? ' · inverted' : ''}`,
  outMeta: (m) => m,
  check: () => [{ level: 'info' as const, message: 'Uses the mask loaded with the sample (Samples ▸ Attach mask). Without one, the image passes through unchanged.' }],
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const mb = ctx.t.mask;
    if (!mb || mb.meta.w !== src.cols || mb.meta.h !== src.rows) return src;
    const sel = ctx.keep(new cv.Mat(src.rows, src.cols, cv.CV_8UC1));
    const md = mb.data;
    const o = sel.data as Uint8Array;
    const c = mb.meta.c;
    for (let i = 0; i < o.length; i++) {
      const v = md[i * c];
      const hit = p.which === 'label' ? v === p.label : v > 0;
      o[i] = hit !== p.invert ? 1 : 0;
    }
    const dst = ctx.keep(cv.Mat.zeros(src.rows, src.cols, src.type()));
    cv.bitwise_and(src, src, dst, sel);
    return dst;
  },
  emitPython: (p, ctx) => {
    const cond = p.which === 'label' ? `(m == ${ctx.c('label', p.label)})` : '(m > 0)';
    return {
      body: [
        'if t.mask is None:',
        '    return img  # no mask loaded',
        'm = t.mask if t.mask.ndim == 2 else t.mask[..., 0]',
        `keep = ${p.invert ? `~${cond}` : cond}`,
        'img = cv2.bitwise_and(img, img, mask=keep.astype(np.uint8))',
      ].join('\n'),
    };
  },
  testCases: [{ targets: true }, { targets: true, params: { which: 'label', label: 2, invert: true } }],
});
