import { border, bool, color, int, select } from '../../params';
import { defineStep } from '../../registry';
import type { ImageMeta, Params } from '../../types';
import { borderCv, borderPy, colorPy, colorScalar } from '../util';

export function padAmounts(m: ImageMeta, p: Params): [number, number, number, number] {
  if (p.mode === 'fixed') return [p.top, p.bottom, p.left, p.right];
  const n = Math.max(1, p.multiple);
  const ph = (n - (m.h % n)) % n;
  const pw = (n - (m.w % n)) % n;
  if (!p.center) return [0, ph, 0, pw];
  return [Math.floor(ph / 2), ph - Math.floor(ph / 2), Math.floor(pw / 2), pw - Math.floor(pw / 2)];
}

defineStep({
  id: 'pad',
  category: 'Geometric',
  label: 'Pad (copyMakeBorder)',
  doc: 'Add a border with cv2.copyMakeBorder: a fixed number of pixels per side, or just enough to make height and width a multiple of N (U-Net needs multiples of 16/32, ViT of the patch size). Border modes: constant colour, reflect, reflect-101, replicate, wrap. Masks are padded with 0.',
  cv2: 'cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_*, value=...)',
  keywords: 'border padding multiple unet vit patch reflect replicate wrap copyMakeBorder',
  params: [
    select('mode', 'Mode', [['fixed', 'Fixed pixels'], ['multiple', 'To a multiple of N']], 'multiple'),
    int('top', 'Top', 16, { min: 0, max: 2048, showIf: { key: 'mode', equals: ['fixed'] } }),
    int('bottom', 'Bottom', 16, { min: 0, max: 2048, showIf: { key: 'mode', equals: ['fixed'] } }),
    int('left', 'Left', 16, { min: 0, max: 2048, showIf: { key: 'mode', equals: ['fixed'] } }),
    int('right', 'Right', 16, { min: 0, max: 2048, showIf: { key: 'mode', equals: ['fixed'] } }),
    int('multiple', 'Multiple of', 32, { min: 1, max: 1024, showIf: { key: 'mode', equals: ['multiple'] } }),
    bool('center', 'Split evenly (else bottom / right)', false, { showIf: { key: 'mode', equals: ['multiple'] } }),
    border('REFLECT_101'),
    color('color', 'Colour (constant)', '#000000', { showIf: { key: 'border', equals: ['CONSTANT'] } }),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: true,
  needs: ['copyMakeBorder'],
  summary: (p) => `${p.mode === 'fixed' ? `${p.top}/${p.bottom}/${p.left}/${p.right}` : `÷${p.multiple}`} · ${String(p.border).toLowerCase()}`,
  outMeta: (m, p) => {
    const [t, b, l, r] = padAmounts(m, p);
    return { ...m, h: m.h + t + b, w: m.w + l + r };
  },
  check: (m, p) => {
    const [t, b, l, r] = padAmounts(m, p);
    if (p.border.startsWith('REFLECT') && (Math.max(t, b) >= m.h || Math.max(l, r) >= m.w)) return [{ level: 'error' as const, message: 'Reflect padding cannot be larger than the image itself.' }];
    return [];
  },
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const [t, b, l, r] = padAmounts(ctx.in, p);
    const dst = ctx.mat();
    cv.copyMakeBorder(src, dst, t, b, l, r, borderCv(cv, p.border), colorScalar(cv, p.color, ctx.in));
    ctx.t.pad(t, b, l, r);
    return dst;
  },
  emitPython: (p, ctx) => {
    const bd = ctx.c('border', borderPy(p.border));
    const val = p.border === 'CONSTANT' ? `, value=${ctx.c('color', { __raw: colorPy(p.color, ctx.in) })}` : '';
    let pre: string;
    if (p.mode === 'fixed') pre = `top, bottom, left, right = ${ctx.c('top', p.top)}, ${ctx.c('bottom', p.bottom)}, ${ctx.c('left', p.left)}, ${ctx.c('right', p.right)}`;
    else {
      const n = ctx.c('multiple', p.multiple);
      pre = `h, w = img.shape[:2]\nph, pw = -h % ${n}, -w % ${n}  # pixels missing to the next multiple\n${p.center ? 'top, left = ph // 2, pw // 2\nbottom, right = ph - top, pw - left' : 'top, bottom, left, right = 0, ph, 0, pw'}`;
    }
    return { body: `${pre}\nimg = cv2.copyMakeBorder(img, top, bottom, left, right, ${bd}${val})\nt.pad(top, bottom, left, right)` };
  },
  emitAlbumentations: (p) => (p.mode === 'multiple' ? `A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=${p.multiple}, pad_width_divisor=${p.multiple}, position=${p.center ? '"center"' : '"top_left"'}, border_mode=cv2.BORDER_${p.border})` : undefined),
  testCases: [{}, { params: { mode: 'fixed', top: 5, bottom: 7, left: 3, right: 11, border: 'CONSTANT', color: '#ff8000' } }, { params: { multiple: 50, center: true, border: 'WRAP' } }, { input: 'gray', params: { mode: 'fixed', border: 'REPLICATE' } }],
});
