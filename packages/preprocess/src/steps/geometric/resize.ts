import { float, int, interp, select, size } from '../../params';
import { defineStep } from '../../registry';
import type { ImageMeta, Params } from '../../types';
import { interpCv, interpPy, roundHalfUp } from '../util';

/** Output (width, height) for the chosen mode. Shared by the browser, outMeta and the emitted Python. */
export function resizeTarget(m: ImageMeta, p: Params): [number, number] {
  if (p.mode === 'exact') return [p.size[0], p.size[1]];
  const s = p.mode === 'scale' ? p.scale : p.mode === 'shorter' ? p.side / Math.min(m.h, m.w) : p.side / Math.max(m.h, m.w);
  return [Math.max(1, roundHalfUp(m.w * s)), Math.max(1, roundHalfUp(m.h * s))];
}

defineStep({
  id: 'resize',
  category: 'Geometric',
  label: 'Resize',
  doc: 'Resize with cv2.resize to an exact size, by a scale factor, or so the shorter / longer side has a given length (keeps the aspect ratio). Use AREA to shrink and LINEAR or CUBIC to enlarge. Masks always use NEAREST, and boxes and keypoints are scaled to match.',
  cv2: 'cv2.resize(img, (width, height), interpolation=cv2.INTER_*)',
  keywords: 'scale size shorter side longer side downsample upsample',
  params: [
    select('mode', 'Mode', [['exact', 'Exact size'], ['scale', 'Scale factor'], ['shorter', 'Shorter side'], ['longer', 'Longer side']], 'exact'),
    size('size', 'Size (W × H)', [512, 512], { showIf: { key: 'mode', equals: ['exact'] } }),
    float('scale', 'Scale', 0.5, { min: 0.01, max: 8, step: 0.05, showIf: { key: 'mode', equals: ['scale'] } }),
    int('side', 'Side length', 256, { min: 1, max: 8192, showIf: { key: 'mode', equals: ['shorter', 'longer'] } }),
    interp('AREA'),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: true,
  accepts: { c: [1, 2, 3, 4] },
  needs: ['resize'],
  tolerance: 1,
  summary: (p) =>
    (p.mode === 'exact' ? `${p.size[0]}×${p.size[1]}` : p.mode === 'scale' ? `×${p.scale}` : `${p.mode} side ${p.side}`) + ` · ${String(p.interp).toLowerCase()}`,
  outMeta: (m, p) => {
    const [w, h] = resizeTarget(m, p);
    return { ...m, w, h };
  },
  check: (m, p) => {
    const [w, h] = resizeTarget(m, p);
    const shrink = w * h < m.w * m.h;
    if (shrink && (p.interp === 'CUBIC' || p.interp === 'LANCZOS4' || p.interp === 'LINEAR') && w * 2 <= m.w)
      return [{ level: 'info' as const, message: `Shrinking ${m.w}×${m.h} → ${w}×${h}: AREA interpolation avoids aliasing (moiré) here.`, fixes: [{ label: 'Use AREA', kind: 'setParams' as const, params: { interp: 'AREA' } }] }];
    if (!shrink && p.interp === 'AREA' && w * h > m.w * m.h)
      return [{ level: 'info' as const, message: 'Enlarging with AREA behaves like NEAREST; LINEAR or CUBIC give smoother results.', fixes: [{ label: 'Use LINEAR', kind: 'setParams' as const, params: { interp: 'LINEAR' } }] }];
    return [];
  },
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const [w, h] = resizeTarget(ctx.in, p);
    const dst = ctx.mat();
    cv.resize(src, dst, new cv.Size(w, h), 0, 0, interpCv(cv, p.interp));
    ctx.t.resize(w, h);
    return dst;
  },
  emitPython: (p, ctx) => {
    const it = ctx.c('interp', interpPy(p.interp));
    let size: string;
    let pre = '';
    if (p.mode === 'exact') {
      size = `(${ctx.c('w', p.size[0])}, ${ctx.c('h', p.size[1])})`;
    } else {
      const k = p.mode === 'scale' ? ctx.c('scale', p.scale) : ctx.c('side', p.side);
      const s = p.mode === 'scale' ? k : `${k} / ${p.mode === 'shorter' ? 'min' : 'max'}(h, w)`;
      pre = `h, w = img.shape[:2]\nscale = ${s}\nsize = (max(1, int(w * scale + 0.5)), max(1, int(h * scale + 0.5)))\n`;
      size = 'size';
    }
    return {
      body: `${pre}img = cv2.resize(img, ${size}, interpolation=${it})  # cv2.resize takes (width, height)\nt.resize(${size})`,
    };
  },
  emitAlbumentations: (p) => (p.mode === 'exact' ? `A.Resize(height=${p.size[1]}, width=${p.size[0]}, interpolation=cv2.INTER_${p.interp})` : p.mode === 'shorter' ? `A.SmallestMaxSize(max_size=${p.side}, interpolation=cv2.INTER_${p.interp})` : p.mode === 'longer' ? `A.LongestMaxSize(max_size=${p.side}, interpolation=cv2.INTER_${p.interp})` : undefined),
  emitTorchvision: (p) => {
    const it = { NEAREST: 'NEAREST', LINEAR: 'BILINEAR', AREA: 'BILINEAR', CUBIC: 'BICUBIC', LANCZOS4: 'LANCZOS' }[p.interp as string];
    if (p.mode === 'exact') return `v2.Resize((${p.size[1]}, ${p.size[0]}), interpolation=v2.InterpolationMode.${it}, antialias=True)`;
    if (p.mode === 'shorter') return `v2.Resize(${p.side}, interpolation=v2.InterpolationMode.${it}, antialias=True)`;
    return undefined;
  },
  emitTf: (p) => (p.mode === 'exact' ? `tf.image.resize(img, (${p.size[1]}, ${p.size[0]}), method=${py_tf(p.interp)}, antialias=True)` : undefined),
});

function py_tf(interp: string): string {
  return { NEAREST: '"nearest"', LINEAR: '"bilinear"', AREA: '"area"', CUBIC: '"bicubic"', LANCZOS4: '"lanczos3"' }[interp] ?? '"bilinear"';
}
