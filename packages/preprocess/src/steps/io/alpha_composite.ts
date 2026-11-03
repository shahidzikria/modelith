import { color } from '../../params';
import { hexToRgb, py } from '../../py';
import { defineStep } from '../../registry';
import { f32, mapPixels } from '../util';

defineStep({
  id: 'alpha_composite',
  category: 'I/O & colour',
  label: 'Alpha-composite on background',
  doc: 'Blend a 4-channel image over a solid background colour using its alpha: out = colour × α + background × (1 − α). Transparent PNG logos and cut-outs then have a defined background instead of whatever the RGB values happen to be.',
  cv2: 'img[..., :3] * a + background * (1 - a)',
  keywords: 'alpha transparency background composite flatten png',
  params: [color('bg', 'Background', '#ffffff')],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [4], dtype: ['uint8'] },
  summary: (p) => `on ${p.bg}`,
  outMeta: (m) => ({ ...m, c: 3, space: m.space === 'RGBA' ? 'RGB' : 'BGR' }),
  applyBrowser: (src, p, ctx) => {
    const [r, g, b] = hexToRgb(p.bg);
    const bg = (ctx.in.space === 'RGBA' ? [r, g, b] : [b, g, r]).map(f32);
    const d = src.data as Uint8Array;
    return mapPixels(
      src,
      ctx,
      'uint8',
      (v, k, i) => {
        const a = f32(d[i + 3] / 255);
        return f32(f32(d[i + k] * a) + f32(bg[k] * f32(1 - a)));
      },
      3,
    );
  },
  emitPython: (p, ctx) => {
    ctx.helper('saturate');
    const [r, g, b] = hexToRgb(p.bg);
    const bg = ctx.c('bg', ctx.in.space === 'RGBA' ? [r, g, b] : [b, g, r], ctx.in.space === 'RGBA' ? 'RGB order' : 'BGR order');
    return {
      body: `a = img[..., 3:4].astype(np.float32) / 255.0\nout = img[..., :3].astype(np.float32) * a + np.array(${bg}, np.float32) * (1.0 - a)\nimg = _saturate(out, np.uint8)`,
    };
  },
  testCases: [{ input: 'rgba' }, { input: 'rgba', params: { bg: '#204080' } }],
});
void py;
