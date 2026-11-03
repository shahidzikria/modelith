import { bool, int } from '../../params';
import { defineStep } from '../../registry';

defineStep({
  id: 'select_channel',
  category: 'I/O & colour',
  label: 'Select channel',
  doc: 'Keep one channel (cv2.split), e.g. V of HSV, L of LAB or the green channel of a fundus image. Optionally replicate it to 3 channels for models that expect colour input.',
  cv2: 'cv2.split(img)[k]',
  keywords: 'split extract channel single band',
  params: [
    int('channel', 'Channel index', 0, { min: 0, max: 3, help: 'In the image\'s own order: BGR → 0 = B, 1 = G, 2 = R; HSV → 0 = H; LAB → 0 = L.' }),
    bool('replicate', 'Replicate to 3 channels', false),
  ],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [2, 3, 4] },
  needs: ['split', 'merge'],
  summary: (p) => `channel ${p.channel}${p.replicate ? ' ×3' : ''}`,
  check: (m, p) => (p.channel >= m.c ? [{ level: 'error' as const, message: `Channel ${p.channel} does not exist; the image has ${m.c} channels (0–${m.c - 1}).` }] : []),
  outMeta: (m, p) => ({ ...m, c: p.replicate ? 3 : 1, space: p.replicate ? 'BGR' : 'GRAY' }),
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const chans = ctx.keep(new cv.MatVector());
    cv.split(src, chans);
    const ch = ctx.keep(chans.get(p.channel));
    if (!p.replicate) return ch;
    const v = ctx.keep(new cv.MatVector());
    for (let i = 0; i < 3; i++) v.push_back(ch);
    const dst = ctx.mat();
    cv.merge(v, dst);
    return dst;
  },
  emitPython: (p, ctx) => {
    const k = ctx.c('channel', p.channel);
    return { body: p.replicate ? `ch = cv2.split(img)[${k}]\nimg = cv2.merge([ch, ch, ch])` : `img = cv2.split(img)[${k}]` };
  },
  testCases: [{}, { params: { channel: 2, replicate: true } }, { input: 'rgba', params: { channel: 3 } }],
});
