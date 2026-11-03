import { defineStep } from '../../registry';

defineStep({
  id: 'channel_shuffle',
  category: 'Augmentation',
  label: 'Channel shuffle',
  doc: 'Randomly permute the colour channels (Fisher-Yates with two draws). A strong colour augmentation that forces shape over colour cues; avoid it when colour is the signal (e.g. ripeness, traffic lights).',
  cv2: 'img[..., order]',
  keywords: 'channel shuffle permute colour augmentation',
  params: [],
  engine: 'both',
  kind: 'random',
  affectsGeometry: false,
  accepts: { c: [3] },
  needs: ['split', 'merge'],
  outMeta: (m) => m,
  applyBrowser: (src, _p, ctx) => {
    const { cv } = ctx;
    const order = [0, 1, 2];
    for (const i of [2, 1]) {
      const j = ctx.rng.randint(0, i);
      [order[i], order[j]] = [order[j], order[i]];
    }
    const parts = ctx.keep(new cv.MatVector());
    cv.split(src, parts);
    const chans = [0, 1, 2].map((i) => ctx.keep(parts.get(i)));
    const v = ctx.keep(new cv.MatVector());
    for (const i of order) v.push_back(chans[i]);
    const dst = ctx.mat();
    cv.merge(v, dst);
    return dst;
  },
  emitPython: (_p, ctx) => {
    ctx.helper('randint');
    return {
      body: ['order = [0, 1, 2]', 'for i in (2, 1):  # Fisher-Yates', '    j = _randint(rng, 0, i)', '    order[i], order[j] = order[j], order[i]', 'img = np.ascontiguousarray(img[..., order])'].join('\n'),
    };
  },
  emitAlbumentations: () => 'A.ChannelShuffle(p={p})',
  testCases: [{ seed: 101 }, { seed: 102 }],
});
