import { select } from '../../params';
import { defineStep } from '../../registry';

defineStep({
  id: 'dct_view',
  category: 'Frequency',
  label: 'DCT view',
  doc: 'Discrete cosine transform (cv2.dct) shown as log(1 + |C|): over the whole image, or per 8×8 block as JPEG does — block artefacts, double compression and quantisation show up as regular patterns. cv2.dct needs even sizes, so a trailing odd row / column is dropped. Not in OpenCV.js; runs on the server.',
  cv2: 'cv2.dct(np.float32(gray))',
  keywords: 'dct discrete cosine transform jpeg blocks 8x8 compression forensics frequency',
  params: [select('mode', 'Mode', [['full', 'Whole image'], ['blocks', '8×8 blocks (JPEG)']], 'blocks')],
  engine: 'server',
  kind: 'deterministic',
  affectsGeometry: true,
  accepts: { c: [1], dtype: ['uint8', 'uint16', 'float32'] },
  summary: (p) => p.mode,
  outMeta: (m, p) => {
    const k = p.mode === 'blocks' ? 8 : 2;
    return { ...m, dtype: 'float32', range: [0, 10], h: m.h - (m.h % k), w: m.w - (m.w % k) };
  },
  emitPython: (p) => {
    const k = p.mode === 'blocks' ? 8 : 2;
    const lines = [`h, w = img.shape[0] // ${k} * ${k}, img.shape[1] // ${k} * ${k}`, 'f = img[:h, :w].astype(np.float32)', 't.crop(0, 0, w, h)'];
    if (p.mode === 'full') lines.push('img = cv2.log(np.abs(cv2.dct(f)) + np.float32(1))');
    else
      lines.push(
        'out = np.empty_like(f)',
        'for y in range(0, h, 8):',
        '    for x in range(0, w, 8):',
        '        out[y : y + 8, x : x + 8] = cv2.dct(f[y : y + 8, x : x + 8])',
        'img = cv2.log(np.abs(out) + np.float32(1))',
      );
    return { body: lines.join('\n') };
  },
  testCases: [{ input: 'gray' }, { input: 'gray', params: { mode: 'full' } }],
});
