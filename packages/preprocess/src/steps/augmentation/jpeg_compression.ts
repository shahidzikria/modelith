import { range } from '../../params';
import { defineStep } from '../../registry';

defineStep({
  id: 'jpeg_compression',
  category: 'Augmentation',
  label: 'JPEG compression',
  doc: 'Re-encode as JPEG with a quality drawn from the range: blocking and ringing artefacts like images shared over messengers and the web. Important for forensics (makes detectors robust to re-compression). JPEG encoding is not in OpenCV.js, so it runs on the server.',
  cv2: 'cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_UNCHANGED)',
  keywords: 'jpeg compression quality artefacts artifacts recompress augmentation forensics',
  params: [range('quality', 'Quality', [50, 95], { min: 1, max: 100, step: 1 })],
  engine: 'server',
  kind: 'random',
  affectsGeometry: false,
  accepts: { c: [1, 3], dtype: ['uint8'] },
  summary: (p) => `Q ${p.quality[0]}–${p.quality[1]}`,
  outMeta: (m) => m,
  emitPython: (p, ctx) => {
    ctx.helper('randint');
    return { body: `q = _randint(rng, *${ctx.c('quality', p.quality)})\nok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])\nimg = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)` };
  },
  emitAlbumentations: (p) => `A.ImageCompression(quality_range=(${p.quality[0]}, ${p.quality[1]}), compression_type="jpeg", p={p})`,
  testCases: [{ seed: 161 }],
});
