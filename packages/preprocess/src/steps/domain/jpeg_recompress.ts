import { int } from '../../params';
import { defineStep } from '../../registry';

defineStep({
  id: 'jpeg_recompress',
  category: 'Domain',
  label: 'JPEG re-compression',
  doc: 'Encode as JPEG at quality Q and decode again (cv2.imencode / imdecode), adding real compression artefacts. Use it to match training data to compressed deployment images, or in forensics pipelines. For a random quality per image use "JPEG compression" in Augmentation. Runs on the server.',
  cv2: 'cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, Q])[1], cv2.IMREAD_UNCHANGED)',
  keywords: 'jpeg compression recompress artefacts artifacts quality imencode',
  params: [int('quality', 'Quality Q', 75, { min: 1, max: 100 })],
  engine: 'server',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1, 3], dtype: ['uint8'] },
  summary: (p) => `Q ${p.quality}`,
  outMeta: (m) => m,
  emitPython: (p, ctx) => ({
    body: `ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, ${ctx.c('quality', p.quality)}])\nimg = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)`,
  }),
  emitAlbumentations: (p) => `A.ImageCompression(quality_range=(${p.quality}, ${p.quality}), compression_type="jpeg", p=1.0)`,
  testCases: [{}],
});
