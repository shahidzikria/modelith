import { float, int } from '../../params';
import { pyFloat } from '../../py';
import { defineStep } from '../../registry';
import { pyRaw } from '../util';

defineStep({
  id: 'ela',
  category: 'Domain',
  label: 'Error Level Analysis (ELA)',
  doc: 'Forensics: re-save the image as JPEG at quality Q and amplify the absolute difference. Regions edited after the last save often re-compress differently and stand out. Q 90–95, scale 10–20. JPEG encoding is not in OpenCV.js, so it runs on the server.',
  cv2: 'cv2.absdiff(img, cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, Q])[1], …)) * scale',
  keywords: 'ela error level analysis forensics tamper manipulation jpeg recompress',
  params: [int('quality', 'JPEG quality Q', 90, { min: 1, max: 100 }), float('scale', 'Amplification', 15, { min: 1, max: 100, step: 1 })],
  engine: 'server',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1, 3], dtype: ['uint8'] },
  summary: (p) => `Q ${p.quality} · ×${p.scale}`,
  outMeta: (m) => m,
  emitPython: (p, ctx) => ({
    body: [
      `ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, ${ctx.c('quality', p.quality)}])`,
      'resaved = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)',
      `img = cv2.convertScaleAbs(cv2.absdiff(img, resaved), alpha=${ctx.c('scale', pyRaw(pyFloat(p.scale)))})`,
    ].join('\n'),
  }),
  testCases: [{}],
});
