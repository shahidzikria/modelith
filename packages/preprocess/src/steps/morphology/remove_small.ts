import { int, select } from '../../params';
import { defineStep } from '../../registry';
import { copyMat } from '../util';

defineStep({
  id: 'remove_small',
  category: 'Morphology',
  label: 'Remove small components',
  doc: 'Delete connected blobs smaller than a minimum area (cv2.connectedComponentsWithStats on the non-zero pixels). Cleans speckles out of threshold / segmentation masks without eroding the large shapes.',
  cv2: 'cv2.connectedComponentsWithStats(binary, connectivity=8)',
  keywords: 'remove small objects components blobs area speckles clean mask connected',
  params: [int('minArea', 'Min area (px)', 50, { min: 1, max: 1000000 }), select('connectivity', 'Connectivity', [['8', '8 (diagonals connect)'], ['4', '4']], '8')],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: false,
  accepts: { c: [1], dtype: ['uint8'] },
  needs: ['connectedComponentsWithStats'],
  summary: (p) => `< ${p.minArea} px`,
  outMeta: (m) => m,
  applyBrowser: (src, p, ctx) => {
    const { cv } = ctx;
    const labels = ctx.mat();
    const stats = ctx.mat();
    const cents = ctx.mat();
    const bin = ctx.mat();
    cv.threshold(src, bin, 0, 255, cv.THRESH_BINARY);
    const n = cv.connectedComponentsWithStats(bin, labels, stats, cents, Number(p.connectivity), cv.CV_32S);
    const keep = new Uint8Array(n);
    const S = stats.data32S as Int32Array;
    for (let i = 1; i < n; i++) keep[i] = S[i * 5 + cv.CC_STAT_AREA] >= p.minArea ? 1 : 0;
    const dst = copyMat(src, ctx);
    const L = labels.data32S as Int32Array;
    const d = dst.data as Uint8Array;
    for (let i = 0; i < L.length; i++) if (!keep[L[i]]) d[i] = 0;
    return dst;
  },
  emitPython: (p, ctx) => ({
    body: [
      `n, labels, stats, _ = cv2.connectedComponentsWithStats((img > 0).astype(np.uint8), connectivity=${p.connectivity})`,
      `keep = stats[:, cv2.CC_STAT_AREA] >= ${ctx.c('min_area', p.minArea)}`,
      'keep[0] = False  # background',
      'img = np.where(keep[labels], img, 0).astype(img.dtype)',
    ].join('\n'),
  }),
  testCases: [{ input: 'doc', before: [{ def: 'threshold', params: { method: 'otsu', type: 'BINARY_INV' } }], params: { minArea: 20 } }],
});
