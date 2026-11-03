import { select } from '../../params';
import { defineStep } from '../../registry';

const CODE: Record<string, string> = { '90': 'ROTATE_90_CLOCKWISE', '180': 'ROTATE_180', '270': 'ROTATE_90_COUNTERCLOCKWISE' };
const K: Record<string, number> = { '90': 1, '180': 2, '270': 3 };

defineStep({
  id: 'rotate90',
  category: 'Geometric',
  label: 'Rotate 90° / 180° / 270°',
  doc: 'Lossless quarter-turn rotation with cv2.rotate (clockwise). Width and height swap for 90° and 270°. For aerial and microscopy images use "Random rotate 90°" as an augmentation.',
  cv2: 'cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)',
  keywords: 'rotate quarter turn 90 180 270 transpose',
  params: [select('angle', 'Angle (clockwise)', [['90', '90°'], ['180', '180°'], ['270', '270° (= 90° counter-clockwise)']], '90')],
  engine: 'both',
  kind: 'deterministic',
  affectsGeometry: true,
  needs: ['rotate'],
  summary: (p) => `${p.angle}°`,
  outMeta: (m, p) => (p.angle === '180' ? m : { ...m, w: m.h, h: m.w }),
  applyBrowser: (src, p, ctx) => {
    const dst = ctx.mat();
    ctx.cv.rotate(src, dst, ctx.cv[CODE[p.angle]]);
    ctx.t.rot90(K[p.angle]);
    return dst;
  },
  emitPython: (p) => ({ body: `img = cv2.rotate(img, cv2.${CODE[p.angle]})\nt.rot90(${K[p.angle]})  # quarter turns clockwise` }),
  emitTorchvision: (p) => `v2.RandomRotation((${-Number(p.angle)}, ${-Number(p.angle)}), expand=True)`,
  emitTf: (p) => `tf.image.rot90(img, k=${4 - K[p.angle]})  # tf rotates counter-clockwise`,
  testCases: [{}, { params: { angle: '180' } }, { params: { angle: '270' } }],
});
