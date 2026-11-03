// Deterministic synthetic test images (integer maths only, so they are identical everywhere).
import type { ImageBuf } from '../../packages/preprocess/src/runtime/buffers';
import { imageMeta } from '../../packages/preprocess/src/meta';

const W = 128;
const H = 96;

/** Small LCG for texture / noise. */
function lcg(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s >>> 24;
  };
}

export function colorImage(): ImageBuf {
  const d = new Uint8Array(W * H * 3);
  const rnd = lcg(7);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 3;
      let b = (x * 2) & 255, g = (y * 2 + 30) & 255, r = ((x + y) * 1) & 255;
      const dx = x - 40, dy = y - 44;
      if (dx * dx + dy * dy < 22 * 22) [b, g, r] = [40, 200, 230];
      if (x > 75 && x < 115 && y > 20 && y < 70) [b, g, r] = [220, 60, 30];
      if (x > 85 && x < 105 && y > 35 && y < 55) [b, g, r] = [250, 250, 250];
      const n = (rnd() % 21) - 10;
      d[i] = Math.min(255, Math.max(0, b + n));
      d[i + 1] = Math.min(255, Math.max(0, g + n));
      d[i + 2] = Math.min(255, Math.max(0, r + n));
    }
  return { meta: imageMeta(H, W, 3), data: d };
}

export function grayImage(): ImageBuf {
  const c = colorImage().data;
  const d = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) d[i] = (c[i * 3] * 29 + c[i * 3 + 1] * 150 + c[i * 3 + 2] * 77) >> 8;
  return { meta: imageMeta(H, W, 1), data: d };
}

export function u16Image(): ImageBuf {
  const g = grayImage().data;
  const d = new Uint16Array(W * H);
  const rnd = lcg(11);
  for (let i = 0; i < W * H; i++) d[i] = Math.min(65535, g[i] * 180 + rnd() * 4 + 3000);
  return { meta: imageMeta(H, W, 1, 'uint16'), data: d };
}

/** Light page with dark text-like strokes, a shadow gradient and a slight skew. */
export function docImage(): ImageBuf {
  const d = new Uint8Array(W * H);
  const rnd = lcg(3);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let v = 225 - Math.floor((x * 50) / W) + (rnd() % 9) - 4;
      const yy = y - Math.floor(x / 16);
      if (x > 8 && x < 120 && yy > 10 && yy < 86 && yy % 12 < 3 && (Math.floor(x / 5) % 4 !== 0)) v = 40 + (rnd() % 20);
      if (x < 4 || y < 3) v = 20;
      d[y * W + x] = Math.max(0, Math.min(255, v));
    }
  return { meta: imageMeta(H, W, 1), data: d };
}

export function floatImage(): ImageBuf {
  const c = colorImage().data;
  const d = new Float32Array(c.length);
  for (let i = 0; i < c.length; i++) d[i] = Math.fround(c[i] / 255);
  return { meta: { ...imageMeta(H, W, 3, 'float32'), range: [0, 1] }, data: d };
}

export function rgbaImage(): ImageBuf {
  const c = colorImage().data;
  const d = new Uint8Array(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    d[i * 4] = c[i * 3];
    d[i * 4 + 1] = c[i * 3 + 1];
    d[i * 4 + 2] = c[i * 3 + 2];
    d[i * 4 + 3] = (i % W) < W / 2 ? 255 : 96;
  }
  return { meta: imageMeta(H, W, 4), data: d };
}

/** Label mask: 0 background, 1 circle, 2 rectangle. */
export function maskImage(): ImageBuf {
  const d = new Uint8Array(W * H);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const dx = x - 40, dy = y - 44;
      if (dx * dx + dy * dy < 22 * 22) d[y * W + x] = 1;
      if (x > 75 && x < 115 && y > 20 && y < 70) d[y * W + x] = 2;
    }
  return { meta: imageMeta(H, W, 1), data: d };
}

export const BOXES = [
  [18, 22, 62, 66, 1],
  [76, 21, 115, 70, 2],
];
export const KEYPOINTS = [
  [40, 44],
  [95, 45],
  [5.5, 90.25],
];

export const INPUTS = { color: colorImage, gray: grayImage, u16: u16Image, doc: docImage, float: floatImage, rgba: rgbaImage };
