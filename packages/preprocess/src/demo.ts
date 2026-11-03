// Built-in demo sample: a small synthetic street-like scene with a label mask, boxes and keypoints,
// so the Studio has something to show before the user loads their own images.
import { imageMeta } from './meta';
import type { ImageBuf } from './runtime/buffers';

export interface DemoSample {
  image: ImageBuf;
  mask: ImageBuf;
  boxes: number[][];
  keypoints: number[][];
}

export function demoSample(w = 512, h = 384): DemoSample {
  const img = new Uint8Array(w * h * 3);
  const mask = new Uint8Array(w * h);
  let s = 12345;
  const rnd = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
  const horizon = Math.round(h * 0.58);
  const sun = { x: w * 0.78, y: h * 0.22, r: h * 0.09 };
  const car = { x1: Math.round(w * 0.16), y1: Math.round(h * 0.55), x2: Math.round(w * 0.46), y2: Math.round(h * 0.8) };
  const sign = { x: Math.round(w * 0.62), y: Math.round(h * 0.42), r: Math.round(h * 0.075) };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let r: number, g: number, b: number;
      let label = 0;
      if (y < horizon) {
        const t = y / horizon;
        r = 70 + 90 * t;
        g = 110 + 80 * t;
        b = 190 + 40 * t;
        const dx = x - sun.x, dy = y - sun.y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < sun.r) (r = 255), (g = 225), (b = 140);
        else if (d < sun.r * 2.2) {
          const k = 1 - (d - sun.r) / (sun.r * 1.2);
          r += (255 - r) * k * 0.5;
          g += (230 - g) * k * 0.5;
        }
      } else {
        const t = (y - horizon) / (h - horizon);
        r = 70 + 40 * t;
        g = 72 + 40 * t;
        b = 78 + 40 * t;
        // Lane markings.
        const lane = Math.abs(x - w / 2 - (y - horizon) * 0.1);
        if (lane < 2 + t * 6 && Math.floor((y - horizon) / 14) % 2 === 0) (r = 235), (g = 225), (b = 170);
      }
      // Buildings.
      const bx = Math.floor(x / (w / 9));
      const bh = horizon - (40 + ((bx * 53) % 70)) * (h / 384);
      if (y < horizon && y > bh && bx % 3 !== 1 && x < w * 0.58) {
        r = 120 + bx * 9;
        g = 105 + bx * 6;
        b = 100 + bx * 4;
        if ((Math.floor(x / 9) + Math.floor(y / 11)) % 3 === 0) (r += 70), (g += 70), (b += 40);
        label = 3;
      }
      // Car.
      if (x >= car.x1 && x < car.x2 && y >= car.y1 && y < car.y2) {
        const roof = y < car.y1 + (car.y2 - car.y1) * 0.4 && (x < car.x1 + (car.x2 - car.x1) * 0.2 || x > car.x2 - (car.x2 - car.x1) * 0.2);
        if (!roof) {
          r = 200;
          g = 40;
          b = 45;
          if (y < car.y1 + (car.y2 - car.y1) * 0.4) (r = 140), (g = 180), (b = 210);
          label = 1;
        }
      }
      for (const wx of [car.x1 + (car.x2 - car.x1) * 0.22, car.x2 - (car.x2 - car.x1) * 0.22]) {
        const dx = x - wx, dy = y - car.y2 + 2;
        if (dx * dx + dy * dy < ((car.y2 - car.y1) * 0.2) ** 2) (r = 25), (g = 25), (b = 28), (label = 1);
      }
      // Round sign on a pole.
      const sx = x - sign.x, sy = y - sign.y;
      if (sx * sx + sy * sy < sign.r * sign.r) {
        r = 230;
        g = 230;
        b = 235;
        if (sx * sx + sy * sy > (sign.r * 0.8) ** 2) (r = 210), (g = 30), (b = 30);
        label = 2;
      } else if (Math.abs(sx) < 3 && y > sign.y && y < horizon + h * 0.12) (r = 90), (g = 90), (b = 95), (label = 2);
      const n = (rnd() - 0.5) * 14;
      const i = (y * w + x) * 3;
      img[i] = clamp(b + n);
      img[i + 1] = clamp(g + n);
      img[i + 2] = clamp(r + n);
      mask[y * w + x] = label;
    }
  }
  return {
    image: { meta: imageMeta(h, w, 3), data: img },
    mask: { meta: imageMeta(h, w, 1), data: mask },
    boxes: [
      [car.x1, car.y1, car.x2, car.y2 + Math.round((car.y2 - car.y1) * 0.2), 1],
      [sign.x - sign.r, sign.y - sign.r, sign.x + sign.r, sign.y + sign.r, 2],
    ],
    keypoints: [
      [car.x1 + (car.x2 - car.x1) * 0.22, car.y2 + 2],
      [car.x2 - (car.x2 - car.x1) * 0.22, car.y2 + 2],
      [sign.x, sign.y],
    ],
  };
}

const clamp = (v: number) => Math.max(0, Math.min(255, Math.round(v)));
