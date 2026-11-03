// Masks, boxes and keypoints that follow the image through geometric steps. Mirrors the `Targets` class
// emitted into preprocess.py (codegen/helpers.ts), so overlays in the viewer match the exported code.
import type { CV } from '../types';
import { bufFromMat, matFromBuf, MatScope, type ImageBuf } from './buffers';

/** 3×3 row-major homography in continuous pixel coordinates (pixel edges at integers). */
export type H3 = number[];

export interface GeomRecord {
  kind: 'resize' | 'warp' | 'flip' | 'crop' | 'pad' | 'rot90' | 'remap';
  /** Maps source → destination coordinates (continuous). Absent for 'remap'. */
  H?: H3;
  src: [number, number];
  dst: [number, number];
  /** kind-specific: flip code, rot90 k, crop [x, y, w, h], pad [top, bottom, left, right]. */
  arg?: number | number[];
}

export interface TargetsData {
  mask?: ImageBuf;
  /** [x1, y1, x2, y2, ...extra] in pixels. */
  boxes: number[][];
  /** [x, y, ...extra] in pixels. */
  keypoints: number[][];
  history: GeomRecord[];
  /** Current image [width, height]. */
  size: [number, number];
}

/** Boxes keeping less than this fraction of their (transformed) area after clipping are dropped. */
export const MIN_BOX_VISIBILITY = 0.1;

export function matmul3(a: number[], b: number[]): number[] {
  const r = new Array(9).fill(0);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) r[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
  return r;
}

/** OpenCV pixel-index matrix (2×3 or 3×3, row-major) → continuous-coordinate homography. */
export function fromPixelMatrix(m: number[]): H3 {
  const M = m.length === 6 ? [...m, 0, 0, 1] : m;
  return matmul3(matmul3([1, 0, 0.5, 0, 1, 0.5, 0, 0, 1], M), [1, 0, -0.5, 0, 1, -0.5, 0, 0, 1]);
}

export function applyH(H: H3, x: number, y: number): [number, number] {
  const d = H[6] * x + H[7] * y + H[8];
  return [(H[0] * x + H[1] * y + H[2]) / d, (H[3] * x + H[4] * y + H[5]) / d];
}

export class Targets {
  mask?: ImageBuf;
  boxes: number[][];
  keypoints: number[][];
  history: GeomRecord[];
  size: [number, number];
  private cv: CV;

  constructor(cv: CV, data: TargetsData) {
    this.cv = cv;
    this.mask = data.mask;
    this.boxes = data.boxes.map((b) => [...b]);
    this.keypoints = data.keypoints.map((k) => [...k]);
    this.history = [...data.history];
    this.size = [...data.size];
  }

  static empty(cv: CV, w: number, h: number): Targets {
    return new Targets(cv, { boxes: [], keypoints: [], history: [], size: [w, h] });
  }

  data(): TargetsData {
    return { mask: this.mask, boxes: this.boxes, keypoints: this.keypoints, history: this.history, size: this.size };
  }

  private withMask(fn: (cv: CV, src: any, dst: any, scope: MatScope) => void): void {
    if (!this.mask) return;
    const cv = this.cv;
    const scope = new MatScope();
    try {
      const src = scope.keep(matFromBuf(cv, this.mask));
      const dst = scope.keep(new cv.Mat());
      fn(cv, src, dst, scope);
      this.mask = bufFromMat(cv, dst, this.mask.meta);
    } finally {
      scope.dispose();
    }
  }

  private applyPoints(H: H3, dst: [number, number]): void {
    const [W, Ht] = dst;
    const boxes: number[][] = [];
    for (const b of this.boxes) {
      const pts = [applyH(H, b[0], b[1]), applyH(H, b[2], b[1]), applyH(H, b[2], b[3]), applyH(H, b[0], b[3])];
      const xs = pts.map((p) => p[0]);
      const ys = pts.map((p) => p[1]);
      const x1 = Math.min(...xs), y1 = Math.min(...ys), x2 = Math.max(...xs), y2 = Math.max(...ys);
      const area = (x2 - x1) * (y2 - y1);
      const cx1 = Math.min(Math.max(x1, 0), W), cy1 = Math.min(Math.max(y1, 0), Ht);
      const cx2 = Math.min(Math.max(x2, 0), W), cy2 = Math.min(Math.max(y2, 0), Ht);
      const clipped = (cx2 - cx1) * (cy2 - cy1);
      if (area > 0 && clipped > 0 && clipped >= MIN_BOX_VISIBILITY * area) boxes.push([cx1, cy1, cx2, cy2, ...b.slice(4)]);
    }
    this.boxes = boxes;
    this.keypoints = this.keypoints.map((k) => [...applyH(H, k[0], k[1]), ...k.slice(2)]);
  }

  private record(r: GeomRecord): void {
    this.history.push(r);
    this.size = r.dst;
  }

  resize(dw: number, dh: number): void {
    const [sw, sh] = this.size;
    this.withMask((cv, s, d) => cv.resize(s, d, new cv.Size(dw, dh), 0, 0, cv.INTER_NEAREST));
    const H = [dw / sw, 0, 0, 0, dh / sh, 0, 0, 0, 1];
    this.applyPoints(H, [dw, dh]);
    this.record({ kind: 'resize', H, src: [sw, sh], dst: [dw, dh] });
  }

  /** `m`: the 2×3 / 3×3 pixel-index matrix passed to warpAffine / warpPerspective. */
  warp(m: number[], dw: number, dh: number): void {
    const src = this.size;
    const M3 = m.length === 6 ? [...m, 0, 0, 1] : m;
    this.withMask((cv, s, d, scope) => {
      const zero = new cv.Scalar(0, 0, 0, 0);
      if (m.length === 6) cv.warpAffine(s, d, scope.keep(cv.matFromArray(2, 3, cv.CV_64F, m)), new cv.Size(dw, dh), cv.INTER_NEAREST, cv.BORDER_CONSTANT, zero);
      else cv.warpPerspective(s, d, scope.keep(cv.matFromArray(3, 3, cv.CV_64F, M3)), new cv.Size(dw, dh), cv.INTER_NEAREST, cv.BORDER_CONSTANT, zero);
    });
    const H = fromPixelMatrix(M3);
    this.applyPoints(H, [dw, dh]);
    this.record({ kind: 'warp', H, src, dst: [dw, dh], arg: M3 });
  }

  /** cv2.flip code: 1 horizontal, 0 vertical, -1 both. */
  flip(code: number): void {
    const [w, h] = this.size;
    this.withMask((cv, s, d) => cv.flip(s, d, code));
    const fx = code !== 0, fy = code !== 1;
    const H = [fx ? -1 : 1, 0, fx ? w : 0, 0, fy ? -1 : 1, fy ? h : 0, 0, 0, 1];
    this.applyPoints(H, [w, h]);
    this.record({ kind: 'flip', H, src: [w, h], dst: [w, h], arg: code });
  }

  crop(x: number, y: number, cw: number, ch: number): void {
    const src = this.size;
    this.withMask((cv, s, d, scope) => scope.keep(s.roi(new cv.Rect(x, y, cw, ch))).copyTo(d));
    const H = [1, 0, -x, 0, 1, -y, 0, 0, 1];
    this.applyPoints(H, [cw, ch]);
    this.record({ kind: 'crop', H, src, dst: [cw, ch], arg: [x, y, cw, ch] });
  }

  pad(top: number, bottom: number, left: number, right: number): void {
    const [w, h] = this.size;
    this.withMask((cv, s, d) => cv.copyMakeBorder(s, d, top, bottom, left, right, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0)));
    const H = [1, 0, left, 0, 1, top, 0, 0, 1];
    const dst: [number, number] = [w + left + right, h + top + bottom];
    this.applyPoints(H, dst);
    this.record({ kind: 'pad', H, src: [w, h], dst, arg: [top, bottom, left, right] });
  }

  /** k quarter turns clockwise (cv2.ROTATE_90_CLOCKWISE for k = 1). */
  rot90(k: number): void {
    k = ((k % 4) + 4) % 4;
    if (!k) return;
    const [w, h] = this.size;
    const code = k === 1 ? 'ROTATE_90_CLOCKWISE' : k === 2 ? 'ROTATE_180' : 'ROTATE_90_COUNTERCLOCKWISE';
    this.withMask((cv, s, d) => cv.rotate(s, d, cv[code]));
    const H = k === 1 ? [0, -1, h, 1, 0, 0, 0, 0, 1] : k === 2 ? [-1, 0, w, 0, -1, h, 0, 0, 1] : [0, 1, 0, -1, 0, w, 0, 0, 1];
    const dst: [number, number] = k === 2 ? [w, h] : [h, w];
    this.applyPoints(H, dst);
    this.record({ kind: 'rot90', H, src: [w, h], dst, arg: k });
  }

  /** Non-linear warp (elastic / grid / optical distortion). Points move to the output pixel that samples nearest to them. */
  remap(mapX: Float32Array, mapY: Float32Array, dw: number, dh: number): void {
    const src = this.size;
    this.withMask((cv, s, d, scope) => {
      const mx = scope.keep(cv.matFromArray(dh, dw, cv.CV_32FC1, mapX));
      const my = scope.keep(cv.matFromArray(dh, dw, cv.CV_32FC1, mapY));
      cv.remap(s, d, mx, my, cv.INTER_NEAREST, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));
    });
    const nearest = (x: number, y: number): [number, number] => {
      let best = Infinity;
      let bi = 0;
      for (let i = 0; i < mapX.length; i++) {
        const dx = mapX[i] + 0.5 - x;
        const dy = mapY[i] + 0.5 - y;
        const dd = dx * dx + dy * dy;
        if (dd < best) {
          best = dd;
          bi = i;
        }
      }
      return [(bi % dw) + 0.5, Math.floor(bi / dw) + 0.5];
    };
    this.keypoints = this.keypoints.map((k) => [...nearest(k[0], k[1]), ...k.slice(2)]);
    this.boxes = this.boxes
      .map((b) => {
        const pts = [nearest(b[0], b[1]), nearest(b[2], b[1]), nearest(b[2], b[3]), nearest(b[0], b[3])];
        const xs = pts.map((p) => p[0]);
        const ys = pts.map((p) => p[1]);
        return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys), ...b.slice(4)];
      })
      .filter((b) => b[2] > b[0] && b[3] > b[1]);
    this.record({ kind: 'remap', src, dst: [dw, dh] });
  }
}
