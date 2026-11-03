// Plain image buffers (transferable between worker and UI) and conversion to / from cv.Mat.
import type { CV, DType, ImageMeta, Mat } from '../types';

export type PixelData = Uint8Array | Uint16Array | Float32Array;

export interface ImageBuf {
  meta: ImageMeta;
  /** HWC, row-major. float16 is held as float32 in the browser. */
  data: PixelData;
}

export function allocData(dtype: DType, n: number): PixelData {
  if (dtype === 'uint8') return new Uint8Array(n);
  if (dtype === 'uint16') return new Uint16Array(n);
  return new Float32Array(n);
}

export function cvDepth(cv: CV, dtype: DType): number {
  return dtype === 'uint8' ? cv.CV_8U : dtype === 'uint16' ? cv.CV_16U : cv.CV_32F;
}

export function cvType(cv: CV, dtype: DType, c: number): number {
  const base = dtype === 'uint8' ? 'CV_8U' : dtype === 'uint16' ? 'CV_16U' : 'CV_32F';
  return cv[`${base}C${c}`];
}

export function dtypeOfMat(cv: CV, m: Mat): DType {
  const d = m.depth();
  if (d === cv.CV_8U) return 'uint8';
  if (d === cv.CV_16U) return 'uint16';
  if (d === cv.CV_32F) return 'float32';
  throw new Error(`Unsupported Mat depth ${d} (only uint8, uint16 and float32 are used).`);
}

/** Copy a buffer into a new Mat (caller owns it). CHW / batch layouts are stored as HWC in the browser. */
export function matFromBuf(cv: CV, buf: ImageBuf): Mat {
  const { h, w, c, dtype } = buf.meta;
  const m = new cv.Mat(h, w, cvType(cv, dtype, c));
  if (dtype === 'uint8') m.data.set(buf.data);
  else if (dtype === 'uint16') m.data16U.set(buf.data);
  else m.data32F.set(buf.data);
  return m;
}

/** Copy a Mat out of WASM memory. `meta` supplies the bookkeeping fields (space, range, layout). */
export function bufFromMat(cv: CV, m: Mat, meta: ImageMeta): ImageBuf {
  if (!m.isContinuous()) {
    // Note: in OpenCV.js, clone() of a ROI stays non-continuous; copyTo() gives a packed Mat.
    const c = new cv.Mat();
    m.copyTo(c);
    try {
      return bufFromMat(cv, c, meta);
    } finally {
      c.delete();
    }
  }
  const dtype = dtypeOfMat(cv, m);
  const data = dtype === 'uint8' ? new Uint8Array(m.data) : dtype === 'uint16' ? new Uint16Array(m.data16U) : new Float32Array(m.data32F);
  return {
    meta: { ...meta, h: m.rows, w: m.cols, c: m.channels(), dtype: meta.dtype === 'float16' && dtype === 'float32' ? 'float16' : dtype },
    data,
  };
}

/** Tracks every Mat / OpenCV object created during a step and deletes them all at the end. */
export class MatScope {
  private items = new Set<{ delete(): void; isDeleted?: () => boolean }>();

  keep<T>(m: T): T {
    if (m && typeof (m as any).delete === 'function') this.items.add(m as any);
    return m;
  }

  get size(): number {
    return this.items.size;
  }

  dispose(): void {
    for (const m of this.items) {
      try {
        if (!m.isDeleted?.()) m.delete();
      } catch {
        /* already deleted */
      }
    }
    this.items.clear();
  }
}
