// Turn any intermediate (uint8/uint16/float, GRAY/BGR/HSV/LAB/…, CHW) into RGBA pixels for a canvas.
import type { CV } from '../types';
import { bufFromMat, matFromBuf, MatScope, type ImageBuf } from './buffers';

export interface DisplayOptions {
  /** 'auto' converts HSV / LAB / … back to RGB; 'raw' shows the stored channels as if they were BGR. */
  color: 'auto' | 'raw';
  /** Show one channel as gray (or false colour). */
  channel: 'all' | 0 | 1 | 2 | 3;
  /** False colour (turbo) for single-channel non-uint8 data. */
  falseColor: boolean;
}

export const DEFAULT_DISPLAY: DisplayOptions = { color: 'auto', channel: 'all', falseColor: true };

export interface Display {
  rgba: Uint8ClampedArray;
  w: number;
  h: number;
  /** Min / max mapped to 0 / 255 when the data was rescaled for display ("display scaled"). */
  scaled?: [number, number];
  falseColor?: boolean;
}

const CONVERT_BACK: Record<string, string> = {
  HSV: 'COLOR_HSV2BGR',
  HLS: 'COLOR_HLS2BGR',
  LAB: 'COLOR_Lab2BGR',
  LUV: 'COLOR_Luv2BGR',
  YCrCb: 'COLOR_YCrCb2BGR',
  XYZ: 'COLOR_XYZ2BGR',
};

/** Turbo colormap (Mikhailov 2019), polynomial approximation; t in [0, 1]. */
export function turbo(t: number): [number, number, number] {
  t = Math.min(1, Math.max(0, t));
  const r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  const b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

export function toDisplay(cv: CV | undefined, input: ImageBuf, opts: DisplayOptions = DEFAULT_DISPLAY): Display {
  let buf = input;
  const { h, w } = buf.meta;
  let space = buf.meta.space;
  // Convert exotic colour spaces back to BGR (needs OpenCV).
  if (opts.color === 'auto' && opts.channel === 'all' && cv && buf.meta.c === 3 && CONVERT_BACK[space] && buf.meta.dtype !== 'uint16') {
    const scope = new MatScope();
    try {
      const src = scope.keep(matFromBuf(cv, buf));
      const dst = scope.keep(new cv.Mat());
      cv.cvtColor(src, dst, cv[CONVERT_BACK[space]]);
      buf = bufFromMat(cv, dst, { ...buf.meta, space: 'BGR', range: buf.meta.dtype === 'uint8' ? [0, 255] : [0, 1] });
      space = 'BGR';
    } finally {
      scope.dispose();
    }
  }
  const c = buf.meta.c;
  const n = h * w;
  const src = buf.data;
  const single = opts.channel !== 'all' || c === 1;
  const ch = opts.channel === 'all' ? 0 : Math.min(opts.channel, c - 1);

  // Range: uint8 shown as-is; floats in [0,1] scaled by 255; anything else min-max scaled.
  let lo = 0;
  let hi = 255;
  let scaled: [number, number] | undefined;
  const isU8 = buf.meta.dtype === 'uint8';
  if (!isU8) {
    let mn = Infinity;
    let mx = -Infinity;
    const stride = single ? c : 1;
    for (let i = single ? ch : 0; i < src.length; i += stride) {
      const v = src[i];
      if (Number.isFinite(v)) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    }
    if (!Number.isFinite(mn)) (mn = 0), (mx = 1);
    if (buf.meta.dtype !== 'uint16' && mn >= 0 && mx <= 1 && !(single && opts.falseColor)) {
      lo = 0;
      hi = 1;
    } else {
      lo = mn;
      hi = mx > mn ? mx : mn + 1;
      scaled = [mn, mx];
    }
  }
  const k = 255 / (hi - lo);
  const rgba = new Uint8ClampedArray(n * 4);
  const useFalse = single && opts.falseColor && !isU8;
  const rgbOrder = space === 'RGB' || space === 'RGBA';
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    if (single) {
      const v = (src[i * c + ch] - lo) * k;
      if (useFalse) {
        const [r, g, b] = turbo(v / 255);
        rgba[o] = r;
        rgba[o + 1] = g;
        rgba[o + 2] = b;
      } else rgba[o] = rgba[o + 1] = rgba[o + 2] = v;
      rgba[o + 3] = 255;
    } else {
      const a = (src[i * c] - lo) * k;
      const b2 = (src[i * c + 1] - lo) * k;
      const c3 = (src[i * c + 2] - lo) * k;
      rgba[o] = rgbOrder ? a : c3;
      rgba[o + 1] = b2;
      rgba[o + 2] = rgbOrder ? c3 : a;
      rgba[o + 3] = c === 4 ? (src[i * c + 3] - lo) * k : 255;
    }
  }
  return { rgba, w, h, scaled, falseColor: useFalse };
}
