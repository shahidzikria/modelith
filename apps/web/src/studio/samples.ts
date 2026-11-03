// Loading sample images, masks and box / keypoint annotations from files.
import { imageMeta, newUid, type ImageBuf } from '@modelith/preprocess';
import { decodeOnServer } from './server';
import type { Sample } from './store';

const NEEDS_SERVER = /\.(tiff?|pgm|ppm|exr|hdr|jp2)$/i;

async function decodeInBrowser(file: Blob, exif = true): Promise<{ w: number; h: number; rgba: Uint8ClampedArray; hasAlpha: boolean }> {
  const bmp = await createImageBitmap(file, { imageOrientation: exif ? 'from-image' : 'none', premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const g = c.getContext('2d', { willReadFrequently: true })!;
  g.drawImage(bmp, 0, 0);
  const { data } = g.getImageData(0, 0, bmp.width, bmp.height);
  bmp.close();
  let hasAlpha = false;
  for (let i = 3; i < data.length; i += 4) if (data[i] !== 255) {
    hasAlpha = true;
    break;
  }
  return { w: c.width, h: c.height, rgba: data, hasAlpha };
}

/** Decode an image file into a Sample. 16-bit PNG and TIFF go through the Python server for exact decoding. */
export async function loadSampleFile(file: File, opts: { exif: boolean; preferServer?: boolean }): Promise<Sample> {
  const is16 = file.type === 'image/png' && (await isPng16(file));
  if (NEEDS_SERVER.test(file.name) || is16 || opts.preferServer) {
    try {
      const exact = await decodeOnServer(file, 'UNCHANGED');
      return { id: newUid('img'), name: file.name, w: exact.meta.w, h: exact.meta.h, exact, boxes: [], keypoints: [] };
    } catch (e) {
      if (NEEDS_SERVER.test(file.name)) throw new Error(`${file.name}: needs the Python server to decode (${(e as Error).message.slice(0, 120)})`);
      // 16-bit PNG without a server: fall back to the browser's 8-bit decode.
    }
  }
  const d = await decodeInBrowser(file, opts.exif);
  return { id: newUid('img'), name: file.name + (is16 ? ' (8-bit preview: start the server for 16-bit)' : ''), w: d.w, h: d.h, rgba: d.rgba, hasAlpha: d.hasAlpha, boxes: [], keypoints: [] };
}

/** PNG bit depth is byte 24 of the file (IHDR). */
async function isPng16(file: File): Promise<boolean> {
  const head = new Uint8Array(await file.slice(0, 26).arrayBuffer());
  return head[24] === 16;
}

/** A label mask image (PNG): uses the first channel as class ids; resized by nearest neighbour if needed. */
export async function loadMaskFile(file: File, w: number, h: number): Promise<ImageBuf> {
  const d = await decodeInBrowser(file, false);
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const sx = Math.min(d.w - 1, Math.floor((x * d.w) / w));
      const sy = Math.min(d.h - 1, Math.floor((y * d.h) / h));
      out[y * w + x] = d.rgba[(sy * d.w + sx) * 4];
    }
  // Masks saved as 0/255 binary images become 0/1 labels.
  let only255 = true;
  for (let i = 0; i < out.length && only255; i++) if (out[i] !== 0 && out[i] !== 255) only255 = false;
  if (only255) for (let i = 0; i < out.length; i++) out[i] = out[i] ? 1 : 0;
  return { meta: imageMeta(h, w, 1), data: out };
}

/**
 * Boxes / keypoints from text:
 * - YOLO txt: `class cx cy w h` (normalised) per line
 * - JSON: `{"boxes": [[x1,y1,x2,y2,cls?],…], "keypoints": [[x,y],…]}` or a COCO-style list of `{"bbox": [x,y,w,h], "category_id"}`
 */
export function parseAnnotations(text: string, w: number, h: number): { boxes: number[][]; keypoints: number[][] } {
  const t = text.trim();
  if (t.startsWith('{') || t.startsWith('[')) {
    const j = JSON.parse(t);
    if (Array.isArray(j)) return { boxes: j.map((a: any) => [a.bbox[0], a.bbox[1], a.bbox[0] + a.bbox[2], a.bbox[1] + a.bbox[3], a.category_id ?? 0]), keypoints: [] };
    const anns = j.annotations as any[] | undefined;
    if (anns) return { boxes: anns.map((a) => [a.bbox[0], a.bbox[1], a.bbox[0] + a.bbox[2], a.bbox[1] + a.bbox[3], a.category_id ?? 0]), keypoints: [] };
    return { boxes: j.boxes ?? [], keypoints: j.keypoints ?? [] };
  }
  const boxes = t
    .split(/\r?\n/)
    .map((l) => l.trim().split(/\s+/).map(Number))
    .filter((v) => v.length >= 5 && v.every(Number.isFinite))
    .map(([c, cx, cy, bw, bh]) => [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h, c]);
  return { boxes, keypoints: [] };
}

export function pickFiles(accept: string, multiple = true, directory = false): Promise<File[]> {
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = accept;
    input.multiple = multiple;
    if (directory) (input as any).webkitdirectory = true;
    input.onchange = () => resolve([...(input.files ?? [])]);
    input.click();
  });
}

export const IMAGE_ACCEPT = 'image/*,.tif,.tiff';
export const isImageName = (n: string) => /\.(png|jpe?g|bmp|webp|gif|tiff?)$/i.test(n);
