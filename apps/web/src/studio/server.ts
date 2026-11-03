// Python backend (apps/studio-server): runs the generated preprocess.py from step k onward when a step has no
// OpenCV.js implementation, decodes 16-bit / TIFF images, and measures exact timings.
import {
  analyzePipeline,
  exportSteps,
  generatePreprocessPy,
  type ImageBuf,
  type ImageMeta,
  type Pipeline,
  type PixelData,
  type RngState,
  type StepOutput,
  type TargetsData,
} from '@modelith/preprocess';
import { requestDisplay, useRuns, type RunView } from './engine';
import { useStudio } from './store';

export const SERVER_URL: string = (import.meta.env.VITE_STUDIO_URL as string | undefined) ?? 'http://localhost:8001';

export const SERVER_HINT = 'Start it with: cd apps/studio-server && pip install -r requirements.txt && uvicorn main:app --port 8001';

// ---------------------------------------------------------------- array transport

export interface WireArray {
  shape: number[];
  dtype: 'uint8' | 'uint16' | 'float32' | 'float16' | 'float64' | 'int32';
  data: string; // base64, little endian, C order
}

function b64(bytes: Uint8Array): string {
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

function unb64(s: string): Uint8Array {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export function toWire(buf: ImageBuf): WireArray {
  const { h, w, c } = buf.meta;
  return {
    shape: c === 1 ? [h, w] : [h, w, c],
    dtype: buf.meta.dtype === 'float16' ? 'float32' : buf.meta.dtype,
    data: b64(new Uint8Array(buf.data.buffer, buf.data.byteOffset, buf.data.byteLength)),
  };
}

function float16ToFloat32(bits: Uint16Array): Float32Array {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) {
    const b = bits[i];
    const s = b & 0x8000 ? -1 : 1;
    const e = (b >> 10) & 0x1f;
    const f = b & 0x3ff;
    out[i] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return out;
}

/** Wire array → HWC ImageBuf, undoing CHW / batch layouts according to the expected meta. */
export function fromWire(a: WireArray, meta: ImageMeta): ImageBuf {
  const bytes = unb64(a.data);
  const ab = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  let data: PixelData;
  if (a.dtype === 'uint8') data = new Uint8Array(ab);
  else if (a.dtype === 'uint16') data = new Uint16Array(ab);
  else if (a.dtype === 'float16') data = float16ToFloat32(new Uint16Array(ab));
  else if (a.dtype === 'float64') data = Float32Array.from(new Float64Array(ab));
  else if (a.dtype === 'int32') data = Float32Array.from(new Int32Array(ab));
  else data = new Float32Array(ab);
  let shape = a.shape.slice();
  if (meta.batch && shape.length === 4) shape = shape.slice(1);
  let h: number, w: number, c: number;
  if (meta.layout === 'CHW' && shape.length === 3) {
    [c, h, w] = shape;
    if (c > 1) {
      const t = new (data.constructor as any)(data.length) as PixelData;
      for (let ch = 0; ch < c; ch++) for (let i = 0; i < h * w; i++) t[i * c + ch] = data[ch * h * w + i];
      data = t;
    }
  } else [h, w, c] = [shape[0], shape[1], shape[2] ?? 1];
  const dtype = a.dtype === 'uint8' || a.dtype === 'uint16' ? a.dtype : meta.dtype === 'float16' ? 'float16' : 'float32';
  return { meta: { ...meta, h, w, c, dtype }, data };
}

// ---------------------------------------------------------------- requests

export interface ServerStep {
  index: number;
  name: string;
  image: WireArray;
  ms: number;
  boxes: number[][];
  keypoints: number[][];
  mask?: WireArray;
}

export async function serverHealth(): Promise<boolean> {
  try {
    const r = await fetch(`${SERVER_URL}/health`, { signal: AbortSignal.timeout(1500) });
    return r.ok;
  } catch {
    return false;
  }
}

export async function runOnServer(
  pipeline: Pipeline,
  input: ImageBuf,
  targets: TargetsData,
  opts: { start: number; train: boolean; rng?: RngState; signal?: AbortSignal; timing?: boolean },
): Promise<{ steps: ServerStep[]; totalMs: number }> {
  const code = generatePreprocessPy(pipeline, { includeVisual: true, input: input.meta }).file.content;
  const r = await fetch(`${SERVER_URL}/preprocess/run`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    signal: opts.signal,
    body: JSON.stringify({
      code,
      image: toWire(input),
      mask: targets.mask ? toWire(targets.mask) : null,
      boxes: targets.boxes,
      keypoints: targets.keypoints,
      start: opts.start,
      train: opts.train,
      seed: pipeline.seed,
      rng_state: opts.rng ?? null,
      timing_runs: opts.timing ? 5 : 0,
    }),
  });
  if (!r.ok) throw new Error(`${r.status}: ${(await r.text()).slice(0, 2000)}`);
  const j = await r.json();
  if (!j.ok) throw new Error(j.error ?? 'server error');
  return { steps: j.steps, totalMs: j.total_ms };
}

/** Pipeline step uids in the order of run_steps() in the server-preview code (visual-only steps included). */
export function exportedUids(p: Pipeline): string[] {
  return exportSteps(p, { includeVisual: true }).map((x) => x.step.uid);
}

/** run_steps() index of the first exported step at or after pipeline step `i`. */
export function exportedIndex(p: Pipeline, i: number): number {
  const later = new Set(p.steps.slice(i).map((s) => s.uid));
  const uids = exportedUids(p);
  const k = uids.findIndex((u) => later.has(u));
  return k < 0 ? uids.length : k;
}

/** The browser stopped at step `from` (no OpenCV.js implementation): continue the run on the server. */
export async function continueOnServer(variant: 'A' | 'B', runId: number, from: number) {
  const run = useRuns.getState()[variant];
  if (!run || run.id !== runId || !run.input) return;
  const st = useStudio.getState();
  const pipeline = st.pipelines.find((p) => p.id === run.pipelineId);
  if (!pipeline) return;
  const prev: StepOutput | undefined = from > 0 ? run.steps[from - 1]?.out : undefined;
  const input = prev?.buf ?? run.input.raw;
  const targets = prev?.targets ?? run.input.targets;
  const patch = (fn: (r: RunView) => RunView) =>
    useRuns.setState((s) => {
      const cur = s[variant];
      return cur?.id === runId ? { [variant]: fn(cur) } : {};
    });
  patch((r) => ({ ...r, serverState: 'running', serverError: undefined }));
  try {
    const { steps } = await runOnServer(pipeline, input, targets, { start: exportedIndex(pipeline, from), train: st.train, rng: prev?.rng });
    const analysis = analyzePipeline(pipeline, run.input.raw.meta, { train: st.train });
    // Map run_steps() indices back to pipeline steps (disabled and batch-level steps are not exported).
    const uids = exportedUids(pipeline);
    let lastBuf = input;
    let lastTargets = targets;
    const byUid = new Map(steps.map((s) => [uids[s.index], s]));
    for (let i = from; i < pipeline.steps.length; i++) {
      const s = pipeline.steps[i];
      const got = byUid.get(s.uid);
      let out: StepOutput;
      if (got) {
        const meta = analysis.byUid[s.uid].out;
        lastBuf = fromWire(got.image, meta);
        lastTargets = {
          mask: got.mask ? fromWire(got.mask, { ...lastBuf.meta, c: 1, dtype: 'uint8', space: 'GRAY', layout: 'HWC', batch: false }) : undefined,
          boxes: got.boxes,
          keypoints: got.keypoints,
          history: [],
          size: [lastBuf.meta.w, lastBuf.meta.h],
        };
        out = { uid: s.uid, key: `server:${runId}:${i}`, buf: lastBuf, targets: lastTargets, rng: prev?.rng ?? { state: '0', inc: '0' }, ms: got.ms, engine: 'server' };
      } else {
        out = { uid: s.uid, key: `server:${runId}:${i}`, buf: lastBuf, targets: lastTargets, rng: { state: '0', inc: '0' }, ms: 0, skipped: true, engine: 'server' };
      }
      const display = await requestDisplay(out.buf, st.display);
      patch((r) => {
        const arr = r.steps.slice();
        arr[i] = { out, display };
        return { ...r, steps: arr };
      });
    }
    patch((r) => ({ ...r, serverState: 'done' }));
  } catch (e) {
    const msg = e instanceof TypeError ? `Python server not reachable at ${SERVER_URL}. ${SERVER_HINT}` : (e as Error).message;
    patch((r) => ({ ...r, serverState: 'error', serverError: msg }));
  }
}

/** Decode a file exactly as cv2.imdecode would (16-bit PNG/TIFF, EXIF orientation, alpha). */
export async function decodeOnServer(file: File, flag: string): Promise<ImageBuf> {
  const form = new FormData();
  form.append('file', file);
  form.append('flag', flag);
  const r = await fetch(`${SERVER_URL}/preprocess/decode`, { method: 'POST', body: form });
  if (!r.ok) throw new Error(await r.text());
  const j = await r.json();
  const a: WireArray = j.image;
  const c = a.shape[2] ?? 1;
  const meta: ImageMeta = {
    h: a.shape[0],
    w: a.shape[1],
    c,
    dtype: a.dtype === 'uint16' ? 'uint16' : a.dtype === 'float32' ? 'float32' : 'uint8',
    space: c === 1 ? 'GRAY' : c === 4 ? 'BGRA' : 'BGR',
    range: a.dtype === 'uint16' ? [0, 65535] : [0, 255],
    layout: 'HWC',
    batch: false,
  };
  return fromWire(a, meta);
}
