// UI side of the OpenCV.js worker: turns samples into input buffers, sends runs, and streams step results
// into a small store the viewer, filmstrip and pipeline rows read from.
import {
  imageMeta,
  type Display,
  type DisplayOptions,
  type ImageBuf,
  type Pipeline,
  type ReadSpec,
  type RunStep,
  type StepOutput,
  type TargetsData,
} from '@modelith/preprocess';
import { useEffect } from 'react';
import { create } from 'zustand';
import type { WorkerIn, WorkerOut } from './protocol';
import { continueOnServer } from './server';
import { useStudio, type Sample } from './store';

export interface StepView {
  out: StepOutput;
  display: Display;
}

export interface RunView {
  id: number;
  pipelineId: string;
  input?: { raw: ImageBuf; display: Display; targets: TargetsData };
  steps: (StepView | undefined)[];
  done: boolean;
  proxy: boolean;
  totalMs?: number;
  error?: string;
  errorAt?: number;
  /** First step routed to the Python server. */
  serverFrom?: number;
  serverState?: 'running' | 'done' | 'error';
  serverError?: string;
}

interface Runs {
  ready: boolean;
  /** Step id → OpenCV.js functions missing from this build (those steps run on the server). */
  missing: Record<string, string[]>;
  A?: RunView;
  B?: RunView;
}

export const useRuns = create<Runs>(() => ({ ready: false, missing: {} }));

// ---------------------------------------------------------------- sample → input buffer

const gray = (b: number, g: number, r: number) => (b * 1868 + g * 9617 + r * 4899 + 8192) >> 14;

/** The image as `load_image()` would return it for this read flag (best effort for browser-decoded files). */
export function sampleInput(s: Sample, read: ReadSpec): ImageBuf {
  const ex = s.exact;
  const { w, h } = s;
  const n = w * h;
  if (ex) {
    const c = ex.meta.c;
    if (read.flag === 'UNCHANGED') return ex;
    if (read.flag === 'ANYDEPTH' && c === 1) return ex;
    const src = ex.data;
    const k = ex.meta.dtype === 'uint16' ? 1 / 257 : 1;
    const px = (i: number, ch: number) => Math.round(src[i * c + Math.min(ch, c - 1)] * k);
    if (read.flag === 'COLOR') {
      if (c === 3 && ex.meta.dtype === 'uint8') return ex;
      const d = new Uint8Array(n * 3);
      for (let i = 0; i < n; i++) for (let ch = 0; ch < 3; ch++) d[i * 3 + ch] = px(i, c === 1 ? 0 : ch);
      return { meta: imageMeta(h, w, 3), data: d };
    }
    if (read.flag === 'ANYDEPTH' && ex.meta.dtype === 'uint16') {
      const d = new Uint16Array(n);
      for (let i = 0; i < n; i++) d[i] = c === 1 ? src[i] : gray(src[i * c], src[i * c + 1], src[i * c + 2]);
      return { meta: imageMeta(h, w, 1, 'uint16'), data: d };
    }
    const d = new Uint8Array(n);
    for (let i = 0; i < n; i++) d[i] = c === 1 ? px(i, 0) : gray(px(i, 0), px(i, 1), px(i, 2));
    return { meta: imageMeta(h, w, 1), data: d };
  }
  const rgba = s.rgba!;
  if (read.flag === 'GRAYSCALE' || read.flag === 'ANYDEPTH') {
    const d = new Uint8Array(n);
    for (let i = 0; i < n; i++) d[i] = gray(rgba[i * 4 + 2], rgba[i * 4 + 1], rgba[i * 4]);
    return { meta: imageMeta(h, w, 1), data: d };
  }
  const c = read.flag === 'UNCHANGED' && s.hasAlpha ? 4 : 3;
  const d = new Uint8Array(n * c);
  for (let i = 0; i < n; i++) {
    d[i * c] = rgba[i * 4 + 2];
    d[i * c + 1] = rgba[i * 4 + 1];
    d[i * c + 2] = rgba[i * 4];
    if (c === 4) d[i * c + 3] = rgba[i * 4 + 3];
  }
  return { meta: imageMeta(h, w, c), data: d };
}

export function sampleTargets(s: Sample): TargetsData {
  return { mask: s.mask, boxes: s.boxes, keypoints: s.keypoints, history: [], size: [s.w, s.h] };
}

export const inputKeyOf = (s: Sample, read: ReadSpec) => `${s.id}:${read.flag}`;

export const runSteps = (p: Pipeline): RunStep[] => p.steps.map(({ uid, def, params, enabled, mode, p: prob }) => ({ uid, def, params, enabled, mode, p: prob }));

// ---------------------------------------------------------------- worker

let worker: Worker | undefined;
let nextId = 1;
const sentInputs = new Set<string>();
const variantHandlers = new Map<number, (m: Extract<WorkerOut, { type: 'variant' }>) => void>();
const displayHandlers = new Map<number, (d: Display) => void>();
const runVariant = new Map<number, 'A' | 'B'>();

function getWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL('./cv.worker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = (e: MessageEvent<WorkerOut>) => onMessage(e.data);
  return worker;
}

function post(msg: WorkerIn) {
  getWorker().postMessage(msg);
}

function patchRun(v: 'A' | 'B', id: number, fn: (r: RunView) => RunView) {
  useRuns.setState((s) => {
    const r = s[v];
    if (!r || r.id !== id) return {};
    return { [v]: fn(r) } as Partial<Runs>;
  });
}

function onMessage(m: WorkerOut) {
  if (m.type === 'ready') {
    useRuns.setState({ ready: true, missing: m.missing });
    return;
  }
  if (m.type === 'variant') {
    variantHandlers.get(m.id)?.(m);
    return;
  }
  if (m.type === 'displayed') {
    displayHandlers.get(m.id)?.(m.display);
    displayHandlers.delete(m.id);
    return;
  }
  const v = runVariant.get(m.id);
  if (!v) return;
  if (m.type === 'input') patchRun(v, m.id, (r) => ({ ...r, input: { raw: m.raw, display: m.display, targets: m.targets } }));
  else if (m.type === 'step')
    patchRun(v, m.id, (r) => {
      const steps = r.steps.slice();
      steps[m.index] = { out: m.out, display: m.display };
      return { ...r, steps };
    });
  else if (m.type === 'done') {
    patchRun(v, m.id, (r) => ({ ...r, done: true, totalMs: m.totalMs, error: m.error, errorAt: m.errorAt, serverFrom: m.serverFrom }));
    if (m.serverFrom !== undefined) void continueOnServer(v, m.id, m.serverFrom);
  }
}

export function ensureInput(sample: Sample, read: ReadSpec): string {
  const key = inputKeyOf(sample, read);
  if (!sentInputs.has(key)) {
    post({ type: 'setInput', inputKey: key, buf: sampleInput(sample, read), targets: sampleTargets(sample) });
    sentInputs.add(key);
  }
  return key;
}

/** Forget a sample's cached input (it was edited, e.g. a mask was attached). */
export function invalidateSample(id: string) {
  for (const k of [...sentInputs])
    if (k.startsWith(`${id}:`)) {
      sentInputs.delete(k);
      post({ type: 'dropInput', inputKey: k });
    }
}

export function startRun(variant: 'A' | 'B', pipeline: Pipeline, sample: Sample, train: boolean, display: DisplayOptions, proxy: boolean): number {
  const id = nextId++;
  const inputKey = ensureInput(sample, pipeline.read);
  runVariant.set(id, variant);
  useRuns.setState((s) => {
    const prev = s[variant];
    // Keep previous results visible until new ones stream in (avoids flicker).
    const steps = prev && prev.pipelineId === pipeline.id ? prev.steps.slice(0, pipeline.steps.length) : [];
    return { [variant]: { id, pipelineId: pipeline.id, input: prev?.input, steps, done: false, proxy } } as Partial<Runs>;
  });
  post({ type: 'run', id, inputKey, steps: runSteps(pipeline), seed: pipeline.seed, train, maxSide: proxy ? 512 : undefined, display });
  return id;
}

/** Run several step lists on the current sample (9 random samples, parameter sweeps). */
export function runVariants(
  sample: Sample,
  read: ReadSpec,
  variants: { steps: RunStep[]; seed: number }[],
  train: boolean,
  display: DisplayOptions,
  onResult: (m: Extract<WorkerOut, { type: 'variant' }>) => void,
  opts: { fullResolution?: boolean; returnRaw?: boolean } = {},
): () => void {
  const id = nextId++;
  variantHandlers.set(id, onResult);
  const maxSide = opts.fullResolution ? undefined : 384;
  post({ type: 'variants', id, inputKey: ensureInput(sample, read), variants, train, display, maxSide, returnRaw: opts.returnRaw });
  return () => variantHandlers.delete(id);
}

/** Convert a raw buffer (e.g. from the server) to a display image in the worker. */
export function requestDisplay(buf: ImageBuf, display: DisplayOptions): Promise<Display> {
  const id = nextId++;
  return new Promise((resolve) => {
    displayHandlers.set(id, resolve);
    post({ type: 'display', id, buf, display });
  });
}

/** Keeps runs A (and B when comparing) in sync with the pipeline, sample and mode; debounced ~60 ms. */
export function useEngine() {
  const pipelines = useStudio((s) => s.pipelines);
  const activeId = useStudio((s) => s.activeId);
  const compareId = useStudio((s) => s.compareId);
  const sample = useStudio((s) => s.samples[s.sampleIndex]);
  const train = useStudio((s) => s.train);
  const dragging = useStudio((s) => s.dragging);
  const display = useStudio((s) => s.display);
  const a = pipelines.find((p) => p.id === activeId)!;
  const b = compareId ? pipelines.find((p) => p.id === compareId) : undefined;

  useEffect(() => {
    getWorker();
  }, []);

  useEffect(() => {
    const t = setTimeout(() => startRun('A', a, sample, train, display, dragging), dragging ? 60 : 0);
    return () => clearTimeout(t);
  }, [a, sample, train, display, dragging]);

  useEffect(() => {
    if (!b) {
      useRuns.setState({ B: undefined });
      return;
    }
    const t = setTimeout(() => startRun('B', b, sample, train, display, dragging), dragging ? 60 : 0);
    return () => clearTimeout(t);
  }, [b, sample, train, display, dragging]);
}
