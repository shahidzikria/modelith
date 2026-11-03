/// <reference lib="webworker" />
// OpenCV.js runs here so the UI never freezes. Each request runs the pipeline one step at a time (yielding
// between steps so newer requests pre-empt older ones); outputs are cached by a key chained from the input
// and every earlier step, so editing step k only recomputes k and later.
import {
  allSteps,
  cvErrorMessage,
  missingInBrowser,
  runSteps,
  Targets,
  toDisplay,
  bufFromMat,
  matFromBuf,
  MatScope,
  type DisplayOptions,
  type ImageBuf,
  type RunStep,
  type StepCache,
  type StepOutput,
  type TargetsData,
} from '@modelith/preprocess';
import cvModule from '@techstark/opencv-js';
import type { WorkerIn, WorkerOut } from './protocol';

let cv: any;
const ready: Promise<void> = new Promise((resolve) => {
  const m: any = cvModule;
  const done = (x: any) => {
    delete x.then;
    cv = x;
    resolve();
  };
  if (m.Mat) done(m);
  else if (typeof m.then === 'function') m.then(done);
  else m.onRuntimeInitialized = () => done(m);
});

const post = (msg: WorkerOut, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(msg, transfer);

// ---------------------------------------------------------------- caches
const inputs = new Map<string, { buf: ImageBuf; targets: TargetsData }>();
/** Step outputs, least recently used evicted beyond a byte budget. */
class LruCache implements StepCache {
  private map = new Map<string, StepOutput>();
  private bytes = 0;
  constructor(private limit: number) {}

  private static size(v: StepOutput): number {
    return v.buf.data.byteLength + (v.targets.mask?.data.byteLength ?? 0);
  }

  get(k: string): StepOutput | undefined {
    const v = this.map.get(k);
    if (v) {
      this.map.delete(k);
      this.map.set(k, v);
    }
    return v;
  }

  set(k: string, v: StepOutput): void {
    const old = this.map.get(k);
    if (old) this.bytes -= LruCache.size(old);
    this.map.delete(k);
    this.map.set(k, v);
    this.bytes += LruCache.size(v);
    for (const [key, x] of this.map) {
      if (this.bytes <= this.limit) break;
      this.map.delete(key);
      this.bytes -= LruCache.size(x);
    }
  }
}

const cacheProxy = new LruCache(384 * 1024 * 1024);

/** Downscale a sample (and its targets) so the longest side ≤ maxSide (slider-drag proxy). */
function proxyOf(key: string, maxSide?: number): { buf: ImageBuf; targets: TargetsData; key: string } {
  const base = inputs.get(key)!;
  const { h, w } = base.buf.meta;
  if (!maxSide || Math.max(h, w) <= maxSide) return { ...base, key };
  const pk = `${key}@${maxSide}`;
  const hit = inputs.get(pk);
  if (hit) return { ...hit, key: pk };
  const s = maxSide / Math.max(h, w);
  const nw = Math.max(1, Math.round(w * s));
  const nh = Math.max(1, Math.round(h * s));
  const scope = new MatScope();
  try {
    const src = scope.keep(matFromBuf(cv, base.buf));
    const dst = scope.keep(new cv.Mat());
    cv.resize(src, dst, new cv.Size(nw, nh), 0, 0, cv.INTER_AREA);
    const buf = bufFromMat(cv, dst, base.buf.meta);
    const t = new Targets(cv, base.targets);
    t.resize(nw, nh);
    t.history = [];
    const v = { buf, targets: t.data() };
    inputs.set(pk, v);
    return { ...v, key: pk };
  } finally {
    scope.dispose();
  }
}

const yieldNow = () => new Promise((r) => setTimeout(r, 0));
let latest = 0;

function display(buf: ImageBuf, opts: DisplayOptions) {
  const d = toDisplay(cv, buf, opts);
  return d;
}

async function run(msg: Extract<WorkerIn, { type: 'run' }>) {
  await ready;
  latest = Math.max(latest, msg.id);
  const { buf, targets, key } = proxyOf(msg.inputKey, msg.maxSide);
  const d0 = display(buf, msg.display);
  post({ type: 'input', id: msg.id, raw: buf, display: d0, targets }, [d0.rgba.buffer]);
  const steps = msg.steps;
  const t0 = performance.now();
  for (let k = 1; k <= steps.length; k++) {
    if (msg.id < latest) return; // superseded
    const res = runSteps(cv, buf, targets, steps.slice(0, k), { train: msg.train, seed: msg.seed, rootKey: key, cache: cacheProxy });
    if (res.error !== undefined || res.serverFrom !== undefined) {
      post({ type: 'done', id: msg.id, error: res.error, errorAt: res.errorAt, serverFrom: res.serverFrom, totalMs: performance.now() - t0 });
      return;
    }
    const o = res.outputs[k - 1];
    const d = display(o.buf, msg.display);
    post({ type: 'step', id: msg.id, index: k - 1, out: o, display: d }, [d.rgba.buffer]);
    await yieldNow();
  }
  post({ type: 'done', id: msg.id, totalMs: performance.now() - t0 });
}

/** Run several step lists (e.g. 9 seeds of one augmentation, or a parameter sweep) and return final displays. */
async function variants(msg: Extract<WorkerIn, { type: 'variants' }>) {
  await ready;
  const { buf, targets, key } = proxyOf(msg.inputKey, msg.maxSide);
  for (let i = 0; i < msg.variants.length; i++) {
    const v = msg.variants[i];
    const res = runSteps(cv, buf, targets, v.steps as RunStep[], { train: msg.train, seed: v.seed, rootKey: key, cache: cacheProxy });
    const last = res.outputs[res.outputs.length - 1];
    if (res.error || res.serverFrom !== undefined || !last) {
      post({ type: 'variant', id: msg.id, index: i, error: res.error ?? 'Needs the Python server' });
      continue;
    }
    const d = display(last.buf, msg.display);
    post({ type: 'variant', id: msg.id, index: i, display: d, targets: last.targets, meta: last.buf.meta, raw: msg.returnRaw ? last.buf : undefined }, [d.rgba.buffer]);
    await yieldNow();
  }
}

self.onmessage = async (e: MessageEvent<WorkerIn>) => {
  const msg = e.data;
  try {
    if (msg.type === 'setInput') {
      inputs.set(msg.inputKey, { buf: msg.buf, targets: msg.targets });
    } else if (msg.type === 'dropInput') {
      for (const k of [...inputs.keys()]) if (k === msg.inputKey || k.startsWith(`${msg.inputKey}@`)) inputs.delete(k);
    } else if (msg.type === 'run') await run(msg);
    else if (msg.type === 'variants') await variants(msg);
    else if (msg.type === 'display') {
      await ready;
      const d = display(msg.buf, msg.display);
      post({ type: 'displayed', id: msg.id, display: d }, [d.rgba.buffer]);
    }
  } catch (err) {
    post({ type: 'done', id: (msg as { id?: number }).id ?? -1, error: cv ? cvErrorMessage(cv, err) : String(err), totalMs: 0 });
  }
};

ready.then(() => {
  const missing: Record<string, string[]> = {};
  for (const def of allSteps()) {
    const m = missingInBrowser(cv, def);
    if (m.length) missing[def.id] = m;
  }
  post({ type: 'ready', missing });
});
