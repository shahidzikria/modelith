// Browser engine: runs pipeline steps with OpenCV.js, one step at a time, caching each output by a key
// chained from the input and every earlier step, so editing step k only recomputes k and later.
import { getStep } from '../registry';
import type { CV, Params, StepDef, StepMode } from '../types';
import { bufFromMat, matFromBuf, MatScope, type ImageBuf } from './buffers';
import { Rng, type RngState } from './rng';
import { Targets, type TargetsData } from './targets';

export interface RunStep {
  uid: string;
  def: string;
  params: Params;
  enabled: boolean;
  mode: StepMode;
  p?: number;
}

export interface StepOutput {
  uid: string;
  key: string;
  buf: ImageBuf;
  targets: TargetsData;
  /** RNG state after this step (lets a later step continue from the cache, or on the server). */
  rng: RngState;
  ms: number;
  /** Disabled or not part of the current mode: output = input. */
  skipped?: boolean;
  /** Random step whose probability draw said "don't apply". */
  notApplied?: boolean;
  engine: 'browser' | 'server';
}

export interface RunResult {
  outputs: StepOutput[];
  /** Index of the first step that needs the Python server; outputs stop before it. */
  serverFrom?: number;
  /** Error from step `errorAt` (outputs stop there). */
  error?: string;
  errorAt?: number;
}

export function isActive(s: RunStep, train: boolean): boolean {
  return s.enabled && (s.mode === 'both' || (s.mode === 'train') === train);
}

/** Functions a step needs that this OpenCV.js build lacks (empty = runs in the browser). */
export function missingInBrowser(cv: CV, def: StepDef): string[] {
  if (def.engine === 'server' || !def.applyBrowser) return ['(server-only step)'];
  return (def.needs ?? []).filter((n) => !(n in cv));
}

/** FNV-1a 32-bit over a string, as 8 hex chars. */
export function hashKey(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

export function stepKey(prev: string, s: RunStep, train: boolean): string {
  return hashKey(`${prev}|${s.def}|${JSON.stringify(s.params)}|${isActive(s, train) ? 1 : 0}|${s.p ?? 1}`) + prev.slice(0, 8);
}

/** OpenCV.js throws numeric pointers for cv::Exception; turn them into readable messages. */
export function cvErrorMessage(cv: CV, err: unknown): string {
  if (typeof err === 'number' && typeof cv.exceptionFromPtr === 'function') {
    try {
      return String(cv.exceptionFromPtr(err).msg).trim();
    } catch {
      /* fall through */
    }
  }
  if (err instanceof Error) return err.message;
  return String(err);
}

/** Anything with Map-like get/set (a Map, or an LRU in the worker). */
export interface StepCache {
  get(key: string): StepOutput | undefined;
  set(key: string, value: StepOutput): unknown;
}

export interface RunOptions {
  train: boolean;
  seed: number;
  /** Identifies the input image (and proxy size); cache keys chain from it. */
  rootKey: string;
  cache?: StepCache;
  /** Called after each step (for streaming progress). */
  onStep?(o: StepOutput, index: number): void;
  /** Cooperative cancellation. */
  cancelled?(): boolean;
}

export function runSteps(cv: CV, input: ImageBuf, targets: TargetsData, steps: RunStep[], opts: RunOptions): RunResult {
  const outputs: StepOutput[] = [];
  let key = hashKey(`${opts.rootKey}|seed=${opts.seed}|train=${opts.train}`);
  let buf = input;
  let tdata = targets;
  let rngState = new Rng(opts.seed).getState();

  for (let i = 0; i < steps.length; i++) {
    if (opts.cancelled?.()) break;
    const s = steps[i];
    key = stepKey(key, s, opts.train);
    const hit = opts.cache?.get(key);
    if (hit) {
      outputs.push({ ...hit, uid: s.uid });
      opts.onStep?.(outputs[i], i);
      buf = hit.buf;
      tdata = hit.targets;
      rngState = hit.rng;
      continue;
    }
    const def = getStep(s.def);
    if (!def || !isActive(s, opts.train)) {
      const o: StepOutput = { uid: s.uid, key, buf, targets: tdata, rng: rngState, ms: 0, skipped: true, engine: 'browser' };
      outputs.push(o);
      opts.onStep?.(o, i);
      continue;
    }
    if (missingInBrowser(cv, def).length) return { outputs, serverFrom: i };

    const rng = new Rng(rngState);
    const t0 = performance.now();
    const p = s.p ?? 1;
    if (def.kind === 'random' && p < 1 && rng.random() >= p) {
      const o: StepOutput = { uid: s.uid, key, buf, targets: tdata, rng: rng.getState(), ms: performance.now() - t0, notApplied: true, engine: 'browser' };
      outputs.push(o);
      opts.cache?.set(key, o);
      opts.onStep?.(o, i);
      rngState = o.rng;
      continue;
    }
    const scope = new MatScope();
    try {
      let outMeta = def.outMeta(buf.meta, s.params);
      const t = new Targets(cv, tdata);
      const src = scope.keep(matFromBuf(cv, buf));
      const res = def.applyBrowser!(src, s.params, {
        cv,
        keep: (m) => scope.keep(m),
        mat: () => scope.keep(new cv.Mat()),
        in: buf.meta,
        out: outMeta,
        rng,
        t,
        train: opts.train,
      });
      scope.keep(res);
      if (def.dynamicSize) outMeta = { ...outMeta, h: res.rows, w: res.cols };
      const next = bufFromMat(cv, res, outMeta);
      if (next.meta.h !== outMeta.h || next.meta.w !== outMeta.w || next.meta.c !== outMeta.c || (next.meta.dtype !== outMeta.dtype && outMeta.dtype !== 'float16'))
        throw new Error(
          `Internal: ${def.label} produced ${next.meta.h}×${next.meta.w}×${next.meta.c} ${next.meta.dtype}, expected ${outMeta.h}×${outMeta.w}×${outMeta.c} ${outMeta.dtype}.`,
        );
      if (def.affectsGeometry) t.size = [outMeta.w, outMeta.h];
      const o: StepOutput = { uid: s.uid, key, buf: next, targets: t.data(), rng: rng.getState(), ms: performance.now() - t0, engine: 'browser' };
      outputs.push(o);
      opts.cache?.set(key, o);
      opts.onStep?.(o, i);
      buf = next;
      tdata = o.targets;
      rngState = o.rng;
    } catch (err) {
      return { outputs, error: cvErrorMessage(cv, err), errorAt: i };
    } finally {
      scope.dispose();
    }
  }
  return { outputs };
}
