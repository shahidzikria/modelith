// pipeline.json import / export with validation and migration from older formats.
import { newUid } from './presets';
import { defaultParams, getStep } from './registry';
import type { Pipeline, PipelineStep, ReadFlag, StepMode } from './types';

export class PipelineFormatError extends Error {}

/** Serialise for `pipeline.json` (stable key order, unknown runtime fields dropped). */
export function serializePipeline(p: Pipeline): string {
  const clean: Pipeline = {
    version: 1,
    id: p.id,
    name: p.name,
    seed: p.seed,
    read: { flag: p.read.flag, exif: p.read.exif },
    steps: p.steps.map((s) => ({
      uid: s.uid,
      def: s.def,
      params: s.params,
      enabled: s.enabled,
      mode: s.mode,
      ...(s.p !== undefined ? { p: s.p } : {}),
      ...(s.note ? { note: s.note } : {}),
      ...(s.exportVisual ? { exportVisual: true } : {}),
    })),
    ...(p.linkedModelId ? { linkedModelId: p.linkedModelId } : {}),
    ...(p.notes ? { notes: p.notes } : {}),
  };
  return JSON.stringify(clean, null, 2) + '\n';
}

const FLAGS: ReadFlag[] = ['COLOR', 'GRAYSCALE', 'UNCHANGED', 'ANYDEPTH'];

/**
 * Parse and migrate a pipeline.
 * - version 0 (draft format): `trainOnly: string[]` list of uids instead of per-step `mode`, `readFlag` string.
 * - version 1: current.
 * Unknown params are dropped, missing params get defaults; unknown steps are kept (flagged by analysis).
 */
export function parsePipeline(raw: unknown): Pipeline {
  if (!raw || typeof raw !== 'object') throw new PipelineFormatError('not a JSON object');
  const o = raw as Record<string, any>;
  const version = o.version ?? 0;
  if (version !== 0 && version !== 1) throw new PipelineFormatError(`unsupported pipeline version ${version}`);
  if (!Array.isArray(o.steps)) throw new PipelineFormatError('missing "steps" array');
  const trainOnly = new Set<string>(version === 0 && Array.isArray(o.trainOnly) ? o.trainOnly : []);
  const steps: PipelineStep[] = o.steps.map((s: any, i: number) => {
    if (!s || typeof s.def !== 'string') throw new PipelineFormatError(`step ${i + 1} has no "def"`);
    const def = getStep(s.def);
    const uid = typeof s.uid === 'string' ? s.uid : newUid();
    const params = def ? { ...defaultParams(s.def), ...Object.fromEntries(Object.entries(s.params ?? {}).filter(([k]) => def.params.some((p) => p.key === k))) } : { ...(s.params ?? {}) };
    let mode: StepMode = s.mode === 'train' || s.mode === 'inference' || s.mode === 'both' ? s.mode : 'both';
    if (version === 0) mode = trainOnly.has(uid) ? 'train' : def?.kind === 'random' ? 'train' : 'both';
    const step: PipelineStep = { uid, def: s.def, params, enabled: s.enabled !== false, mode };
    if (typeof s.p === 'number') step.p = Math.min(1, Math.max(0, s.p));
    else if (def?.kind === 'random') step.p = 0.5;
    if (typeof s.note === 'string') step.note = s.note;
    if (s.exportVisual) step.exportVisual = true;
    return step;
  });
  const flag: ReadFlag = FLAGS.includes(o.read?.flag) ? o.read.flag : FLAGS.includes(o.readFlag) ? o.readFlag : 'COLOR';
  return {
    version: 1,
    id: typeof o.id === 'string' ? o.id : newUid('p'),
    name: typeof o.name === 'string' && o.name.trim() ? o.name : 'Imported pipeline',
    seed: Number.isInteger(o.seed) && o.seed >= 0 ? o.seed : 0,
    read: { flag, exif: o.read?.exif !== false },
    steps,
    ...(typeof o.linkedModelId === 'string' ? { linkedModelId: o.linkedModelId } : {}),
    ...(typeof o.notes === 'string' ? { notes: o.notes } : {}),
  };
}
