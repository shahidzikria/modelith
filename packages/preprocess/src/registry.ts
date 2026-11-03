import type { Params, StepDef } from './types';

const steps = new Map<string, StepDef>();

/** Register a step. Step files call this at import time; `steps/index.ts` imports them all. */
export function defineStep(def: StepDef): StepDef {
  if (steps.has(def.id)) throw new Error(`Duplicate step id: ${def.id}`);
  steps.set(def.id, def);
  return def;
}

export function getStep(id: string): StepDef | undefined {
  return steps.get(id);
}

export function requireStep(id: string): StepDef {
  const s = steps.get(id);
  if (!s) throw new Error(`Unknown step: ${id}`);
  return s;
}

export function allSteps(): StepDef[] {
  return [...steps.values()];
}

/** Default param values for a step. */
export function defaultParams(id: string): Params {
  return Object.fromEntries(requireStep(id).params.map((p) => [p.key, structuredClone(p.default)]));
}
