// Version history (§12.5): a step-level diff between two versions of a pipeline.
import { getStep } from '../registry';
import type { Pipeline } from '../types';

export type DiffEntry =
  | { kind: 'added'; uid: string; label: string; index: number }
  | { kind: 'removed'; uid: string; label: string; index: number }
  | { kind: 'moved'; uid: string; label: string; from: number; to: number }
  | { kind: 'params'; uid: string; label: string; changes: { key: string; from: unknown; to: unknown }[] }
  | { kind: 'toggled'; uid: string; label: string; enabled: boolean }
  | { kind: 'mode'; uid: string; label: string; from: string; to: string }
  | { kind: 'pipeline'; field: string; from: unknown; to: unknown };

const label = (def: string) => getStep(def)?.label ?? def;

export function diffPipelines(a: Pipeline, b: Pipeline): DiffEntry[] {
  const out: DiffEntry[] = [];
  for (const field of ['name', 'seed'] as const) if (a[field] !== b[field]) out.push({ kind: 'pipeline', field, from: a[field], to: b[field] });
  if (a.read.flag !== b.read.flag) out.push({ kind: 'pipeline', field: 'read', from: a.read.flag, to: b.read.flag });
  const ia = new Map(a.steps.map((s, i) => [s.uid, i]));
  const ib = new Map(b.steps.map((s, i) => [s.uid, i]));
  a.steps.forEach((s, i) => !ib.has(s.uid) && out.push({ kind: 'removed', uid: s.uid, label: label(s.def), index: i }));
  b.steps.forEach((s, i) => !ia.has(s.uid) && out.push({ kind: 'added', uid: s.uid, label: label(s.def), index: i }));
  // Steps present in both: the longest run that kept its relative order stayed put; the rest moved.
  const common = b.steps.filter((s) => ia.has(s.uid));
  const stay = new Set(longestIncreasing(common.map((s) => ia.get(s.uid)!)).map((k) => common[k].uid));
  for (const s of common) if (!stay.has(s.uid)) out.push({ kind: 'moved', uid: s.uid, label: label(s.def), from: ia.get(s.uid)!, to: ib.get(s.uid)! });
  for (const s of common) {
    const o = a.steps[ia.get(s.uid)!];
    const changes = Object.keys({ ...o.params, ...s.params })
      .filter((k) => JSON.stringify(o.params[k]) !== JSON.stringify(s.params[k]))
      .map((key) => ({ key, from: o.params[key], to: s.params[key] }));
    if (o.p !== s.p) changes.push({ key: 'p', from: o.p, to: s.p });
    if (changes.length) out.push({ kind: 'params', uid: s.uid, label: label(s.def), changes });
    if (o.enabled !== s.enabled) out.push({ kind: 'toggled', uid: s.uid, label: label(s.def), enabled: s.enabled });
    if (o.mode !== s.mode) out.push({ kind: 'mode', uid: s.uid, label: label(s.def), from: o.mode, to: s.mode });
  }
  return out;
}

export function describeDiff(d: DiffEntry): string {
  const v = (x: unknown) => (typeof x === 'string' ? x : JSON.stringify(x));
  switch (d.kind) {
    case 'added':
      return `+ ${d.label} (step ${d.index + 1})`;
    case 'removed':
      return `− ${d.label} (was step ${d.index + 1})`;
    case 'moved':
      return `↕ ${d.label}: step ${d.from + 1} → ${d.to + 1}`;
    case 'params':
      return `✎ ${d.label}: ${d.changes.map((c) => `${c.key} ${v(c.from)} → ${v(c.to)}`).join(', ')}`;
    case 'toggled':
      return `${d.enabled ? '● enabled' : '○ disabled'} ${d.label}`;
    case 'mode':
      return `⇄ ${d.label}: ${d.from} → ${d.to}`;
    case 'pipeline':
      return `✎ ${d.field}: ${v(d.from)} → ${v(d.to)}`;
  }
}

/** Indices of one longest strictly increasing subsequence (O(n²); pipelines are short). */
function longestIncreasing(v: number[]): number[] {
  const len = v.map(() => 1);
  const prev = v.map(() => -1);
  for (let i = 0; i < v.length; i++)
    for (let j = 0; j < i; j++)
      if (v[j] < v[i] && len[j] + 1 > len[i]) {
        len[i] = len[j] + 1;
        prev[i] = j;
      }
  let end = len.indexOf(Math.max(0, ...len));
  const out: number[] = [];
  while (end >= 0) {
    out.unshift(end);
    end = prev[end];
  }
  return out;
}
