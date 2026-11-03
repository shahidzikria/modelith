// Param factories used by step files.
import type { ParamSpec, SelectOption } from './types';

type Extra = Partial<Omit<ParamSpec, 'key' | 'label' | 'type' | 'default'>>;

export const int = (key: string, label: string, def: number, extra: Extra = {}): ParamSpec => ({ key, label, type: 'int', default: def, step: 1, ...extra });
/** Odd integer ≥ 1 (kernel sizes); the UI only offers 1, 3, 5, … */
export const oddInt = (key: string, label: string, def: number, extra: Extra = {}): ParamSpec => ({ key, label, type: 'oddInt', default: def, min: 1, max: 31, step: 2, ...extra });
export const float = (key: string, label: string, def: number, extra: Extra = {}): ParamSpec => ({ key, label, type: 'float', default: def, ...extra });
export const bool = (key: string, label: string, def: boolean, extra: Extra = {}): ParamSpec => ({ key, label, type: 'bool', default: def, ...extra });
export const str = (key: string, label: string, def: string, extra: Extra = {}): ParamSpec => ({ key, label, type: 'string', default: def, ...extra });
export const color = (key: string, label: string, def: string, extra: Extra = {}): ParamSpec => ({ key, label, type: 'color', default: def, ...extra });
export const range = (key: string, label: string, def: [number, number], extra: Extra = {}): ParamSpec => ({ key, label, type: 'range', default: def, ...extra });
export const size = (key: string, label: string, def: [number, number], extra: Extra = {}): ParamSpec => ({ key, label, type: 'size', default: def, min: 1, max: 8192, ...extra });
export const floats = (key: string, label: string, def: number[], extra: Extra = {}): ParamSpec => ({ key, label, type: 'floats', default: def, ...extra });
export const rect = (key: string, label: string, def: [number, number, number, number], extra: Extra = {}): ParamSpec => ({ key, label, type: 'rect', default: def, tool: 'rect', ...extra });
export const points = (key: string, label: string, def: number[][], extra: Extra = {}): ParamSpec => ({ key, label, type: 'points', default: def, tool: 'points', ...extra });
export const matrix = (key: string, label: string, def: number[][], extra: Extra = {}): ParamSpec => ({ key, label, type: 'matrix', default: def, ...extra });

/** Select from `[value, label]` pairs or plain values. */
export function select(key: string, label: string, options: (string | [string, string])[], def: string, extra: Extra = {}): ParamSpec {
  const opts: SelectOption[] = options.map((o) => (Array.isArray(o) ? { value: o[0], label: o[1] } : { value: o, label: o }));
  return { key, label, type: 'select', options: opts, default: def, ...extra };
}

export const INTERPOLATIONS: [string, string][] = [
  ['NEAREST', 'Nearest'],
  ['LINEAR', 'Linear'],
  ['AREA', 'Area (best for shrinking)'],
  ['CUBIC', 'Cubic'],
  ['LANCZOS4', 'Lanczos-4'],
];

export const interp = (def = 'LINEAR', key = 'interp') =>
  select(key, 'Interpolation', INTERPOLATIONS, def, { help: 'AREA for shrinking, LINEAR or CUBIC for enlarging. Masks always use NEAREST.' });

export const BORDERS: [string, string][] = [
  ['CONSTANT', 'Constant (colour)'],
  ['REFLECT', 'Reflect (abc|cba)'],
  ['REFLECT_101', 'Reflect 101 (abc|ba)'],
  ['REPLICATE', 'Replicate (abc|cc)'],
  ['WRAP', 'Wrap (abc|ab)'],
];

export const border = (def = 'CONSTANT', key = 'border') => select(key, 'Border', BORDERS, def);

