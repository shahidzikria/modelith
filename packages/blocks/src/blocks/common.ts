import type { ParamSpec, Shape } from '../types';
import { py } from '../py';

// ---------------------------------------------------------------------------
// Param factories
// ---------------------------------------------------------------------------

type Extra = Partial<Omit<ParamSpec, 'key' | 'label' | 'type' | 'default'>>;

export const int = (key: string, label: string, def: number, extra: Extra = {}): ParamSpec => ({
  key,
  label,
  type: 'int',
  default: def,
  min: 0,
  ...extra,
});
export const float = (key: string, label: string, def: number, extra: Extra = {}): ParamSpec => ({
  key,
  label,
  type: 'float',
  default: def,
  ...extra,
});
export const bool = (key: string, label: string, def: boolean, extra: Extra = {}): ParamSpec => ({
  key,
  label,
  type: 'bool',
  default: def,
  ...extra,
});
export const select = (key: string, label: string, options: string[], def: string, extra: Extra = {}): ParamSpec => ({
  key,
  label,
  type: 'select',
  options,
  default: def,
  ...extra,
});
export const intList = (key: string, label: string, def: number[], extra: Extra = {}): ParamSpec => ({
  key,
  label,
  type: 'intList',
  default: def,
  ...extra,
});
export const str = (key: string, label: string, def: string, extra: Extra = {}): ParamSpec => ({
  key,
  label,
  type: 'string',
  default: def,
  ...extra,
});

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

export const ACTS = ['relu', 'relu6', 'silu', 'gelu', 'leaky_relu', 'hardswish', 'mish', 'sigmoid', 'tanh', 'none'];

export const actParam = (def = 'relu', key = 'act'): ParamSpec =>
  select(key, 'Activation', ACTS, def, { help: 'Non-linearity applied after normalisation.' });

export function torchAct(name: string): string {
  switch (name) {
    case 'relu':
      return 'nn.ReLU(inplace=True)';
    case 'relu6':
      return 'nn.ReLU6(inplace=True)';
    case 'silu':
      return 'nn.SiLU(inplace=True)';
    case 'gelu':
      return 'nn.GELU()';
    case 'leaky_relu':
      return 'nn.LeakyReLU(0.1, inplace=True)';
    case 'hardswish':
      return 'nn.Hardswish(inplace=True)';
    case 'mish':
      return 'nn.Mish(inplace=True)';
    case 'sigmoid':
      return 'nn.Sigmoid()';
    case 'tanh':
      return 'nn.Tanh()';
    default:
      return 'nn.Identity()';
  }
}

/** Keras activation layer expression (without the call). */
export function kerasAct(name: string, layerName: string): string {
  const n = `name=${py(layerName)}`;
  switch (name) {
    case 'leaky_relu':
      return `layers.LeakyReLU(negative_slope=0.1, ${n})`;
    case 'relu6':
      return `layers.ReLU(max_value=6.0, ${n})`;
    case 'hardswish':
      return `layers.Activation("hard_swish", ${n})`;
    case 'none':
      return `layers.Identity(${n})`;
    default:
      return `layers.Activation(${py(name)}, ${n})`;
  }
}

/** Keras activation name string used by helper functions. */
export function kerasActName(name: string): string {
  return name === 'hardswish' ? 'hard_swish' : name;
}

// ---------------------------------------------------------------------------
// Param-count formulas (A6)
// ---------------------------------------------------------------------------

export const convParams = (cin: number, cout: number, k: number, groups = 1, bias = false): number =>
  (cin / groups) * cout * k * k + (bias ? cout : 0);

/** Conv (no bias) + BatchNorm affine params. */
export const cbaParams = (cin: number, cout: number, k: number, groups = 1): number => convParams(cin, cout, k, groups) + 2 * cout;

export const linearParams = (din: number, dout: number, bias = true): number => din * dout + (bias ? dout : 0);

export const transformerLayerParams = (d: number, mlp: number): number =>
  4 * d * d + 4 * d + 2 * d * mlp + mlp + d + 4 * d;

export const lstmParams = (din: number, h: number, gates = 4): number => gates * (h * (din + h) + 2 * h);

export function inputImageSize(ins: Shape[]): { h: number; w: number } | undefined {
  const s = ins[0];
  if (s?.kind === 'image') return { h: Math.round(s.h * s.stride), w: Math.round(s.w * s.stride) };
  return undefined;
}
