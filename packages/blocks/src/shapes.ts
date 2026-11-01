import {
  ShapeError,
  type Framework,
  type ImageShape,
  type Level,
  type PyramidShape,
  type SequenceShape,
  type Shape,
  type ShapeKind,
  type TokensShape,
  type VectorShape,
} from './types';

// ---------------------------------------------------------------------------
// Formulas
// ---------------------------------------------------------------------------

/** PyTorch padding used for `padding: "same"` by the generated code. */
export function torchSamePad(k: number, d = 1): number {
  return Math.floor((d * (k - 1)) / 2);
}

/**
 * Output size of a convolution / pooling window along one axis.
 * PyTorch: floor((h + 2p − d(k−1) − 1)/s) + 1 with p = d(k−1)//2 for "same".
 * Keras "same": ceil(h / s) regardless of kernel size; "valid": ceil((h − d(k−1)) / s).
 */
export function convOut(
  h: number,
  k: number,
  s: number,
  padding: 'same' | 'valid',
  d: number,
  fw: Framework,
): number {
  if (fw === 'tensorflow') {
    return padding === 'same' ? Math.ceil(h / s) : Math.ceil((h - d * (k - 1)) / s);
  }
  const p = padding === 'same' ? torchSamePad(k, d) : 0;
  return Math.floor((h + 2 * p - d * (k - 1) - 1) / s) + 1;
}

/** ConvTranspose: (h−1)s − 2p + d(k−1) + output_padding + 1. */
export function convTransposeOut(h: number, k: number, s: number, p: number, op: number, d = 1): number {
  return (h - 1) * s - 2 * p + d * (k - 1) + op + 1;
}

// ---------------------------------------------------------------------------
// Assertions
// ---------------------------------------------------------------------------

const KIND_NAMES: Record<ShapeKind, string> = {
  image: 'a feature map',
  sequence: 'a sequence',
  vector: 'a vector',
  tokens: 'token ids',
  pyramid: 'a feature pyramid',
  video: 'a video tensor',
  output: 'a task output',
};

export function kindName(k: ShapeKind): string {
  return KIND_NAMES[k];
}

export function expectKind<K extends ShapeKind>(s: Shape | undefined, kind: K, what = 'Input'): Extract<Shape, { kind: K }> {
  if (!s) throw new ShapeError(`${what} is not connected.`);
  if (s.kind !== kind) {
    throw new ShapeError(`${what} must be ${kindName(kind)}, got ${kindName(s.kind)}.`, fixFor(s.kind, kind));
  }
  return s as Extract<Shape, { kind: K }>;
}

/** How to turn one tensor kind into another, if there is a standard block for it. */
export function fixFor(have: ShapeKind, want: ShapeKind): string | undefined {
  if (have === 'image' && want === 'vector') return 'Add Global average pool or Flatten first.';
  if (have === 'image' && want === 'sequence') return 'Add Patch embed or Image → sequence first.';
  if (have === 'sequence' && want === 'vector') return 'Add Sequence pooling first.';
  if (have === 'sequence' && want === 'image') return 'Add Sequence → image first.';
  if (have === 'tokens') return 'Add an Embedding layer first.';
  if (have === 'pyramid' && want === 'image') return 'Add Select level to pick one pyramid level.';
  if (have === 'image' && want === 'pyramid') return 'Bundle three feature maps with Make pyramid, or use a backbone in pyramid mode.';
  return undefined;
}

export const img = (s: Shape | undefined, what?: string): ImageShape => expectKind(s, 'image', what);
export const seq = (s: Shape | undefined, what?: string): SequenceShape => expectKind(s, 'sequence', what);
export const vec = (s: Shape | undefined, what?: string): VectorShape => expectKind(s, 'vector', what);
export const tok = (s: Shape | undefined, what?: string): TokensShape => expectKind(s, 'tokens', what);
export const pyr = (s: Shape | undefined, what?: string): PyramidShape => expectKind(s, 'pyramid', what);

/** Feature dimension of the last axis (channels for images). */
export function featureDim(s: Shape): number {
  switch (s.kind) {
    case 'image':
      return s.c;
    case 'sequence':
      return s.d;
    case 'vector':
      return s.d;
    default:
      throw new ShapeError(`Expected a feature map, sequence or vector, got ${kindName(s.kind)}.`);
  }
}

export function withFeature(s: Shape, d: number): Shape {
  switch (s.kind) {
    case 'image':
      return { ...s, c: d };
    case 'sequence':
      return { ...s, d };
    case 'vector':
      return { ...s, d };
    default:
      return s;
  }
}

export function positions(s: Shape): number {
  switch (s.kind) {
    case 'image':
      return s.h * s.w;
    case 'sequence':
      return s.l;
    case 'tokens':
      return s.l;
    case 'video':
      return s.t * s.h * s.w;
    case 'pyramid':
      return s.levels.reduce((a, l) => a + l.h * l.w, 0);
    default:
      return 1;
  }
}

export function numel(s: Shape): number {
  switch (s.kind) {
    case 'image':
      return s.c * s.h * s.w;
    case 'sequence':
      return s.l * s.d;
    case 'vector':
      return s.d;
    case 'tokens':
      return s.l;
    case 'video':
      return s.t * s.c * s.h * s.w;
    case 'pyramid':
      return s.levels.reduce((a, l) => a + l.c * l.h * l.w, 0);
    case 'output':
      return s.dims.reduce((a, b) => a * b, 1);
  }
}

export function sameShape(a: Shape, b: Shape): boolean {
  return JSON.stringify(stripMeta(a)) === JSON.stringify(stripMeta(b));
}

function stripMeta(s: Shape): unknown {
  if (s.kind === 'image') return [s.c, s.h, s.w];
  if (s.kind === 'sequence') return [s.l, s.d];
  if (s.kind === 'vector') return [s.d];
  if (s.kind === 'pyramid') return s.levels.map((l) => [l.c, l.h, l.w]);
  return s;
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

export function formatShape(s: Shape): string {
  switch (s.kind) {
    case 'image':
      return `${s.c}×${s.h}×${s.w}`;
    case 'sequence':
      return `${s.l}×${s.d}`;
    case 'vector':
      return `${s.d}`;
    case 'tokens':
      return `${s.l} tokens`;
    case 'video':
      return `${s.t}×${s.c}×${s.h}×${s.w}`;
    case 'pyramid': {
      const lv = s.levels.map((l) => Math.round(Math.log2(l.stride)));
      const range = lv.length > 1 ? `P${lv[0]}–P${lv[lv.length - 1]}` : `P${lv[0]}`;
      return `${range} · ${s.levels.map((l) => l.c).join('/')}`;
    }
    case 'output':
      return s.label;
  }
}

export function formatCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return `${n}`;
}

export function pyramidOf(levels: Level[]): PyramidShape {
  return { kind: 'pyramid', levels };
}

/** Python tuple literal for the shape with an explicit batch dimension, framework layout. */
export function pyShape(s: Shape, fw: Framework, batch: string | number = 'B'): string {
  const b = String(batch);
  switch (s.kind) {
    case 'image':
      return fw === 'pytorch' ? `(${b}, ${s.c}, ${s.h}, ${s.w})` : `(${b}, ${s.h}, ${s.w}, ${s.c})`;
    case 'sequence':
      return `(${b}, ${s.l}, ${s.d})`;
    case 'vector':
      return `(${b}, ${s.d})`;
    case 'tokens':
      return `(${b}, ${s.l})`;
    case 'video':
      return fw === 'pytorch' ? `(${b}, ${s.c}, ${s.t}, ${s.h}, ${s.w})` : `(${b}, ${s.t}, ${s.h}, ${s.w}, ${s.c})`;
    default:
      return `(${b},)`;
  }
}
