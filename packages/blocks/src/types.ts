// Core types shared by the registry, shape inference, emitters and the UI.

export type Framework = 'pytorch' | 'tensorflow';

export type Category =
  | 'Inputs'
  | 'Layers'
  | 'CNN blocks'
  | 'Seq & attn'
  | 'Backbones'
  | 'Necks/merge'
  | 'Heads'
  | 'Detectors'
  | 'My blocks';

export const CATEGORIES: Category[] = [
  'Inputs',
  'Layers',
  'CNN blocks',
  'Seq & attn',
  'Backbones',
  'Necks/merge',
  'Heads',
  'Detectors',
  'My blocks',
];

// ---------------------------------------------------------------------------
// Tensor shapes (batch dimension is always implicit)
// ---------------------------------------------------------------------------

export interface Level {
  c: number;
  h: number;
  w: number;
  stride: number;
}

export type TaskKind =
  | 'classification'
  | 'multilabel'
  | 'token'
  | 'segmentation'
  | 'dense'
  | 'detection'
  | 'instance'
  | 'embedding'
  | 'regression'
  | 'contrastive';

export type Shape =
  | { kind: 'image'; c: number; h: number; w: number; stride: number }
  | { kind: 'sequence'; l: number; d: number; grid?: { h: number; w: number; stride: number }; prefix?: number }
  | { kind: 'vector'; d: number }
  | { kind: 'tokens'; l: number; vocab: number }
  | { kind: 'pyramid'; levels: Level[] }
  | { kind: 'video'; t: number; c: number; h: number; w: number }
  | { kind: 'output'; task: TaskKind; label: string; dims: number[] };

export type ShapeKind = Shape['kind'];

export type ImageShape = Extract<Shape, { kind: 'image' }>;
export type SequenceShape = Extract<Shape, { kind: 'sequence' }>;
export type VectorShape = Extract<Shape, { kind: 'vector' }>;
export type TokensShape = Extract<Shape, { kind: 'tokens' }>;
export type PyramidShape = Extract<Shape, { kind: 'pyramid' }>;
export type OutputShape = Extract<Shape, { kind: 'output' }>;

export class ShapeError extends Error {
  fix?: string;
  constructor(message: string, fix?: string) {
    super(message);
    this.name = 'ShapeError';
    this.fix = fix;
  }
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

export type ParamType = 'int' | 'float' | 'bool' | 'select' | 'intList' | 'string';

export interface ParamSpec {
  key: string;
  label: string;
  type: ParamType;
  default: unknown;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  help?: string;
  /** Only show when another param has a given value. */
  showIf?: { key: string; equals: unknown[] };
}

export type Params = Record<string, any>;

// ---------------------------------------------------------------------------
// Emission contexts
// ---------------------------------------------------------------------------

export interface InferCtx {
  framework: Framework;
  /** Height × width of the first image input (task heads upsample to it). */
  imageSize?: { h: number; w: number };
}

export interface EmitCtx {
  framework: Framework;
  /** Python variable / attribute name of this node, e.g. `c2f_2`. */
  name: string;
  /** Node id, used for `# node:` back-links. */
  id: string;
  params: Params;
  ins: Shape[];
  out: Shape;
  /** Name of the declared input (for input nodes). */
  inputNames: string[];
  imageSize?: { h: number; w: number };
}

export interface TorchEmit {
  /** Constructor expression assigned to `self.<name>`. */
  init?: string;
  /** Forward expression. `self` is the module attribute (`self.<name>`) when `init` is set. */
  forward?: (args: string[], self: string) => string;
  helpers?: string[];
  imports?: string[];
}

export interface KerasEmit {
  /** Functional-API expression producing this node's output tensor. */
  expr: (args: string[]) => string;
  helpers?: string[];
  imports?: string[];
}

export interface Unsupported {
  unsupported: string;
}

export function isUnsupported(x: unknown): x is Unsupported {
  return typeof x === 'object' && x !== null && 'unsupported' in x;
}

// ---------------------------------------------------------------------------
// Heads: how train.py builds targets, losses and metrics
// ---------------------------------------------------------------------------

export interface HeadTrainSpec {
  task: TaskKind;
  /** Python expression building the loss module (PyTorch). */
  torchLoss: string;
  /** Python expression producing ONE synthetic target sample. Uses `CONFIG`. */
  torchTarget: string;
  /** Metric family for evaluation. */
  metric: 'accuracy' | 'multilabel' | 'miou' | 'map' | 'rmse' | 'depth' | 'loss';
  /** Keras loss expression (when compile/fit is possible). */
  kerasLoss?: string;
  kerasTarget?: string;
  kerasMetrics?: string;
  /** Torchvision detectors compute losses inside the model. */
  internalLoss?: boolean;
  /** Real-data format documentation for the dataset docstring. */
  targetDoc: string;
  /** Optional post-processing function name (defined in helpers). */
  postprocess?: string;
  /** Extra keyword arguments for the post-processing call during evaluation. */
  postprocessArgs?: string;
}

// ---------------------------------------------------------------------------
// Blocks
// ---------------------------------------------------------------------------

export interface PortDef {
  name: string;
  accepts: ShapeKind[];
  optional?: boolean;
}

export interface BlockDef {
  type: string;
  category: Category;
  label: string;
  /** Short description shown in palette tooltips and the inspector. */
  doc: string;
  paper?: string;
  /** Prefix for generated variable names. */
  varName: string;
  inputs: PortDef[];
  /** Accept any number of inputs on a single port (Concat, Add…). */
  variadic?: boolean;
  params: ParamSpec[];
  inferShape(ins: Shape[], p: Params, ctx: InferCtx): Shape;
  paramCount?(ins: Shape[], p: Params, out: Shape): number;
  /** Multiply-accumulates at the declared input size. */
  macs?(ins: Shape[], p: Params, out: Shape, params: number): number;
  emitTorch(ctx: EmitCtx): TorchEmit | Unsupported;
  emitKeras(ctx: EmitCtx): KerasEmit | Unsupported;
  head?: (p: Params, ins: Shape[], out: Shape, ctx: InferCtx) => HeadTrainSpec;
  /** torchvision detectors: whole upstream graph becomes the backbone. */
  wrapsGraph?: boolean;
  /** Approximate train-mode activation bytes multiplier (default 1 × output size). */
  isInput?: boolean;
  /** Uses pretrained weights (affects LR and validation). */
  pretrained?: (p: Params) => boolean;
  /** Extra pip requirements. */
  requires?: string[];
}

// ---------------------------------------------------------------------------
// Graph / project
// ---------------------------------------------------------------------------

export interface GraphNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  params: Params;
  label?: string;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  targetPort: number;
}

export interface CustomBlock {
  type: string; // `custom:<slug>`
  label: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  /** Node ids inside the block that receive the block's inputs, in port order. */
  inputs: { node: string; port: number }[];
  output: string;
  repeat?: number;
}

export interface Project {
  version: 2;
  name: string;
  framework: Framework;
  nodes: GraphNode[];
  edges: GraphEdge[];
  customBlocks: CustomBlock[];
  notes?: string;
}
