// Core types shared by the step registry, analysis, browser runtime, emitters and the UI.

import type { Rng } from './runtime/rng';
import type { Targets } from './runtime/targets';

export type DType = 'uint8' | 'uint16' | 'float32' | 'float16';
export type ColorSpace = 'BGR' | 'RGB' | 'GRAY' | 'HSV' | 'HLS' | 'LAB' | 'LUV' | 'YCrCb' | 'XYZ' | 'BGRA' | 'RGBA';

export interface ImageMeta {
  h: number;
  w: number;
  c: number;
  dtype: DType;
  space: ColorSpace;
  /** Value range the data is expected to occupy (e.g. [0, 255], [0, 1], [-2.1, 2.6]). */
  range: [number, number];
  /** Memory layout; tensor-conversion steps switch to CHW. */
  layout: 'HWC' | 'CHW';
  /** A leading batch dimension was added. */
  batch: boolean;
}

export type Category =
  | 'I/O & colour'
  | 'Geometric'
  | 'Intensity'
  | 'Filtering'
  | 'Denoising'
  | 'Morphology'
  | 'Edges'
  | 'Threshold'
  | 'Frequency'
  | 'Domain'
  | 'Augmentation'
  | 'Normalise & tensor';

export const CATEGORIES: Category[] = [
  'I/O & colour',
  'Geometric',
  'Intensity',
  'Filtering',
  'Denoising',
  'Morphology',
  'Edges',
  'Threshold',
  'Frequency',
  'Domain',
  'Augmentation',
  'Normalise & tensor',
];

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

export type ParamType =
  | 'int'
  | 'oddInt'
  | 'float'
  | 'bool'
  | 'select'
  | 'string'
  /** '#rrggbb'; emitted as a tuple in the image's channel order. */
  | 'color'
  /** [lo, hi] */
  | 'range'
  /** [w, h] with an aspect lock in the UI. */
  | 'size'
  /** Fixed-length number list, e.g. mean (3 values). */
  | 'floats'
  /** [x, y, w, h] in input pixels; drawn on the viewer. */
  | 'rect'
  /** [[x, y], …] in input pixels; dragged / clicked on the viewer. */
  | 'points'
  /** Small editable numeric grid (e.g. a filter2D kernel). */
  | 'matrix';

export interface SelectOption {
  value: string;
  label: string;
}

export interface ParamSpec {
  key: string;
  label: string;
  type: ParamType;
  default: unknown;
  min?: number;
  max?: number;
  step?: number;
  options?: SelectOption[];
  help?: string;
  /** Only show when another param has one of these values. */
  showIf?: { key: string; equals: unknown[] };
  /** For 'points': expected count (4 for perspective corners); undefined = any number. */
  count?: number;
  /** Viewer interaction used to edit this param. */
  tool?: 'rect' | 'points' | 'eyedropper' | 'spectrum-points';
}

export type Params = Record<string, any>;

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

export type Fix =
  | { label: string; kind: 'insertBefore'; def: string; params?: Params }
  | { label: string; kind: 'insertAfter'; def: string; params?: Params }
  | { label: string; kind: 'replace'; def: string; params?: Params }
  | { label: string; kind: 'setParams'; params: Params }
  | { label: string; kind: 'moveToEnd' }
  | { label: string; kind: 'remove' }
  | { label: string; kind: 'setMode'; mode: StepMode }
  | { label: string; kind: 'append'; def: string; params?: Params }
  | { label: string; kind: 'updateModelInput'; c: number; h: number; w: number };

export interface Issue {
  level: 'error' | 'warning' | 'info';
  message: string;
  /** Step the issue belongs to (undefined = pipeline-level). */
  uid?: string;
  fixes?: Fix[];
}

/** What a step accepts. Anything left out is unconstrained. */
export interface Accepts {
  c?: number[];
  dtype?: DType[];
  space?: ColorSpace[];
  /** Tensor-conversion steps accept CHW; everything else needs HWC (default). */
  layout?: ('HWC' | 'CHW')[];
}

// ---------------------------------------------------------------------------
// Emission contexts
// ---------------------------------------------------------------------------

export interface PyCtx {
  in: ImageMeta;
  out: ImageMeta;
  /** 1-based position among exported steps. */
  index: number;
  /** Register a named constant at the top of preprocess.py and return its Python name. */
  c(name: string, value: unknown, comment?: string): string;
  /** Include a shared Python helper (see codegen/helpers.ts). */
  helper(name: string): void;
}

export interface PyEmit {
  /** Function body operating on `img` (and `t`, `rng` for geometric / random steps). `return img` is appended. */
  body: string;
  /** Extra docstring line(s) after the summary. */
  doc?: string;
}

export interface XCtx {
  in: ImageMeta;
  out: ImageMeta;
  /** Python name of this step's function in preprocess.py (for Lambda wrappers). */
  fn: string;
  index: number;
}

// ---------------------------------------------------------------------------
// Browser runtime context
// ---------------------------------------------------------------------------

/* OpenCV.js objects are typed loosely so this package does not depend on the WASM build. */
export type CV = any;
export type Mat = any;

export interface BrowserCtx {
  cv: CV;
  /** Track a Mat (or other deletable) so it is freed after the step. Returns it. */
  keep<T>(m: T): T;
  /** New tracked empty Mat. */
  mat(): Mat;
  in: ImageMeta;
  out: ImageMeta;
  rng: Rng;
  t: Targets;
  train: boolean;
}

// ---------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------

export interface StepDef {
  id: string;
  category: Category;
  label: string;
  /** What it does, when to use it, typical values. */
  doc: string;
  /** The OpenCV / NumPy call(s) it maps to. */
  cv2: string;
  params: ParamSpec[];
  engine: 'browser' | 'server' | 'both';
  kind: 'deterministic' | 'random';
  /** Masks, boxes and keypoints must be transformed too. */
  affectsGeometry: boolean;
  /** Reads masks / boxes (e.g. "Apply mask") without moving them: the Python function receives `t`. */
  usesTargets?: boolean;
  /** Output height / width depend on the pixels (content crop, rectification); outMeta gives an estimate. */
  dynamicSize?: boolean;
  accepts?: Accepts;
  /** Shape / dtype / colour-space after the step, computed without running it. */
  outMeta(input: ImageMeta, p: Params): ImageMeta;
  /** Extra checks (kernel larger than image, thresholds out of order, …). */
  check?(input: ImageMeta, p: Params): Omit<Issue, 'uid'>[];
  /** OpenCV.js functions this step needs; missing ones route the step to the server. */
  needs?: string[];
  applyBrowser?(src: Mat, p: Params, ctx: BrowserCtx): Mat;
  emitPython(p: Params, ctx: PyCtx): PyEmit;
  /** Albumentations transform expression; undefined → wrapped in A.Lambda around the cv2 function. */
  emitAlbumentations?(p: Params, ctx: XCtx): string | undefined;
  /** torchvision.transforms.v2 expression on an RGB tensor; undefined → Cv2Step wrapper. */
  emitTorchvision?(p: Params, ctx: XCtx): string | undefined;
  /** tf.image expression on `img` (float or uint8 HWC tensor); undefined → tf.numpy_function. */
  emitTf?(p: Params, ctx: XCtx): string | undefined;
  /** One-line parameter summary for the pipeline row, e.g. `clip 2.0 · 8×8`. */
  summary?(p: Params): string;
  /** Visualisation-only: excluded from export unless the user ticks "include". */
  visualOnly?: boolean;
  /** Batch-level augmentation (MixUp, CutMix, Mosaic): not part of the per-image function; dataset.py applies it. */
  batch?: { emitBatchPython(p: Params): string };
  /** Default train/inference mode when added. */
  defaultMode?: StepMode;
  /** Max abs difference allowed between browser and Python outputs (0 = exact). */
  tolerance?: number;
  /** Prefix for generated constant names (defaults to the id in upper case). */
  constPrefix?: string;
  /** Hidden from the library (internal / deprecated). */
  hidden?: boolean;
  /** Search keywords. */
  keywords?: string;
  /** Eyedropper: params to set when the user clicks a pixel (values in the step's input space). */
  onPick?(px: number[], input: ImageMeta, p: Params): Params;
  /** Parity-test cases (browser vs generated Python). Default: one case with default params. */
  testCases?: TestCase[];
}

export interface TestCase {
  params?: Params;
  /** Test image: 'color' BGR uint8, 'gray', 'u16' gray uint16, 'doc' document-like gray, 'float' BGR float32 [0,1], 'rgba'. */
  input?: 'color' | 'gray' | 'u16' | 'doc' | 'float' | 'rgba';
  /** Steps (def + params) inserted before this one, e.g. a conversion it needs. */
  before?: { def: string; params?: Params }[];
  p?: number;
  train?: boolean;
  seed?: number;
  /** Override the step's tolerance for this case. */
  tolerance?: number;
  /** Also run browser targets (mask/boxes/keypoints) and compare with Python Targets. */
  targets?: boolean;
}

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

export type StepMode = 'both' | 'train' | 'inference';

export interface PipelineStep {
  uid: string;
  def: string;
  params: Params;
  enabled: boolean;
  /** Probability for random steps (default 1 for deterministic, 0.5 for most augmentations). */
  p?: number;
  mode: StepMode;
  note?: string;
  /** Visualisation-only steps: include in the export anyway. */
  exportVisual?: boolean;
}

export type ReadFlag = 'COLOR' | 'GRAYSCALE' | 'UNCHANGED' | 'ANYDEPTH';

export interface ReadSpec {
  flag: ReadFlag;
  /** Apply EXIF orientation (cv2.imread does this for COLOR/GRAYSCALE unless IMREAD_IGNORE_ORIENTATION). */
  exif: boolean;
}

export interface Pipeline {
  version: 1;
  id: string;
  name: string;
  seed: number;
  read: ReadSpec;
  steps: PipelineStep[];
  /** Image-input node id of the Modelith model this pipeline feeds. */
  linkedModelId?: string;
  notes?: string;
}

export interface GeneratedFile {
  path: string;
  content: string;
  /** 1-based line number → step uid (two-way code ↔ pipeline linking). */
  lineMap?: Record<number, string>;
  language: 'python' | 'json' | 'markdown' | 'text';
}
