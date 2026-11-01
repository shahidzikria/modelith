// Recommendations for the editor: what a block would output if appended after a node, which
// blocks commonly come next, and parameter values that keep the whole graph valid.
import { allBlocks, defaultParams, getBlock } from '../registry';
import { fixFor, kindName } from '../shapes';
import { ShapeError, type Params, type Shape, type ShapeKind } from '../types';
import { analyze, type Analysis, type GraphLike } from './analyze';

export interface AppendPreview {
  /** False when the block can't take this output (wrong kind or a shape error). */
  ok: boolean;
  shape?: Shape;
  params?: number;
  reason?: string;
  fix?: string;
  /** Block needs more inputs than the one being connected (merges, detectors with two ports). */
  needsMore?: boolean;
}

function imageSizeOf(g: GraphLike): { h: number; w: number } | undefined {
  const n = g.nodes.find((x) => x.type === 'image_input');
  return n ? { h: Number(n.params.height), w: Number(n.params.width) } : undefined;
}

/** Preview block `type` (default params) connected to an output of shape `from`. */
export function previewBlock(type: string, from: Shape, g: GraphLike, params?: Params): AppendPreview {
  const def = getBlock(type);
  if (!def || def.isInput || !def.inputs.length) return { ok: false, reason: 'Input blocks have no input port.' };
  if (!def.inputs[0].accepts.includes(from.kind)) {
    const fix = def.inputs[0].accepts.map((k) => fixFor(from.kind, k)).find(Boolean);
    return { ok: false, reason: `Needs ${def.inputs[0].accepts.map(kindName).join(' or ')}, got ${kindName(from.kind)}.`, fix };
  }
  const required = def.variadic ? 2 : def.inputs.filter((p) => !p.optional).length;
  if (required > 1) return { ok: true, needsMore: true, reason: `Needs ${required} inputs.` };
  const p = { ...defaultParams(type), ...params };
  try {
    const shape = def.inferShape([from], p, { framework: g.framework, imageSize: imageSizeOf(g) });
    return { ok: true, shape, params: Math.round(def.paramCount?.([from], p, shape) ?? 0) };
  } catch (err) {
    if (err instanceof ShapeError) return { ok: false, reason: err.message, fix: err.fix };
    return { ok: false, reason: (err as Error).message };
  }
}

/** Preview every non-input block appended after node `fromId`. Empty when that node has no shape yet. */
export function previewAppends(g: GraphLike, analysis: Analysis, fromId: string): Record<string, AppendPreview> {
  const from = analysis.nodes[fromId]?.shape;
  const out: Record<string, AppendPreview> = {};
  if (!from || from.kind === 'output') return out;
  for (const b of allBlocks()) if (!b.isInput) out[b.type] = previewBlock(b.type, from, g);
  return out;
}

/** Blocks that usually follow an output of each kind, most common first. */
const NEXT: Record<ShapeKind, string[]> = {
  image: ['conv_bn_act', 'maxpool', 'resnet_basic', 'mbconv', 'timm_backbone', 'gap', 'cls_head', 'seg_head', 'image_to_seq', 'make_pyramid'],
  sequence: ['transformer_encoder', 'layernorm', 'seq_pool', 'token_head', 'rnn', 'linear', 'seq_to_image'],
  vector: ['linear', 'dropout', 'activation', 'cls_head', 'regression_head', 'embedding_head'],
  tokens: ['embedding'],
  pyramid: ['fpn', 'pan', 'yolo_head', 'select_level', 'retinanet', 'fcos'],
  video: [],
  output: [],
};

export interface Suggestion extends AppendPreview {
  type: string;
}

/** Up to `limit` valid next blocks for node `fromId`, each with its predicted output. */
export function suggestNext(g: GraphLike, analysis: Analysis, fromId: string, limit = 6): Suggestion[] {
  const from = analysis.nodes[fromId]?.shape;
  if (!from) return [];
  return NEXT[from.kind]
    .filter((t) => getBlock(t))
    .map((type) => ({ type, ...previewBlock(type, from, g) }))
    .filter((s) => s.ok && !s.needsMore)
    .slice(0, limit);
}

/** Largest downsampling factor among image / pyramid shapes: input H and W should be multiples of it. */
export function requiredMultiple(analysis: Analysis): number {
  let m = 1;
  for (const ni of Object.values(analysis.nodes)) {
    const s = ni.shape;
    if (s?.kind === 'image') m = Math.max(m, s.stride);
    else if (s?.kind === 'pyramid') for (const l of s.levels) m = Math.max(m, l.stride);
    else if (s?.kind === 'sequence' && s.grid) m = Math.max(m, s.grid.stride);
  }
  return m;
}

const SIZES = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 384, 416, 448, 512, 640, 768, 1024];
const CHANNELS = [16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048];

function errorIds(a: Analysis): Set<string> {
  return new Set(Object.entries(a.nodes).filter(([, ni]) => ni.error && !ni.waiting).map(([id]) => id));
}

/** Width of the tensor feeding a node (channels / embedding size). */
function inputWidth(s: Shape | undefined): number | undefined {
  if (!s) return undefined;
  if (s.kind === 'image' || s.kind === 'video') return s.c;
  if (s.kind === 'sequence' || s.kind === 'vector') return s.d;
  return undefined;
}

function candidates(key: string, type: string, current: number, inW: number | undefined): number[] | undefined {
  if (type === 'image_input' && (key === 'height' || key === 'width')) return SIZES;
  switch (key) {
    case 'out':
    case 'dim':
    case 'channels':
    case 'mid':
    case 'features': {
      if (type === 'tabular_input') return undefined;
      const base = inW ? [inW / 2, inW, inW * 2, inW * 4] : [];
      return [...base, ...CHANNELS.filter((c) => c >= current / 4 && c <= current * 4)];
    }
    case 'hidden':
    case 'ffn':
      return inW ? [inW, inW * 2, inW * 4, ...CHANNELS] : CHANNELS;
    case 'kernel':
      return type.endsWith('pool') ? [2, 3, 5] : [1, 3, 5, 7];
    case 'stride':
      return [1, 2];
    case 'heads':
      return [1, 2, 4, 6, 8, 12, 16];
    case 'patch':
      return [4, 8, 14, 16, 32];
    default:
      return undefined;
  }
}

export interface ParamSuggestion {
  value: number;
  /** Matches the node's current value. */
  current: boolean;
}

/**
 * Values for integer param `key` of node `nodeId` that introduce no new errors anywhere in the
 * graph (e.g. image sizes divisible by the network stride, heads that divide the model dim).
 * Returns undefined for params we have no sensible candidates for.
 */
export function suggestParamValues(g: GraphLike, nodeId: string, key: string, limit = 8): ParamSuggestion[] | undefined {
  const node = g.nodes.find((n) => n.id === nodeId);
  const def = node && getBlock(node.type);
  const spec = def?.params.find((p) => p.key === key);
  if (!node || !spec || spec.type !== 'int') return undefined;
  const base = analyze(g);
  const current = Number(node.params[key]);
  const raw = candidates(key, node.type, current, inputWidth(base.nodes[nodeId]?.ins[0]));
  if (!raw) return undefined;
  const before = errorIds(base);
  const values = [...new Set(raw.map(Math.round))]
    .filter((v) => Number.isFinite(v) && v >= (spec.min ?? 1) && v <= (spec.max ?? Infinity))
    .sort((a, b) => a - b);
  const ok = values.filter((v) => {
    if (v === current) return !before.has(nodeId);
    const trial = analyze({ ...g, nodes: g.nodes.map((n) => (n.id === nodeId ? { ...n, params: { ...n.params, [key]: v } } : n)) });
    for (const id of errorIds(trial)) if (!before.has(id)) return false;
    return !trial.nodes[nodeId]?.error;
  });
  // Keep the values closest to the current one.
  const picked = [...ok].sort((a, b) => Math.abs(Math.log(a / current)) - Math.abs(Math.log(b / current))).slice(0, limit);
  return picked.sort((a, b) => a - b).map((value) => ({ value, current: value === current }));
}
