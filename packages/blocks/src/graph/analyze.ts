import { getBlock } from '../registry';
import { numel, positions } from '../shapes';
import {
  isUnsupported,
  ShapeError,
  type BlockDef,
  type Framework,
  type GraphEdge,
  type GraphNode,
  type Shape,
} from '../types';

export interface NodeInfo {
  shape?: Shape;
  error?: { message: string; fix?: string };
  /** Upstream is broken or missing; this node can't be evaluated yet. */
  waiting?: boolean;
  ins: Shape[];
  params: number;
  macs: number;
  /** Feeds at least one head. */
  used: boolean;
  /** Reason this block can't be emitted in the selected framework. */
  unsupported?: string;
}

export interface Issue {
  level: 'error' | 'warning';
  message: string;
  nodeId?: string;
}

export interface Analysis {
  framework: Framework;
  nodes: Record<string, NodeInfo>;
  /** Deterministic topological order of all evaluable nodes. */
  order: string[];
  /** Ids of head nodes (blocks with a `head` spec). */
  heads: string[];
  issues: Issue[];
  totalParams: number;
  totalMacs: number;
  /** Rough training activation memory per sample, bytes (fp32, forward + saved for backward). */
  activationBytesPerSample: number;
  ok: boolean;
}

export interface GraphLike {
  nodes: GraphNode[];
  edges: GraphEdge[];
  framework: Framework;
}

/** Edges into a node grouped by port; variadic blocks get all edges compacted by port order. */
export function incoming(nodeId: string, edges: GraphEdge[], def: BlockDef | undefined): (GraphEdge | undefined)[] {
  const into = edges.filter((e) => e.target === nodeId).sort((a, b) => a.targetPort - b.targetPort || a.id.localeCompare(b.id));
  if (def?.variadic) return into;
  const ports: (GraphEdge | undefined)[] = (def?.inputs ?? []).map(() => undefined);
  for (const e of into) if (e.targetPort < ports.length) ports[e.targetPort] = e;
  return ports;
}

/** Kahn's algorithm with deterministic tie-breaking (x, then y, then id). Returns order and cyclic node ids. */
export function topoSort(nodes: GraphNode[], edges: GraphEdge[]): { order: string[]; cyclic: string[] } {
  const ids = new Set(nodes.map((n) => n.id));
  const indeg = new Map<string, number>();
  const out = new Map<string, string[]>();
  for (const n of nodes) {
    indeg.set(n.id, 0);
    out.set(n.id, []);
  }
  for (const e of edges) {
    if (!ids.has(e.source) || !ids.has(e.target)) continue;
    indeg.set(e.target, indeg.get(e.target)! + 1);
    out.get(e.source)!.push(e.target);
  }
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const cmp = (a: string, b: string) => {
    const na = byId.get(a)!;
    const nb = byId.get(b)!;
    return na.position.x - nb.position.x || na.position.y - nb.position.y || a.localeCompare(b);
  };
  const ready = nodes.filter((n) => indeg.get(n.id) === 0).map((n) => n.id);
  ready.sort(cmp);
  const order: string[] = [];
  while (ready.length) {
    const id = ready.shift()!;
    order.push(id);
    for (const t of out.get(id)!) {
      indeg.set(t, indeg.get(t)! - 1);
      if (indeg.get(t) === 0) {
        ready.push(t);
        ready.sort(cmp);
      }
    }
  }
  const cyclic = nodes.filter((n) => !order.includes(n.id)).map((n) => n.id);
  return { order, cyclic };
}

export function analyze(g: GraphLike): Analysis {
  const { nodes, edges, framework } = g;
  const info: Record<string, NodeInfo> = {};
  const issues: Issue[] = [];
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const { order, cyclic } = topoSort(nodes, edges);
  const imgIn = order.map((id) => byId.get(id)!).find((n) => n.type === "image_input");
  const imageSize = imgIn ? { h: Number(imgIn.params.height), w: Number(imgIn.params.width) } : undefined;

  if (cyclic.length) {
    issues.push({ level: 'error', message: `The graph has a cycle through ${cyclic.length} node(s). Models must be acyclic.` });
    for (const id of cyclic) {
      info[id] = { ins: [], params: 0, macs: 0, used: false, error: { message: 'Part of a cycle.' } };
    }
  }

  for (const id of order) {
    const node = byId.get(id)!;
    const def = getBlock(node.type);
    const ni: NodeInfo = { ins: [], params: 0, macs: 0, used: false };
    info[id] = ni;
    if (!def) {
      ni.error = { message: `Unknown block type "${node.type}".` };
      continue;
    }
    const port = incoming(id, edges, def);
    const ins: Shape[] = [];
    let missing = false;
    let waiting = false;
    port.forEach((e, i) => {
      if (!e) {
        if (!def.inputs[i]?.optional) missing = true;
        return;
      }
      const src = info[e.source];
      if (!src?.shape) waiting = true;
      else ins.push(src.shape);
    });
    if (def.variadic && port.length < 2) missing = true;
    if (waiting) {
      ni.waiting = true;
      ni.error = { message: 'Waiting for upstream.' };
      continue;
    }
    if (missing) {
      ni.error = {
        message: def.variadic ? `${def.label} needs at least two inputs.` : `Connect all inputs (${def.inputs.map((p) => p.name).join(', ')}).`,
      };
      continue;
    }
    ni.ins = ins;
    try {
      const shape = def.inferShape(ins, node.params, { framework, imageSize });
      ni.shape = shape;
      ni.params = Math.round(def.paramCount?.(ins, node.params, shape) ?? 0);
      ni.macs = Math.round(def.macs ? def.macs(ins, node.params, shape, ni.params) : ni.params * positions(shape.kind === 'output' ? ins[0] ?? shape : shape));
    } catch (err) {
      if (err instanceof ShapeError) ni.error = { message: err.message, fix: err.fix };
      else ni.error = { message: `Internal error: ${(err as Error).message}` };
    }
  }

  // Heads and reachability (which nodes feed a head).
  const heads = order.filter((id) => getBlock(byId.get(id)!.type)?.head);
  const rev = new Map<string, string[]>();
  for (const e of edges) {
    if (!rev.has(e.target)) rev.set(e.target, []);
    rev.get(e.target)!.push(e.source);
  }
  const stack = [...heads];
  while (stack.length) {
    const id = stack.pop()!;
    if (!info[id] || info[id].used) continue;
    info[id].used = true;
    stack.push(...(rev.get(id) ?? []));
  }

  // Framework support.
  for (const id of order) {
    const node = byId.get(id)!;
    const def = getBlock(node.type);
    const ni = info[id];
    if (!def || !ni.shape) continue;
    const ctx = { framework, name: 'x', id, params: node.params, ins: ni.ins, out: ni.shape, inputNames: [], imageSize };
    try {
      const r = framework === 'pytorch' ? def.emitTorch(ctx) : def.emitKeras(ctx);
      if (isUnsupported(r)) ni.unsupported = r.unsupported;
    } catch (err) {
      ni.unsupported = `Emitter failed: ${(err as Error).message}`;
    }
  }

  // Graph-level checks.
  if (nodes.length && !heads.length) {
    issues.push({ level: 'error', message: 'Add a task head (Heads or Detectors) so the model has an output.' });
  }
  for (const n of nodes) {
    const def = getBlock(n.type);
    const ni = info[n.id];
    if (!def || !ni) continue;
    if (def.isInput && !edges.some((e) => e.source === n.id)) {
      issues.push({ level: 'warning', nodeId: n.id, message: `Input "${n.params.name}" is not connected to anything.` });
    } else if (!ni.used && heads.length && !def.head) {
      issues.push({ level: 'warning', nodeId: n.id, message: `${n.label ?? def.label} does not feed any head and will be left out of the code.` });
    }
    if (ni.error && !ni.waiting) issues.push({ level: 'error', nodeId: n.id, message: `${n.label ?? def.label}: ${ni.error.message}` });
    if (ni.unsupported && ni.used) {
      issues.push({ level: 'error', nodeId: n.id, message: `${n.label ?? def.label} is not available in ${framework === 'pytorch' ? 'PyTorch' : 'TensorFlow'}: ${ni.unsupported}` });
    }
  }
  const wrapping = heads.filter((id) => getBlock(byId.get(id)!.type)?.wrapsGraph);
  if (wrapping.length > 1) issues.push({ level: 'error', message: 'Only one torchvision detector (Faster/Mask R-CNN, RetinaNet, FCOS) is allowed per model.' });
  if (wrapping.length === 1 && heads.length > 1) {
    issues.push({ level: 'error', nodeId: wrapping[0], message: 'torchvision detectors own the whole model; remove the other heads or use the YOLO/SSD/DETR head for multi-task models.' });
  }

  let totalParams = 0;
  let totalMacs = 0;
  let act = 0;
  for (const id of order) {
    const ni = info[id];
    if (!ni.used) continue;
    totalParams += ni.params;
    totalMacs += ni.macs;
    if (ni.shape && ni.shape.kind !== 'output') act += numel(ni.shape) * 4 * 2;
  }

  return {
    framework,
    nodes: info,
    order,
    heads,
    issues,
    totalParams,
    totalMacs,
    activationBytesPerSample: act,
    ok: !issues.some((i) => i.level === 'error'),
  };
}

/** Training memory estimate in bytes: params + grads + AdamW state (fp32) + activations × batch. */
export function trainingMemoryBytes(a: Analysis, batch: number): number {
  return a.totalParams * 4 * 4 + a.activationBytesPerSample * batch;
}
