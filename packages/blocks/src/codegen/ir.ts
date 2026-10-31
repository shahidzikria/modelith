import { analyze, incoming, type Analysis } from '../graph/analyze';
import { ident } from '../py';
import { requireBlock } from '../registry';
import type { BlockDef, Framework, GraphEdge, GraphNode, HeadTrainSpec, Params, Shape } from '../types';

export interface IRNode {
  id: string;
  type: string;
  def: BlockDef;
  /** Python variable / attribute name. */
  name: string;
  params: Params;
  /** Variable names of the inputs, in port order. */
  inputs: string[];
  inputIds: string[];
  ins: Shape[];
  out: Shape;
  head?: HeadTrainSpec;
}

export interface IR {
  name: string;
  framework: Framework;
  /** Pruned, topologically ordered nodes (inputs included). */
  nodes: IRNode[];
  inputs: IRNode[];
  heads: IRNode[];
  /** torchvision detector that wraps the rest of the graph, if any. */
  wrapper?: IRNode;
  analysis: Analysis;
  imageSize?: { h: number; w: number };
  /** Uses a pretrained backbone or transformer blocks (affects LR defaults). */
  pretrained: boolean;
  hasTransformer: boolean;
}

export class CodegenError extends Error {
  constructor(public issues: string[]) {
    super(`Cannot generate code:\n- ${issues.join('\n- ')}`);
  }
}

/** Names the generated code itself uses; node variables must not shadow them. */
const RESERVED = new Set(['x', 'model', 'nn', 'F', 'torch', 'math', 'keras', 'layers', 'ops', 'np', 'tf', 'timm', 'outputs', 'targets', 'inputs', 'config', 'out', 'level', 'levels']);

export function buildIR(project: { name: string; framework: Framework; nodes: GraphNode[]; edges: GraphEdge[] }): IR {
  const analysis = analyze(project);
  const errors = analysis.issues.filter((i) => i.level === 'error').map((i) => i.message);
  if (errors.length) throw new CodegenError(errors);

  const byId = new Map(project.nodes.map((n) => [n.id, n]));
  const used = analysis.order.filter((id) => analysis.nodes[id].used);

  // Deterministic names: input name, user label, or `<prefix>` / `<prefix>_<k>` in topological order.
  const prefixCount = new Map<string, number>();
  for (const id of used) {
    const n = byId.get(id)!;
    const def = requireBlock(n.type);
    if (!def.isInput && !n.label) prefixCount.set(def.varName, (prefixCount.get(def.varName) ?? 0) + 1);
  }
  const taken = new Set<string>();
  const seen = new Map<string, number>();
  const names = new Map<string, string>();
  for (const id of used) {
    const n = byId.get(id)!;
    const def = requireBlock(n.type);
    let base: string;
    if (def.isInput) base = ident(String(n.params.name ?? def.varName));
    else if (n.label) base = ident(n.label);
    else if (prefixCount.get(def.varName) === 1) base = def.varName;
    else {
      const k = (seen.get(def.varName) ?? 0) + 1;
      seen.set(def.varName, k);
      base = `${def.varName}_${k}`;
    }
    let name = RESERVED.has(base) ? `${base}_` : base;
    let i = 2;
    while (taken.has(name)) name = `${base}_${i++}`;
    taken.add(name);
    names.set(id, name);
  }

  const nodes: IRNode[] = used.map((id) => {
    const n = byId.get(id)!;
    const def = requireBlock(n.type);
    const ni = analysis.nodes[id];
    const inEdges = incoming(id, project.edges, def).filter((e): e is GraphEdge => !!e);
    return {
      id,
      type: n.type,
      def,
      name: names.get(id)!,
      params: n.params,
      inputs: inEdges.map((e) => names.get(e.source)!),
      inputIds: inEdges.map((e) => e.source),
      ins: ni.ins,
      out: ni.shape!,
      head: def.head?.(n.params, ni.ins, ni.shape!),
    };
  });

  const imgIn = nodes.find((n) => n.type === 'image_input');
  return {
    name: project.name,
    framework: project.framework,
    nodes,
    inputs: nodes.filter((n) => n.def.isInput),
    heads: nodes.filter((n) => n.head),
    wrapper: nodes.find((n) => n.def.wrapsGraph),
    analysis,
    imageSize: imgIn ? { h: imgIn.params.height, w: imgIn.params.width } : undefined,
    pretrained: nodes.some((n) => n.def.pretrained?.(n.params)),
    hasTransformer: nodes.some((n) => /transformer|vit|detr|swin/.test(n.type) || (n.type === 'timm_backbone' && /vit|swin|dinov2/.test(String(n.params.model)))),
  };
}
