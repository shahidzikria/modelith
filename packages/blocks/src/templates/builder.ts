import { defaultParams, requireBlock } from '../registry';
import type { Framework, GraphEdge, GraphNode, Params, Project } from '../types';

/** Fluent helper for building graphs in code (templates, tests, AI patches). */
export class GraphBuilder {
  nodes: GraphNode[] = [];
  edges: GraphEdge[] = [];
  private n = 0;

  constructor(private prefix = 'n') {}

  /** Add a node; `from` lists upstream node ids in port order. Returns the new node id. */
  add(type: string, params: Params = {}, from: string | string[] = [], label?: string): string {
    requireBlock(type);
    const id = `${this.prefix}${++this.n}`;
    this.nodes.push({ id, type, position: { x: 0, y: 0 }, params: { ...defaultParams(type), ...params }, ...(label ? { label } : {}) });
    const srcs = Array.isArray(from) ? from : [from];
    srcs.forEach((s, port) => this.edges.push({ id: `e_${s}_${id}_${port}`, source: s, target: id, targetPort: port }));
    return id;
  }

  /** Add a linear chain of [type, params] after `from`; returns the last id. */
  chain(from: string, ...steps: [string, Params?, string?][]): string {
    let cur = from;
    for (const [type, params, label] of steps) cur = this.add(type, params ?? {}, cur, label);
    return cur;
  }

  /** Lay nodes out in columns by longest path from the sources, wrapping into bands of `perRow` columns. */
  layout(dx = 260, dy = 130, perRow = 6): this {
    const depth = new Map<string, number>();
    const preds = new Map<string, string[]>();
    for (const e of this.edges) preds.set(e.target, [...(preds.get(e.target) ?? []), e.source]);
    const visit = (id: string): number => {
      if (depth.has(id)) return depth.get(id)!;
      const d = Math.max(-1, ...(preds.get(id) ?? []).map(visit)) + 1;
      depth.set(id, d);
      return d;
    };
    const rows = new Map<number, number>();
    const rowOf = new Map<string, number>();
    for (const node of this.nodes) {
      const d = visit(node.id);
      const r = rows.get(d) ?? 0;
      rows.set(d, r + 1);
      rowOf.set(node.id, r);
    }
    // Band b holds depths [b·perRow, (b+1)·perRow); its height is its tallest column.
    const maxDepth = Math.max(0, ...depth.values());
    const bandTop: number[] = [0];
    for (let b = 0; b * perRow <= maxDepth; b++) {
      let tallest = 1;
      for (let d = b * perRow; d < (b + 1) * perRow; d++) tallest = Math.max(tallest, rows.get(d) ?? 0);
      bandTop.push(bandTop[b] + tallest * dy + dy * 0.6);
    }
    for (const node of this.nodes) {
      const d = depth.get(node.id)!;
      const band = Math.floor(d / perRow);
      node.position = { x: (d % perRow) * dx, y: bandTop[band] + rowOf.get(node.id)! * dy };
    }
    return this;
  }

  project(name: string, framework: Framework = 'pytorch'): Project {
    this.layout();
    return { version: 2, name, framework, nodes: this.nodes, edges: this.edges, customBlocks: [] };
  }
}
