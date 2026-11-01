import type { GraphEdge, GraphNode } from '@netloom/blocks';

type Elk = { layout: (g: unknown) => Promise<{ children?: { id: string; x?: number; y?: number }[] }> };
let elk: Promise<Elk> | undefined;

/** Layered left→right auto-layout with ELK (loaded on first use). Returns new positions by node id. */
export async function autoLayout(nodes: GraphNode[], edges: GraphEdge[]): Promise<Record<string, { x: number; y: number }>> {
  elk ??= import('elkjs/lib/elk.bundled.js').then((m) => new m.default() as unknown as Elk);
  const graph = {
    id: 'root',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': 'RIGHT',
      'elk.layered.spacing.nodeNodeBetweenLayers': '70',
      'elk.spacing.nodeNode': '40',
      'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF',
    },
    children: nodes.map((n) => ({ id: n.id, width: 200, height: 86 })),
    edges: edges.map((e) => ({ id: e.id, sources: [e.source], targets: [e.target] })),
  };
  const res = await (await elk).layout(graph);
  return Object.fromEntries((res.children ?? []).map((c) => [c.id, { x: c.x ?? 0, y: c.y ?? 0 }]));
}
