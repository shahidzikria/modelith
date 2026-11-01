import { formatShape, getBlock, incoming, kindName, type Analysis } from '@netloom/blocks';
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  applyNodeChanges,
  useReactFlow,
  type Connection,
  type Edge,
  type EdgeChange,
  type NodeChange,
} from '@xyflow/react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CATEGORY_COLORS } from '../colors';
import { createsCycle, useStore } from '../store';
import { BlockNode, type BlockFlowNode } from './BlockNode';

const nodeTypes = { block: BlockNode };

export function Canvas({ analysis }: { analysis: Analysis }) {
  const project = useStore((s) => s.project);
  const selected = useStore((s) => s.selected);
  const codeHighlight = useStore((s) => s.codeHighlight);
  const viewNonce = useStore((s) => s.viewNonce);
  const { screenToFlowPosition, fitView } = useReactFlow();
  const [rfNodes, setRfNodes] = useState<BlockFlowNode[]>([]);
  const [selectedEdges, setSelectedEdges] = useState<string[]>([]);
  const dragging = useRef(false);

  // Sync store → React Flow nodes, keeping measured sizes (and positions mid-drag).
  useEffect(() => {
    setRfNodes((prev) => {
      const prevById = new Map(prev.map((n) => [n.id, n]));
      return project.nodes.map((n) => {
        const def = getBlock(n.type)!;
        const p = prevById.get(n.id);
        const connected = def.variadic ? incoming(n.id, project.edges, def).length : 0;
        return {
          id: n.id,
          type: 'block' as const,
          position: dragging.current && p ? p.position : n.position,
          selected: selected.includes(n.id),
          measured: p?.measured,
          data: {
            def,
            title: n.label ?? def.label,
            info: analysis.nodes[n.id],
            ports: def.variadic ? Math.max(2, connected + 1) : def.inputs.length,
            highlighted: codeHighlight === n.id,
          },
        };
      });
    });
  }, [project.nodes, project.edges, analysis, selected, codeHighlight]);

  useEffect(() => {
    if (viewNonce) requestAnimationFrame(() => fitView({ padding: 0.15, duration: 300 }));
  }, [viewNonce, fitView]);

  const edges: Edge[] = useMemo(
    () =>
      project.edges.map((e) => {
        const src = analysis.nodes[e.source]?.shape;
        const tgt = analysis.nodes[e.target];
        const bad = !!tgt?.error && !tgt.waiting;
        return {
          id: e.id,
          source: e.source,
          target: e.target,
          targetHandle: `in-${e.targetPort}`,
          sourceHandle: 'out',
          label: src ? formatShape(src) : undefined,
          labelBgPadding: [4, 2] as [number, number],
          labelBgBorderRadius: 4,
          selected: selectedEdges.includes(e.id),
          style: bad ? { stroke: 'var(--danger)', strokeWidth: 1.5 } : { strokeWidth: 1.5 },
          ariaLabel: `Edge carrying ${src ? formatShape(src) : 'unknown shape'}`,
        };
      }),
    [project.edges, analysis, selectedEdges],
  );

  const onNodesChange = useCallback(
    (changes: NodeChange<BlockFlowNode>[]) => {
      setRfNodes((nds) => applyNodeChanges(changes, nds));
      const st = useStore.getState();
      const removed = changes.filter((c) => c.type === 'remove').map((c) => c.id);
      if (removed.length) st.removeNodes(removed);
      if (changes.some((c) => c.type === 'position' && c.dragging)) dragging.current = true;
      const sel = changes.filter((c) => c.type === 'select');
      if (sel.length) {
        const next = new Set(st.selected);
        for (const c of sel) {
          if (c.type !== 'select') continue;
          if (c.selected) next.add(c.id);
          else next.delete(c.id);
        }
        st.select([...next]);
      }
    },
    [],
  );

  // Commit the final position of every dragged node once (one undo step per drag).
  const onNodeDragStop = useCallback((_: unknown, _node: BlockFlowNode, dragged: BlockFlowNode[]) => {
    dragging.current = false;
    useStore.getState().moveNodes(dragged.map((n) => ({ id: n.id, position: n.position })));
  }, []);

  const onEdgesChange = useCallback((changes: EdgeChange[]) => {
    const removed = changes.filter((c) => c.type === 'remove').map((c) => c.id);
    if (removed.length) useStore.getState().removeEdges(removed);
    const sel = changes.filter((c) => c.type === 'select');
    if (sel.length) {
      setSelectedEdges((cur) => {
        const next = new Set(cur);
        for (const c of sel) if (c.type === 'select') (c.selected ? next.add(c.id) : next.delete(c.id));
        return [...next];
      });
    }
  }, []);

  const onConnect = useCallback(
    (c: Connection) => {
      const st = useStore.getState();
      const target = st.project.nodes.find((n) => n.id === c.target);
      const def = target && getBlock(target.type);
      if (!def || !c.targetHandle) return;
      const port = Number(c.targetHandle.replace('in-', ''));
      const portDef = def.inputs[def.variadic ? 0 : port];
      const shape = analysis.nodes[c.source]?.shape;
      if (shape && portDef && !portDef.accepts.includes(shape.kind)) {
        const want = portDef.accepts.map(kindName).join(' or ');
        st.toast(`${def.label} (${portDef.name}) needs ${want}, got ${kindName(shape.kind)}.`, 'error');
        return;
      }
      if (createsCycle(st.project.edges, c.source, c.target)) {
        st.toast('That connection would create a cycle.', 'error');
        return;
      }
      st.connect({ source: c.source, target: c.target, targetPort: port });
    },
    [analysis],
  );

  const isValidConnection = useCallback(
    (c: Connection | Edge) => !createsCycle(useStore.getState().project.edges, c.source, c.target),
    [],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      const type = e.dataTransfer.getData('application/netloom-block');
      if (!type) return;
      e.preventDefault();
      const pos = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      useStore.getState().addNode(type, { x: pos.x - 100, y: pos.y - 30 });
    },
    [screenToFlowPosition],
  );

  return (
    <ReactFlow<BlockFlowNode, Edge>
      nodes={rfNodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeDragStop={onNodeDragStop}
      onConnect={onConnect}
      isValidConnection={isValidConnection}
      onDrop={onDrop}
      onDragOver={(e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
      }}
      onPaneClick={() => useStore.getState().select([])}
      deleteKeyCode={['Delete', 'Backspace']}
      multiSelectionKeyCode={['Shift', 'Meta', 'Control']}
      snapToGrid
      snapGrid={[10, 10]}
      minZoom={0.15}
      maxZoom={2}
      fitView
      proOptions={{ hideAttribution: true }}
      aria-label="Model graph canvas"
    >
      <Background gap={20} color="var(--grid)" />
      <Controls showInteractive={false} position="bottom-left" />
      <MiniMap
        pannable
        zoomable
        position="bottom-right"
        nodeColor={(n) => CATEGORY_COLORS[(n as BlockFlowNode).data.def.category]}
        nodeStrokeWidth={0}
        ariaLabel="Mini map"
        className="!hidden md:!block"
      />
    </ReactFlow>
  );
}
