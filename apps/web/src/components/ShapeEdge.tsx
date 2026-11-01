import { BaseEdge, EdgeLabelRenderer, getBezierPath, type Edge, type EdgeProps } from '@xyflow/react';
import { useStore } from '../store';

export type ShapeEdgeData = { shape?: string; bad?: boolean };
export type ShapeFlowEdge = Edge<ShapeEdgeData, 'shape'>;

/** Edge labelled with the tensor shape it carries; a delete button appears while selected. */
export function ShapeEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data, selected, style }: EdgeProps<ShapeFlowEdge>) {
  const [path, x, y] = getBezierPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition });
  return (
    <>
      <BaseEdge id={id} path={path} style={style} interactionWidth={16} />
      {(data?.shape || selected) && (
        <EdgeLabelRenderer>
          <div
            className="nodrag nopan pointer-events-auto absolute flex items-center gap-1"
            style={{ transform: `translate(-50%, -50%) translate(${x}px, ${y}px)` }}
          >
            {data?.shape && (
              <span className={`rounded border bg-panel px-1 font-mono text-[10px] ${data.bad ? 'border-danger/60 text-danger' : 'border-line text-muted'}`}>{data.shape}</span>
            )}
            {selected && (
              <button
                onClick={() => useStore.getState().removeEdges([id])}
                className="flex h-5 w-5 items-center justify-center rounded-full border border-danger/60 bg-panel text-xs text-danger hover:bg-danger/10"
                aria-label="Delete connection"
                title="Delete connection (Del)"
              >
                ×
              </button>
            )}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}
