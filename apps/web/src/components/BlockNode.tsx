import { formatCount, formatShape, kindName, type BlockDef, type NodeInfo, type Shape } from '@netloom/blocks';
import { Handle, NodeToolbar, Position, type Node, type NodeProps } from '@xyflow/react';
import { memo } from 'react';
import { CATEGORY_COLORS } from '../colors';
import { useStore } from '../store';

export interface BlockNodeData extends Record<string, unknown> {
  def: BlockDef;
  title: string;
  info?: NodeInfo;
  ports: number;
  /** Shape arriving at each input port (undefined = not connected / not inferred yet). */
  inShapes: (Shape | undefined)[];
  highlighted: boolean;
}

export type BlockFlowNode = Node<BlockNodeData, 'block'>;

/** "↓2×" / "↑4×" when a feature map changes resolution. */
function resize(a: Shape | undefined, b: Shape | undefined): string | undefined {
  if (a?.kind !== 'image' || b?.kind !== 'image' || a.h === b.h) return undefined;
  return a.h > b.h ? `↓${+(a.h / b.h).toFixed(1)}×` : `↑${+(b.h / a.h).toFixed(1)}×`;
}

function BlockNodeImpl({ id, data, selected }: NodeProps<BlockFlowNode>) {
  const { def, info, ports, title, highlighted, inShapes } = data;
  const color = CATEGORY_COLORS[def.category];
  const error = info?.error && !info.waiting ? info.error : undefined;
  const unsupported = info?.unsupported;
  const shape = info?.shape;
  const portNames = Array.from({ length: ports }, (_, i) => (def.variadic ? `in ${i + 1}` : def.inputs[i]?.name ?? `in ${i + 1}`));
  const ring = selected ? 'ring-2 ring-accent' : highlighted ? 'ring-2 ring-[var(--warn)]' : '';
  const status = error ? `Error: ${error.message}` : unsupported ? `Unsupported: ${unsupported}` : info?.waiting ? 'Waiting for upstream' : shape ? `Output ${formatShape(shape)}` : '';
  const change = resize(inShapes[0], shape);

  return (
    <div
      className={`relative w-[220px] rounded-lg border bg-panel text-fg shadow-sm ${error ? 'border-danger' : 'border-line'} ${ring} ${info && !info.used && !def.head && !def.isInput ? 'opacity-60' : ''}`}
      role="group"
      aria-label={`${title} (${def.label}). ${status}`}
    >
      <NodeToolbar position={Position.Top} offset={6} className="flex gap-1">
        <button
          onClick={() => {
            useStore.getState().select([id]);
            useStore.getState().duplicate();
          }}
          className="rounded-md border border-line bg-panel px-2 py-0.5 text-[11px] shadow-sm hover:bg-panel-2"
          aria-label={`Duplicate ${title}`}
        >
          Duplicate
        </button>
        <button
          onClick={() => useStore.getState().removeNodes([id])}
          className="rounded-md border border-danger/50 bg-panel px-2 py-0.5 text-[11px] text-danger shadow-sm hover:bg-danger/10"
          aria-label={`Delete ${title}`}
        >
          Delete
        </button>
      </NodeToolbar>
      <div className="h-1.5 rounded-t-lg" style={{ background: color }} />
      <div className="px-3 pt-1.5 pb-2">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="truncate text-[13px] font-semibold leading-tight" title={title}>
              {title}
            </div>
            {title !== def.label && <div className="truncate text-[11px] text-muted">{def.label}</div>}
          </div>
          {(error || unsupported) && (
            <span
              className="mt-0.5 inline-flex h-4 min-w-4 shrink-0 items-center justify-center rounded-full bg-danger px-1 text-[10px] font-bold text-white"
              title={error ? `${error.message}${error.fix ? `\nFix: ${error.fix}` : ''}` : unsupported}
              aria-hidden
            >
              !
            </span>
          )}
        </div>
        <dl className="mt-1.5 grid grid-cols-[auto_1fr] items-center gap-x-2 gap-y-0.5 text-[11px]">
          {portNames.map((n, i) => (
            <div key={n} className="contents">
              <dt className="text-muted">{ports > 1 ? n : 'in'}</dt>
              <dd className={`truncate font-mono ${inShapes[i] ? '' : 'text-muted italic'}`}>
                {inShapes[i] ? formatShape(inShapes[i]!) : `connect ${(def.inputs[def.variadic ? 0 : i]?.accepts ?? []).map(kindName).join(' / ')}`}
              </dd>
            </div>
          ))}
          <dt className="text-muted">out</dt>
          <dd className="flex min-w-0 items-center gap-1.5">
            <span
              className={`truncate rounded px-1 font-mono ${error ? 'bg-danger/10 text-danger' : 'bg-panel-2'}`}
              title={shape ? `${kindName(shape.kind)}: ${formatShape(shape)}` : error?.message}
            >
              {shape ? formatShape(shape) : info?.waiting ? 'waiting…' : error ? 'error' : '—'}
            </span>
            {change && <span className="shrink-0 text-muted">{change}</span>}
            {info && info.params > 0 && <span className="ml-auto shrink-0 text-muted">{formatCount(info.params)}</span>}
          </dd>
        </dl>
        {error && <p className="mt-1 line-clamp-2 text-[10px] leading-snug text-danger">{error.fix ?? error.message}</p>}
      </div>
      {portNames.map((name, i) => (
        <Handle
          key={i}
          type="target"
          position={Position.Left}
          id={`in-${i}`}
          style={{ top: ports === 1 ? '50%' : `${((i + 1) / (ports + 1)) * 100}%` }}
          aria-label={`Input port ${name} of ${title}`}
          title={`${name}: accepts ${(def.inputs[def.variadic ? 0 : i]?.accepts ?? []).map(kindName).join(' / ')}`}
        />
      ))}
      {!def.head && <Handle type="source" position={Position.Right} id="out" aria-label={`Output port of ${title}`} title="Output — drag to an input port" />}
    </div>
  );
}

export const BlockNode = memo(BlockNodeImpl);
