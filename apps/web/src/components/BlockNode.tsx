import { formatCount, formatShape, kindName, type BlockDef, type NodeInfo } from '@netloom/blocks';
import { Handle, Position, type Node, type NodeProps } from '@xyflow/react';
import { memo } from 'react';
import { CATEGORY_COLORS } from '../colors';

export interface BlockNodeData extends Record<string, unknown> {
  def: BlockDef;
  title: string;
  info?: NodeInfo;
  ports: number;
  highlighted: boolean;
}

export type BlockFlowNode = Node<BlockNodeData, 'block'>;

function BlockNodeImpl({ data, selected }: NodeProps<BlockFlowNode>) {
  const { def, info, ports, title, highlighted } = data;
  const color = CATEGORY_COLORS[def.category];
  const error = info?.error && !info.waiting ? info.error : undefined;
  const unsupported = info?.unsupported;
  const shape = info?.shape;
  const portNames = Array.from({ length: ports }, (_, i) => (def.variadic ? `in ${i + 1}` : def.inputs[i]?.name ?? `in ${i + 1}`));
  const ring = selected ? 'ring-2 ring-accent' : highlighted ? 'ring-2 ring-[var(--warn)]' : '';
  const status = error ? `Error: ${error.message}` : unsupported ? `Unsupported: ${unsupported}` : info?.waiting ? 'Waiting for upstream' : shape ? `Output ${formatShape(shape)}` : '';

  return (
    <div
      className={`relative w-[200px] rounded-lg border bg-panel text-fg shadow-sm ${error ? 'border-danger' : 'border-line'} ${ring} ${info && !info.used && !def.head && !def.isInput ? 'opacity-60' : ''}`}
      role="group"
      aria-label={`${title} (${def.label}). ${status}`}
    >
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
        <div className="mt-1.5 flex items-center justify-between gap-2 text-[11px]">
          <span
            className={`truncate rounded px-1.5 py-0.5 font-mono ${error ? 'bg-danger/10 text-danger' : 'bg-panel-2 text-fg'}`}
            title={shape ? `${kindName(shape.kind)}: ${formatShape(shape)}` : error?.message}
          >
            {shape ? formatShape(shape) : info?.waiting ? 'waiting…' : error ? 'error' : '—'}
          </span>
          {info && info.params > 0 && <span className="shrink-0 text-muted">{formatCount(info.params)}</span>}
        </div>
        {ports > 1 && (
          <div className="mt-1 flex flex-col gap-0.5 text-[10px] text-muted">
            {portNames.map((n) => (
              <span key={n}>◂ {n}</span>
            ))}
          </div>
        )}
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
