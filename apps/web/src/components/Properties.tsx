import { formatCount, formatShape, getBlock, type Analysis, type ParamSpec } from '@netloom/blocks';
import { useStore } from '../store';
import { CATEGORY_COLORS } from '../colors';

function Field({ spec, value, onChange, id, describedBy }: { spec: ParamSpec; value: unknown; onChange: (v: unknown) => void; id: string; describedBy?: string }) {
  const a = { id, "aria-describedby": describedBy };
  const base = 'w-full rounded-md border border-line bg-panel-2 px-2 py-1 text-sm outline-none focus:border-accent';
  switch (spec.type) {
    case 'bool':
      return (
        <input
          {...a}
          type="checkbox"
          checked={!!value}
          onChange={(e) => onChange(e.target.checked)}
          className="h-4 w-4 accent-[var(--accent)]"
        />
      );
    case 'select':
      return (
        <select {...a} value={String(value)} onChange={(e) => onChange(e.target.value)} className={base}>
          {spec.options!.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      );
    case 'int':
    case 'float':
      return (
        <input
          {...a}
          type="number"
          value={Number.isFinite(value as number) ? (value as number) : ''}
          min={spec.min}
          max={spec.max}
          step={spec.step ?? (spec.type === 'int' ? 1 : 0.01)}
          onChange={(e) => {
            if (e.target.value === '') return;
            let v = spec.type === 'int' ? Math.round(Number(e.target.value)) : Number(e.target.value);
            if (spec.min !== undefined) v = Math.max(spec.min, v);
            if (spec.max !== undefined) v = Math.min(spec.max, v);
            onChange(v);
          }}
          className={base}
        />
      );
    case 'intList':
      return (
        <input
          {...a}
          type="text"
          defaultValue={(value as number[]).join(', ')}
          key={(value as number[]).join(',')}
          onBlur={(e) => {
            const parts = e.target.value
              .split(/[\s,]+/)
              .filter(Boolean)
              .map((x) => Math.round(Number(x)))
              .filter((x) => Number.isFinite(x));
            if (parts.length) onChange(parts);
          }}
          className={`${base} font-mono`}
          placeholder="e.g. 6, 12, 18"
        />
      );
    default:
      return <input {...a} type="text" value={String(value ?? '')} onChange={(e) => onChange(e.target.value)} className={base} />;
  }
}

export function Properties({ analysis }: { analysis: Analysis }) {
  const selected = useStore((s) => s.selected);
  const node = useStore((s) => s.project.nodes.find((n) => n.id === s.selected[s.selected.length - 1]));
  const updateParams = useStore((s) => s.updateParams);
  const setLabel = useStore((s) => s.setLabel);

  if (!node) {
    return (
      <div className="p-4 text-sm text-muted">
        <p>Select a block to edit its hyper-parameters.</p>
        <p className="mt-3">
          Tip: click a palette item to append it after the selected block and connect it automatically.
        </p>
      </div>
    );
  }
  if (selected.length > 1) {
    return <div className="p-4 text-sm text-muted">{selected.length} blocks selected. Ctrl+C / Ctrl+V to copy, Ctrl+D to duplicate, Del to delete.</div>;
  }
  const def = getBlock(node.type)!;
  const info = analysis.nodes[node.id];
  const visible = def.params.filter((p) => !p.showIf || p.showIf.equals.includes(node.params[p.showIf.key]));

  return (
    <div className="p-4">
      <div className="flex items-center gap-2">
        <span className="h-3 w-3 rounded-sm" style={{ background: CATEGORY_COLORS[def.category] }} aria-hidden />
        <h2 className="text-sm font-semibold">{def.label}</h2>
        <span className="text-xs text-muted">· {def.category}</span>
      </div>
      <p className="mt-1.5 text-xs leading-relaxed text-muted">{def.doc}</p>
      {def.paper && <p className="mt-1 text-xs text-muted italic">Paper: {def.paper}</p>}

      {info?.error && !info.waiting && (
        <div className="mt-3 rounded-md border border-danger/40 bg-danger/10 p-2.5 text-xs text-danger" role="alert">
          <p className="font-medium">{info.error.message}</p>
          {info.error.fix && <p className="mt-1">Fix: {info.error.fix}</p>}
        </div>
      )}
      {info?.unsupported && (
        <div className="mt-3 rounded-md border border-warn/40 bg-warn/10 p-2.5 text-xs text-warn" role="alert">
          Not available in this framework: {info.unsupported}
        </div>
      )}

      <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 rounded-md bg-panel-2 p-2.5 text-xs">
        {info?.ins.map((s, i) => (
          <div key={i} className="contents">
            <dt className="text-muted">in {def.variadic ? i + 1 : def.inputs[i]?.name}</dt>
            <dd className="font-mono">{formatShape(s)}</dd>
          </div>
        ))}
        <dt className="text-muted">out</dt>
        <dd className="font-mono">{info?.shape ? formatShape(info.shape) : '—'}</dd>
        <dt className="text-muted">params</dt>
        <dd className="font-mono">{info ? `${formatCount(info.params)} (${info.params.toLocaleString()})` : '—'}</dd>
        {info && info.macs > 0 && (
          <>
            <dt className="text-muted">FLOPs</dt>
            <dd className="font-mono">{formatCount(info.macs * 2)}</dd>
          </>
        )}
      </dl>

      <div className="mt-4 space-y-3">
        {!def.isInput && (
          <div>
            <label htmlFor="node-label" className="mb-1 block text-xs font-medium">
              Name in code
            </label>
            <input
              id="node-label"
              type="text"
              value={node.label ?? ''}
              placeholder={def.varName}
              onChange={(e) => setLabel(node.id, e.target.value)}
              className="w-full rounded-md border border-line bg-panel-2 px-2 py-1 font-mono text-sm outline-none focus:border-accent"
            />
          </div>
        )}
        {visible.map((spec) => {
          const id = `param-${spec.key}`;
          return (
            <div key={spec.key} className={spec.type === 'bool' ? 'flex items-center justify-between gap-3' : ''}>
              <div className="mb-1 flex items-center gap-1.5 text-xs font-medium">
                <label htmlFor={id}>{spec.label}</label>
                {spec.help && (
                  <>
                    <span className="cursor-help rounded-full border border-line px-1 text-[10px] text-muted" title={spec.help} aria-hidden>
                      ?
                    </span>
                    <span id={`${id}-help`} className="sr-only">
                      {spec.help}
                    </span>
                  </>
                )}
              </div>
              <Field
                spec={spec}
                id={id}
                describedBy={spec.help ? `${id}-help` : undefined}
                value={node.params[spec.key]}
                onChange={(v) => updateParams(node.id, { [spec.key]: v })}
              />
            </div>
          );
        })}
        {!visible.length && def.isInput === undefined && <p className="text-xs text-muted">This block has no hyper-parameters.</p>}
      </div>
    </div>
  );
}
