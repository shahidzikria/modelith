import { formatCount, formatShape, getBlock, trainingMemoryBytes, type Analysis } from '@modelith/blocks';
import { useState } from 'react';
import { useStore } from '../store';
import { Verify } from './Verify';

const GB = 1024 ** 3;

export function Summary({ analysis }: { analysis: Analysis }) {
  const nodes = useStore((s) => s.project.nodes);
  const select = useStore((s) => s.select);
  const [batch, setBatch] = useState(16);
  const mem = trainingMemoryBytes(analysis, batch);
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const errors = analysis.issues.filter((i) => i.level === 'error');
  const warnings = analysis.issues.filter((i) => i.level === 'warning');

  return (
    <div className="space-y-4 p-4 text-sm">
      <div className="grid grid-cols-2 gap-2">
        <Stat label="Parameters" value={formatCount(analysis.totalParams)} sub={analysis.totalParams.toLocaleString()} />
        <Stat label="FLOPs / sample" value={`${((analysis.totalMacs * 2) / 1e9).toFixed(2)} G`} sub="at declared input size" />
        <Stat label="Weights (fp32)" value={`${((analysis.totalParams * 4) / 1024 ** 2).toFixed(1)} MB`} />
        <Stat label="Heads" value={String(analysis.heads.length)} />
      </div>

      <div className="rounded-md border border-line p-3">
        <div className="flex items-center justify-between gap-2">
          <label htmlFor="mem-batch" className="text-xs font-medium">
            Training memory at batch
          </label>
          <input
            id="mem-batch"
            type="number"
            min={1}
            max={4096}
            value={batch}
            onChange={(e) => setBatch(Math.max(1, Number(e.target.value) || 1))}
            className="w-20 rounded-md border border-line bg-panel-2 px-2 py-0.5 text-right text-sm"
          />
        </div>
        <p className={`mt-2 font-mono text-lg ${mem > 24 * GB ? 'text-danger' : ''}`}>≈ {(mem / GB).toFixed(2)} GB</p>
        <p className="mt-1 text-[11px] leading-snug text-muted">
          Weights + gradients + AdamW state + stored activations (fp32, rough). AMP roughly halves activations.
          {mem > 24 * GB && <span className="text-danger"> Above 24 GB: lower the batch size or input resolution.</span>}
        </p>
      </div>

      <Verify analysis={analysis} />

      <section aria-label="Validation">
        <h3 className="mb-2 text-xs font-semibold tracking-wide text-muted uppercase">
          Validation {analysis.ok ? <span className="text-ok">· ready to generate</span> : <span className="text-danger">· {errors.length} error(s)</span>}
        </h3>
        {!analysis.issues.length && <p className="text-xs text-muted">No issues.</p>}
        <ul className="space-y-1.5">
          {[...errors, ...warnings].map((i, k) => (
            <li key={k}>
              <button
                className={`w-full rounded-md border px-2.5 py-1.5 text-left text-xs ${i.level === 'error' ? 'border-danger/40 bg-danger/10 text-danger' : 'border-warn/40 bg-warn/10 text-warn'} ${i.nodeId ? 'hover:brightness-110' : 'cursor-default'}`}
                onClick={() => i.nodeId && select([i.nodeId])}
              >
                {i.message}
              </button>
            </li>
          ))}
        </ul>
      </section>

      <section aria-label="Layers">
        <h3 className="mb-2 text-xs font-semibold tracking-wide text-muted uppercase">Layers</h3>
        <table className="w-full text-xs">
          <thead className="text-left text-muted">
            <tr>
              <th className="py-1 font-medium">Block</th>
              <th className="py-1 font-medium">Output</th>
              <th className="py-1 text-right font-medium">Params</th>
            </tr>
          </thead>
          <tbody>
            {analysis.order.map((id) => {
              const n = byId.get(id);
              const info = analysis.nodes[id];
              if (!n) return null;
              const def = getBlock(n.type)!;
              return (
                <tr key={id} className={`cursor-pointer border-t border-line/60 hover:bg-panel-2 ${info.used ? '' : 'opacity-50'}`} onClick={() => select([id])}>
                  <td className="py-1 pr-2">{n.label ?? def.label}</td>
                  <td className="py-1 pr-2 font-mono">{info.shape ? formatShape(info.shape) : info.waiting ? '…' : '✕'}</td>
                  <td className="py-1 text-right font-mono">{info.params ? formatCount(info.params) : ''}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-md bg-panel-2 p-2.5">
      <div className="text-[11px] text-muted">{label}</div>
      <div className="font-mono text-base font-semibold">{value}</div>
      {sub && <div className="truncate text-[10px] text-muted">{sub}</div>}
    </div>
  );
}
