import { formatCount, generateProject, type Analysis } from '@netloom/blocks';
import { useState } from 'react';
import { useStore } from '../store';

const VERIFY_URL = (import.meta.env.VITE_VERIFY_URL as string | undefined) ?? 'http://localhost:8000';

interface Result {
  ok: boolean;
  params?: number;
  outputs?: unknown;
  error?: string;
  seconds?: number;
}

/** Runs the generated model.py in the verify service and diffs real vs predicted shapes. */
export function Verify({ analysis }: { analysis: Analysis }) {
  const project = useStore((s) => s.project);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Result>();
  const [predicted, setPredicted] = useState<{ params: number; heads: Record<string, number[]> }>();

  const run = async () => {
    setBusy(true);
    setResult(undefined);
    try {
      const { files, ir } = generateProject(project);
      setPredicted({
        params: ir.analysis.totalParams,
        heads: Object.fromEntries(ir.heads.map((h) => [h.name, h.out.kind === 'output' ? [2, ...h.out.dims] : []])),
      });
      const body = { framework: project.framework, files: Object.fromEntries(files.filter((f) => f.path.endsWith('.py')).map((f) => [f.path, f.content])) };
      const res = await fetch(`${VERIFY_URL}/verify`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
      setResult(await res.json());
    } catch (e) {
      setResult({ ok: false, error: `Could not reach the verify service at ${VERIFY_URL} (${(e as Error).message}). Start it with: uvicorn main:app --port 8000 (apps/verify).` });
    } finally {
      setBusy(false);
    }
  };

  const outputs = result?.outputs;
  const realHeads: Record<string, unknown> = Array.isArray(outputs) && predicted ? { [Object.keys(predicted.heads)[0]]: outputs } : ((outputs as Record<string, unknown>) ?? {});

  return (
    <section aria-label="Verify with real framework" className="rounded-md border border-line p-3">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-xs font-semibold">Verify with real framework</h3>
        <button onClick={run} disabled={busy || !analysis.ok} className="rounded-md border border-line px-2 py-1 text-xs hover:bg-panel-2">
          {busy ? 'Running…' : 'Verify'}
        </button>
      </div>
      <p className="mt-1 text-[11px] text-muted">Instantiates the generated model in the sandboxed verify service and runs a dummy forward pass.</p>
      {result && !result.ok && <pre className="mt-2 max-h-40 overflow-auto rounded bg-danger/10 p-2 text-[11px] whitespace-pre-wrap text-danger">{result.error}</pre>}
      {result?.ok && predicted && (
        <table className="mt-2 w-full text-xs">
          <thead className="text-left text-muted">
            <tr>
              <th className="font-medium">Item</th>
              <th className="font-medium">Predicted</th>
              <th className="font-medium">Real</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            <Row label="params" a={formatCount(predicted.params)} b={formatCount(result.params ?? 0)} same={Math.abs((result.params ?? 0) - predicted.params) <= 0.03 * (result.params ?? 1)} />
            {Object.entries(predicted.heads).map(([name, dims]) => {
              const real = JSON.stringify(realHeads[name]);
              const pred = JSON.stringify(dims);
              return <Row key={name} label={name} a={dims.length > 1 ? pred : '—'} b={real ?? '—'} same={dims.length <= 1 || real === pred} />;
            })}
          </tbody>
        </table>
      )}
      {result?.ok && <p className="mt-1 text-[11px] text-muted">Ran in {result.seconds}s.</p>}
    </section>
  );
}

function Row({ label, a, b, same }: { label: string; a: string; b: string; same: boolean }) {
  return (
    <tr className="border-t border-line/60">
      <td className="py-1 font-sans">{label}</td>
      <td className="py-1">{a}</td>
      <td className={`py-1 ${same ? 'text-ok' : 'text-danger'}`}>
        {b} {same ? '✓' : '✕'}
      </td>
    </tr>
  );
}
