import { getStep, type Display } from '@modelith/preprocess';
import { useEffect, useMemo, useState } from 'react';
import { runSteps, runVariants } from './engine';
import { useActivePipeline, useStudio } from './store';
import { Thumb } from './Thumb';

/** Grid of results across a range of one numeric parameter (§12.2), e.g. Canny thresholds 50 → 200. */
export function ParamSweep({ uid, onClose }: { uid: string; onClose(): void }) {
  const pipeline = useActivePipeline();
  const sample = useStudio((s) => s.samples[s.sampleIndex]);
  const display = useStudio((s) => s.display);
  const train = useStudio((s) => s.train);
  const updateParams = useStudio((s) => s.updateParams);
  const index = pipeline.steps.findIndex((s) => s.uid === uid);
  const step = pipeline.steps[index];
  const def = getStep(step.def)!;
  const numeric = def.params.filter((p) => p.type === 'int' || p.type === 'float' || p.type === 'oddInt');
  const [key, setKey] = useState(numeric[0]?.key);
  const spec = numeric.find((p) => p.key === key)!;
  const cur = Number(step.params[key]);
  const [lo, setLo] = useState(spec.min ?? Math.min(0, cur));
  const [hi, setHi] = useState(spec.max !== undefined ? Math.min(spec.max, Math.max(cur * 4, cur + 10)) : cur * 4 || 10);
  const [n, setN] = useState(6);
  const values = useMemo(() => {
    const vs: number[] = [];
    for (let i = 0; i < n; i++) {
      let v = lo + ((hi - lo) * i) / Math.max(1, n - 1);
      if (spec.type === 'int') v = Math.round(v);
      if (spec.type === 'oddInt') v = Math.max(1, Math.round(v) | 1);
      if (spec.type === 'float') v = Math.round(v * 1000) / 1000;
      vs.push(v);
    }
    return [...new Set(vs)];
  }, [lo, hi, n, spec]);
  const [items, setItems] = useState<(Display | string | undefined)[]>([]);

  useEffect(() => {
    setItems(new Array(values.length).fill(undefined));
    const base = runSteps(pipeline).slice(0, index + 1);
    const variants = values.map((v) => ({ steps: base.map((s, i) => (i === index ? { ...s, params: { ...s.params, [key]: v } } : s)), seed: pipeline.seed }));
    return runVariants(sample, pipeline.read, variants, train, display, (m) =>
      setItems((prev) => {
        const next = prev.slice();
        next[m.index] = m.display ?? m.error;
        return next;
      }),
    );
  }, [values, key, pipeline, sample, index, train, display]);

  const num = 'w-20 rounded-md border border-line bg-panel-2 px-2 py-1 text-xs';
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" role="dialog" aria-modal="true" aria-label="Parameter sweep" onClick={onClose}>
      <div className="max-h-full w-full max-w-4xl overflow-auto rounded-xl border border-line bg-panel p-4 shadow-xl" onClick={(e) => e.stopPropagation()}>
        <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
          <h3 className="mr-2 text-sm font-semibold">Parameter sweep · {def.label}</h3>
          <select aria-label="Parameter" value={key} onChange={(e) => setKey(e.target.value)} className="rounded-md border border-line bg-panel-2 px-2 py-1">
            {numeric.map((p) => (
              <option key={p.key} value={p.key}>
                {p.label}
              </option>
            ))}
          </select>
          <label>
            from <input type="number" value={lo} onChange={(e) => setLo(Number(e.target.value))} className={num} />
          </label>
          <label>
            to <input type="number" value={hi} onChange={(e) => setHi(Number(e.target.value))} className={num} />
          </label>
          <label>
            steps <input type="number" min={2} max={12} value={n} onChange={(e) => setN(Math.max(2, Math.min(12, Number(e.target.value))))} className={num} />
          </label>
          <button onClick={onClose} className="ml-auto rounded-md border border-line px-2 py-1 hover:bg-panel-2" aria-label="Close">
            ✕
          </button>
        </div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {values.map((v, i) => {
            const d = items[i];
            return (
              <button
                key={v}
                onClick={() => {
                  updateParams(uid, { [key]: v });
                  onClose();
                }}
                className="overflow-hidden rounded-md border border-line bg-panel-2 text-left hover:border-accent"
                title="Use this value"
              >
                <div className="flex aspect-[4/3] items-center justify-center">
                  {typeof d === 'string' ? <p className="p-2 text-xs text-danger">{d}</p> : d ? <Thumb display={d} label={`${spec.label} = ${v}`} /> : <span className="text-xs text-muted">…</span>}
                </div>
                <p className={`px-2 py-1 font-mono text-[11px] ${v === cur ? 'text-accent' : ''}`}>
                  {spec.label} = {v}
                  {v === cur ? ' (current)' : ''}
                </p>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
