import type { Display } from '@modelith/preprocess';
import { useEffect, useState } from 'react';
import { runSteps, runVariants } from './engine';
import { useActivePipeline, useStudio } from './store';
import { Thumb } from './Thumb';

/** 9 random draws of the pipeline up to (and including) an augmentation step, each with a different seed. */
export function NineSamples({ uid, onClose }: { uid: string; onClose(): void }) {
  const pipeline = useActivePipeline();
  const sample = useStudio((s) => s.samples[s.sampleIndex]);
  const display = useStudio((s) => s.display);
  const [base, setBase] = useState(() => Math.floor(Math.random() * 1e6));
  const [items, setItems] = useState<(Display | string | undefined)[]>([]);
  const index = pipeline.steps.findIndex((s) => s.uid === uid);

  useEffect(() => {
    setItems(new Array(9).fill(undefined));
    // Only the steps up to this one; force this step on so every draw shows it (p is still honoured).
    const steps = runSteps(pipeline).slice(0, index + 1).map((s, i) => (i === index ? { ...s, enabled: true, mode: 'both' as const } : s));
    const variants = Array.from({ length: 9 }, (_, i) => ({ steps, seed: base + i }));
    return runVariants(sample, pipeline.read, variants, true, display, (m) =>
      setItems((prev) => {
        const next = prev.slice();
        next[m.index] = m.display ?? m.error;
        return next;
      }),
    );
  }, [pipeline, sample, base, index, display]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" role="dialog" aria-modal="true" aria-label="9 random samples" onClick={onClose}>
      <div className="max-h-full w-full max-w-3xl overflow-auto rounded-xl border border-line bg-panel p-4 shadow-xl" onClick={(e) => e.stopPropagation()}>
        <div className="mb-3 flex items-center gap-2">
          <h3 className="font-semibold">9 random samples</h3>
          <span className="text-xs text-muted">
            seeds {base}–{base + 8} · train mode · steps 1–{index + 1}
          </span>
          <button onClick={() => setBase(Math.floor(Math.random() * 1e6))} className="ml-auto rounded-md border border-line px-2 py-1 text-xs hover:bg-panel-2">
            🎲 Reroll
          </button>
          <button onClick={onClose} className="rounded-md border border-line px-2 py-1 text-xs hover:bg-panel-2" aria-label="Close">
            ✕
          </button>
        </div>
        <div className="grid grid-cols-3 gap-2">
          {items.map((d, i) => (
            <div key={i} className="flex aspect-[4/3] items-center justify-center overflow-hidden rounded-md border border-line bg-panel-2">
              {typeof d === 'string' ? <p className="p-2 text-xs text-danger">{d}</p> : d ? <Thumb display={d} label={`Sample with seed ${base + i}`} /> : <span className="text-xs text-muted">…</span>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
