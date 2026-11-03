// Inspector extras: auto-suggest (§12.1), dataset statistics (§12.3), batch-augmentation preview (MixUp / CutMix /
// Mosaic with the loaded samples), version history (§12.5).
import { describeDiff, diffPipelines, getStep, suggestSteps, type Display, type ImageBuf, type Pipeline } from '@modelith/preprocess';
import { useEffect, useMemo, useState } from 'react';
import { useStore as useModelStore } from '../store';
import { runSteps, runVariants, sampleInput } from './engine';
import { useActivePipeline, useStudio } from './store';
import { Thumb } from './Thumb';

const btn = 'rounded-md border border-line px-2 py-1 text-xs hover:bg-panel-2';

/** Suggestions for the current sample, shown on the Read image row. */
export function SuggestionsPanel() {
  const pipeline = useActivePipeline();
  const sample = useStudio((s) => s.samples[s.sampleIndex]);
  const addStep = useStudio((s) => s.addStep);
  const { stats, suggestions } = useMemo(() => suggestSteps(sampleInput(sample, pipeline.read)), [sample, pipeline.read]);
  const tensorAt = pipeline.steps.findIndex((s) => getStep(s.def)?.category === 'Normalise & tensor');
  return (
    <section className="space-y-2 rounded-md border border-line p-2" aria-label="Suggestions">
      <h4 className="text-xs font-semibold">Suggestions for “{sample.name}”</h4>
      <p className="font-mono text-[11px] text-muted">
        mean {stats.mean.toFixed(0)} · std {stats.std.toFixed(0)} · noise σ {stats.noise.toFixed(1)} · sharpness {stats.sharpness.toFixed(0)}
      </p>
      {suggestions.length ? (
        <ul className="space-y-1">
          {suggestions.map((s, i) => (
            <li key={i} className="flex items-start gap-2 text-xs">
              <span className="flex-1">
                <b>{getStep(s.def)?.label}</b>: {s.reason}
              </span>
              <button className={btn} onClick={() => addStep(s.def, s.at === 'start' ? 0 : tensorAt < 0 ? undefined : tensorAt, s.params)}>
                Add
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-xs text-muted">Nothing stands out: brightness, contrast, noise and sharpness look normal.</p>
      )}
    </section>
  );
}

/** Channel mean / std over all loaded samples at the input of a Standardise step (§12.3). */
export function DatasetStats({ uid }: { uid: string }) {
  const pipeline = useActivePipeline();
  const samples = useStudio((s) => s.samples);
  const display = useStudio((s) => s.display);
  const updateParams = useStudio((s) => s.updateParams);
  const toast = useModelStore((s) => s.toast);
  const [busy, setBusy] = useState(false);
  const index = pipeline.steps.findIndex((s) => s.uid === uid);
  const run = async () => {
    setBusy(true);
    const steps = runSteps(pipeline).slice(0, index);
    const bufs: ImageBuf[] = [];
    for (const s of samples) {
      if (!steps.length) {
        bufs.push(sampleInput(s, pipeline.read));
        continue;
      }
      await new Promise<void>((resolve) => {
        const stop = runVariants(s, pipeline.read, [{ steps, seed: pipeline.seed }], false, display, (m) => {
          stop();
          if (m.raw) bufs.push(m.raw);
          resolve();
        }, { fullResolution: true, returnRaw: true });
      });
    }
    setBusy(false);
    const c = bufs[0]?.meta.c ?? 0;
    if (!bufs.length || (c !== 3 && c !== 1)) return toast('Needs 1- or 3-channel images at this step', 'error');
    const k = bufs[0].meta.dtype === 'uint8' ? 255 : bufs[0].meta.dtype === 'uint16' ? 65535 : 1;
    const sum = new Array(c).fill(0), sq = new Array(c).fill(0);
    let n = 0;
    for (const b of bufs) {
      for (let i = 0; i < b.data.length; i++) {
        const v = b.data[i] / k;
        sum[i % c] += v;
        sq[i % c] += v * v;
      }
      n += b.data.length / c;
    }
    const mean = sum.map((s) => s / n);
    const std = sq.map((s, i) => Math.sqrt(Math.max(1e-12, s / n - mean[i] ** 2)));
    // Standardise takes RGB-order values; BGR images are measured in BGR order.
    const rgb = bufs[0].meta.space.startsWith('RGB') || c === 1;
    const r = (v: number[]) => (c === 1 ? [v[0], v[0], v[0]] : rgb ? v : [...v].reverse()).map((x) => Math.round(x * 1e4) / 1e4);
    updateParams(uid, { preset: 'custom', mean: r(mean), std: r(std) });
    toast(`Mean / std from ${bufs.length} sample(s) set on the Standardise step`, 'success');
  };
  return (
    <button className={btn} onClick={run} disabled={busy} title="Compute per-channel mean / std of the loaded samples at this step's input">
      {busy ? 'Computing…' : `Mean / std from ${samples.length} sample${samples.length === 1 ? '' : 's'}`}
    </button>
  );
}

/** MixUp / CutMix / Mosaic preview using the final outputs of the loaded samples (they run per batch in dataset.py). */
export function BatchPreview({ def, params, onClose }: { def: string; params: Record<string, any>; onClose(): void }) {
  const pipeline = useActivePipeline();
  const samples = useStudio((s) => s.samples);
  const display = useStudio((s) => s.display);
  const [tiles, setTiles] = useState<Display[]>([]);
  const [seed, setSeed] = useState(1);
  useEffect(() => {
    const steps = runSteps(pipeline).filter((s) => !getStep(s.def)?.batch);
    const out: Display[] = [];
    let cancelled = false;
    (async () => {
      for (const s of samples.slice(0, 4)) {
        await new Promise<void>((resolve) => {
          const stop = runVariants(s, pipeline.read, [{ steps, seed: pipeline.seed }], true, display, (m) => {
            stop();
            if (m.display) out.push(m.display);
            resolve();
          });
        });
      }
      if (!cancelled) setTiles(out);
    })();
    return () => {
      cancelled = true;
    };
  }, [pipeline, samples, display]);

  const composed = useMemo(() => composeBatch(def, tiles, params, seed), [def, tiles, params, seed]);
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" role="dialog" aria-modal="true" aria-label="Batch augmentation preview" onClick={onClose}>
      <div className="max-h-full w-full max-w-2xl overflow-auto rounded-xl border border-line bg-panel p-4 shadow-xl" onClick={(e) => e.stopPropagation()}>
        <div className="mb-2 flex items-center gap-2">
          <h3 className="font-semibold">{getStep(def)?.label} preview</h3>
          <span className="text-xs text-muted">using {Math.min(4, samples.length)} loaded sample(s) · applied per batch in dataset.py</span>
          <button className={`${btn} ml-auto`} onClick={() => setSeed((x) => x + 1)}>
            🎲
          </button>
          <button className={btn} onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        {samples.length < 2 ? (
          <p className="text-sm text-muted">Load at least 2 images (Samples ▸ Load images…) to preview a batch-level augmentation.</p>
        ) : composed ? (
          <div className="flex justify-center">
            <Thumb display={composed} label="Batch augmentation result" className="max-h-[60vh]" />
          </div>
        ) : (
          <p className="text-sm text-muted">Rendering samples…</p>
        )}
      </div>
    </div>
  );
}

function resizeNearest(d: Display, w: number, h: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const sx = Math.min(d.w - 1, Math.floor((x * d.w) / w)), sy = Math.min(d.h - 1, Math.floor((y * d.h) / h));
      out.set(d.rgba.subarray((sy * d.w + sx) * 4, (sy * d.w + sx) * 4 + 4), (y * w + x) * 4);
    }
  return out;
}

function composeBatch(def: string, tiles: Display[], params: Record<string, any>, seed: number): Display | undefined {
  if (tiles.length < 2) return undefined;
  const rnd = (k: number) => {
    const x = Math.sin(seed * 9301 + k * 49297) * 233280;
    return x - Math.floor(x);
  };
  const w = tiles[0].w, h = tiles[0].h;
  const a = tiles[0].rgba, b = resizeNearest(tiles[1], w, h);
  if (def === 'mixup') {
    const lam = 0.2 + 0.6 * rnd(1);
    const out = new Uint8ClampedArray(a.length);
    for (let i = 0; i < a.length; i++) out[i] = lam * a[i] + (1 - lam) * b[i];
    return { rgba: out, w, h };
  }
  if (def === 'cutmix') {
    const out = new Uint8ClampedArray(a);
    const cw = Math.floor(w * (0.3 + 0.4 * rnd(2))), ch = Math.floor(h * (0.3 + 0.4 * rnd(3)));
    const x0 = Math.floor((w - cw) * rnd(4)), y0 = Math.floor((h - ch) * rnd(5));
    for (let y = y0; y < y0 + ch; y++) out.set(b.subarray((y * w + x0) * 4, (y * w + x0 + cw) * 4), (y * w + x0) * 4);
    return { rgba: out, w, h };
  }
  // Mosaic: 2×2 around a random centre.
  const W = 2 * w, H = 2 * h;
  const out = new Uint8ClampedArray(W * H * 4).fill(114);
  for (let i = 3; i < out.length; i += 4) out[i] = 255;
  const cx = Math.floor(w * (0.5 + rnd(6))), cy = Math.floor(h * (0.5 + rnd(7)));
  const quads = [
    [cx - w, cy - h],
    [cx, cy - h],
    [cx - w, cy],
    [cx, cy],
  ];
  quads.forEach(([ox, oy], k) => {
    const t = tiles[k % tiles.length];
    const px = resizeNearest(t, w, h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const X = ox + x, Y = oy + y;
        if (X >= 0 && Y >= 0 && X < W && Y < H) out.set(px.subarray((y * w + x) * 4, (y * w + x) * 4 + 4), (Y * W + X) * 4);
      }
  });
  void params;
  return { rgba: out, w: W, h: H };
}

// ----------------------------------------------------------------------------- version history

const VERSIONS_KEY = 'modelith:pipeline-versions';
interface Version {
  at: number;
  label: string;
  pipeline: Pipeline;
}

function loadVersions(): Record<string, Version[]> {
  try {
    return JSON.parse(localStorage.getItem(VERSIONS_KEY) ?? '{}');
  } catch {
    return {};
  }
}

export function VersionsDialog({ onClose }: { onClose(): void }) {
  const pipeline = useActivePipeline();
  const [all, setAll] = useState(loadVersions);
  const [label, setLabel] = useState('');
  const versions = all[pipeline.id] ?? [];
  const save = (next: Record<string, Version[]>) => {
    setAll(next);
    try {
      localStorage.setItem(VERSIONS_KEY, JSON.stringify(next));
    } catch {
      /* storage unavailable */
    }
  };
  const restore = (v: Version) => {
    useStudio.setState((s) => ({ pipelines: s.pipelines.map((p) => (p.id === pipeline.id ? { ...structuredClone(v.pipeline), id: pipeline.id } : p)) }));
    onClose();
  };
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" role="dialog" aria-modal="true" aria-label="Version history" onClick={onClose}>
      <div className="max-h-full w-full max-w-xl overflow-auto rounded-xl border border-line bg-panel p-4 shadow-xl" onClick={(e) => e.stopPropagation()}>
        <div className="mb-3 flex items-center gap-2">
          <h3 className="font-semibold">Versions of “{pipeline.name}”</h3>
          <button className={`${btn} ml-auto`} onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        <div className="mb-3 flex gap-2">
          <input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="Label, e.g. before CLAHE tuning" className="min-w-0 flex-1 rounded-md border border-line bg-panel-2 px-2 py-1 text-xs" aria-label="Version label" />
          <button
            className={`${btn} bg-accent font-medium text-accent-fg`}
            onClick={() => {
              save({ ...all, [pipeline.id]: [{ at: Date.now(), label: label.trim() || `Version ${versions.length + 1}`, pipeline: structuredClone(pipeline) }, ...versions] });
              setLabel('');
            }}
          >
            Save version
          </button>
        </div>
        {!versions.length && <p className="text-sm text-muted">No saved versions yet.</p>}
        <ul className="space-y-2">
          {versions.map((v, i) => {
            const d = diffPipelines(v.pipeline, pipeline);
            return (
              <li key={v.at} className="rounded-md border border-line p-2 text-xs">
                <div className="flex items-center gap-2">
                  <b>{v.label}</b>
                  <span className="text-muted">{new Date(v.at).toLocaleString()}</span>
                  <button className={`${btn} ml-auto`} onClick={() => restore(v)}>
                    Restore
                  </button>
                  <button className={btn} onClick={() => save({ ...all, [pipeline.id]: versions.filter((_, k) => k !== i) })} aria-label={`Delete ${v.label}`}>
                    🗑
                  </button>
                </div>
                <p className="mt-1 text-[11px] text-muted">{d.length ? 'Changes since then:' : 'Same as now.'}</p>
                <ul className="mt-0.5 font-mono text-[11px]">
                  {d.map((e, k) => (
                    <li key={k}>{describeDiff(e)}</li>
                  ))}
                </ul>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}

