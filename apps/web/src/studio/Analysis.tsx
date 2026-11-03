import { fmtNum, getStep, quality, type ImageBuf } from '@modelith/preprocess';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useRuns } from './engine';
import { useActivePipeline, useStudio } from './store';
import { frameAt, frameBefore } from './Viewer';
import { useViewer } from './viewerState';

const CH_COLORS: Record<string, string[]> = {
  BGR: ['#3b82f6', '#22c55e', '#ef4444'],
  RGB: ['#ef4444', '#22c55e', '#3b82f6'],
  BGRA: ['#3b82f6', '#22c55e', '#ef4444', '#9ca3af'],
  RGBA: ['#ef4444', '#22c55e', '#3b82f6', '#9ca3af'],
};
const chColors = (space: string, c: number) => CH_COLORS[space]?.slice(0, c) ?? ['#a3a3a3', '#f59e0b', '#8b5cf6', '#9ca3af'].slice(0, c);
const chNames = (space: string, c: number) => {
  const m: Record<string, string[]> = { HSV: ['H', 'S', 'V'], HLS: ['H', 'L', 'S'], LAB: ['L', 'a', 'b'], LUV: ['L', 'u', 'v'], YCrCb: ['Y', 'Cr', 'Cb'], XYZ: ['X', 'Y', 'Z'], GRAY: ['gray'] };
  return (m[space] ?? space.split('')).slice(0, c);
};

export interface ChannelStats {
  min: number;
  max: number;
  mean: number;
  std: number;
}

export function stats(buf: ImageBuf): { ch: ChannelStats[]; unique?: number } {
  const c = buf.meta.c;
  const d = buf.data;
  const out: ChannelStats[] = [];
  for (let k = 0; k < c; k++) {
    let mn = Infinity, mx = -Infinity, s = 0, s2 = 0, n = 0;
    for (let i = k; i < d.length; i += c) {
      const v = d[i];
      if (!Number.isFinite(v)) continue;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      s += v;
      s2 += v * v;
      n++;
    }
    const mean = s / Math.max(1, n);
    out.push({ min: mn, max: mx, mean, std: Math.sqrt(Math.max(0, s2 / Math.max(1, n) - mean * mean)) });
  }
  let unique: number | undefined;
  if (c === 1 && buf.meta.dtype !== 'float32') {
    const seen = new Set<number>();
    for (let i = 0; i < d.length && seen.size <= 64; i++) seen.add(d[i]);
    if (seen.size <= 64) unique = seen.size;
  }
  return { ch: out, unique };
}

/** 256-bin histogram per channel over [lo, hi]. */
function histogram(buf: ImageBuf, lo: number, hi: number): Float64Array[] {
  const c = buf.meta.c;
  const hs = Array.from({ length: c }, () => new Float64Array(256));
  const k = 255 / (hi - lo || 1);
  const d = buf.data;
  for (let i = 0; i < d.length; i++) {
    const b = Math.round((d[i] - lo) * k);
    if (b >= 0 && b < 256) hs[i % c][b]++;
  }
  return hs;
}

function rangeOf(bufs: (ImageBuf | undefined)[]): [number, number] {
  const b = bufs.find(Boolean)!;
  if (b.meta.dtype === 'uint8') return [0, 255];
  let lo = Infinity, hi = -Infinity;
  for (const x of bufs) {
    if (!x) continue;
    for (const s of stats(x).ch) {
      lo = Math.min(lo, s.min);
      hi = Math.max(hi, s.max);
    }
  }
  return [lo, hi > lo ? hi : lo + 1];
}

function Histogram({ after, before }: { after: ImageBuf; before?: ImageBuf }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const comparable = before && before.meta.c === after.meta.c && before.meta.dtype === after.meta.dtype;
  const [lo, hi] = useMemo(() => rangeOf([after, comparable ? before : undefined]), [after, before, comparable]);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth, H = c.clientHeight;
    c.width = W * dpr;
    c.height = H * dpr;
    const g = c.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const ha = histogram(after, lo, hi);
    const hb = comparable ? histogram(before!, lo, hi) : undefined;
    const peak = Math.max(1, ...ha.map((h) => Math.max(...h)), ...(hb?.map((h) => Math.max(...h)) ?? []));
    const colors = chColors(after.meta.space, after.meta.c);
    const draw = (h: Float64Array, color: string, dashed: boolean) => {
      g.beginPath();
      for (let i = 0; i < 256; i++) {
        const x = (i / 255) * W;
        const y = H - (Math.sqrt(h[i] / peak) * (H - 4));
        if (i) g.lineTo(x, y);
        else g.moveTo(x, y);
      }
      g.strokeStyle = color;
      g.globalAlpha = dashed ? 0.45 : 0.95;
      g.setLineDash(dashed ? [3, 3] : []);
      g.lineWidth = 1.25;
      g.stroke();
    };
    hb?.forEach((h, i) => draw(h, colors[i], true));
    ha.forEach((h, i) => draw(h, colors[i], false));
    g.globalAlpha = 1;
    g.setLineDash([]);
  }, [after, before, lo, hi, comparable]);
  return (
    <figure className="flex min-h-0 min-w-0 flex-1 flex-col">
      <canvas ref={ref} className="min-h-0 w-full flex-1 rounded border border-line bg-panel-2" role="img" aria-label="Histogram per channel, before (dashed) and after" />
      <figcaption className="flex justify-between font-mono text-[10px] text-muted">
        <span>{fmtNum(lo)}</span>
        <span>{comparable ? 'solid: after · dashed: before (√ scale)' : 'after (√ scale)'}</span>
        <span>{fmtNum(hi)}</span>
      </figcaption>
    </figure>
  );
}

function LineProfile({ after, before }: { after: ImageBuf; before?: ImageBuf }) {
  const line = useViewer((s) => s.line);
  const setLine = useViewer((s) => s.setLine);
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || !line) return;
    const dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth, H = c.clientHeight;
    c.width = W * dpr;
    c.height = H * dpr;
    const g = c.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const n = Math.max(2, Math.round(Math.hypot(line[2] - line[0], line[3] - line[1])));
    const sampleAlong = (b: ImageBuf) => {
      const { w, h, c: ch } = b.meta;
      const vals: number[][] = Array.from({ length: ch }, () => []);
      for (let i = 0; i < n; i++) {
        const x = Math.floor(line[0] + ((line[2] - line[0]) * i) / (n - 1));
        const y = Math.floor(line[1] + ((line[3] - line[1]) * i) / (n - 1));
        for (let k = 0; k < ch; k++) vals[k].push(x >= 0 && y >= 0 && x < w && y < h ? b.data[(y * w + x) * ch + k] : NaN);
      }
      return vals;
    };
    const va = sampleAlong(after);
    const vb = before && before.meta.w === after.meta.w && before.meta.h === after.meta.h && before.meta.c === after.meta.c ? sampleAlong(before) : undefined;
    const all = [...va, ...(vb ?? [])].flat().filter(Number.isFinite);
    const lo = Math.min(...all), hi = Math.max(...all);
    const colors = chColors(after.meta.space, after.meta.c);
    const plot = (v: number[], color: string, dashed: boolean) => {
      g.beginPath();
      v.forEach((y, i) => {
        const px = (i / (n - 1)) * W;
        const py = H - 2 - ((y - lo) / (hi - lo || 1)) * (H - 4);
        if (i) g.lineTo(px, py);
        else g.moveTo(px, py);
      });
      g.strokeStyle = color;
      g.setLineDash(dashed ? [3, 3] : []);
      g.globalAlpha = dashed ? 0.45 : 1;
      g.stroke();
    };
    vb?.forEach((v, i) => plot(v, colors[i], true));
    va.forEach((v, i) => plot(v, colors[i], false));
    g.globalAlpha = 1;
    g.setLineDash([]);
  }, [line, after, before]);
  if (!line)
    return (
      <div className="flex min-w-0 flex-1 items-center justify-center rounded border border-dashed border-line p-2 text-center text-[11px] text-muted">
        Shift+drag on the image to plot intensity along a line (good for judging blur, sharpening and edges).
      </div>
    );
  return (
    <figure className="flex min-h-0 min-w-0 flex-1 flex-col">
      <canvas ref={ref} className="min-h-0 w-full flex-1 rounded border border-line bg-panel-2" role="img" aria-label="Intensity along the drawn line" />
      <figcaption className="flex justify-between text-[10px] text-muted">
        <span>line profile ({Math.round(Math.hypot(line[2] - line[0], line[3] - line[1]))} px)</span>
        <button onClick={() => setLine(undefined)} className="underline">
          clear
        </button>
      </figcaption>
    </figure>
  );
}

export function Analysis() {
  const pipeline = useActivePipeline();
  const run = useRuns((s) => (s.A?.pipelineId === pipeline.id ? s.A : s.B));
  const selectedUid = useStudio((s) => s.selectedUid);
  const display = useStudio((s) => s.display);
  const setDisplay = useStudio((s) => s.setDisplay);
  const [tab, setTab] = useState<'histogram' | 'quality' | 'timing'>('histogram');
  const index = selectedUid && selectedUid !== 'read' ? pipeline.steps.findIndex((s) => s.uid === selectedUid) : -1;
  const after = frameAt(run, index);
  const before = index >= 0 ? frameBefore(run, index) : undefined;
  const st = useMemo(() => (after ? stats(after.raw) : undefined), [after?.raw]);
  const total = run?.steps.reduce((a, s) => a + (s && !s.out.skipped ? s.out.ms : 0), 0) ?? 0;

  return (
    <section className="flex h-full flex-col" aria-label="Analysis">
      <div className="flex items-center gap-1 border-b border-line px-2 py-1 text-[11px]">
        {(['histogram', 'quality', 'timing'] as const).map((t) => (
          <button key={t} onClick={() => setTab(t)} aria-pressed={tab === t} className={`rounded px-2 py-0.5 ${tab === t ? 'bg-panel-2 text-fg' : 'text-muted hover:text-fg'}`}>
            {t === 'histogram' ? 'Histogram & stats' : t === 'quality' ? 'Quality' : 'Timing'}
          </button>
        ))}
        <span className="ml-auto flex items-center gap-1.5 text-muted">
          <label className="flex items-center gap-1">
            show
            <select
              aria-label="Channel to display"
              value={String(display.channel)}
              onChange={(e) => setDisplay({ channel: e.target.value === 'all' ? 'all' : (Number(e.target.value) as 0 | 1 | 2 | 3) })}
              className="rounded border border-line bg-panel px-1"
            >
              <option value="all">all channels</option>
              {after && after.raw.meta.c > 1 && chNames(after.raw.meta.space, after.raw.meta.c).map((n, i) => <option key={i} value={i}>channel {n}</option>)}
            </select>
          </label>
          <label className="flex items-center gap-1" title="HSV / LAB / … shown converted back to RGB, or the raw stored channels">
            <input type="checkbox" checked={display.color === 'raw'} onChange={(e) => setDisplay({ color: e.target.checked ? 'raw' : 'auto' })} className="accent-[var(--accent)]" />
            raw channels
          </label>
          <label className="flex items-center gap-1">
            <input type="checkbox" checked={display.falseColor} onChange={(e) => setDisplay({ falseColor: e.target.checked })} className="accent-[var(--accent)]" />
            false colour
          </label>
        </span>
      </div>
      {tab === 'quality' ? (
        <QualityPanel after={after?.raw} before={before?.raw} input={run?.input?.raw} />
      ) : tab === 'histogram' ? (
        after && st ? (
          <div className="flex min-h-0 flex-1 flex-col gap-2 p-2 md:flex-row">
            <Histogram after={after.raw} before={before?.raw} />
            <LineProfile after={after.raw} before={before?.raw} />
            <div className="scroll-thin shrink-0 overflow-auto md:w-56">
              <p className="font-mono text-[11px]">
                {after.raw.meta.layout === 'CHW' ? `${after.raw.meta.c}×${after.raw.meta.h}×${after.raw.meta.w}` : `${after.raw.meta.h}×${after.raw.meta.w}×${after.raw.meta.c}`} {after.raw.meta.dtype} · {after.raw.meta.space}
                {st.unique !== undefined ? ` · ${st.unique} unique` : ''}
              </p>
              <table className="mt-1 w-full font-mono text-[10.5px]">
                <thead className="text-muted">
                  <tr>
                    <th className="text-left font-normal">ch</th>
                    <th className="text-right font-normal">min</th>
                    <th className="text-right font-normal">max</th>
                    <th className="text-right font-normal">mean</th>
                    <th className="text-right font-normal">std</th>
                  </tr>
                </thead>
                <tbody>
                  {st.ch.map((s, i) => (
                    <tr key={i}>
                      <td style={{ color: chColors(after.raw.meta.space, after.raw.meta.c)[i] }}>{chNames(after.raw.meta.space, after.raw.meta.c)[i]}</td>
                      <td className="text-right">{fmtNum(s.min)}</td>
                      <td className="text-right">{fmtNum(s.max)}</td>
                      <td className="text-right">{fmtNum(s.mean)}</td>
                      <td className="text-right">{fmtNum(s.std)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <p className="p-3 text-xs text-muted">No result for this step yet.</p>
        )
      ) : (
        <div className="scroll-thin min-h-0 flex-1 overflow-auto p-2">
          <table className="w-full text-[11px]">
            <tbody>
              {pipeline.steps.map((s, i) => {
                const r = run?.steps[i]?.out;
                const ms = r && !r.skipped ? r.ms : 0;
                return (
                  <tr key={s.uid} className={r?.skipped ? 'text-muted' : ''}>
                    <td className="w-6 text-right font-mono text-muted">{i + 1}</td>
                    <td className="truncate px-2">{getStep(s.def)?.label ?? s.def}</td>
                    <td className="w-1/3">
                      <div className="h-2 rounded bg-accent/70" style={{ width: `${total ? (100 * ms) / total : 0}%` }} />
                    </td>
                    <td className="w-20 text-right font-mono">{r ? (r.skipped ? 'off' : `${ms.toFixed(1)} ms`) : '…'}</td>
                    <td className="w-14 text-right text-muted">{r?.engine === 'server' ? 'server' : r ? 'browser' : ''}</td>
                  </tr>
                );
              })}
            </tbody>
            <tfoot>
              <tr className="border-t border-line font-medium">
                <td />
                <td className="px-2">Total{run?.proxy ? ' (low-res proxy)' : ''}</td>
                <td />
                <td className="text-right font-mono">{total.toFixed(1)} ms</td>
                <td className="text-right text-muted">{total ? `${Math.round(1000 / total)} img/s` : ''}</td>
              </tr>
            </tfoot>
          </table>
          <p className="mt-2 text-[11px] text-muted">Browser times (OpenCV.js, single thread). The Python timings for the exact exported code come from “Measure on server” in the Export panel.</p>
        </div>
      )}
    </section>
  );
}

function QualityPanel({ after, before, input }: { after?: ImageBuf; before?: ImageBuf; input?: ImageBuf }) {
  const q = useMemo(() => (after && after.meta.layout === 'HWC' ? { prev: quality(before, after), orig: input && input !== after ? quality(input, after) : undefined, inq: input ? quality(undefined, input) : undefined } : undefined), [after, before, input]);
  if (!q) return <p className="p-3 text-xs text-muted">No result for this step (or the output is a CHW tensor).</p>;
  const f = (v?: number, d = 2) => (v === undefined ? '–' : v === Infinity ? '∞' : v.toFixed(d));
  return (
    <div className="p-3 text-xs">
      <table className="font-mono text-[11.5px]">
        <tbody>
          <tr><td className="pr-4 text-muted">PSNR vs previous step</td><td>{f(q.prev.psnr)} dB</td></tr>
          <tr><td className="pr-4 text-muted">SSIM vs previous step</td><td>{f(q.prev.ssim, 4)}</td></tr>
          <tr><td className="pr-4 text-muted">PSNR / SSIM vs input</td><td>{f(q.orig?.psnr)} dB / {f(q.orig?.ssim, 4)}</td></tr>
          <tr><td className="pr-4 text-muted">Entropy (input → this step)</td><td>{f(q.inq?.entropy)} → {f(q.prev.entropy)} bits</td></tr>
          <tr><td className="pr-4 text-muted">Sharpness, var. of Laplacian (input → this step)</td><td>{f(q.inq?.sharpness, 0)} → {f(q.prev.sharpness, 0)}</td></tr>
        </tbody>
      </table>
      <p className="mt-2 text-[11px] text-muted">PSNR / SSIM need the same size, channels and dtype; “–” means the step changed them.</p>
    </div>
  );
}
