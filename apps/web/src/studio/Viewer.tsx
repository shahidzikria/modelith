import { applyH, fmtNum, getStep, type Display, type ImageBuf, type TargetsData } from '@modelith/preprocess';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useRuns, type RunView } from './engine';

import { useActivePipeline, useStudio, type ViewMode } from './store';
import { Thumb } from './Thumb';
import { useViewer } from './viewerState';

export interface Frame {
  label: string;
  display: Display;
  raw: ImageBuf;
  targets: TargetsData;
}

const LABEL_COLORS = [
  [0, 0, 0],
  [230, 75, 75],
  [60, 180, 75],
  [70, 130, 230],
  [240, 190, 40],
  [170, 90, 220],
  [70, 210, 210],
  [240, 120, 200],
];
const labelColor = (k: number) => LABEL_COLORS[k % LABEL_COLORS.length] ?? [255, 255, 255];

/** Input + every step output of a run as frames (index -1 = input). */
export function frameAt(run: RunView | undefined, index: number): Frame | undefined {
  if (!run?.input) return undefined;
  if (index < 0) return { label: 'Input', display: run.input.display, raw: run.input.raw, targets: run.input.targets };
  const s = run.steps[index];
  if (!s) return undefined;
  return { label: `Step ${index + 1}`, display: s.display, raw: s.out.buf, targets: s.out.targets };
}

/** The frame before step `index` (previous step that has a result, else the input). */
export function frameBefore(run: RunView | undefined, index: number): Frame | undefined {
  for (let i = index - 1; i >= 0; i--) {
    const f = frameAt(run, i);
    if (f) return f;
  }
  return frameAt(run, -1);
}

export function pixelValues(buf: ImageBuf, x: number, y: number): number[] | undefined {
  const { w, h, c } = buf.meta;
  if (x < 0 || y < 0 || x >= w || y >= h) return undefined;
  const i = (Math.floor(y) * w + Math.floor(x)) * c;
  return Array.from(buf.data.subarray(i, i + c));
}

/** Map a point in the current frame back to the original image through the recorded geometric transforms. */
export function toOriginal(t: TargetsData, x: number, y: number): [number, number] | undefined {
  let p: [number, number] = [x + 0.5, y + 0.5];
  for (const r of [...t.history].reverse()) {
    if (!r.H) return undefined;
    const H = r.H;
    // 3×3 inverse.
    const [a, b, c, d, e, f, g, h, i] = H;
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const det = a * A + b * B + c * C;
    const inv = [A, -(b * i - c * h), b * f - c * e, B, a * i - c * g, -(a * f - c * d), C, -(a * h - b * g), a * e - b * d].map((v) => v / det);
    p = applyH(inv, p[0], p[1]);
  }
  return [p[0] - 0.5, p[1] - 0.5];
}

/** Absolute difference of two displays (same size), amplified. */
function diffDisplay(a: Display, b: Display, gain: number): Display | undefined {
  if (a.w !== b.w || a.h !== b.h) return undefined;
  const out = new Uint8ClampedArray(a.rgba.length);
  for (let i = 0; i < out.length; i += 4) {
    out[i] = Math.abs(a.rgba[i] - b.rgba[i]) * gain;
    out[i + 1] = Math.abs(a.rgba[i + 1] - b.rgba[i + 1]) * gain;
    out[i + 2] = Math.abs(a.rgba[i + 2] - b.rgba[i + 2]) * gain;
    out[i + 3] = 255;
  }
  return { rgba: out, w: a.w, h: a.h };
}

function toBitmapCanvas(d: Display): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = d.w;
  c.height = d.h;
  c.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(d.rgba), d.w, d.h), 0, 0);
  return c;
}

function maskCanvas(mask: ImageBuf): HTMLCanvasElement {
  const { w, h, c } = mask.meta;
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const k = mask.data[i * c];
    if (!k) continue;
    const [r, g, b] = labelColor(k);
    rgba[i * 4] = r;
    rgba[i * 4 + 1] = g;
    rgba[i * 4 + 2] = b;
    rgba[i * 4 + 3] = 255;
  }
  const cv = document.createElement('canvas');
  cv.width = w;
  cv.height = h;
  cv.getContext('2d')!.putImageData(new ImageData(rgba, w, h), 0, 0);
  return cv;
}

const VIEW_MODES: [ViewMode, string][] = [
  ['output', 'Output'],
  ['before-after', 'Before / after'],
  ['side', 'Side by side'],
  ['diff', 'Difference'],
  ['grid', 'Step grid'],
];

function fmtPx(v: number[] | undefined, dtype: string): string {
  if (!v) return '–';
  return v.map((x) => (dtype.startsWith('float') ? fmtNum(Math.round(x * 1000) / 1000) : String(x))).join(', ');
}

export function Viewer() {
  const pipeline = useActivePipeline();
  const compareId = useStudio((s) => s.compareId);
  const selectedUid = useStudio((s) => s.selectedUid);
  const view = useStudio((s) => s.view);
  const overlays = useStudio((s) => s.overlays);
  const maskAlpha = useStudio((s) => s.maskAlpha);
  const diffGain = useStudio((s) => s.diffGain);
  const activeTool = useStudio((s) => s.activeTool);
  const sample = useStudio((s) => s.samples[s.sampleIndex]);
  const samplesCount = useStudio((s) => s.samples.length);
  const sampleIndex = useStudio((s) => s.sampleIndex);
  const { setView, setOverlay, setMaskAlpha, setDiffGain, setSampleIndex, select, updateParams, setActiveTool } = useStudio.getState();
  const run = useRuns((s) => (s.A?.pipelineId === pipeline.id ? s.A : s.B?.pipelineId === pipeline.id ? s.B : undefined));
  const runOther = useRuns((s) => (compareId ? (s.A?.pipelineId === compareId ? s.A : s.B) : undefined));
  const line = useViewer((s) => s.line);
  const setLine = useViewer((s) => s.setLine);

  const index = selectedUid && selectedUid !== 'read' ? pipeline.steps.findIndex((s) => s.uid === selectedUid) : -1;
  const step = index >= 0 ? pipeline.steps[index] : undefined;
  const def = step ? getStep(step.def) : undefined;
  const toolSpec = activeTool && def ? def.params.find((p) => p.key === activeTool) : undefined;

  // Editing a param on the image happens on the step's input (on its output for spectrum points).
  const after = toolSpec && toolSpec.tool !== 'spectrum-points' ? frameBefore(run, index) : (frameAt(run, index) ?? (run?.steps.length ? undefined : frameAt(run, -1)));
  const before = frameBefore(run, index);

  // ---------------------------------------------------------------- zoom / pan
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tf, setTf] = useState({ s: 1, x: 0, y: 0 });
  const [split, setSplit] = useState(0.5);
  const [hover, setHover] = useState<{ x: number; y: number }>();
  const [size, setSize] = useState({ w: 0, h: 0 });
  const drag = useRef<{ kind: 'pan' | 'line' | 'rect' | 'point' | 'split'; sx: number; sy: number; ox: number; oy: number; pi?: number } | undefined>(undefined);
  const imgW = after?.display.w ?? 0;
  const imgH = after?.display.h ?? 0;

  const fit = useCallback(() => {
    const el = wrapRef.current;
    if (!el || !imgW) return;
    const s = Math.min((el.clientWidth - 16) / imgW, (el.clientHeight - 16) / imgH);
    setTf({ s, x: (el.clientWidth - imgW * s) / 2, y: (el.clientHeight - imgH * s) / 2 });
  }, [imgW, imgH]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useEffect(fit, [fit, size.w, size.h, sample.id]);

  const toImage = (cx: number, cy: number) => {
    const r = canvasRef.current!.getBoundingClientRect();
    return { x: (cx - r.left - tf.x) / tf.s, y: (cy - r.top - tf.y) / tf.s };
  };

  // ---------------------------------------------------------------- drawing
  const afterCanvas = useMemo(() => (after ? toBitmapCanvas(after.display) : undefined), [after?.display]);
  const beforeCanvas = useMemo(() => (before ? toBitmapCanvas(before.display) : undefined), [before?.display]);
  const diff = useMemo(() => (after && before ? diffDisplay(after.display, before.display, diffGain) : undefined), [after?.display, before?.display, diffGain]);
  const diffCanvas = useMemo(() => (diff ? toBitmapCanvas(diff) : undefined), [diff]);
  const maskCv = useMemo(() => (after?.targets.mask ? maskCanvas(after.targets.mask) : undefined), [after?.targets.mask]);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c || view === 'side' || view === 'grid' || compareId) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = size.w * dpr;
    c.height = size.h * dpr;
    const g = c.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, size.w, size.h);
    if (!afterCanvas || !after) return;
    g.save();
    g.translate(tf.x, tf.y);
    g.scale(tf.s, tf.s);
    g.imageSmoothingEnabled = tf.s < 2;
    // Checkerboard under transparent images.
    g.fillStyle = '#8882';
    g.fillRect(0, 0, imgW, imgH);
    if (view === 'diff') {
      if (diffCanvas) g.drawImage(diffCanvas, 0, 0);
      else g.drawImage(afterCanvas, 0, 0);
    } else g.drawImage(afterCanvas, 0, 0);
    if (view === 'before-after' && beforeCanvas && !toolSpec) {
      const sx = imgW * split;
      g.save();
      g.beginPath();
      g.rect(0, 0, sx, imgH);
      g.clip();
      g.drawImage(beforeCanvas, 0, 0, imgW, imgH);
      g.restore();
    }
    const lw = 1.5 / tf.s;
    if (overlays.mask && maskCv && maskCv.width === imgW) {
      g.globalAlpha = maskAlpha;
      g.imageSmoothingEnabled = false;
      g.drawImage(maskCv, 0, 0);
      g.globalAlpha = 1;
    }
    if (overlays.grid) {
      g.strokeStyle = 'rgba(255,255,255,0.45)';
      g.lineWidth = lw;
      for (let k = 1; k < 8; k++) {
        g.beginPath();
        g.moveTo((imgW * k) / 8, 0);
        g.lineTo((imgW * k) / 8, imgH);
        g.moveTo(0, (imgH * k) / 8);
        g.lineTo(imgW, (imgH * k) / 8);
        g.stroke();
      }
    }
    if (overlays.boxes) {
      g.lineWidth = 2 / tf.s;
      g.font = `${12 / tf.s}px sans-serif`;
      for (const b of after.targets.boxes) {
        const [r, gg, bb] = labelColor(Math.round(b[4] ?? 1));
        g.strokeStyle = `rgb(${r},${gg},${bb})`;
        g.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
        if (b[4] !== undefined) {
          g.fillStyle = `rgb(${r},${gg},${bb})`;
          g.fillText(`class ${b[4]}`, b[0] + 2 / tf.s, b[1] - 3 / tf.s);
        }
      }
    }
    if (overlays.keypoints) {
      for (const k of after.targets.keypoints) {
        g.beginPath();
        g.arc(k[0], k[1], 4 / tf.s, 0, Math.PI * 2);
        g.fillStyle = '#ffeb3b';
        g.fill();
        g.lineWidth = 1.5 / tf.s;
        g.strokeStyle = '#000';
        g.stroke();
      }
    }
    // Param editing overlays.
    if (toolSpec && step) {
      const v = step.params[toolSpec.key];
      g.strokeStyle = '#00e5ff';
      g.lineWidth = 2 / tf.s;
      if (toolSpec.type === 'rect') {
        g.setLineDash([6 / tf.s, 4 / tf.s]);
        g.strokeRect(v[0], v[1], v[2], v[3]);
        g.setLineDash([]);
      } else if (toolSpec.type === 'points' || toolSpec.tool === 'spectrum-points') {
        (v as number[][]).forEach((p, i) => {
          g.beginPath();
          g.arc(p[0], p[1], 6 / tf.s, 0, Math.PI * 2);
          g.stroke();
          g.fillStyle = '#00e5ff';
          g.font = `${12 / tf.s}px sans-serif`;
          g.fillText(String(i + 1), p[0] + 8 / tf.s, p[1] - 8 / tf.s);
        });
        if ((v as number[][]).length > 2 && toolSpec.count === 4) {
          g.beginPath();
          (v as number[][]).forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])));
          g.closePath();
          g.stroke();
        }
      }
    }
    if (line) {
      g.strokeStyle = '#ff4081';
      g.lineWidth = 2 / tf.s;
      g.beginPath();
      g.moveTo(line[0], line[1]);
      g.lineTo(line[2], line[3]);
      g.stroke();
    }
    g.restore();
    if (view === 'before-after' && beforeCanvas && !toolSpec) {
      const x = tf.x + imgW * split * tf.s;
      g.fillStyle = '#fff';
      g.fillRect(x - 1, 0, 2, size.h);
      g.beginPath();
      g.arc(x, size.h / 2, 9, 0, Math.PI * 2);
      g.fill();
      g.fillStyle = '#000';
      g.font = '10px sans-serif';
      g.fillText('⇔', x - 5, size.h / 2 + 3);
    }
  }, [afterCanvas, beforeCanvas, diffCanvas, maskCv, tf, size, view, overlays, maskAlpha, split, after, imgW, imgH, line, toolSpec, step, compareId]);

  // ---------------------------------------------------------------- interaction
  const onWheel = (e: React.WheelEvent) => {
    const r = canvasRef.current!.getBoundingClientRect();
    const mx = e.clientX - r.left;
    const my = e.clientY - r.top;
    const k = Math.exp(-e.deltaY * 0.0015);
    setTf((t) => {
      const s = Math.min(64, Math.max(0.05, t.s * k));
      return { s, x: mx - ((mx - t.x) * s) / t.s, y: my - ((my - t.y) * s) / t.s };
    });
  };

  const onPointerDown = (e: React.PointerEvent) => {
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    const p = toImage(e.clientX, e.clientY);
    const splitX = tf.x + imgW * split * tf.s;
    const r = canvasRef.current!.getBoundingClientRect();
    if (view === 'before-after' && !toolSpec && Math.abs(e.clientX - r.left - splitX) < 12) {
      drag.current = { kind: 'split', sx: e.clientX, sy: e.clientY, ox: 0, oy: 0 };
      return;
    }
    if (toolSpec && step) {
      if (toolSpec.type === 'rect') drag.current = { kind: 'rect', sx: p.x, sy: p.y, ox: 0, oy: 0 };
      else if (toolSpec.type === 'points') {
        const pts = step.params[toolSpec.key] as number[][];
        const near = pts.findIndex((q) => Math.hypot(q[0] - p.x, q[1] - p.y) * tf.s < 12);
        if (e.button === 2 && near >= 0) {
          updateParams(step.uid, { [toolSpec.key]: pts.filter((_, i) => i !== near) });
          return;
        }
        if (near >= 0) drag.current = { kind: 'point', sx: p.x, sy: p.y, ox: 0, oy: 0, pi: near };
        else if (!toolSpec.count || pts.length < toolSpec.count) updateParams(step.uid, { [toolSpec.key]: [...pts, [Math.round(p.x), Math.round(p.y)]] });
      } else if (toolSpec.tool === 'eyedropper' && def?.onPick && after) {
        const px = pixelValues(after.raw, p.x, p.y);
        if (px) updateParams(step.uid, def.onPick(px, after.raw.meta, step.params));
      }
      return;
    }
    if (e.shiftKey) {
      drag.current = { kind: 'line', sx: p.x, sy: p.y, ox: 0, oy: 0 };
      setLine([p.x, p.y, p.x, p.y]);
      return;
    }
    drag.current = { kind: 'pan', sx: e.clientX, sy: e.clientY, ox: tf.x, oy: tf.y };
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const p = toImage(e.clientX, e.clientY);
    setHover(p);
    const d = drag.current;
    if (!d) return;
    if (d.kind === 'pan') setTf((t) => ({ ...t, x: d.ox + e.clientX - d.sx, y: d.oy + e.clientY - d.sy }));
    else if (d.kind === 'split') {
      const r = canvasRef.current!.getBoundingClientRect();
      setSplit(Math.min(1, Math.max(0, (e.clientX - r.left - tf.x) / (imgW * tf.s))));
    } else if (d.kind === 'line') setLine([d.sx, d.sy, p.x, p.y]);
    else if (d.kind === 'rect' && step && toolSpec) {
      const x0 = Math.max(0, Math.round(Math.min(d.sx, p.x)));
      const y0 = Math.max(0, Math.round(Math.min(d.sy, p.y)));
      const x1 = Math.min(imgW, Math.round(Math.max(d.sx, p.x)));
      const y1 = Math.min(imgH, Math.round(Math.max(d.sy, p.y)));
      if (x1 - x0 >= 2 && y1 - y0 >= 2) updateParams(step.uid, { [toolSpec.key]: [x0, y0, x1 - x0, y1 - y0] });
    } else if (d.kind === 'point' && step && toolSpec) {
      const pts = (step.params[toolSpec.key] as number[][]).map((q, i) => (i === d.pi ? [Math.round(Math.min(imgW, Math.max(0, p.x))), Math.round(Math.min(imgH, Math.max(0, p.y)))] : q));
      updateParams(step.uid, { [toolSpec.key]: pts });
    }
  };

  const onPointerUp = () => {
    if (drag.current?.kind === 'line' && line && Math.hypot(line[2] - line[0], line[3] - line[1]) < 2) setLine(undefined);
    drag.current = undefined;
  };

  // ---------------------------------------------------------------- inspector readout
  const hv = hover && after ? pixelValues(after.raw, hover.x, hover.y) : undefined;
  const orig = hover && after && run?.input ? toOriginal(after.targets, hover.x, hover.y) : undefined;
  const ov = orig && run?.input ? pixelValues(run.input.raw, orig[0], orig[1]) : undefined;

  const btn = (on: boolean) => `rounded px-2 py-1 text-[11px] ${on ? 'bg-accent text-accent-fg' : 'text-muted hover:text-fg'}`;
  const serverMsg = run?.serverState === 'error' ? run.serverError : run?.serverState === 'running' ? 'Running the rest of the pipeline on the Python server…' : undefined;

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-wrap items-center gap-1 border-b border-line px-2 py-1">
        {!compareId && (
          <div className="flex rounded-md border border-line p-0.5" role="radiogroup" aria-label="View mode">
            {VIEW_MODES.map(([m, label]) => (
              <button key={m} role="radio" aria-checked={view === m} onClick={() => setView(m)} className={btn(view === m)}>
                {label}
              </button>
            ))}
          </div>
        )}
        {compareId && <span className="px-1 text-[11px] font-medium">Comparing variant A and B (final outputs)</span>}
        <div className="flex items-center gap-1.5 px-1 text-[11px]" role="group" aria-label="Overlays">
          {(['mask', 'boxes', 'keypoints', 'grid'] as const).map((k) => (
            <label key={k} className="flex items-center gap-1 text-muted">
              <input type="checkbox" checked={overlays[k]} onChange={(e) => setOverlay(k, e.target.checked)} className="accent-[var(--accent)]" />
              {k}
            </label>
          ))}
          {overlays.mask && (
            <input aria-label="Mask opacity" type="range" min={0.1} max={1} step={0.05} value={maskAlpha} onChange={(e) => setMaskAlpha(Number(e.target.value))} className="w-16 accent-[var(--accent)]" />
          )}
          {view === 'diff' && (
            <label className="flex items-center gap-1 text-muted">
              gain
              <input aria-label="Difference amplification" type="range" min={1} max={32} step={1} value={diffGain} onChange={(e) => setDiffGain(Number(e.target.value))} className="w-16 accent-[var(--accent)]" />×{diffGain}
            </label>
          )}
        </div>
        <div className="ml-auto flex items-center gap-1 text-[11px]">
          {samplesCount > 1 && (
            <>
              <button onClick={() => setSampleIndex(sampleIndex - 1)} className="rounded border border-line px-1.5" aria-label="Previous sample (←)">
                ◀
              </button>
              <span className="text-muted">
                {sampleIndex + 1}/{samplesCount}
              </span>
              <button onClick={() => setSampleIndex(sampleIndex + 1)} className="rounded border border-line px-1.5" aria-label="Next sample (→)">
                ▶
              </button>
            </>
          )}
          <span className="max-w-40 truncate text-muted" title={sample.name}>
            {sample.name}
          </span>
          <button onClick={fit} className="rounded border border-line px-1.5 py-0.5" title="Fit (double-click the image)">
            Fit
          </button>
          <button onClick={() => setTf((t) => ({ s: 1, x: t.x, y: t.y }))} className="rounded border border-line px-1.5 py-0.5" title="Actual pixels">
            1:1
          </button>
        </div>
      </div>

      <div ref={wrapRef} className="checker-soft relative min-h-0 flex-1 overflow-hidden">
        {compareId ? (
          <ComparePanes a={runOther} b={run} gain={diffGain} />
        ) : view === 'side' ? (
          <div className="grid h-full grid-cols-2 gap-2 p-2">
            {[before, after].map((f, i) => (
              <figure key={i} className="flex min-h-0 flex-col items-center justify-center">
                {f ? <Thumb display={f.display} label={i ? 'After' : 'Before'} /> : null}
                <figcaption className="mt-1 text-[11px] text-muted">{i ? `After: ${step ? `${index + 1}. ${def?.label}` : 'input'}` : `Before: ${before?.label ?? '–'}`}</figcaption>
              </figure>
            ))}
          </div>
        ) : view === 'grid' ? (
          <StepGrid run={run} />
        ) : (
          <canvas
            ref={canvasRef}
            className={`absolute inset-0 h-full w-full touch-none ${toolSpec ? 'cursor-crosshair' : 'cursor-grab'}`}
            role="img"
            aria-label={after ? `${after.label} result` : 'No result yet'}
            onWheel={onWheel}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerLeave={() => setHover(undefined)}
            onDoubleClick={fit}
            onContextMenu={(e) => toolSpec && e.preventDefault()}
          />
        )}
        {!run?.input && <p className="absolute inset-0 flex items-center justify-center text-sm text-muted">Loading OpenCV.js…</p>}
        {toolSpec && (
          <div className="absolute top-2 left-2 rounded-md bg-black/70 px-2 py-1 text-[11px] text-white">
            Editing “{toolSpec.label}” on the step's input ·{' '}
            {toolSpec.type === 'rect' ? 'drag a rectangle' : toolSpec.type === 'points' ? 'click to add, drag to move, right-click to delete' : 'click a pixel'} ·{' '}
            <button className="underline" onClick={() => setActiveTool(undefined)}>
              done
            </button>
          </div>
        )}
        {after?.display.scaled && (
          <span className="absolute top-2 right-2 rounded bg-black/70 px-1.5 py-0.5 text-[10px] text-white" title={`Values ${fmtNum(after.display.scaled[0])} … ${fmtNum(after.display.scaled[1])} mapped to black … white for display only`}>
            display scaled{after.display.falseColor ? ' · false colour' : ''}
          </span>
        )}
        {run?.proxy && <span className="absolute right-2 bottom-2 rounded bg-black/70 px-1.5 py-0.5 text-[10px] text-white">low-res preview</span>}
        {serverMsg && (
          <div className={`absolute inset-x-2 bottom-2 rounded-md px-2 py-1.5 text-xs ${run?.serverState === 'error' ? 'bg-danger/90 text-white' : 'bg-black/70 text-white'}`}>
            {serverMsg}
          </div>
        )}
        {run?.error && run.errorAt !== undefined && (
          <div className="absolute inset-x-2 bottom-2 rounded-md bg-danger/90 px-2 py-1.5 text-xs text-white">
            Step {run.errorAt + 1} failed: {run.error}
          </div>
        )}
      </div>
      <div className="flex min-h-6 items-center gap-3 border-t border-line px-2 py-0.5 font-mono text-[11px] text-muted" aria-live="off">
        {after && hover && hv ? (
          <>
            <span>
              x {Math.floor(hover.x)}, y {Math.floor(hover.y)}
            </span>
            <span className="text-fg">
              {after.raw.meta.space}({fmtPx(hv, after.raw.meta.dtype)})
            </span>
            <span>{after.raw.meta.dtype}</span>
            {ov && run?.input && (
              <span>
                original ({Math.floor(orig![0])}, {Math.floor(orig![1])}) {run.input.raw.meta.space}({fmtPx(ov, run.input.raw.meta.dtype)})
              </span>
            )}
          </>
        ) : (
          <span>
            {after ? `${after.raw.meta.w}×${after.raw.meta.h}×${after.raw.meta.c} ${after.raw.meta.dtype} ${after.raw.meta.space}` : ''} · scroll to zoom, drag to pan, Shift+drag for a line profile
          </span>
        )}
      </div>
    </div>
  );
}

function StepGrid({ run }: { run?: RunView }) {
  const pipeline = useActivePipeline();
  const select = useStudio((s) => s.select);
  const selectedUid = useStudio((s) => s.selectedUid);
  const frames = [{ uid: 'read', label: '0 · input', f: frameAt(run, -1) }, ...pipeline.steps.map((s, i) => ({ uid: s.uid, label: `${i + 1} · ${getStep(s.def)?.label ?? s.def}`, f: frameAt(run, i), skipped: run?.steps[i]?.out.skipped }))];
  return (
    <div className="scroll-thin grid h-full auto-rows-min grid-cols-2 gap-2 overflow-y-auto p-2 sm:grid-cols-3 xl:grid-cols-4">
      {frames.map(({ uid, label, f, skipped }: any) => (
        <button key={uid} onClick={() => select(uid)} className={`overflow-hidden rounded-md border bg-panel text-left ${selectedUid === uid ? 'border-accent' : 'border-line hover:border-accent/60'} ${skipped ? 'opacity-50' : ''}`}>
          <div className="flex aspect-[4/3] items-center justify-center bg-panel-2">{f ? <Thumb display={f.display} label={label} /> : <span className="text-xs text-muted">…</span>}</div>
          <p className="truncate px-2 py-1 text-[11px]">
            {label}
            {skipped ? ' (off)' : ''}
          </p>
        </button>
      ))}
    </div>
  );
}

function lastFrame(run?: RunView): Frame | undefined {
  if (!run) return undefined;
  for (let i = run.steps.length - 1; i >= 0; i--) {
    const f = frameAt(run, i);
    if (f) return f;
  }
  return frameAt(run, -1);
}

function ComparePanes({ a, b, gain }: { a?: RunView; b?: RunView; gain: number }) {
  const pipelines = useStudio((s) => s.pipelines);
  const fa = lastFrame(a);
  const fb = lastFrame(b);
  const d = fa && fb ? diffDisplay(fa.display, fb.display, gain) : undefined;
  const name = (r?: RunView) => pipelines.find((p) => p.id === r?.pipelineId)?.name ?? '';
  return (
    <div className="grid h-full grid-cols-3 gap-2 p-2">
      {[
        [fa, `A · ${name(a)}`],
        [fb, `B · ${name(b)} (editing)`],
      ].map(([f, label], i) => (
        <figure key={i} className="flex min-h-0 flex-col items-center justify-center">
          {f ? <Thumb display={(f as Frame).display} label={label as string} /> : <span className="text-xs text-muted">…</span>}
          <figcaption className="mt-1 truncate text-[11px] text-muted">{label as string}</figcaption>
        </figure>
      ))}
      <figure className="flex min-h-0 flex-col items-center justify-center">
        {d ? <Thumb display={d} label="Difference A − B" /> : <span className="p-2 text-center text-xs text-muted">{fa && fb ? 'Outputs differ in size' : '…'}</span>}
        <figcaption className="mt-1 text-[11px] text-muted">|A − B| × {gain}</figcaption>
      </figure>
    </div>
  );
}
