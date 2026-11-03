import type { ParamSpec } from '@modelith/preprocess';
import { useState } from 'react';
import { beginGesture, endGesture, useStudio } from './store';

const inputCls = 'w-full min-w-0 rounded-md border border-line bg-panel-2 px-2 py-1 text-xs outline-none focus:border-accent';

const toOdd = (v: number, min = 1) => {
  const n = Math.max(min, Math.round(v));
  return n % 2 ? n : n + 1;
};

function clamp(spec: ParamSpec, v: number): number {
  let x = v;
  if (spec.min !== undefined) x = Math.max(spec.min, x);
  if (spec.max !== undefined) x = Math.min(spec.max, x);
  if (spec.type === 'int') x = Math.round(x);
  if (spec.type === 'oddInt') x = toOdd(x, spec.min ?? 1);
  return x;
}

/** Slider + number box; the slider previews on a proxy while dragging and records one undo step. */
function NumberField({ spec, value, onChange, id }: { spec: ParamSpec; value: number; onChange(v: number): void; id: string }) {
  const [text, setText] = useState<string>();
  const hasSlider = spec.min !== undefined && spec.max !== undefined;
  const step = spec.step ?? (spec.type === 'float' ? 0.01 : spec.type === 'oddInt' ? 2 : 1);
  return (
    <div className="flex items-center gap-2">
      {hasSlider && (
        <input
          type="range"
          aria-label={`${spec.label} slider`}
          min={spec.min}
          max={spec.max}
          step={step}
          value={value}
          onPointerDown={beginGesture}
          onPointerUp={endGesture}
          onKeyUp={endGesture}
          onBlur={endGesture}
          onChange={(e) => onChange(clamp(spec, Number(e.target.value)))}
          className="min-w-0 flex-1 accent-[var(--accent)]"
        />
      )}
      <input
        id={id}
        type="number"
        inputMode="decimal"
        min={spec.min}
        max={spec.max}
        step={step}
        value={text ?? String(value)}
        onChange={(e) => {
          setText(e.target.value);
          const v = Number(e.target.value);
          if (e.target.value.trim() !== '' && Number.isFinite(v) && (spec.type !== 'oddInt' || v % 2 === 1)) onChange(clamp(spec, v));
        }}
        onBlur={(e) => {
          setText(undefined);
          const v = Number(e.target.value);
          if (Number.isFinite(v)) onChange(clamp(spec, v));
        }}
        className={`${inputCls} ${hasSlider ? '!w-20 shrink-0' : ''}`}
      />
    </div>
  );
}

function SizeField({ spec, value, onChange, id }: { spec: ParamSpec; value: [number, number]; onChange(v: [number, number]): void; id: string }) {
  const [locked, setLocked] = useState(false);
  const set = (i: 0 | 1, v: number) => {
    if (!Number.isFinite(v) || v < 1) return;
    const next: [number, number] = [...value] as [number, number];
    next[i] = Math.round(v);
    if (locked) next[1 - i] = Math.max(1, Math.round((v * value[1 - i]) / value[i]));
    onChange(next);
  };
  return (
    <div className="flex items-center gap-1.5">
      <input id={id} aria-label={`${spec.label} width`} type="number" min={1} value={value[0]} onChange={(e) => set(0, Number(e.target.value))} className={inputCls} />
      <button
        type="button"
        aria-label={locked ? 'Unlock aspect ratio' : 'Lock aspect ratio'}
        aria-pressed={locked}
        title={locked ? 'Width and height are linked' : 'Link width and height'}
        onClick={() => setLocked(!locked)}
        className={`shrink-0 rounded border px-1.5 py-0.5 text-xs ${locked ? 'border-accent text-accent' : 'border-line text-muted'}`}
      >
        {locked ? '🔒' : '🔓'}
      </button>
      <input aria-label={`${spec.label} height`} type="number" min={1} value={value[1]} onChange={(e) => set(1, Number(e.target.value))} className={inputCls} />
    </div>
  );
}

function MatrixField({ spec, value, onChange }: { spec: ParamSpec; value: number[][]; onChange(v: number[][]): void }) {
  const rows = value.length;
  const cols = value[0]?.length ?? 0;
  const resize = (r: number, c: number) => {
    if (r < 1 || c < 1 || r > 9 || c > 9) return;
    onChange(Array.from({ length: r }, (_, i) => Array.from({ length: c }, (_, j) => value[i]?.[j] ?? 0)));
  };
  return (
    <div>
      <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
        {value.map((row, i) =>
          row.map((v, j) => (
            <input
              key={`${i}-${j}`}
              aria-label={`${spec.label} row ${i + 1} column ${j + 1}`}
              type="number"
              step={0.1}
              value={v}
              onChange={(e) => {
                const n = Number(e.target.value);
                if (!Number.isFinite(n)) return;
                onChange(value.map((r, a) => r.map((x, b) => (a === i && b === j ? n : x))));
              }}
              className={`${inputCls} px-1 text-center font-mono`}
            />
          )),
        )}
      </div>
      <div className="mt-1 flex items-center gap-1 text-[11px] text-muted">
        {rows}×{cols}
        <button type="button" className="rounded border border-line px-1" onClick={() => resize(rows + 2, cols + 2)}>
          bigger
        </button>
        <button type="button" className="rounded border border-line px-1" onClick={() => resize(rows - 2, cols - 2)}>
          smaller
        </button>
        <span className="ml-auto">sum {value.flat().reduce((a, b) => a + b, 0).toFixed(3)}</span>
      </div>
    </div>
  );
}

export function ParamField({ uid, spec, value, isDefault }: { uid: string; spec: ParamSpec; value: any; isDefault: boolean }) {
  const updateParams = useStudio((s) => s.updateParams);
  const resetParam = useStudio((s) => s.resetParam);
  const tool = useStudio((s) => (s.selectedUid === uid ? s.activeTool : undefined));
  const setTool = useStudio((s) => s.setActiveTool);
  const set = (v: unknown) => updateParams(uid, { [spec.key]: v });
  const id = `param-${uid}-${spec.key}`;

  let control: React.ReactNode;
  switch (spec.type) {
    case 'int':
    case 'float':
    case 'oddInt':
      control = <NumberField spec={spec} value={value} onChange={set} id={id} />;
      break;
    case 'bool':
      control = <input id={id} type="checkbox" checked={!!value} onChange={(e) => set(e.target.checked)} className="accent-[var(--accent)]" />;
      break;
    case 'select':
      control = (
        <select id={id} value={value} onChange={(e) => set(e.target.value)} className={inputCls}>
          {spec.options!.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      );
      break;
    case 'string':
      control = <input id={id} value={value} onChange={(e) => set(e.target.value)} className={inputCls} />;
      break;
    case 'color':
      control = (
        <div className="flex items-center gap-2">
          <input id={id} type="color" value={value} onChange={(e) => set(e.target.value)} className="h-7 w-10 cursor-pointer rounded border border-line bg-transparent" />
          <span className="font-mono text-xs text-muted">{value}</span>
        </div>
      );
      break;
    case 'size':
      control = <SizeField spec={spec} value={value} onChange={set} id={id} />;
      break;
    case 'range':
    case 'floats':
    case 'rect':
      control = (
        <div className="flex gap-1">
          {(value as number[]).map((v, i) => (
            <input
              key={i}
              id={i === 0 ? id : undefined}
              aria-label={`${spec.label} ${spec.type === 'range' ? (i ? 'max' : 'min') : spec.type === 'rect' ? ['x', 'y', 'width', 'height'][i] : i + 1}`}
              type="number"
              step={spec.step ?? (spec.type === 'floats' ? 0.001 : 1)}
              min={spec.min}
              max={spec.max}
              value={v}
              onChange={(e) => {
                const n = Number(e.target.value);
                if (!Number.isFinite(n)) return;
                const next = [...(value as number[])];
                next[i] = n;
                set(next);
              }}
              className={`${inputCls} px-1`}
            />
          ))}
        </div>
      );
      break;
    case 'points':
      control = (
        <p className="font-mono text-[11px] text-muted">
          {(value as number[][]).length ? (value as number[][]).map((p) => `(${p.map((x) => Math.round(x)).join(', ')})`).join(' ') : 'none'}
        </p>
      );
      break;
    case 'matrix':
      control = <MatrixField spec={spec} value={value} onChange={set} />;
      break;
  }

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-1">
        <label htmlFor={id} className="flex-1 text-[11px] font-medium text-muted" title={spec.help}>
          {spec.label}
          {spec.help && <span className="ml-1 cursor-help" aria-hidden>ⓘ</span>}
        </label>
        {spec.tool && (
          <button
            type="button"
            aria-pressed={tool === spec.key}
            onClick={() => setTool(tool === spec.key ? undefined : spec.key)}
            className={`rounded border px-1.5 py-0.5 text-[11px] ${tool === spec.key ? 'border-accent bg-accent/10 text-accent' : 'border-line text-muted hover:text-fg'}`}
            title={spec.tool === 'eyedropper' ? 'Click the image to pick a colour' : spec.tool === 'rect' ? 'Drag a rectangle on the image' : 'Drag / click points on the image'}
          >
            {spec.tool === 'eyedropper' ? '💧 pick' : spec.tool === 'rect' ? '▭ draw' : '✥ edit on image'}
          </button>
        )}
        {!isDefault && (
          <button type="button" onClick={() => resetParam(uid, spec.key)} className="rounded px-1 text-[11px] text-muted hover:text-fg" aria-label={`Reset ${spec.label} to default`} title="Reset to default">
            ↺
          </button>
        )}
      </div>
      {control}
      {spec.help && <p className="sr-only">{spec.help}</p>}
    </div>
  );
}
