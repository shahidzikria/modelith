import { useDroppable } from '@dnd-kit/core';
import { SortableContext, useSortable, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { formatMeta, getStep, READ_FLAGS, type PipelineStep, type StepInfo } from '@modelith/preprocess';
import { useRuns } from './engine';
import { STEP_COLORS } from './StepLibrary';
import { useActivePipeline, useStudio } from './store';
import { usePipelineAnalysis } from './useStudioAnalysis';

export function stepSummary(s: PipelineStep): string {
  const def = getStep(s.def);
  if (!def) return s.def;
  const sum = def.summary?.(s.params) ?? '';
  const prob = def.kind === 'random' ? `p ${s.p ?? 1}` : '';
  return [def.label, sum, prob].filter(Boolean).join(' · ');
}

function StepRow({ step, index, info, ms, serverOnly, ran, failed }: { step: PipelineStep; index: number; info?: StepInfo; ms?: number; serverOnly: boolean; ran?: 'skipped' | 'not-applied'; failed?: boolean }) {
  const def = getStep(step.def);
  const { attributes, listeners, setNodeRef, setActivatorNodeRef, transform, transition, isDragging, isOver, active } = useSortable({ id: step.uid });
  // A step dragged in from the library must not shuffle the rows; show where it will be inserted instead.
  const fromLibrary = String(active?.id ?? '').startsWith('lib:');
  const selected = useStudio((s) => s.selectedUid === step.uid);
  const highlighted = useStudio((s) => s.codeHighlight === step.uid);
  const { select, toggleStep, nudgeStep, removeStep, setHover } = useStudio.getState();
  const issues = info?.issues ?? [];
  const worst = issues.some((i) => i.level === 'error') ? 'error' : issues.some((i) => i.level === 'warning') ? 'warning' : undefined;
  const inactive = !info?.active;

  return (
    <li
      ref={setNodeRef}
      style={fromLibrary ? undefined : { transform: CSS.Transform.toString(transform), transition }}
      className={`relative ${isDragging ? 'z-10 opacity-60' : ''}`}
      onMouseEnter={() => setHover(step.uid)}
      onMouseLeave={() => setHover(undefined)}
    >
      {fromLibrary && isOver && <div className="absolute inset-x-1 -top-0.5 z-10 h-0.5 rounded bg-accent" aria-hidden />}
      <div
        data-step-row
        tabIndex={0}
        role="button"
        aria-pressed={selected}
        aria-label={`Step ${index + 1}: ${def?.label ?? step.def}${step.enabled ? '' : ' (disabled)'}`}
        onClick={() => select(step.uid)}
        onKeyDown={(e) => {
          if (e.altKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
            e.preventDefault();
            nudgeStep(step.uid, e.key === 'ArrowUp' ? -1 : 1);
            requestAnimationFrame(() => (document.querySelector(`[data-uid="${step.uid}"]`) as HTMLElement | null)?.focus());
          } else if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            select(step.uid);
          } else if (e.key === 'Delete') {
            e.preventDefault();
            removeStep(step.uid);
          }
        }}
        data-uid={step.uid}
        className={`group flex cursor-pointer items-start gap-1.5 rounded-md border px-1.5 py-1.5 text-left ${
          selected ? 'border-accent bg-accent/10' : highlighted ? 'border-accent/60 bg-panel-2' : 'border-transparent hover:bg-panel-2'
        } ${inactive ? 'opacity-55' : ''}`}
      >
        <button
          ref={setActivatorNodeRef}
          {...listeners}
          {...attributes}
          aria-label={`Drag to reorder ${def?.label ?? step.def} (or focus the row and press Alt+↑/↓)`}
          className="mt-0.5 cursor-grab touch-none rounded px-0.5 text-muted hover:text-fg active:cursor-grabbing"
          onClick={(e) => e.stopPropagation()}
        >
          ⋮⋮
        </button>
        <span className="mt-0.5 w-5 shrink-0 text-right font-mono text-[11px] text-muted">{index + 1}</span>
        <span className="mt-1.5 h-2 w-2 shrink-0 rounded-sm" style={{ background: def ? STEP_COLORS[def.category] : '#999' }} aria-hidden />
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-1">
            <span className={`truncate text-[13px] ${step.enabled ? '' : 'line-through'}`}>{stepSummary(step)}</span>
          </span>
          <span className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-[10.5px] text-muted">
            {info && info.active && <span className="font-mono">{formatMeta(info.out)}</span>}
            {ms !== undefined && !ran && <span className="font-mono">{ms < 10 ? ms.toFixed(1) : Math.round(ms)} ms</span>}
            {ran === 'not-applied' && <span title="The probability draw skipped it for this image">not applied</span>}
            {step.mode !== 'both' && <span className="rounded bg-panel-2 px-1">{step.mode === 'train' ? 'train only' : 'inference only'}</span>}
            {serverOnly && <span className="rounded bg-panel-2 px-1" title="No OpenCV.js implementation: runs on the Python server">server</span>}
            {def?.visualOnly && <span className="rounded bg-panel-2 px-1">visual</span>}
          </span>
        </span>
        {(worst || failed) && (
          <span
            className={`mt-0.5 text-sm ${worst === 'error' || failed ? 'text-danger' : 'text-warn'}`}
            title={issues.map((i) => i.message).join('\n') || 'This step failed; see the inspector.'}
            aria-label={worst === 'error' || failed ? 'error' : 'warning'}
          >
            {worst === 'error' || failed ? '⛔' : '⚠'}
          </span>
        )}
        <label className="mt-0.5 flex items-center" onClick={(e) => e.stopPropagation()}>
          <span className="sr-only">Enable {def?.label}</span>
          <input type="checkbox" checked={step.enabled} onChange={() => toggleStep(step.uid)} className="accent-[var(--accent)]" />
        </label>
      </div>
    </li>
  );
}

function EndZone() {
  const { setNodeRef, isOver } = useDroppable({ id: 'pipeline-end' });
  const setMobilePanel = useStudio((s) => s.setMobilePanel);
  return (
    <div ref={setNodeRef} className={`mx-2 mt-1 rounded-md border border-dashed px-3 py-2 text-center text-xs ${isOver ? 'border-accent bg-accent/10' : 'border-line text-muted'}`}>
      <button
        onClick={() => {
          setMobilePanel('library');
          setTimeout(() => (document.getElementById('step-search') as HTMLInputElement | null)?.focus(), 50);
        }}
        className="hover:text-fg"
      >
        + add step
      </button>{' '}
      <span className="hidden lg:inline">(drag one here from the library)</span>
    </div>
  );
}

export function PipelineList() {
  const pipeline = useActivePipeline();
  const analysis = usePipelineAnalysis(pipeline);
  const run = useRuns((s) => (s.A?.pipelineId === pipeline.id ? s.A : s.B?.pipelineId === pipeline.id ? s.B : undefined));
  const missing = useRuns((s) => s.missing);
  const selectedRead = useStudio((s) => s.selectedUid === 'read');
  const select = useStudio((s) => s.select);
  const train = useStudio((s) => s.train);
  const total = run?.steps.reduce((a, s) => a + (s && !s.out.skipped ? s.out.ms : 0), 0) ?? 0;
  const pipelineIssues = analysis.issues.filter((i) => !i.uid);

  return (
    <section className="flex h-full flex-col" aria-label="Pipeline">
      <div className="flex items-center gap-2 border-b border-line px-3 py-2">
        <h2 className="text-xs font-semibold tracking-wide text-muted uppercase">Pipeline</h2>
        <span className="ml-auto font-mono text-[11px] text-muted" title="Sum of per-step browser times for this image">
          {run?.done ? `${total.toFixed(1)} ms · ${total > 0 ? Math.round(1000 / total) : '∞'} img/s${run.proxy ? ' (proxy)' : ''}` : 'running…'}
        </span>
      </div>
      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto py-1.5">
        <div
          role="button"
          tabIndex={0}
          aria-pressed={selectedRead}
          onClick={() => select('read')}
          onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && select('read')}
          className={`mx-2 mb-1 flex cursor-pointer items-center gap-2 rounded-md border px-2 py-1.5 text-[13px] ${selectedRead ? 'border-accent bg-accent/10' : 'border-transparent hover:bg-panel-2'}`}
        >
          <span className="text-muted" aria-hidden>
            📂
          </span>
          <span className="min-w-0 flex-1">
            <span className="block">Read image · {pipeline.read.flag.toLowerCase()}</span>
            <span className="block font-mono text-[10.5px] text-muted">
              {formatMeta(analysis.input)} · {READ_FLAGS[pipeline.read.flag].doc}
            </span>
          </span>
        </div>
        <SortableContext items={pipeline.steps.map((s) => s.uid)} strategy={verticalListSortingStrategy}>
          <ol className="px-1" aria-label="Pipeline steps">
            {pipeline.steps.map((s, i) => {
              const r = run?.steps[i];
              return (
                <StepRow
                  key={s.uid}
                  step={s}
                  index={i}
                  info={analysis.byUid[s.uid]}
                  ms={r?.out.uid === s.uid ? r.out.ms : undefined}
                  serverOnly={!!missing[s.def]}
                  ran={r?.out.uid === s.uid ? (r.out.skipped ? 'skipped' : r.out.notApplied ? 'not-applied' : undefined) : undefined}
                  failed={run?.errorAt === i}
                />
              );
            })}
          </ol>
        </SortableContext>
        {!pipeline.steps.length && <p className="px-4 py-3 text-xs text-muted">No steps yet. Pick one from the library or load a preset.</p>}
        <EndZone />
        {pipelineIssues.length > 0 && (
          <ul className="mx-2 mt-2 space-y-1">
            {pipelineIssues.map((i, k) => (
              <IssueItem key={k} issue={i} />
            ))}
          </ul>
        )}
      </div>
      <div className="border-t border-line px-3 py-1.5 font-mono text-[11px] text-muted">
        Output ({train ? 'train' : 'inference'}): <span className="text-fg">{formatMeta(analysis.final)}</span> · {analysis.final.space}
        {analysis.final.dtype.startsWith('float') ? ` · ~[${analysis.final.range.join(', ')}]` : ''}
      </div>
    </section>
  );
}

export function IssueItem({ issue }: { issue: import('@modelith/preprocess').Issue }) {
  const applyFix = useStudio((s) => s.applyFix);
  const color = issue.level === 'error' ? 'text-danger' : issue.level === 'warning' ? 'text-warn' : 'text-muted';
  return (
    <li className="rounded-md border border-line bg-panel-2/50 px-2 py-1.5 text-xs">
      <p className={color}>
        <span aria-hidden>{issue.level === 'error' ? '⛔ ' : issue.level === 'warning' ? '⚠ ' : 'ℹ '}</span>
        {issue.message}
      </p>
      {!!issue.fixes?.length && (
        <div className="mt-1 flex flex-wrap gap-1">
          {issue.fixes.map((f, k) => (
            <button key={k} onClick={() => applyFix(f, issue.uid)} className="rounded border border-line bg-panel px-1.5 py-0.5 text-[11px] hover:border-accent">
              {f.label}
            </button>
          ))}
        </div>
      )}
    </li>
  );
}
