import { defaultParams, getStep, READ_FLAGS, type ReadFlag, type StepMode } from '@modelith/preprocess';
import { useState } from 'react';
import { useRuns } from './engine';
import { NineSamples } from './NineSamples';
import { ParamField } from './ParamField';
import { IssueItem, stepSummary } from './PipelineList';
import { STEP_COLORS } from './StepLibrary';
import { beginGesture, endGesture, useActivePipeline, useStudio } from './store';
import { usePipelineAnalysis } from './useStudioAnalysis';
import { ParamSweep } from './ParamSweep';
import { BatchPreview, DatasetStats, SuggestionsPanel } from './Extras';

function ReadInspector() {
  const pipeline = useActivePipeline();
  const setRead = useStudio((s) => s.setRead);
  return (
    <div className="space-y-3 p-3 text-sm">
      <h3 className="font-semibold">Read image</h3>
      <p className="text-xs text-muted">
        How <code>load_image()</code> decodes files (<code>cv2.imdecode</code>, which also handles non-ASCII paths). OpenCV returns <b>BGR</b>; most models want RGB, so
        add <i>Convert colour → RGB</i> before normalising.
      </p>
      <label className="block text-[11px] font-medium text-muted" htmlFor="read-flag">
        Flag
      </label>
      <select id="read-flag" value={pipeline.read.flag} onChange={(e) => setRead({ flag: e.target.value as ReadFlag })} className="w-full rounded-md border border-line bg-panel-2 px-2 py-1 text-xs">
        {(Object.keys(READ_FLAGS) as ReadFlag[]).map((f) => (
          <option key={f} value={f}>
            IMREAD_{f} — {READ_FLAGS[f].doc}
          </option>
        ))}
      </select>
      <label className="flex items-center gap-2 text-xs">
        <input type="checkbox" checked={pipeline.read.exif} onChange={(e) => setRead({ exif: e.target.checked })} className="accent-[var(--accent)]" />
        Apply EXIF orientation (phone photos)
      </label>
      <p className="text-[11px] text-muted">
        16-bit PNG/TIFF and exact EXIF handling need the Python server for decoding; without it the browser decodes images to 8-bit.
      </p>
      <SuggestionsPanel />
    </div>
  );
}

export function StepInspector() {
  const pipeline = useActivePipeline();
  const selectedUid = useStudio((s) => s.selectedUid);
  const analysis = usePipelineAnalysis(pipeline);
  const run = useRuns((s) => (s.A?.pipelineId === pipeline.id ? s.A : s.B));
  const missing = useRuns((s) => s.missing);
  const { setStepMode, setStepP, toggleStep, removeStep, duplicateStep, updateParams, setStepField } = useStudio.getState();
  const [showSamples, setShowSamples] = useState(false);
  const [showSweep, setShowSweep] = useState(false);
  const [showBatch, setShowBatch] = useState(false);

  if (selectedUid === 'read') return <ReadInspector />;
  const index = pipeline.steps.findIndex((s) => s.uid === selectedUid);
  const step = pipeline.steps[index];
  if (!step) return <p className="p-4 text-sm text-muted">Select a step to edit its parameters.</p>;
  const def = getStep(step.def);
  if (!def) return <p className="p-4 text-sm text-danger">Unknown step “{step.def}”.</p>;
  const info = analysis.byUid[step.uid];
  const defaults = defaultParams(step.def);
  const visible = def.params.filter((p) => !p.showIf || p.showIf.equals.includes(step.params[p.showIf.key]));
  const error = run?.errorAt === index ? run.error : undefined;
  const serverOnly = missing[def.id];

  return (
    <div className="scroll-thin h-full overflow-y-auto">
      <div className="space-y-3 p-3">
        <div className="flex items-start gap-2">
          <span className="mt-1.5 h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: STEP_COLORS[def.category] }} aria-hidden />
          <div className="min-w-0 flex-1">
            <h3 className="text-sm font-semibold">
              {index + 1}. {def.label}
              {def.kind === 'random' && <span className="ml-1" aria-label="random">🎲</span>}
            </h3>
            <p className="text-[11px] text-muted">
              {def.category} · {stepSummary(step)}
            </p>
          </div>
          <label className="flex items-center gap-1 text-xs">
            <input type="checkbox" checked={step.enabled} onChange={() => toggleStep(step.uid)} className="accent-[var(--accent)]" />
            On
          </label>
        </div>

        {info && (
          <p className="rounded-md bg-panel-2 px-2 py-1 font-mono text-[11px]">
            {info.in.space} {info.in.dtype} {info.in.w}×{info.in.h}×{info.in.c} → {info.out.space} {info.out.dtype} {info.out.w}×{info.out.h}×{info.out.c}
            {info.out.layout === 'CHW' ? ' (CHW)' : ''}
          </p>
        )}
        {serverOnly && (
          <p className="rounded-md border border-line px-2 py-1 text-[11px] text-muted">
            <b>Server step</b>: not in this OpenCV.js build ({serverOnly.join(', ')}), so the pipeline from here on runs on the Python server.
          </p>
        )}
        {error && <p className="rounded-md border border-danger/50 px-2 py-1 text-xs text-danger">Failed: {error}</p>}
        {info?.issues.length ? (
          <ul className="space-y-1">
            {info.issues.map((i, k) => (
              <IssueItem key={k} issue={i} />
            ))}
          </ul>
        ) : null}

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label htmlFor="step-mode" className="text-[11px] font-medium text-muted">
              Applies in
            </label>
            <select id="step-mode" value={step.mode} onChange={(e) => setStepMode(step.uid, e.target.value as StepMode)} className="w-full rounded-md border border-line bg-panel-2 px-2 py-1 text-xs">
              <option value="both">Both</option>
              <option value="train">Train only</option>
              <option value="inference">Inference only</option>
            </select>
          </div>
          {def.kind === 'random' && (
            <div>
              <label htmlFor="step-p" className="text-[11px] font-medium text-muted">
                Probability p = {step.p ?? 1}
              </label>
              <input
                id="step-p"
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={step.p ?? 1}
                onPointerDown={beginGesture}
                onPointerUp={endGesture}
                onChange={(e) => setStepP(step.uid, Number(e.target.value))}
                className="w-full accent-[var(--accent)]"
              />
            </div>
          )}
        </div>

        {visible.length > 0 && (
          <fieldset className="space-y-2.5">
            <legend className="mb-1 text-[11px] font-semibold tracking-wide text-muted uppercase">Parameters</legend>
            {visible.map((spec) => (
              <ParamField key={spec.key} uid={step.uid} spec={spec} value={step.params[spec.key]} isDefault={JSON.stringify(step.params[spec.key]) === JSON.stringify(defaults[spec.key])} />
            ))}
          </fieldset>
        )}

        <div className="flex flex-wrap gap-1.5">
          {def.kind === 'random' && (
            <button onClick={() => setShowSamples(true)} className="rounded-md border border-line px-2 py-1 text-xs hover:bg-panel-2">
              🎲 Show 9 samples
            </button>
          )}
          {def.params.some((p) => p.type === 'int' || p.type === 'float' || p.type === 'oddInt') && (
            <button onClick={() => setShowSweep(true)} className="rounded-md border border-line px-2 py-1 text-xs hover:bg-panel-2">
              Parameter sweep
            </button>
          )}
          {def.batch && (
            <button onClick={() => setShowBatch(true)} className="rounded-md border border-line px-2 py-1 text-xs hover:bg-panel-2">
              Preview with loaded samples
            </button>
          )}
          {def.id === 'standardize' && <DatasetStats uid={step.uid} />}
          <button onClick={() => updateParams(step.uid, defaults)} className="rounded-md border border-line px-2 py-1 text-xs hover:bg-panel-2">
            Reset all
          </button>
          <button onClick={() => duplicateStep(step.uid)} className="rounded-md border border-line px-2 py-1 text-xs hover:bg-panel-2">
            Duplicate
          </button>
          <button onClick={() => removeStep(step.uid)} className="rounded-md border border-danger/50 px-2 py-1 text-xs text-danger hover:bg-danger/10">
            Delete step
          </button>
        </div>
        {def.visualOnly && (
          <label className="flex items-center gap-2 text-xs">
            <input type="checkbox" checked={!!step.exportVisual} onChange={(e) => setStepField(step.uid, { exportVisual: e.target.checked })} className="accent-[var(--accent)]" />
            Include this visualisation step in the exported code
          </label>
        )}

        <details className="rounded-md border border-line px-2 py-1.5 text-xs" open>
          <summary className="cursor-pointer font-medium">About this step</summary>
          <p className="mt-1.5 leading-relaxed text-muted">{def.doc}</p>
          <p className="mt-1.5 font-mono text-[11px] break-words">{def.cv2}</p>
          <p className="mt-1.5 text-[11px] text-muted">
            Engine: {def.engine === 'server' || serverOnly ? 'Python server' : def.engine === 'browser' ? 'browser' : 'browser (OpenCV.js) + Python'}
            {def.affectsGeometry ? ' · transforms masks, boxes and keypoints' : ''}
            {def.tolerance ? ` · browser vs Python within ±${def.tolerance}` : ''}
          </p>
        </details>
        <div>
          <label htmlFor="step-note" className="text-[11px] font-medium text-muted">
            Note
          </label>
          <textarea
            id="step-note"
            rows={2}
            value={step.note ?? ''}
            onChange={(e) => setStepField(step.uid, { note: e.target.value || undefined })}
            placeholder="Why this step is here…"
            className="w-full rounded-md border border-line bg-panel-2 px-2 py-1 text-xs"
          />
        </div>
      </div>
      {showSamples && <NineSamples uid={step.uid} onClose={() => setShowSamples(false)} />}
      {showSweep && <ParamSweep uid={step.uid} onClose={() => setShowSweep(false)} />}
      {showBatch && <BatchPreview def={def.id} params={step.params} onClose={() => setShowBatch(false)} />}
    </div>
  );
}
