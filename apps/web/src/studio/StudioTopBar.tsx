import * as Menu from '@radix-ui/react-dropdown-menu';
import { ident, presets, serializePipeline } from '@modelith/preprocess';
import { forwardRef, useState } from 'react';
import { useStore as useZustand } from 'zustand';
import { useStore as useModelStore, type Theme } from '../store';
import { download } from '../zip';
import { BatchDialog } from './BatchDialog';
import { VersionsDialog } from './Extras';
import { invalidateSample } from './engine';
import { IMAGE_ACCEPT, loadMaskFile, loadSampleFile, parseAnnotations, pickFiles } from './samples';
import { studioRedo, studioUndo, useActivePipeline, useStudio } from './store';
import { useModelInputs } from './useStudioAnalysis';
import { ViewSwitch } from './ViewSwitch';

// forwardRef: Radix menu triggers (asChild) need the button's ref to open and position the menu.
const Btn = forwardRef<HTMLButtonElement, React.ButtonHTMLAttributes<HTMLButtonElement> & { label: string }>(function Btn(props, ref) {
  const { label, children, className = '', ...rest } = props;
  return (
    <button ref={ref} aria-label={label} title={label} className={`inline-flex h-8 shrink-0 items-center justify-center gap-1 rounded-md border border-line px-2 text-xs hover:bg-panel-2 ${className}`} {...rest}>
      {children}
    </button>
  );
});

const itemCls = 'flex cursor-pointer items-start gap-2 rounded px-2 py-1.5 text-xs outline-none data-[highlighted]:bg-panel-2';
const contentCls = 'z-50 max-h-[70vh] min-w-56 overflow-y-auto rounded-lg border border-line bg-panel p-1 shadow-xl';

export function StudioTopBar() {
  const pipeline = useActivePipeline();
  const pipelines = useStudio((s) => s.pipelines);
  const train = useStudio((s) => s.train);
  const compareId = useStudio((s) => s.compareId);
  const sampleIndex = useStudio((s) => s.sampleIndex);
  const sample = useStudio((s) => s.samples[s.sampleIndex]);
  const st = useStudio.getState();
  const theme = useModelStore((s) => s.theme);
  const setTheme = useModelStore((s) => s.setTheme);
  const toast = useModelStore((s) => s.toast);
  const canUndo = useZustand(useStudio.temporal, (s) => s.pastStates.length > 0);
  const canRedo = useZustand(useStudio.temporal, (s) => s.futureStates.length > 0);
  const inputs = useModelInputs();
  const [batch, setBatch] = useState(false);
  const [versions, setVersions] = useState(false);

  const loadImages = async () => {
    const files = await pickFiles(IMAGE_ACCEPT);
    if (!files.length) return;
    const loaded = [];
    for (const f of files) {
      try {
        loaded.push(await loadSampleFile(f, { exif: pipeline.read.exif }));
      } catch (e) {
        toast((e as Error).message, 'error');
      }
    }
    if (loaded.length) {
      st.addSamples(loaded);
      toast(`Loaded ${loaded.length} image${loaded.length > 1 ? 's' : ''} (← / → to switch)`, 'success');
    }
  };
  const attachMask = async () => {
    const [f] = await pickFiles('image/png,image/*', false);
    if (!f) return;
    st.updateSample(sampleIndex, { mask: await loadMaskFile(f, sample.w, sample.h) });
    invalidateSample(sample.id);
    st.setOverlay('mask', true);
    toast('Mask attached', 'success');
  };
  const attachBoxes = async () => {
    const [f] = await pickFiles('.txt,.json', false);
    if (!f) return;
    try {
      const a = parseAnnotations(await f.text(), sample.w, sample.h);
      st.updateSample(sampleIndex, a);
      invalidateSample(sample.id);
      toast(`Attached ${a.boxes.length} box(es), ${a.keypoints.length} keypoint(s)`, 'success');
    } catch (e) {
      toast(`Could not read annotations: ${(e as Error).message}`, 'error');
    }
  };
  const exportJson = () => download(new Blob([serializePipeline(pipeline)], { type: 'application/json' }), `${ident(pipeline.name)}.pipeline.json`);
  const importJson = async () => {
    const [f] = await pickFiles('.json,application/json', false);
    if (!f) return;
    try {
      const p = st.importPipeline(JSON.parse(await f.text()));
      toast(`Imported ${p.name}`, 'success');
    } catch (e) {
      toast(`Import failed: ${(e as Error).message}`, 'error');
    }
  };

  return (
    <header className="flex h-12 shrink-0 items-center gap-1.5 overflow-x-auto border-b border-line bg-panel px-2 sm:px-3">
      <button className="lg:hidden" onClick={() => st.setMobilePanel('library')} aria-label="Open step library">
        <span className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-line">☰</span>
      </button>
      <img src="/favicon.svg" alt="" className="hidden h-6 w-6 sm:block" />
      <ViewSwitch active="studio" />
      <label htmlFor="pipeline-name" className="sr-only">
        Pipeline name
      </label>
      <input
        id="pipeline-name"
        value={pipeline.name}
        onChange={(e) => st.rename(e.target.value)}
        className="w-24 min-w-0 shrink rounded-md border border-transparent bg-transparent px-2 py-1 text-sm hover:border-line focus:border-accent focus:outline-none sm:w-40"
      />
      <Menu.Root>
        <Menu.Trigger asChild>
          <Btn label="Pipelines">Pipelines ▾</Btn>
        </Menu.Trigger>
        <Menu.Portal>
          <Menu.Content className={contentCls} sideOffset={4} align="start">
            {pipelines.map((p) => (
              <Menu.Item key={p.id} className={itemCls} onSelect={() => st.setActive(p.id)}>
                <span className="w-3">{p.id === pipeline.id ? '●' : p.id === compareId ? 'A' : ''}</span>
                <span className="flex-1">{p.name}</span>
                <span className="text-muted">{p.steps.length} steps</span>
              </Menu.Item>
            ))}
            <Menu.Separator className="my-1 h-px bg-line" />
            <Menu.Item className={itemCls} onSelect={st.newPipeline}>
              New empty pipeline
            </Menu.Item>
            <Menu.Item className={itemCls} onSelect={importJson}>
              Import .pipeline.json…
            </Menu.Item>
            <Menu.Item className={itemCls} onSelect={exportJson}>
              Export .pipeline.json
            </Menu.Item>
            <Menu.Item className={itemCls} onSelect={() => setVersions(true)}>
              Versions…
            </Menu.Item>
            <Menu.Item className={`${itemCls} text-danger`} disabled={pipelines.length <= 1} onSelect={() => st.deletePipeline(pipeline.id)}>
              Delete this pipeline
            </Menu.Item>
          </Menu.Content>
        </Menu.Portal>
      </Menu.Root>
      <Menu.Root>
        <Menu.Trigger asChild>
          <Btn label="Presets">Presets ▾</Btn>
        </Menu.Trigger>
        <Menu.Portal>
          <Menu.Content className={`${contentCls} w-80`} sideOffset={4} align="start">
            <Menu.Label className="px-2 py-1 text-[11px] text-muted">Load as a new, editable pipeline</Menu.Label>
            {presets.map((p) => (
              <Menu.Item key={p.id} className={itemCls} onSelect={() => st.loadPreset(p.id)}>
                <span>
                  <span className="block font-medium">{p.label}</span>
                  <span className="block text-[11px] text-muted">{p.description}</span>
                </span>
              </Menu.Item>
            ))}
          </Menu.Content>
        </Menu.Portal>
      </Menu.Root>
      <div className="mx-0.5 hidden h-5 w-px bg-line sm:block" />
      <Btn label="Undo (Ctrl+Z)" onClick={studioUndo} disabled={!canUndo}>
        ↶
      </Btn>
      <Btn label="Redo (Ctrl+Shift+Z)" onClick={studioRedo} disabled={!canRedo}>
        ↷
      </Btn>
      <div className="flex h-8 shrink-0 items-center rounded-md border border-line text-xs">
        <label htmlFor="seed" className="pl-2 text-muted">
          Seed
        </label>
        <input id="seed" type="number" min={0} value={pipeline.seed} onChange={(e) => st.setSeed(Number(e.target.value))} className="w-16 bg-transparent px-1 outline-none" />
        <button aria-label="Random seed" title="Random seed" onClick={() => st.setSeed(Math.floor(Math.random() * 100000))} className="h-full border-l border-line px-1.5 hover:bg-panel-2">
          🎲
        </button>
      </div>
      <div className="flex shrink-0 rounded-md border border-line p-0.5 text-xs" role="radiogroup" aria-label="Train or inference preview">
        {[false, true].map((t) => (
          <button key={String(t)} role="radio" aria-checked={train === t} onClick={() => st.setTrain(t)} className={`rounded px-2 py-1 ${train === t ? 'bg-accent text-accent-fg' : 'text-muted hover:text-fg'}`}>
            {t ? 'Train' : 'Inference'}
          </button>
        ))}
      </div>
      <label className="sr-only" htmlFor="link-model">
        Link to model
      </label>
      <select
        id="link-model"
        value={pipeline.linkedModelId ?? ''}
        onChange={(e) => {
          st.link(e.target.value || undefined);
          if (e.target.value) toast('Linked: train.py will load data through this pipeline', 'success');
        }}
        className="h-8 max-w-44 shrink-0 rounded-md border border-line bg-panel px-1 text-xs"
        title="Link this pipeline to an Image input of the current Modelith model"
      >
        <option value="">{inputs.length ? 'Link to model…' : 'No model with an Image input'}</option>
        {inputs.map((i) => (
          <option key={i.id} value={i.id}>
            🔗 {i.label}
          </option>
        ))}
      </select>
      <Menu.Root>
        <Menu.Trigger asChild>
          <Btn label="Samples">Samples ▾</Btn>
        </Menu.Trigger>
        <Menu.Portal>
          <Menu.Content className={contentCls} sideOffset={4} align="end">
            <Menu.Item className={itemCls} onSelect={loadImages}>
              Load images…
            </Menu.Item>
            <Menu.Item className={itemCls} onSelect={attachMask}>
              Attach mask to this image (label PNG)…
            </Menu.Item>
            <Menu.Item className={itemCls} onSelect={attachBoxes}>
              Attach boxes (YOLO .txt / JSON)…
            </Menu.Item>
            <Menu.Item className={`${itemCls} text-danger`} disabled={useStudio.getState().samples.length <= 1} onSelect={() => st.removeSample(sampleIndex)}>
              Remove this image
            </Menu.Item>
          </Menu.Content>
        </Menu.Portal>
      </Menu.Root>
      <Btn label="Run on batch" onClick={() => setBatch(true)}>
        Batch
      </Btn>
      {compareId ? (
        <>
          <Btn label="Swap A and B" onClick={st.swapVariant}>
            ⇄ A/B
          </Btn>
          <Btn label="Stop comparing" onClick={st.closeVariant} className="border-accent text-accent">
            ✕ Compare
          </Btn>
        </>
      ) : (
        <Btn label="Duplicate pipeline as variant B and compare" onClick={st.duplicateAsVariant}>
          + Variant B
        </Btn>
      )}
      <div className="ml-auto flex shrink-0 items-center gap-1.5">
        <button
          onClick={() => {
            st.setRightTab('code');
            st.setMobilePanel('pipeline');
          }}
          className="inline-flex h-8 shrink-0 items-center rounded-md bg-accent px-2.5 text-xs font-semibold whitespace-nowrap text-accent-fg hover:brightness-110"
        >
          Export code ▸
        </button>
        <select aria-label="Theme" value={theme} onChange={(e) => setTheme(e.target.value as Theme)} className="hidden h-8 rounded-md border border-line bg-panel px-1 text-xs sm:block">
          <option value="system">Auto</option>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
        <button className="md:hidden" onClick={() => st.setMobilePanel('pipeline')} aria-label="Open pipeline panel">
          <span className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-line">⚙</span>
        </button>
      </div>
      {batch && <BatchDialog onClose={() => setBatch(false)} />}
      {versions && <VersionsDialog onClose={() => setVersions(false)} />}
    </header>
  );
}
