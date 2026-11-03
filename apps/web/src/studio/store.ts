import {
  DEFAULT_DISPLAY,
  defaultParams,
  demoSample,
  getPreset,
  makePipeline,
  makeStep,
  newUid,
  parsePipeline,
  type DisplayOptions,
  type Fix,
  type ImageBuf,
  type Params,
  type Pipeline,
  type PipelineStep,
  type ReadSpec,
  type StepMode,
} from '@modelith/preprocess';
import { current } from 'immer';
import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { temporal } from 'zundo';
import { useStore as useModelStore } from '../store';
import { onImportPipelines, PIPELINES_KEY, publishPipelines, registerStudio } from './links';

export type ViewMode = 'output' | 'before-after' | 'side' | 'diff' | 'grid';

/** A loaded sample: RGBA as decoded by the browser (or an exact buffer decoded by the server). */
export interface Sample {
  id: string;
  name: string;
  w: number;
  h: number;
  rgba?: Uint8ClampedArray;
  hasAlpha?: boolean;
  /** Exact decoded image (16-bit / TIFF / EXIF via the server, or the built-in demo). */
  exact?: ImageBuf;
  mask?: ImageBuf;
  boxes: number[][];
  keypoints: number[][];
}

export interface Overlays {
  mask: boolean;
  boxes: boolean;
  keypoints: boolean;
  grid: boolean;
}

interface StudioState {
  pipelines: Pipeline[];
  activeId: string;
  /** Variant B shown side by side for comparing orders. */
  compareId?: string;
  selectedUid?: string;
  train: boolean;
  samples: Sample[];
  sampleIndex: number;
  view: ViewMode;
  overlays: Overlays;
  maskAlpha: number;
  diffGain: number;
  display: DisplayOptions;
  /** A slider is being dragged: preview on a low-resolution proxy. */
  dragging: boolean;
  rightTab: 'inspector' | 'code';
  /** Step under the cursor in the code view (code → pipeline highlight). */
  codeHighlight?: string;
  /** Step the pointer is over in the pipeline list (pipeline → code highlight). */
  hoverUid?: string;
  mobilePanel: 'none' | 'library' | 'pipeline';
  /** Param key of the selected step being edited on the viewer (rect / points / eyedropper). */
  activeTool?: string;

  // ---- pipeline edits (undoable)
  addStep(def: string, index?: number, params?: Params): string;
  moveStep(from: number, to: number): void;
  nudgeStep(uid: string, delta: number): void;
  removeStep(uid: string): void;
  duplicateStep(uid: string): void;
  toggleStep(uid: string): void;
  updateParams(uid: string, patch: Params): void;
  resetParam(uid: string, key: string): void;
  setStepMode(uid: string, mode: StepMode): void;
  setStepP(uid: string, p: number): void;
  setStepField(uid: string, patch: Partial<Pick<PipelineStep, 'note' | 'exportVisual'>>): void;
  setRead(patch: Partial<ReadSpec>): void;
  setSeed(seed: number): void;
  rename(name: string): void;
  applyFix(fix: Fix, uid?: string): void;
  loadPreset(id: string): void;
  newPipeline(): void;
  importPipeline(raw: unknown): Pipeline;
  setActive(id: string): void;
  deletePipeline(id: string): void;
  duplicateAsVariant(): void;
  closeVariant(): void;
  swapVariant(): void;
  link(modelNodeId?: string): void;

  // ---- UI state (not undoable)
  select(uid?: string): void;
  setTrain(train: boolean): void;
  addSamples(samples: Sample[]): void;
  removeSample(index: number): void;
  setSampleIndex(i: number): void;
  updateSample(index: number, patch: Partial<Sample>): void;
  setView(v: ViewMode): void;
  setOverlay(k: keyof Overlays, v: boolean): void;
  setMaskAlpha(a: number): void;
  setDiffGain(g: number): void;
  setDisplay(patch: Partial<DisplayOptions>): void;
  setDragging(d: boolean): void;
  setRightTab(t: StudioState['rightTab']): void;
  setCodeHighlight(uid?: string): void;
  setHover(uid?: string): void;
  setMobilePanel(p: StudioState['mobilePanel']): void;
  setActiveTool(key?: string): void;
}

const STORAGE_KEY = PIPELINES_KEY;

function loadSaved(): { pipelines: Pipeline[]; activeId: string } | undefined {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return undefined;
    const o = JSON.parse(raw);
    const pipelines = (o.pipelines as unknown[]).map(parsePipeline);
    if (!pipelines.length) return undefined;
    return { pipelines, activeId: pipelines.some((p) => p.id === o.activeId) ? o.activeId : pipelines[0].id };
  } catch {
    return undefined;
  }
}

export function demoAsSample(): Sample {
  const d = demoSample();
  return { id: 'demo', name: 'demo-street.png', w: d.image.meta.w, h: d.image.meta.h, exact: d.image, mask: d.mask, boxes: d.boxes, keypoints: d.keypoints };
}

const initialPipeline = () => getPreset('imagenet_cls')!.build();
const saved = loadSaved();
const first = saved?.pipelines ?? [initialPipeline()];

/** Find a step in the active pipeline. */
function findStep(s: StudioState, uid: string): PipelineStep | undefined {
  return s.pipelines.find((p) => p.id === s.activeId)?.steps.find((x) => x.uid === uid);
}

function active(s: StudioState): Pipeline {
  return s.pipelines.find((p) => p.id === s.activeId)!;
}

export const useStudio = create<StudioState>()(
  temporal(
    immer((set, get) => ({
      pipelines: first,
      activeId: saved?.activeId ?? first[0].id,
      selectedUid: first[0].steps[0]?.uid,
      train: false,
      samples: [demoAsSample()],
      sampleIndex: 0,
      view: 'output',
      overlays: { mask: false, boxes: true, keypoints: true, grid: false },
      maskAlpha: 0.45,
      diffGain: 4,
      display: DEFAULT_DISPLAY,
      dragging: false,
      rightTab: 'inspector',
      mobilePanel: 'none',

      addStep: (def, index, params) => {
        const step = makeStep(def, params);
        set((s) => {
          const p = active(s);
          const i = index === undefined ? p.steps.length : Math.max(0, Math.min(index, p.steps.length));
          p.steps.splice(i, 0, step);
          s.selectedUid = step.uid;
        });
        return step.uid;
      },
      moveStep: (from, to) =>
        set((s) => {
          const p = active(s);
          if (from === to || from < 0 || to < 0 || from >= p.steps.length || to >= p.steps.length) return;
          const [x] = p.steps.splice(from, 1);
          p.steps.splice(to, 0, x);
        }),
      nudgeStep: (uid, delta) => {
        const p = active(get());
        const i = p.steps.findIndex((x) => x.uid === uid);
        if (i >= 0) get().moveStep(i, Math.max(0, Math.min(p.steps.length - 1, i + delta)));
      },
      removeStep: (uid) =>
        set((s) => {
          const p = active(s);
          const i = p.steps.findIndex((x) => x.uid === uid);
          if (i < 0) return;
          p.steps.splice(i, 1);
          if (s.selectedUid === uid) s.selectedUid = p.steps[Math.min(i, p.steps.length - 1)]?.uid;
        }),
      duplicateStep: (uid) =>
        set((s) => {
          const p = active(s);
          const i = p.steps.findIndex((x) => x.uid === uid);
          if (i < 0) return;
          const copy = { ...structuredClone(current(p.steps[i])), uid: newUid() };
          p.steps.splice(i + 1, 0, copy);
          s.selectedUid = copy.uid;
        }),
      toggleStep: (uid) =>
        set((s) => {
          const st = findStep(s, uid);
          if (st) st.enabled = !st.enabled;
        }),
      updateParams: (uid, patch) =>
        set((s) => {
          const st = findStep(s, uid);
          if (st) Object.assign(st.params, patch);
        }),
      resetParam: (uid, key) =>
        set((s) => {
          const st = findStep(s, uid);
          if (st) st.params[key] = defaultParams(st.def)[key];
        }),
      setStepMode: (uid, mode) =>
        set((s) => {
          const st = findStep(s, uid);
          if (st) st.mode = mode;
        }),
      setStepP: (uid, p) =>
        set((s) => {
          const st = findStep(s, uid);
          if (st) st.p = Math.min(1, Math.max(0, p));
        }),
      setStepField: (uid, patch) =>
        set((s) => {
          const st = findStep(s, uid);
          if (st) Object.assign(st, patch);
        }),
      setRead: (patch) =>
        set((s) => {
          Object.assign(active(s).read, patch);
        }),
      setSeed: (seed) =>
        set((s) => {
          active(s).seed = Math.max(0, Math.floor(seed));
        }),
      rename: (name) =>
        set((s) => {
          active(s).name = name;
        }),

      applyFix: (fix, uid) => {
        const st = get();
        const p = active(st);
        const i = uid ? p.steps.findIndex((x) => x.uid === uid) : -1;
        switch (fix.kind) {
          case 'insertBefore':
            st.addStep(fix.def, Math.max(0, i), fix.params);
            break;
          case 'insertAfter':
            st.addStep(fix.def, i + 1, fix.params);
            break;
          case 'append':
            st.addStep(fix.def, undefined, fix.params);
            break;
          case 'replace':
            if (uid) {
              set((s) => {
                const ps = active(s);
                const k = ps.steps.findIndex((x) => x.uid === uid);
                ps.steps[k] = makeStep(fix.def, fix.params);
                s.selectedUid = ps.steps[k].uid;
              });
            }
            break;
          case 'setParams':
            if (uid) st.updateParams(uid, fix.params);
            break;
          case 'setMode':
            if (uid) st.setStepMode(uid, fix.mode);
            break;
          case 'moveToEnd':
            if (i >= 0) {
              // Keep tensor conversions after it (e.g. Normalise → HWC→CHW stays in order).
              st.moveStep(i, p.steps.length - 1);
            }
            break;
          case 'remove':
            if (uid) st.removeStep(uid);
            break;
          case 'updateModelInput': {
            const node = useModelStore.getState().project.nodes.find((n) => n.id === p.linkedModelId);
            if (node) useModelStore.getState().updateParams(node.id, { channels: fix.c, height: fix.h, width: fix.w });
            break;
          }
        }
      },

      loadPreset: (id) => {
        const preset = getPreset(id);
        if (!preset) return;
        const p = preset.build();
        set((s) => {
          s.pipelines.push(p);
          s.activeId = p.id;
          s.compareId = undefined;
          s.selectedUid = p.steps[0]?.uid;
        });
      },
      newPipeline: () => {
        const p = makePipeline('Untitled pipeline', []);
        set((s) => {
          s.pipelines.push(p);
          s.activeId = p.id;
          s.compareId = undefined;
          s.selectedUid = 'read';
        });
      },
      importPipeline: (raw) => {
        const p = parsePipeline(raw);
        if (get().pipelines.some((x) => x.id === p.id)) p.id = newUid('p');
        set((s) => {
          s.pipelines.push(p);
          s.activeId = p.id;
          s.compareId = undefined;
          s.selectedUid = p.steps[0]?.uid;
        });
        return p;
      },
      setActive: (id) =>
        set((s) => {
          if (!s.pipelines.some((p) => p.id === id)) return;
          s.activeId = id;
          if (s.compareId === id) s.compareId = undefined;
          s.selectedUid = active(s).steps[0]?.uid;
        }),
      deletePipeline: (id) =>
        set((s) => {
          if (s.pipelines.length <= 1) return;
          s.pipelines = s.pipelines.filter((p) => p.id !== id);
          if (s.compareId === id) s.compareId = undefined;
          if (s.activeId === id) {
            s.activeId = s.pipelines[0].id;
            s.selectedUid = s.pipelines[0].steps[0]?.uid;
          }
        }),
      duplicateAsVariant: () =>
        set((s) => {
          const a = active(s);
          const b: Pipeline = { ...structuredClone(current(a)), id: newUid('p'), name: `${a.name.replace(/ \(B\)$/, '')} (B)` };
          delete b.linkedModelId;
          s.pipelines.push(b);
          // Edit B straight away (it is the one being reordered); A stays alongside for comparison.
          s.activeId = b.id;
          s.compareId = a.id;
        }),
      closeVariant: () =>
        set((s) => {
          s.compareId = undefined;
        }),
      swapVariant: () =>
        set((s) => {
          if (!s.compareId) return;
          const a = s.activeId;
          s.activeId = s.compareId;
          s.compareId = a;
          s.selectedUid = active(s).steps[0]?.uid;
        }),
      link: (modelNodeId) =>
        set((s) => {
          // One pipeline per model input.
          for (const p of s.pipelines) if (modelNodeId && p.linkedModelId === modelNodeId) delete p.linkedModelId;
          const a = active(s);
          if (modelNodeId) a.linkedModelId = modelNodeId;
          else delete a.linkedModelId;
        }),

      select: (uid) =>
        set((s) => {
          if (s.selectedUid !== uid) s.activeTool = undefined;
          s.selectedUid = uid;
        }),
      setTrain: (train) =>
        set((s) => {
          s.train = train;
        }),
      addSamples: (samples) =>
        set((s) => {
          s.samples.push(...(samples as any));
          s.sampleIndex = s.samples.length - samples.length;
        }),
      removeSample: (index) =>
        set((s) => {
          if (s.samples.length <= 1) return;
          s.samples.splice(index, 1);
          s.sampleIndex = Math.min(s.sampleIndex, s.samples.length - 1);
        }),
      setSampleIndex: (i) =>
        set((s) => {
          s.sampleIndex = (i + s.samples.length) % s.samples.length;
        }),
      updateSample: (index, patch) =>
        set((s) => {
          Object.assign(s.samples[index], patch);
        }),
      setView: (v) =>
        set((s) => {
          s.view = v;
        }),
      setOverlay: (k, v) =>
        set((s) => {
          s.overlays[k] = v;
        }),
      setMaskAlpha: (a) =>
        set((s) => {
          s.maskAlpha = a;
        }),
      setDiffGain: (g) =>
        set((s) => {
          s.diffGain = g;
        }),
      setDisplay: (patch) =>
        set((s) => {
          Object.assign(s.display, patch);
        }),
      setDragging: (d) =>
        set((s) => {
          s.dragging = d;
        }),
      setRightTab: (t) =>
        set((s) => {
          s.rightTab = t;
        }),
      setCodeHighlight: (uid) =>
        set((s) => {
          s.codeHighlight = uid;
        }),
      setHover: (uid) =>
        set((s) => {
          s.hoverUid = uid;
        }),
      setActiveTool: (key) =>
        set((s) => {
          s.activeTool = key;
        }),
      setMobilePanel: (p) =>
        set((s) => {
          s.mobilePanel = p;
        }),
    })),
    {
      partialize: (s) => ({ pipelines: s.pipelines, activeId: s.activeId, compareId: s.compareId }),
      equality: (a, b) => a.pipelines === b.pipelines && a.activeId === b.activeId && a.compareId === b.compareId,
      limit: 200,
    },
  ),
);

export const studioUndo = () => useStudio.temporal.getState().undo();
export const studioRedo = () => useStudio.temporal.getState().redo();

export const useActivePipeline = () => useStudio((s) => s.pipelines.find((p) => p.id === s.activeId)!);

/**
 * Slider / drag gestures: many intermediate updates, one undo entry. Call begin() on pointer down and
 * end() on release; updates in between are not recorded.
 */
let gestureStart: Pipeline[] | undefined;
export function beginGesture(): void {
  if (gestureStart) return;
  gestureStart = useStudio.getState().pipelines;
  useStudio.temporal.getState().pause();
  useStudio.getState().setDragging(true);
}
export function endGesture(): void {
  if (!gestureStart) return;
  const final = useStudio.getState().pipelines;
  const start = gestureStart;
  gestureStart = undefined;
  useStudio.setState({ pipelines: start });
  useStudio.temporal.getState().resume();
  useStudio.setState({ pipelines: final });
  useStudio.getState().setDragging(false);
}

// The model builder reads links through ./links; edits made there go through this store's undo history.
registerStudio((nodeId, pipelineId) => {
  const s = useStudio.getState();
  const prev = s.activeId;
  s.setActive(pipelineId);
  s.link(nodeId);
  if (prev !== pipelineId) s.setActive(prev);
}, useStudio.getState().pipelines);
useStudio.subscribe((s, prev) => {
  if (s.pipelines !== prev.pipelines) publishPipelines(s.pipelines);
});
onImportPipelines((incoming) => useStudio.setState((s) => ({ pipelines: [...s.pipelines.filter((p) => !incoming.some((x) => x.id === p.id)), ...incoming] })));

// Autosave pipelines (debounced).
let saveTimer: ReturnType<typeof setTimeout> | undefined;
useStudio.subscribe((s, prev) => {
  if (s.pipelines === prev.pipelines && s.activeId === prev.activeId) return;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ pipelines: s.pipelines, activeId: s.activeId }));
    } catch {
      /* storage unavailable */
    }
  }, 300);
});
