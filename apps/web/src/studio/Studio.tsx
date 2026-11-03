import {
  DndContext,
  DragOverlay,
  KeyboardSensor,
  PointerSensor,
  closestCenter,
  MeasuringStrategy,
  pointerWithin,
  type CollisionDetection,
  useSensor,
  useSensors,
  type DragEndEvent,
  type DragStartEvent,
} from '@dnd-kit/core';
import { sortableKeyboardCoordinates } from '@dnd-kit/sortable';
import { getStep } from '@modelith/preprocess';
import { useEffect, useState } from 'react';
import { Toasts } from '../components/Toasts';
import { Analysis } from './Analysis';
import { useEngine } from './engine';
import { ExportPanel } from './ExportPanel';
import { Filmstrip } from './Filmstrip';
import { PipelineList } from './PipelineList';
import { StepInspector } from './StepInspector';
import { StepLibrary } from './StepLibrary';
import { studioRedo, studioUndo, useActivePipeline, useStudio } from './store';
import { StudioTopBar } from './StudioTopBar';
import { Viewer } from './Viewer';

/** Drop where the pointer is (row under the cursor); fall back to the nearest row when between rows. */
const collision: CollisionDetection = (args) => {
  const hits = pointerWithin(args);
  return hits.length ? hits : closestCenter(args);
};

function isTyping(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null;
  return !!t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable || !!t.closest('.monaco-editor'));
}

function useStudioShortcuts() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e)) return;
      const mod = e.ctrlKey || e.metaKey;
      const k = e.key.toLowerCase();
      const st = useStudio.getState();
      if (mod && k === 'z' && !e.shiftKey) studioUndo();
      else if (mod && ((k === 'z' && e.shiftKey) || k === 'y')) studioRedo();
      // Arrow keys switch samples (Alt+arrows reorder steps; handled on the rows).
      else if (!mod && !e.altKey && k === 'arrowright' && !(e.target as HTMLElement)?.closest('[data-step-row]')) st.setSampleIndex(st.sampleIndex + 1);
      else if (!mod && !e.altKey && k === 'arrowleft' && !(e.target as HTMLElement)?.closest('[data-step-row]')) st.setSampleIndex(st.sampleIndex - 1);
      else return;
      e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
}

export default function Studio() {
  useEngine();
  useStudioShortcuts();
  const pipeline = useActivePipeline();
  const rightTab = useStudio((s) => s.rightTab);
  const setRightTab = useStudio((s) => s.setRightTab);
  const mobilePanel = useStudio((s) => s.mobilePanel);
  const setMobilePanel = useStudio((s) => s.setMobilePanel);
  const [dragDef, setDragDef] = useState<string>();

  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 5 } }), useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }));

  const onDragStart = (e: DragStartEvent) => {
    const id = String(e.active.id);
    if (id.startsWith('lib:')) setDragDef(id.slice(4));
  };
  const onDragEnd = (e: DragEndEvent) => {
    setDragDef(undefined);
    const st = useStudio.getState();
    const steps = pipeline.steps;
    const activeId = String(e.active.id);
    const overId = e.over ? String(e.over.id) : undefined;
    if (!overId) return;
    const overIndex = overId === 'pipeline-end' ? steps.length : overId === 'pipeline-start' ? 0 : steps.findIndex((s) => s.uid === overId);
    if (activeId.startsWith('lib:')) {
      // Dropping on a row inserts before it (between it and the previous step); "+ add step" appends.
      st.addStep(activeId.slice(4), overIndex < 0 ? steps.length : overIndex);
      st.setMobilePanel('none');
      return;
    }
    const from = steps.findIndex((s) => s.uid === activeId);
    if (from >= 0 && overIndex >= 0 && from !== overIndex) st.moveStep(from, Math.min(overIndex, steps.length - 1));
  };

  const drawer = (side: 'left' | 'right') =>
    `fixed inset-y-0 ${side}-0 z-40 w-[86vw] max-w-sm border-line bg-panel shadow-xl lg:static lg:z-auto lg:w-auto lg:max-w-none lg:shadow-none`;

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <StudioTopBar />
      <DndContext sensors={sensors} collisionDetection={collision} measuring={{ droppable: { strategy: MeasuringStrategy.Always } }}
        // Only the pipeline list may auto-scroll during a drag (scrolling the library would shift the drop point).
        autoScroll={{ canScroll: (el) => !!el.closest('[aria-label="Pipeline"]') }} onDragStart={onDragStart} onDragEnd={onDragEnd} onDragCancel={() => setDragDef(undefined)}>
        <div className="relative flex min-h-0 flex-1">
          <aside className={`${drawer('left')} min-h-0 border-r lg:block lg:w-56 xl:w-60 ${mobilePanel === 'library' ? 'block' : 'hidden'}`} aria-label="Step library">
            <StepLibrary />
          </aside>
          <main className="flex min-h-0 min-w-0 flex-1 flex-col" aria-label="Viewer">
            <div className="relative min-h-0 flex-[3]">
              <Viewer />
            </div>
            <Filmstrip />
            <div className="min-h-0 flex-[2] overflow-hidden border-t border-line">
              <Analysis />
            </div>
          </main>
          <aside
            className={`${drawer('right')} min-h-0 flex-col border-l md:static md:z-auto md:flex md:w-[340px] md:max-w-none md:shadow-none xl:w-[400px] ${mobilePanel === 'pipeline' ? 'flex' : 'hidden'}`}
            aria-label="Pipeline panel"
          >
            <div className="flex items-center justify-end border-b border-line p-1 md:hidden">
              <button onClick={() => setMobilePanel('none')} className="rounded px-2 py-1 text-sm" aria-label="Close pipeline panel">
                Close ✕
              </button>
            </div>
            <div className="min-h-0 flex-[5] overflow-hidden">
              <PipelineList />
            </div>
            <div className="flex min-h-0 flex-[4] flex-col border-t border-line">
              <div className="flex shrink-0 gap-1 border-b border-line px-2 pt-1.5" role="tablist" aria-label="Step panel">
                {(['inspector', 'code'] as const).map((t) => (
                  <button
                    key={t}
                    role="tab"
                    aria-selected={rightTab === t}
                    onClick={() => setRightTab(t)}
                    className={`rounded-t-md px-3 py-1.5 text-xs font-medium ${rightTab === t ? 'border border-b-0 border-line bg-panel text-fg' : 'text-muted hover:text-fg'}`}
                  >
                    {t === 'inspector' ? 'Step inspector' : 'Code'}
                  </button>
                ))}
              </div>
              <div className="min-h-0 flex-1">{rightTab === 'inspector' ? <StepInspector /> : <ExportPanel />}</div>
            </div>
          </aside>
          {mobilePanel !== 'none' && <div className="fixed inset-0 z-30 bg-black/30 lg:hidden" onClick={() => setMobilePanel('none')} aria-hidden />}
        </div>
        <DragOverlay dropAnimation={null}>
          {dragDef ? <div className="rounded-md border border-accent bg-panel px-3 py-1.5 text-sm shadow-lg">{getStep(dragDef)?.label}</div> : null}
        </DragOverlay>
      </DndContext>
      <Toasts />
    </div>
  );
}
