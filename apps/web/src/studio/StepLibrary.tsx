import { useDraggable } from '@dnd-kit/core';
import { allSteps, CATEGORIES, type Category, type StepDef } from '@modelith/preprocess';
import { useMemo, useState } from 'react';
import { useStore as useModelStore } from '../store';
import { useRuns } from './engine';
import { useStudio } from './store';

/** Okabe–Ito based palette, one colour per category. */
export const STEP_COLORS: Record<Category, string> = {
  'I/O & colour': '#0072B2',
  Geometric: '#009E73',
  Intensity: '#E69F00',
  Filtering: '#56B4E9',
  Denoising: '#8C6BB1',
  Morphology: '#B8A600',
  Edges: '#D55E00',
  Threshold: '#CC79A7',
  Frequency: '#117733',
  Domain: '#882255',
  Augmentation: '#E15759',
  'Normalise & tensor': '#7A7A7A',
};

function LibraryItem({ def, serverOnly }: { def: StepDef; serverOnly: boolean }) {
  const { attributes, listeners, setNodeRef, isDragging } = useDraggable({ id: `lib:${def.id}` });
  const addStep = useStudio((s) => s.addStep);
  const toast = useModelStore((s) => s.toast);
  return (
    <li>
      <button
        ref={setNodeRef}
        {...listeners}
        {...attributes}
        onClick={() => {
          addStep(def.id);
          useStudio.getState().setMobilePanel('none');
          toast(`Added ${def.label}`, 'success');
        }}
        aria-label={def.label}
        aria-roledescription="draggable step"
        title={`${def.doc}\n\n${def.cv2}`}
        className={`group flex w-full items-start gap-2 rounded-md px-2 py-1.5 text-left hover:bg-panel-2 ${isDragging ? 'opacity-40' : ''}`}
      >
        <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: STEP_COLORS[def.category] }} aria-hidden />
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-1.5">
            <span className="truncate text-[13px]">{def.label}</span>
            {def.kind === 'random' && <span title="Random (augmentation)" aria-label="random">🎲</span>}
            {serverOnly && <span className="rounded bg-panel-2 px-1 text-[10px] text-muted" title="Runs on the Python server">server</span>}
          </span>
          <span className="block truncate text-[11px] text-muted">{def.doc}</span>
        </span>
      </button>
    </li>
  );
}

export function StepLibrary() {
  const [query, setQuery] = useState('');
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const missing = useRuns((s) => s.missing);
  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    const steps = allSteps().filter(
      (d) =>
        !d.hidden &&
        (!q || d.label.toLowerCase().includes(q) || d.id.includes(q) || d.doc.toLowerCase().includes(q) || d.cv2.toLowerCase().includes(q) || (d.keywords ?? '').includes(q) || d.category.toLowerCase().includes(q)),
    );
    return CATEGORIES.map((c) => [c, steps.filter((d) => d.category === c).sort((a, b) => a.label.localeCompare(b.label))] as [Category, StepDef[]]).filter(([, ds]) => ds.length);
  }, [query]);

  return (
    <nav className="flex h-full flex-col" aria-label="Step library">
      <div className="border-b border-line p-3">
        <label className="sr-only" htmlFor="step-search">
          Search steps
        </label>
        <input
          id="step-search"
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search steps, e.g. blur, cv2.Canny…"
          className="w-full rounded-md border border-line bg-panel-2 px-3 py-1.5 text-sm outline-none placeholder:text-muted focus:border-accent"
        />
        <p className="mt-2 text-[11px] leading-snug text-muted">Drag a step onto a pipeline row to insert it there (or onto “+ add step” to append), or click to append.</p>
      </div>
      <div className="scroll-thin flex-1 overflow-y-auto pb-6">
        {groups.map(([cat, defs]) => {
          const open = query ? true : !collapsed[cat];
          return (
            <section key={cat} className="border-b border-line/60">
              <button
                className="flex w-full items-center gap-2 px-3 py-2 text-left text-[11px] font-semibold tracking-wide text-muted uppercase hover:text-fg"
                onClick={() => setCollapsed((c) => ({ ...c, [cat]: open }))}
                aria-expanded={open}
              >
                <span className="h-2.5 w-2.5 rounded-sm" style={{ background: STEP_COLORS[cat] }} aria-hidden />
                <span className="flex-1">{cat}</span>
                <span className="font-normal normal-case">{defs.length}</span>
                <span aria-hidden>{open ? '−' : '+'}</span>
              </button>
              {open && (
                <ul className="px-2 pb-2">
                  {defs.map((d) => (
                    <LibraryItem key={d.id} def={d} serverOnly={!!missing[d.id]} />
                  ))}
                </ul>
              )}
            </section>
          );
        })}
        {!groups.length && <p className="p-4 text-sm text-muted">No steps match “{query}”.</p>}
      </div>
    </nav>
  );
}
