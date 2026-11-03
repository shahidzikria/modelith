import { getStep } from '@modelith/preprocess';
import { useRuns } from './engine';
import { useActivePipeline, useStudio } from './store';
import { Thumb } from './Thumb';
import { frameAt } from './Viewer';

/** Thumbnail after each step; click one to view the image at that step. */
export function Filmstrip() {
  const pipeline = useActivePipeline();
  const run = useRuns((s) => (s.A?.pipelineId === pipeline.id ? s.A : s.B));
  const selectedUid = useStudio((s) => s.selectedUid);
  const select = useStudio((s) => s.select);
  const items = [
    { uid: 'read', n: 0, label: 'Input', f: frameAt(run, -1), off: false },
    ...pipeline.steps.map((s, i) => ({ uid: s.uid, n: i + 1, label: getStep(s.def)?.label ?? s.def, f: frameAt(run, i), off: !!run?.steps[i]?.out.skipped })),
  ];
  return (
    <div className="scroll-thin flex shrink-0 gap-1.5 overflow-x-auto border-t border-line bg-panel px-2 py-1.5" role="listbox" aria-label="Filmstrip: result after each step" aria-orientation="horizontal">
      {items.map((it) => (
        <button
          key={it.uid}
          role="option"
          aria-selected={selectedUid === it.uid || (it.uid === 'read' && !selectedUid)}
          onClick={() => select(it.uid)}
          className={`flex w-[88px] shrink-0 flex-col overflow-hidden rounded-md border text-left ${selectedUid === it.uid ? 'border-accent ring-1 ring-accent' : 'border-line hover:border-accent/60'} ${it.off ? 'opacity-45' : ''}`}
          title={`${it.n}. ${it.label}`}
        >
          <div className="flex h-[52px] items-center justify-center bg-panel-2">{it.f ? <Thumb display={it.f.display} label={`${it.label} thumbnail`} /> : <span className="text-[10px] text-muted">…</span>}</div>
          <span className="truncate px-1 py-0.5 text-[10px]">
            {it.n}. {it.label}
          </span>
        </button>
      ))}
    </div>
  );
}
