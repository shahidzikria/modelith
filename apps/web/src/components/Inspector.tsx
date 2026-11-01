import type { Analysis } from '@netloom/blocks';
import * as Tabs from '@radix-ui/react-tabs';
import { lazy, Suspense } from 'react';
import { useStore, type InspectorTab } from '../store';
import { Properties } from './Properties';
import { Summary } from './Summary';

const CodeView = lazy(() => import('./CodeView'));

const TABS: { id: InspectorTab; label: string }[] = [
  { id: 'properties', label: 'Properties' },
  { id: 'summary', label: 'Summary' },
  { id: 'code', label: 'Code' },
  { id: 'notes', label: 'Notes' },
];

export function Inspector({ analysis }: { analysis: Analysis }) {
  const tab = useStore((s) => s.inspectorTab);
  const setTab = useStore((s) => s.setTab);
  const notes = useStore((s) => s.project.notes ?? '');
  const setNotes = useStore((s) => s.setNotes);
  const errorCount = analysis.issues.filter((i) => i.level === 'error').length;

  return (
    <Tabs.Root value={tab} onValueChange={(v) => setTab(v as InspectorTab)} className="flex h-full flex-col" aria-label="Inspector">
      <Tabs.List className="flex shrink-0 border-b border-line px-2">
        {TABS.map((t) => (
          <Tabs.Trigger
            key={t.id}
            value={t.id}
            className="relative px-3 py-2.5 text-xs font-medium text-muted hover:text-fg data-[state=active]:text-fg data-[state=active]:after:absolute data-[state=active]:after:inset-x-2 data-[state=active]:after:bottom-0 data-[state=active]:after:h-0.5 data-[state=active]:after:bg-accent"
          >
            {t.label}
            {t.id === 'summary' && errorCount > 0 && (
              <span className="ml-1 rounded-full bg-danger px-1.5 text-[10px] text-white" aria-label={`${errorCount} errors`}>
                {errorCount}
              </span>
            )}
          </Tabs.Trigger>
        ))}
      </Tabs.List>
      <Tabs.Content value="properties" className="scroll-thin min-h-0 flex-1 overflow-y-auto">
        <Properties analysis={analysis} />
      </Tabs.Content>
      <Tabs.Content value="summary" className="scroll-thin min-h-0 flex-1 overflow-y-auto">
        <Summary analysis={analysis} />
      </Tabs.Content>
      <Tabs.Content value="code" className="min-h-0 flex-1">
        <Suspense fallback={<p className="p-4 text-sm text-muted">Loading editor…</p>}>
          <CodeView />
        </Suspense>
      </Tabs.Content>
      <Tabs.Content value="notes" className="flex min-h-0 flex-1 flex-col p-3">
        <label htmlFor="notes" className="mb-2 text-xs text-muted">
          Notes are saved with the project (and in graph.json).
        </label>
        <textarea
          id="notes"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          className="min-h-0 flex-1 resize-none rounded-md border border-line bg-panel-2 p-2 text-sm outline-none focus:border-accent"
          placeholder="Experiment ideas, dataset notes, results…"
        />
      </Tabs.Content>
    </Tabs.Root>
  );
}
