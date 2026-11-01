import { allBlocks, CATEGORIES, type BlockDef, type Category } from '@netloom/blocks';
import { useMemo, useState } from 'react';
import { CATEGORY_COLORS } from '../colors';
import { redo, undo, useStore } from '../store';
import { runAutoLayout } from './TopBar';

export function Palette() {
  const [query, setQuery] = useState('');
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const addNode = useStore((s) => s.addNode);
  const toast = useStore((s) => s.toast);
  const setMobilePanel = useStore((s) => s.setMobilePanel);

  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    const blocks = allBlocks().filter(
      (b) => !q || b.label.toLowerCase().includes(q) || b.type.includes(q) || b.doc.toLowerCase().includes(q) || b.category.toLowerCase().includes(q),
    );
    return CATEGORIES.map((c) => [c, blocks.filter((b) => b.category === c)] as [Category, BlockDef[]]).filter(([, bs]) => bs.length);
  }, [query]);

  const add = (b: BlockDef) => {
    const id = addNode(b.type);
    setMobilePanel('none');
    const sel = useStore.getState().project.edges.some((e) => e.target === id);
    toast(sel ? `Added ${b.label} after the selected block` : `Added ${b.label}`, 'success');
  };

  return (
    <nav className="flex h-full flex-col" aria-label="Block palette">
      <div className="border-b border-line p-3">
        <div className="mb-3 flex flex-wrap gap-1.5 lg:hidden">
          {[
            ['Templates', () => useStore.getState().setShowGallery(true)],
            ['Undo', undo],
            ['Redo', redo],
            ['Layout', () => void runAutoLayout()],
          ].map(([label, fn]) => (
            <button
              key={label as string}
              onClick={() => {
                (fn as () => void)();
                setMobilePanel('none');
              }}
              className="rounded-md border border-line px-2 py-1 text-xs hover:bg-panel-2"
            >
              {label as string}
            </button>
          ))}
        </div>
        <label className="sr-only" htmlFor="palette-search">
          Search blocks
        </label>
        <input
          id="palette-search"
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search blocks…"
          className="w-full rounded-md border border-line bg-panel-2 px-3 py-1.5 text-sm outline-none placeholder:text-muted focus:border-accent"
        />
        <p className="mt-2 text-[11px] leading-snug text-muted">Drag onto the canvas, or click to append after the selected block.</p>
      </div>
      <div className="scroll-thin flex-1 overflow-y-auto pb-6">
        {groups.map(([cat, blocks]) => {
          const open = query ? true : !collapsed[cat];
          return (
            <section key={cat} className="border-b border-line/60">
              <button
                className="flex w-full items-center gap-2 px-3 py-2 text-left text-[11px] font-semibold tracking-wide text-muted uppercase hover:text-fg"
                onClick={() => setCollapsed((c) => ({ ...c, [cat]: open }))}
                aria-expanded={open}
              >
                <span className="h-2.5 w-2.5 rounded-sm" style={{ background: CATEGORY_COLORS[cat] }} aria-hidden />
                <span className="flex-1">{cat}</span>
                <span aria-hidden>{open ? '−' : '+'}</span>
              </button>
              {open && (
                <ul className="px-2 pb-2">
                  {blocks.map((b) => (
                    <li key={b.type}>
                      <button
                        draggable
                        onDragStart={(e) => {
                          e.dataTransfer.setData('application/netloom-block', b.type);
                          e.dataTransfer.effectAllowed = 'copy';
                        }}
                        onClick={() => add(b)}
                        aria-label={b.label}
                        className="group flex w-full items-start gap-2 rounded-md px-2 py-1.5 text-left hover:bg-panel-2"
                        title={`${b.doc}${b.paper ? `\n\nPaper: ${b.paper}` : ''}`}
                      >
                        <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: CATEGORY_COLORS[cat] }} aria-hidden />
                        <span className="min-w-0">
                          <span className="block truncate text-[13px]">{b.label}</span>
                          <span className="block truncate text-[11px] text-muted">{b.doc}</span>
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          );
        })}
        {!groups.length && <p className="p-4 text-sm text-muted">No blocks match “{query}”.</p>}
      </div>
    </nav>
  );
}
