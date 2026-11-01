import { getBlock, templates, type Project, type Template } from '@netloom/blocks';
import { useMemo } from 'react';
import { CATEGORY_COLORS } from '../colors';
import { emptyProject, useStore } from '../store';

function Thumbnail({ project }: { project: Project }) {
  const { nodes, edges } = project;
  const xs = nodes.map((n) => n.position.x);
  const ys = nodes.map((n) => n.position.y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const w = Math.max(...xs) - minX + 200;
  const h = Math.max(...ys) - minY + 90;
  const byId = new Map(nodes.map((n) => [n.id, n]));
  return (
    <svg viewBox={`${minX - 20} ${minY - 20} ${w + 40} ${h + 40}`} className="h-24 w-full" preserveAspectRatio="xMidYMid meet" aria-hidden>
      {edges.map((e) => {
        const a = byId.get(e.source);
        const b = byId.get(e.target);
        if (!a || !b) return null;
        const x1 = a.position.x + 200;
        const y1 = a.position.y + 40;
        const x2 = b.position.x;
        const y2 = b.position.y + 40;
        const mx = (x1 + x2) / 2;
        return <path key={e.id} d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`} stroke="var(--edge)" strokeWidth={6} fill="none" />;
      })}
      {nodes.map((n) => (
        <rect
          key={n.id}
          x={n.position.x}
          y={n.position.y}
          width={200}
          height={80}
          rx={14}
          fill="var(--panel)"
          stroke={CATEGORY_COLORS[getBlock(n.type)!.category]}
          strokeWidth={10}
        />
      ))}
    </svg>
  );
}

export function TemplateGallery({ onClose }: { onClose?: () => void }) {
  const framework = useStore((s) => s.project.framework);
  const setProject = useStore((s) => s.setProject);
  const built = useMemo(() => templates.map((t) => [t, t.build()] as [Template, Project]), []);

  return (
    <div className="absolute inset-0 z-20 overflow-y-auto bg-bg/95 backdrop-blur-sm">
      <div className="mx-auto max-w-5xl px-4 py-8">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-xl font-semibold">Start from a template</h2>
            <p className="mt-1 text-sm text-muted">Every template loads as an editable graph. You can also start empty and drag blocks from the palette.</p>
          </div>
          {onClose && (
            <button onClick={onClose} className="rounded-md border border-line px-3 py-1.5 text-sm hover:bg-panel-2" aria-label="Close template gallery">
              Close
            </button>
          )}
        </div>
        <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <button
            onClick={() => setProject({ ...emptyProject(), framework })}
            className="flex flex-col items-center justify-center rounded-xl border border-dashed border-line p-4 text-sm text-muted hover:border-accent hover:text-fg"
          >
            <span className="text-2xl" aria-hidden>
              +
            </span>
            Empty canvas
          </button>
          {built.map(([t, p]) => (
            <button
              key={t.id}
              onClick={() => setProject({ ...p, framework })}
              className="group flex flex-col rounded-xl border border-line bg-panel p-3 text-left shadow-sm transition hover:border-accent hover:shadow-md"
            >
              <Thumbnail project={p} />
              <div className="mt-2 flex items-center justify-between gap-2">
                <span className="font-medium">{t.name}</span>
                <span className="shrink-0 rounded bg-panel-2 px-1.5 py-0.5 text-[11px] text-muted">{t.task}</span>
              </div>
              <p className="mt-1 text-xs leading-snug text-muted">{t.description}</p>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
