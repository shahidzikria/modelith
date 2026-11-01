import type { Analysis, Project } from '@netloom/blocks';
import { useStore as useZustand } from 'zustand';
import { autoLayout } from '../layout';
import { redo, undo, useStore, type Theme } from '../store';
import { download } from '../zip';

export async function runAutoLayout() {
  const { project, moveNodes, requestFit } = useStore.getState();
  const pos = await autoLayout(project.nodes, project.edges);
  moveNodes(Object.entries(pos).map(([id, position]) => ({ id, position })));
  requestFit();
}

function Btn(props: React.ButtonHTMLAttributes<HTMLButtonElement> & { label: string }) {
  const { label, children, className = '', ...rest } = props;
  return (
    <button
      aria-label={label}
      title={label}
      className={`inline-flex h-8 items-center justify-center gap-1 rounded-md border border-line px-2 text-xs hover:bg-panel-2 ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}

export function TopBar({ analysis }: { analysis: Analysis }) {
  const name = useStore((s) => s.project.name);
  const framework = useStore((s) => s.project.framework);
  const theme = useStore((s) => s.theme);
  const { rename, setFramework, setTab, setShowGallery, setTheme, setProject, toast, requestFit, setMobilePanel } = useStore.getState();
  const canUndo = useZustand(useStore.temporal, (s) => s.pastStates.length > 0);
  const canRedo = useZustand(useStore.temporal, (s) => s.futureStates.length > 0);
  const errors = analysis.issues.filter((i) => i.level === 'error').length;

  const exportProject = () => {
    const p = useStore.getState().project;
    download(new Blob([JSON.stringify(p, null, 2)], { type: 'application/json' }), `${p.name.replace(/\W+/g, '_')}.netloom.json`);
  };
  const importProject = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json,.netloom.json,application/json';
    input.onchange = async () => {
      const f = input.files?.[0];
      if (!f) return;
      try {
        const p = JSON.parse(await f.text()) as Project;
        if (p.version !== 2 || !Array.isArray(p.nodes)) throw new Error('not a Netloom v2 project');
        setProject(p);
        toast(`Imported ${p.name}`, 'success');
      } catch (e) {
        toast(`Import failed: ${(e as Error).message}`, 'error');
      }
    };
    input.click();
  };

  return (
    <header className="flex h-12 shrink-0 items-center gap-2 border-b border-line bg-panel px-2 sm:px-3">
      <button className="md:hidden" onClick={() => setMobilePanel('palette')} aria-label="Open block palette">
        <span className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-line">☰</span>
      </button>
      <div className="flex shrink-0 items-center gap-2">
        <img src="/favicon.svg" alt="" className="h-6 w-6" />
        <span className="hidden font-semibold sm:inline">Netloom</span>
      </div>
      <label htmlFor="project-name" className="sr-only">
        Project name
      </label>
      <input
        id="project-name"
        value={name}
        onChange={(e) => rename(e.target.value)}
        className="w-24 min-w-0 flex-shrink rounded-md border border-transparent bg-transparent px-2 py-1 text-sm hover:border-line focus:border-accent focus:outline-none sm:w-44"
      />
      <div className="hidden items-center gap-1 lg:flex">
        <Btn label="Templates" onClick={() => setShowGallery(true)}>
          Templates
        </Btn>
        <Btn label="Import .netloom.json" onClick={importProject}>
          Import
        </Btn>
        <Btn label="Export .netloom.json" onClick={exportProject}>
          Export
        </Btn>
      </div>
      <div className="mx-1 hidden h-5 w-px bg-line sm:block" />
      <div className="hidden items-center gap-1 sm:flex">
        <Btn label="Undo (Ctrl+Z)" onClick={undo} disabled={!canUndo}>
          ↶
        </Btn>
        <Btn label="Redo (Ctrl+Shift+Z)" onClick={redo} disabled={!canRedo}>
          ↷
        </Btn>
        <Btn label="Auto-layout (L)" onClick={runAutoLayout}>
          Layout
        </Btn>
        <Btn label="Fit view (F)" onClick={requestFit}>
          Fit
        </Btn>
      </div>
      <div className="ml-auto flex shrink-0 items-center gap-1.5 sm:gap-2">
        <div className="flex rounded-md border border-line p-0.5 text-xs" role="radiogroup" aria-label="Framework">
          {(['pytorch', 'tensorflow'] as const).map((f) => (
            <button
              key={f}
              role="radio"
              aria-checked={framework === f}
              onClick={() => setFramework(f)}
              className={`rounded px-2 py-1 ${framework === f ? 'bg-accent text-accent-fg' : 'text-muted hover:text-fg'}`}
            >
              <span className="sm:hidden">{f === "pytorch" ? "PT" : "TF"}</span>
              <span className="hidden sm:inline">{f === "pytorch" ? "PyTorch" : "TensorFlow"}</span>
            </button>
          ))}
        </div>
        <Btn
          label="Validate"
          onClick={() => {
            setTab('summary');
            setMobilePanel('inspector');
            toast(errors ? `${errors} error(s) — see Summary` : 'Graph is valid', errors ? 'error' : 'success');
          }}
          className={`max-sm:!hidden ${errors ? "text-danger" : ""}`}
        >
          {errors ? `✕ ${errors}` : '✓'} Validate
        </Btn>
        <button
          onClick={() => {
            setTab('code');
            setMobilePanel('inspector');
          }}
          className="inline-flex h-8 shrink-0 items-center rounded-md bg-accent px-2.5 text-xs font-semibold whitespace-nowrap text-accent-fg hover:brightness-110"
        >
          <span className="sm:hidden">Code ▸</span>
          <span className="hidden sm:inline">Generate code ▸</span>
        </button>
        <select
          aria-label="Theme"
          value={theme}
          onChange={(e) => setTheme(e.target.value as Theme)}
          className="hidden h-8 rounded-md border border-line bg-panel px-1 text-xs sm:block"
        >
          <option value="system">Auto</option>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
        <button className="md:hidden" onClick={() => setMobilePanel('inspector')} aria-label="Open inspector">
          <span className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-line">⚙</span>
        </button>
      </div>
    </header>
  );
}
