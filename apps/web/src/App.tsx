import { ReactFlowProvider, useReactFlow } from '@xyflow/react';
import { useEffect } from 'react';
import { Canvas } from './components/Canvas';
import { Inspector } from './components/Inspector';
import { Palette } from './components/Palette';
import { TemplateGallery } from './components/TemplateGallery';
import { Toasts } from './components/Toasts';
import { runAutoLayout, TopBar } from './components/TopBar';
import { redo, undo, useStore } from './store';
import { useAnalysis } from './useAnalysis';

function isTyping(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null;
  return !!t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable || !!t.closest('.monaco-editor'));
}

function useShortcuts() {
  const { fitView } = useReactFlow();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e)) return;
      const mod = e.ctrlKey || e.metaKey;
      const k = e.key.toLowerCase();
      const st = useStore.getState();
      if (mod && k === 'z' && !e.shiftKey) undo();
      else if (mod && ((k === 'z' && e.shiftKey) || k === 'y')) redo();
      else if (mod && k === 'd') st.duplicate();
      else if (mod && k === 'c') st.copy();
      else if (mod && k === 'v') st.paste();
      else if (!mod && k === 'f') fitView({ padding: 0.15, duration: 300 });
      else if (!mod && k === 'l') void runAutoLayout();
      else if (!mod && k === 'escape') st.select([]);
      else return;
      e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [fitView]);
}

function useThemeAttr() {
  const theme = useStore((s) => s.theme);
  useEffect(() => {
    if (theme === 'system') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);
}

function Shell() {
  const analysis = useAnalysis();
  const showGallery = useStore((s) => s.showGallery);
  const hasNodes = useStore((s) => s.project.nodes.length > 0);
  const mobilePanel = useStore((s) => s.mobilePanel);
  const setMobilePanel = useStore((s) => s.setMobilePanel);
  const setShowGallery = useStore((s) => s.setShowGallery);
  useShortcuts();
  useThemeAttr();

  const drawer = (side: 'left' | 'right') =>
    `fixed inset-y-0 ${side}-0 z-40 w-[86vw] max-w-sm border-line bg-panel shadow-xl md:static md:z-auto md:w-auto md:max-w-none md:shadow-none`;

  return (
    <div className="flex h-full flex-col">
      <TopBar analysis={analysis} />
      <div className="relative flex min-h-0 flex-1">
        <aside
          className={`${drawer('left')} border-r md:block md:w-60 lg:w-64 ${mobilePanel === 'palette' ? 'block' : 'hidden'}`}
          aria-label="Palette"
        >
          <Palette />
        </aside>
        <main className="relative min-w-0 flex-1">
          <Canvas analysis={analysis} />
          {!hasNodes && !showGallery && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center p-6">
              <div className="pointer-events-auto max-w-sm rounded-xl border border-dashed border-line bg-panel/80 p-5 text-center text-sm text-muted">
                <p className="font-medium text-fg">Empty canvas</p>
                <p className="mt-1">Drag blocks from the palette (start with an Input), or click them to append.</p>
                <button onClick={() => setShowGallery(true)} className="mt-3 rounded-md border border-line px-3 py-1.5 text-xs text-fg hover:bg-panel-2">
                  Browse templates
                </button>
              </div>
            </div>
          )}
          {showGallery && <TemplateGallery onClose={() => setShowGallery(false)} />}
        </main>
        <aside
          className={`${drawer('right')} border-l md:block md:w-[380px] xl:w-[460px] ${mobilePanel === 'inspector' ? 'block' : 'hidden'}`}
          aria-label="Inspector panel"
        >
          <div className="flex items-center justify-end border-b border-line p-1 md:hidden">
            <button onClick={() => setMobilePanel('none')} className="rounded px-2 py-1 text-sm" aria-label="Close inspector">
              Close ✕
            </button>
          </div>
          <div className="h-full md:h-full">
            <Inspector analysis={analysis} />
          </div>
        </aside>
        {mobilePanel !== 'none' && <div className="fixed inset-0 z-30 bg-black/30 md:hidden" onClick={() => setMobilePanel('none')} aria-hidden />}
      </div>
      <Toasts />
    </div>
  );
}

export default function App() {
  return (
    <ReactFlowProvider>
      <Shell />
    </ReactFlowProvider>
  );
}
