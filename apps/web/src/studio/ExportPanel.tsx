import Editor, { type OnMount } from '@monaco-editor/react';
import { generatePipelineFiles, ident, type GeneratedFile } from '@modelith/preprocess';
import { useEffect, useMemo, useRef, useState } from 'react';
import '../monaco';
import { useStore as useModelStore } from '../store';
import { downloadZip } from '../zip';
import { useRuns } from './engine';
import { runOnServer, SERVER_HINT } from './server';
import { useActivePipeline, useStudio } from './store';
import { useModelInputs } from './useStudioAnalysis';

type Mono = Parameters<OnMount>[1];
type Ed = Parameters<OnMount>[0];

function useResolvedDark(): boolean {
  const theme = useModelStore((s) => s.theme);
  const [sysDark, setSysDark] = useState(() => window.matchMedia('(prefers-color-scheme: dark)').matches);
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const fn = () => setSysDark(mq.matches);
    mq.addEventListener('change', fn);
    return () => mq.removeEventListener('change', fn);
  }, []);
  return theme === 'dark' || (theme === 'system' && sysDark);
}

export function usePipelineFiles(): GeneratedFile[] {
  const pipeline = useActivePipeline();
  const inputs = useModelInputs();
  const framework = useModelStore((s) => s.project.framework);
  const [debounced, setDebounced] = useState(pipeline);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(pipeline), 150);
    return () => clearTimeout(t);
  }, [pipeline]);
  const model = debounced.linkedModelId ? inputs.find((i) => i.id === debounced.linkedModelId)?.input : undefined;
  return useMemo(() => generatePipelineFiles(debounced, { framework: model?.framework ?? framework, model }), [debounced, model, framework]);
}

export function ExportPanel() {
  const pipeline = useActivePipeline();
  const files = usePipelineFiles();
  const selectedUid = useStudio((s) => s.selectedUid);
  const hoverUid = useStudio((s) => s.hoverUid);
  const setCodeHighlight = useStudio((s) => s.setCodeHighlight);
  const toast = useModelStore((s) => s.toast);
  const dark = useResolvedDark();
  const [active, setActive] = useState('preprocess.py');
  const [timing, setTiming] = useState<string>();
  const editorRef = useRef<Ed | null>(null);
  const monacoRef = useRef<Mono | null>(null);
  const decorations = useRef<string[]>([]);
  const file = files.find((f) => f.path === active) ?? files[0];
  const lineMapRef = useRef<Record<number, string>>({});
  lineMapRef.current = file?.lineMap ?? {};

  // Pipeline → code: highlight the hovered (else selected) step's lines.
  const focusUid = hoverUid ?? selectedUid;
  useEffect(() => {
    const ed = editorRef.current;
    const mono = monacoRef.current;
    if (!ed || !mono || !file) return;
    const lines = Object.entries(file.lineMap ?? {})
      .filter(([, u]) => u === focusUid)
      .map(([l]) => Number(l));
    decorations.current = ed.deltaDecorations(
      decorations.current,
      lines.map((l) => ({ range: new mono.Range(l, 1, l, 1), options: { isWholeLine: true, className: 'modelith-code-highlight' } })),
    );
    const fnLine = lines.find((l) => /^def step_/.test(file.content.split('\n')[l - 1] ?? '')) ?? lines[0];
    if (fnLine) ed.revealLineInCenterIfOutsideViewport(fnLine);
  }, [focusUid, file]);

  const onMount: OnMount = (ed, mono) => {
    editorRef.current = ed;
    monacoRef.current = mono;
    // Code → pipeline: highlight the step owning the cursor line.
    ed.onDidChangeCursorPosition((e) => setCodeHighlight(lineMapRef.current[e.position.lineNumber]));
    ed.onDidBlurEditorText(() => setCodeHighlight(undefined));
  };

  const measure = async () => {
    const run = useRuns.getState().A;
    if (!run?.input) return;
    setTiming('Measuring on the Python server…');
    try {
      const { steps, totalMs } = await runOnServer(pipeline, run.input.raw, run.input.targets, { start: 0, train: useStudio.getState().train, timing: true });
      setTiming(`Python (exported code, median of 5): ${totalMs.toFixed(1)} ms/image ≈ ${Math.round(1000 / Math.max(totalMs, 1e-3))} img/s · ${steps.map((s) => `${s.name} ${s.ms.toFixed(1)}`).join(' · ')}`);
    } catch (e) {
      setTiming(e instanceof TypeError ? `Python server not reachable. ${SERVER_HINT}` : (e as Error).message);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="scroll-thin flex items-center gap-1 overflow-x-auto border-b border-line px-2 py-1.5" role="tablist" aria-label="Exported files">
        {files.map((f) => (
          <button
            key={f.path}
            role="tab"
            aria-selected={f.path === file?.path}
            onClick={() => setActive(f.path)}
            className={`shrink-0 rounded px-2 py-1 font-mono text-[11px] ${f.path === file?.path ? 'bg-panel-2 text-fg' : 'text-muted hover:text-fg'}`}
          >
            {f.path}
          </button>
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-1.5 border-b border-line px-2 py-1.5 text-xs">
        <button
          className="rounded border border-line px-2 py-1 hover:bg-panel-2"
          onClick={async () => {
            await navigator.clipboard.writeText(file?.content ?? '');
            toast(`Copied ${file?.path}`, 'success');
          }}
        >
          Copy file
        </button>
        <button className="rounded bg-accent px-2 py-1 font-medium text-accent-fg hover:brightness-110" onClick={() => downloadZip(`${ident(pipeline.name)}_preprocessing`, files).then(() => toast('Downloaded ZIP', 'success'))}>
          Download ZIP
        </button>
        <button className="rounded border border-line px-2 py-1 hover:bg-panel-2" onClick={measure} title="Run the exported preprocess.py on the server and time each step">
          Measure on server
        </button>
      </div>
      {timing && <p className="border-b border-line px-2 py-1 text-[11px] text-muted">{timing}</p>}
      <div className="min-h-0 flex-1">
        <Editor
          path={`studio/${file?.path}`}
          value={file?.content}
          language={file?.language === 'json' || file?.language === 'text' ? 'plaintext' : file?.language}
          theme={dark ? 'vs-dark' : 'vs'}
          onMount={onMount}
          options={{ readOnly: true, minimap: { enabled: false }, fontSize: 12, lineNumbersMinChars: 3, scrollBeyondLastLine: false, automaticLayout: true, tabSize: 4 }}
        />
      </div>
    </div>
  );
}
