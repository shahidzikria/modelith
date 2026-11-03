import Editor, { type OnMount } from '@monaco-editor/react';
import { CodegenError, generateProject, type GeneratedFile, type Project, type ProjectOptions } from '@modelith/blocks';
import type { Pipeline } from '@modelith/preprocess';
import { projectPreprocessing, useLinks } from '../studio/links';
import { useEffect, useMemo, useRef, useState } from 'react';
import '../monaco';
import { useStore } from '../store';
import { downloadZip } from '../zip';

type Mono = Parameters<OnMount>[1];
type Ed = Parameters<OnMount>[0];

function useResolvedDark(): boolean {
  const theme = useStore((s) => s.theme);
  const [sysDark, setSysDark] = useState(() => window.matchMedia('(prefers-color-scheme: dark)').matches);
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const fn = () => setSysDark(mq.matches);
    mq.addEventListener('change', fn);
    return () => mq.removeEventListener('change', fn);
  }, []);
  return theme === 'dark' || (theme === 'system' && sysDark);
}

export function generate(project: Project, pipelines: Pipeline[] = []): { files?: GeneratedFile[]; errors?: string[] } {
  try {
    return { files: generateProject(project, { preprocessing: projectPreprocessing(project, pipelines) as ProjectOptions['preprocessing'] }).files };
  } catch (err) {
    if (err instanceof CodegenError) return { errors: err.issues };
    return { errors: [(err as Error).message] };
  }
}

export default function CodeView() {
  const project = useStore((s) => s.project);
  const selected = useStore((s) => s.selected);
  const toast = useStore((s) => s.toast);
  const setCodeHighlight = useStore((s) => s.setCodeHighlight);
  const dark = useResolvedDark();
  const [active, setActive] = useState('model.py');
  const [editable, setEditable] = useState(false);
  const editorRef = useRef<Ed | null>(null);
  const monacoRef = useRef<Mono | null>(null);
  const decorations = useRef<string[]>([]);

  // Debounced generation so typing in the inspector stays snappy.
  const [debounced, setDebounced] = useState(project);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(project), 150);
    return () => clearTimeout(t);
  }, [project]);
  const pipelines = useLinks((s) => s.pipelines);
  const result = useMemo(() => generate(debounced, pipelines), [debounced, pipelines]);
  const file = result.files?.find((f) => f.path === active) ?? result.files?.[0];
  const lineMapRef = useRef<Record<number, string>>({});
  lineMapRef.current = file?.lineMap ?? {};

  // Graph → code: reveal the first line of the selected node.
  useEffect(() => {
    const ed = editorRef.current;
    const mono = monacoRef.current;
    if (!ed || !mono || !file) return;
    const id = selected[selected.length - 1];
    const lines = Object.entries(file.lineMap ?? {})
      .filter(([, n]) => n === id)
      .map(([l]) => Number(l));
    decorations.current = ed.deltaDecorations(
      decorations.current,
      lines.map((l) => ({ range: new mono.Range(l, 1, l, 1), options: { isWholeLine: true, className: 'modelith-code-highlight' } })),
    );
    if (lines.length) ed.revealLineInCenterIfOutsideViewport(lines[0]);
  }, [selected, file]);

  const onMount: OnMount = (ed, mono) => {
    editorRef.current = ed;
    monacoRef.current = mono;
    // Code → graph: highlight the node owning the cursor line.
    ed.onDidChangeCursorPosition((e) => setCodeHighlight(lineMapRef.current[e.position.lineNumber]));
    ed.onDidBlurEditorText(() => setCodeHighlight(undefined));
  };

  if (result.errors) {
    return (
      <div className="p-4 text-sm">
        <p className="font-medium">Fix these issues to generate code:</p>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-xs text-danger">
          {result.errors.map((e, i) => (
            <li key={i}>{e}</li>
          ))}
        </ul>
      </div>
    );
  }
  const files = result.files!;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-1 overflow-x-auto border-b border-line px-2 py-1.5" role="tablist" aria-label="Generated files">
        {files.map((f) => (
          <button
            key={f.path}
            role="tab"
            aria-selected={f.path === file?.path}
            onClick={() => setActive(f.path)}
            className={`shrink-0 rounded px-2 py-1 font-mono text-xs ${f.path === file?.path ? 'bg-panel-2 text-fg' : 'text-muted hover:text-fg'}`}
          >
            {f.path}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-2 border-b border-line px-2 py-1.5 text-xs">
        <button
          className="rounded border border-line px-2 py-1 hover:bg-panel-2"
          onClick={async () => {
            await navigator.clipboard.writeText(editorRef.current?.getValue() ?? file?.content ?? '');
            toast(`Copied ${file?.path}`, 'success');
          }}
        >
          Copy file
        </button>
        <button
          className="rounded bg-accent px-2 py-1 font-medium text-accent-fg hover:brightness-110"
          onClick={() => downloadZip(project.name, files).then(() => toast('Downloaded project ZIP', 'success'))}
        >
          Download ZIP
        </button>
        <label className="ml-auto flex items-center gap-1.5 text-muted">
          <input type="checkbox" checked={editable} onChange={(e) => setEditable(e.target.checked)} className="accent-[var(--accent)]" />
          Edit
        </label>
      </div>
      <div className="min-h-0 flex-1">
        <Editor
          path={file?.path}
          value={file?.content}
          language={file?.language === 'json' || file?.language === 'text' ? 'plaintext' : file?.language}
          theme={dark ? 'vs-dark' : 'vs'}
          onMount={onMount}
          options={{
            readOnly: !editable,
            minimap: { enabled: false },
            fontSize: 12,
            lineNumbersMinChars: 3,
            scrollBeyondLastLine: false,
            wordWrap: 'off',
            renderLineHighlight: 'line',
            automaticLayout: true,
            tabSize: 4,
          }}
        />
      </div>
    </div>
  );
}
