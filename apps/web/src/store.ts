import {
  analyze,
  defaultParams,
  kindName,
  getBlock,
  incoming,
  type Framework,
  type GraphEdge,
  type GraphNode,
  type Params,
  type Project,
} from '@modelith/blocks';
import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { temporal } from 'zundo';

export type InspectorTab = 'properties' | 'summary' | 'code' | 'notes';
export type Theme = 'system' | 'light' | 'dark';

export interface Toast {
  id: number;
  message: string;
  kind: 'info' | 'error' | 'success';
}

interface Clipboard {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface State {
  project: Project;
  selected: string[];
  inspectorTab: InspectorTab;
  theme: Theme;
  /** Node highlighted because the code cursor is on one of its lines. */
  codeHighlight?: string;
  toasts: Toast[];
  clipboard?: Clipboard;
  showGallery: boolean;
  mobilePanel: "none" | "palette" | "inspector";
  /** Bumped when the canvas should fit the view (project loaded, auto-layout). */
  viewNonce: number;
  requestFit(): void;

  setProject(p: Project): void;
  /** Clear the canvas (undoable); keeps the framework choice. */
  resetProject(): void;
  resetParams(id: string): void;
  rename(name: string): void;
  setFramework(f: Framework): void;
  setNotes(notes: string): void;
  addNode(type: string, position?: { x: number; y: number }, opts?: { connectFrom?: string; params?: Params }): string;
  updateParams(id: string, patch: Params): void;
  setLabel(id: string, label: string): void;
  moveNodes(moves: { id: string; position: { x: number; y: number } }[]): void;
  connect(e: Omit<GraphEdge, 'id'>): void;
  removeEdges(ids: string[]): void;
  removeNodes(ids: string[]): void;
  select(ids: string[]): void;
  duplicate(): void;
  copy(): void;
  paste(): void;
  setTab(t: InspectorTab): void;
  setTheme(t: Theme): void;
  setCodeHighlight(id?: string): void;
  toast(message: string, kind?: Toast['kind']): void;
  dismissToast(id: number): void;
  setShowGallery(v: boolean): void;
  setMobilePanel(p: State['mobilePanel']): void;
}

let counter = 0;
export const newId = (prefix = 'n') => `${prefix}${Date.now().toString(36)}${(counter++).toString(36)}`;

export const emptyProject = (): Project => ({
  version: 2,
  name: 'Untitled model',
  framework: 'pytorch',
  nodes: [],
  edges: [],
  customBlocks: [],
  notes: '',
});

const STORAGE_KEY = 'modelith:project';
const THEME_KEY = 'modelith:theme';

function loadSaved(): Project | undefined {
  try {
    // Fall back to the key used before the rename from Netloom.
    const raw = localStorage.getItem(STORAGE_KEY) ?? localStorage.getItem('netloom:project');
    if (!raw) return undefined;
    const p = JSON.parse(raw) as Project;
    if (p.version !== 2 || !Array.isArray(p.nodes)) return undefined;
    return { ...emptyProject(), ...p, nodes: p.nodes.filter((n) => getBlock(n.type)) };
  } catch {
    return undefined;
  }
}

function loadTheme(): Theme {
  try {
    return (localStorage.getItem(THEME_KEY) as Theme) || 'system';
  } catch {
    return 'system';
  }
}

/** Would adding source → target create a cycle? */
export function createsCycle(edges: GraphEdge[], source: string, target: string): boolean {
  if (source === target) return true;
  const stack = [target];
  const seen = new Set<string>();
  while (stack.length) {
    const id = stack.pop()!;
    if (id === source) return true;
    if (seen.has(id)) continue;
    seen.add(id);
    for (const e of edges) if (e.source === id) stack.push(e.target);
  }
  return false;
}

const saved = loadSaved();

export const useStore = create<State>()(
  temporal(
    immer((set, get) => ({
      project: saved ?? emptyProject(),
      selected: [],
      inspectorTab: 'properties',
      theme: loadTheme(),
      toasts: [],
      showGallery: !saved || saved.nodes.length === 0,
      mobilePanel: "none",
      viewNonce: 0,

      setProject: (p) =>
        set((s) => {
          s.project = { ...emptyProject(), ...structuredClone(p) };
          s.selected = [];
          s.showGallery = false;
          s.viewNonce++;
        }),
      resetProject: () =>
        set((s) => {
          s.project = { ...emptyProject(), framework: s.project.framework };
          s.selected = [];
          s.codeHighlight = undefined;
        }),
      resetParams: (id) =>
        set((s) => {
          const n = s.project.nodes.find((x) => x.id === id);
          if (n) n.params = { ...defaultParams(n.type), ...(getBlock(n.type)?.isInput ? { name: n.params.name } : {}) };
        }),
      rename: (name) =>
        set((s) => {
          s.project.name = name;
        }),
      setFramework: (f) =>
        set((s) => {
          s.project.framework = f;
        }),
      setNotes: (notes) =>
        set((s) => {
          s.project.notes = notes;
        }),

      addNode: (type, position, opts = {}) => {
        const id = newId();
        const { project, selected } = get();
        const from = opts.connectFrom ?? (position ? undefined : selected[selected.length - 1]);
        const anchor = from ? project.nodes.find((n) => n.id === from) : undefined;
        let pos = position;
        if (!pos) {
          if (anchor) {
            pos = { x: anchor.position.x + 260, y: anchor.position.y };
            // Avoid stacking on an existing node.
            while (project.nodes.some((n) => Math.abs(n.position.x - pos!.x) < 40 && Math.abs(n.position.y - pos!.y) < 40)) {
              pos = { x: pos.x, y: pos.y + 120 };
            }
          } else {
            const maxX = Math.max(-260, ...project.nodes.map((n) => n.position.x));
            pos = { x: maxX + 260, y: 0 };
          }
        }
        const def = getBlock(type)!;
        // Only auto-connect when the selected block's output fits the new block's first port.
        let connect = !!anchor && def.inputs.length > 0;
        if (connect && anchor) {
          const shape = analyze({ nodes: project.nodes, edges: project.edges, framework: project.framework }).nodes[anchor.id]?.shape;
          if (shape && !def.inputs[0].accepts.includes(shape.kind)) {
            connect = false;
            setTimeout(() => get().toast(`Not connected: ${def.label} needs ${def.inputs[0].accepts.map(kindName).join(" or ")}, got ${kindName(shape.kind)}.`, "error"));
          }
        }
        set((s) => {
          s.project.nodes.push({ id, type, position: pos!, params: { ...defaultParams(type), ...(opts.params ?? {}) } });
          if (connect && anchor) {
            s.project.edges.push({ id: newId('e'), source: anchor.id, target: id, targetPort: 0 });
          }
          s.selected = [id];
          s.showGallery = false;
        });
        return id;
      },

      updateParams: (id, patch) =>
        set((s) => {
          const n = s.project.nodes.find((x) => x.id === id);
          if (n) Object.assign(n.params, patch);
        }),
      setLabel: (id, label) =>
        set((s) => {
          const n = s.project.nodes.find((x) => x.id === id);
          if (!n) return;
          if (label.trim()) n.label = label;
          else delete n.label;
        }),
      moveNodes: (moves) =>
        set((s) => {
          for (const m of moves) {
            const n = s.project.nodes.find((x) => x.id === m.id);
            if (n) n.position = m.position;
          }
        }),

      connect: (e) =>
        set((s) => {
          const def = getBlock(s.project.nodes.find((n) => n.id === e.target)?.type ?? '');
          if (!def) return;
          // Dropping on an occupied port replaces its edge.
          s.project.edges = s.project.edges.filter((x) => !(x.target === e.target && x.targetPort === e.targetPort));
          s.project.edges.push({ ...e, id: newId('e') });
          if (def.variadic) {
            // Compact port indices so the node always shows one free port.
            const ins = incoming(e.target, s.project.edges, def);
            ins.forEach((x, i) => {
              const edge = s.project.edges.find((y) => y.id === x?.id);
              if (edge) edge.targetPort = i;
            });
          }
        }),
      removeEdges: (ids) =>
        set((s) => {
          s.project.edges = s.project.edges.filter((e) => !ids.includes(e.id));
        }),
      removeNodes: (ids) =>
        set((s) => {
          s.project.nodes = s.project.nodes.filter((n) => !ids.includes(n.id));
          s.project.edges = s.project.edges.filter((e) => !ids.includes(e.source) && !ids.includes(e.target));
          s.selected = s.selected.filter((id) => !ids.includes(id));
        }),
      select: (ids) =>
        set((s) => {
          s.selected = ids;
        }),

      duplicate: () => {
        get().copy();
        get().paste();
      },
      copy: () => {
        const { project, selected } = get();
        const nodes = project.nodes.filter((n) => selected.includes(n.id));
        const edges = project.edges.filter((e) => selected.includes(e.source) && selected.includes(e.target));
        set((s) => {
          s.clipboard = structuredClone({ nodes, edges });
        });
      },
      paste: () => {
        const clip = get().clipboard;
        if (!clip?.nodes.length) return;
        const map = new Map<string, string>();
        set((s) => {
          for (const n of clip.nodes) {
            const id = newId();
            map.set(n.id, id);
            s.project.nodes.push({ ...structuredClone(n), id, position: { x: n.position.x + 40, y: n.position.y + 140 } });
          }
          for (const e of clip.edges) {
            s.project.edges.push({ ...e, id: newId('e'), source: map.get(e.source)!, target: map.get(e.target)! });
          }
          s.selected = [...map.values()];
        });
      },

      setTab: (t) =>
        set((s) => {
          s.inspectorTab = t;
        }),
      setTheme: (t) => {
        try {
          localStorage.setItem(THEME_KEY, t);
        } catch {
          /* storage unavailable */
        }
        set((s) => {
          s.theme = t;
        });
      },
      setCodeHighlight: (id) =>
        set((s) => {
          s.codeHighlight = id;
        }),
      toast: (message, kind = 'info') => {
        const id = ++counter;
        set((s) => {
          s.toasts.push({ id, message, kind });
        });
        setTimeout(() => get().dismissToast(id), kind === 'error' ? 6000 : 3500);
      },
      dismissToast: (id) =>
        set((s) => {
          s.toasts = s.toasts.filter((t) => t.id !== id);
        }),
      setShowGallery: (v) =>
        set((s) => {
          s.showGallery = v;
        }),
      requestFit: () =>
        set((s) => {
          s.viewNonce++;
        }),
      setMobilePanel: (p) =>
        set((s) => {
          s.mobilePanel = p;
        }),
    })),
    {
      partialize: (s) => ({ project: s.project }),
      equality: (a, b) => a.project === b.project,
      limit: 200,
    },
  ),
);

export const undo = () => useStore.temporal.getState().undo();
export const redo = () => useStore.temporal.getState().redo();

// Autosave (debounced).
let saveTimer: ReturnType<typeof setTimeout> | undefined;
useStore.subscribe((s, prev) => {
  if (s.project === prev.project) return;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(s.project));
    } catch {
      /* storage unavailable (private mode / quota) */
    }
  }, 300);
});
