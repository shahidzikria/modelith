import { create } from 'zustand';

/** Viewer state shared with the analysis panel (line profile) — not undoable, not persisted. */
interface ViewerState {
  /** Line drawn with Shift+drag, in image pixels of the shown step: [x0, y0, x1, y1]. */
  line?: [number, number, number, number];
  setLine(l?: ViewerState['line']): void;
}

export const useViewer = create<ViewerState>((set) => ({
  setLine: (line) => set({ line }),
}));
