import { useStore } from '../store';

export function Toasts() {
  const toasts = useStore((s) => s.toasts);
  const dismiss = useStore((s) => s.dismissToast);
  return (
    <div className="pointer-events-none fixed bottom-4 left-1/2 z-50 flex -translate-x-1/2 flex-col items-center gap-2" role="status" aria-live="polite">
      {toasts.map((t) => (
        <button
          key={t.id}
          onClick={() => dismiss(t.id)}
          className={`pointer-events-auto max-w-[min(92vw,520px)] rounded-lg border px-3 py-2 text-left text-sm shadow-lg ${
            t.kind === 'error' ? 'border-danger/50 bg-panel text-danger' : t.kind === 'success' ? 'border-ok/40 bg-panel text-fg' : 'border-line bg-panel text-fg'
          }`}
        >
          {t.message}
        </button>
      ))}
    </div>
  );
}
