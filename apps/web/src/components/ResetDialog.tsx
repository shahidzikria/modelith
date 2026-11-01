import * as Dialog from '@radix-ui/react-dialog';
import { useStore } from '../store';

/** "Reset" button: confirms, then clears the canvas (undoable with Ctrl+Z). */
export function ResetButton({ className = '', onDone }: { className?: string; onDone?: () => void }) {
  const count = useStore((s) => s.project.nodes.length);
  const reset = () => {
    const { resetProject, toast } = useStore.getState();
    resetProject();
    toast('Canvas cleared — Ctrl+Z to undo', 'info');
    onDone?.();
  };
  return (
    <Dialog.Root>
      <Dialog.Trigger asChild>
        <button
          disabled={!count}
          title="Clear the canvas"
          className={`inline-flex h-8 items-center justify-center rounded-md border border-line px-2 text-xs hover:bg-panel-2 disabled:opacity-40 ${className}`}
        >
          Reset
        </button>
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/40" />
        <Dialog.Content className="fixed top-1/2 left-1/2 z-50 w-[min(92vw,380px)] -translate-x-1/2 -translate-y-1/2 rounded-xl border border-line bg-panel p-5 text-fg shadow-xl">
          <Dialog.Title className="text-sm font-semibold">Reset the canvas?</Dialog.Title>
          <Dialog.Description className="mt-2 text-sm text-muted">
            This removes all {count} block{count === 1 ? '' : 's'} and their connections. You can undo it with Ctrl+Z.
          </Dialog.Description>
          <div className="mt-5 flex justify-end gap-2">
            <Dialog.Close asChild>
              <button className="rounded-md border border-line px-3 py-1.5 text-xs hover:bg-panel-2">Cancel</button>
            </Dialog.Close>
            <Dialog.Close asChild>
              <button onClick={reset} className="rounded-md bg-danger px-3 py-1.5 text-xs font-semibold text-white hover:brightness-110">
                Reset canvas
              </button>
            </Dialog.Close>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
