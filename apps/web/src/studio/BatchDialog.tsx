import { analyzePipeline, ident, type Display, type ImageBuf } from '@modelith/preprocess';
import JSZip from 'jszip';
import { useRef, useState } from 'react';
import { download } from '../zip';
import { invalidateSample, requestDisplay, runSteps, runVariants, sampleInput, sampleTargets, useRuns } from './engine';
import { IMAGE_ACCEPT, isImageName, loadSampleFile, pickFiles } from './samples';
import { runOnServer, fromWire } from './server';
import { useActivePipeline, useStudio, type Sample } from './store';
import { Thumb } from './Thumb';

interface Item {
  name: string;
  display?: Display;
  raw?: ImageBuf;
  error?: string;
}

/** Encode a result for download: PNG for 8-bit images (BGR → RGB handled), else .npy. */
async function encode(buf: ImageBuf, display: Display): Promise<{ ext: string; data: Uint8Array }> {
  const viewable = buf.meta.dtype === 'uint8' && buf.meta.layout === 'HWC' && !buf.meta.batch && [1, 3, 4].includes(buf.meta.c);
  if (viewable) {
    const c = new OffscreenCanvas(display.w, display.h);
    c.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(display.rgba), display.w, display.h), 0, 0);
    const blob = await c.convertToBlob({ type: 'image/png' });
    return { ext: 'png', data: new Uint8Array(await blob.arrayBuffer()) };
  }
  const { h, w, c } = buf.meta;
  const descr = buf.meta.dtype === 'uint8' ? '|u1' : buf.meta.dtype === 'uint16' ? '<u2' : '<f4';
  let shape = c === 1 ? [h, w] : [h, w, c];
  let data = buf.data;
  if (buf.meta.layout === 'CHW') {
    shape = [c, h, w];
    const t = new (data.constructor as any)(data.length);
    for (let i = 0; i < h * w; i++) for (let k = 0; k < c; k++) t[k * h * w + i] = data[i * c + k];
    data = t;
  }
  if (buf.meta.batch) shape = [1, ...shape];
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${shape.join(', ')}), }`;
  header += ' '.repeat((64 - ((11 + header.length) % 64)) % 64) + '\n';
  const head = new Uint8Array(10 + header.length);
  head.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, header.length & 255, header.length >> 8]);
  for (let i = 0; i < header.length; i++) head[10 + i] = header.charCodeAt(i);
  const body = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const out = new Uint8Array(head.length + body.length);
  out.set(head);
  out.set(body, head.length);
  return { ext: 'npy', data: out };
}

/** "Run on batch": process a folder / several files / a zip, show a gallery and download the outputs. */
export function BatchDialog({ onClose }: { onClose(): void }) {
  const pipeline = useActivePipeline();
  const train = useStudio((s) => s.train);
  const display = useStudio((s) => s.display);
  const missing = useRuns((s) => s.missing);
  const needsServer = pipeline.steps.some((s) => s.enabled && missing[s.def]);
  const [engine, setEngine] = useState<'browser' | 'server'>(needsServer ? 'server' : 'browser');
  const [items, setItems] = useState<Item[]>([]);
  const [done, setDone] = useState(0);
  const [running, setRunning] = useState(false);
  const cancel = useRef<AbortController | undefined>(undefined);

  const collect = async (kind: 'files' | 'folder' | 'zip'): Promise<File[]> => {
    if (kind === 'zip') {
      const [z] = await pickFiles('.zip', false);
      if (!z) return [];
      const zip = await JSZip.loadAsync(z);
      const out: File[] = [];
      for (const e of Object.values(zip.files)) if (!e.dir && isImageName(e.name)) out.push(new File([await e.async('blob')], e.name.split('/').pop()!));
      return out;
    }
    return (await pickFiles(IMAGE_ACCEPT, true, kind === 'folder')).filter((f) => isImageName(f.name));
  };

  const runOne = (s: Sample, signal: AbortSignal): Promise<Item> => {
    if (engine === 'server') {
      const input = sampleInput(s, pipeline.read);
      return runOnServer(pipeline, input, sampleTargets(s), { start: 0, train, signal }).then(async ({ steps }) => {
        const last = steps[steps.length - 1];
        const meta = analyzePipeline(pipeline, input.meta, { train }).final;
        const raw = last ? fromWire(last.image, meta) : input;
        return { name: s.name, raw, display: await requestDisplay(raw, display) };
      });
    }
    return new Promise((resolve) => {
      const stop = runVariants(
        s,
        pipeline.read,
        [{ steps: runSteps(pipeline), seed: pipeline.seed }],
        train,
        display,
        (m) => {
          stop();
          invalidateSample(s.id);
          resolve({ name: s.name, display: m.display, raw: m.raw, error: m.error });
        },
        { fullResolution: true, returnRaw: true },
      );
      signal.addEventListener('abort', () => resolve({ name: s.name, error: 'cancelled' }));
    });
  };

  const start = async (kind: 'files' | 'folder' | 'zip') => {
    const files = await collect(kind);
    if (!files.length) return;
    const ac = new AbortController();
    cancel.current = ac;
    setItems(files.map((f) => ({ name: f.name })));
    setDone(0);
    setRunning(true);
    for (let i = 0; i < files.length; i++) {
      if (ac.signal.aborted) break;
      let item: Item;
      try {
        const s = await loadSampleFile(files[i], { exif: pipeline.read.exif });
        item = await runOne(s, ac.signal);
      } catch (e) {
        item = { name: files[i].name, error: (e as Error).message };
      }
      setItems((prev) => prev.map((x, k) => (k === i ? item : x)));
      setDone(i + 1);
    }
    setRunning(false);
  };

  const downloadAll = async () => {
    const zip = new JSZip();
    for (const it of items) {
      if (!it.raw || !it.display) continue;
      const { ext, data } = await encode(it.raw, it.display);
      zip.file(`${it.name.replace(/\.[^.]+$/, '')}.${ext}`, data);
    }
    download(await zip.generateAsync({ type: 'blob' }), `${ident(pipeline.name)}_batch.zip`);
  };

  const btn = 'rounded-md border border-line px-2 py-1 text-xs hover:bg-panel-2 disabled:opacity-40';
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" role="dialog" aria-modal="true" aria-label="Run on batch">
      <div className="flex max-h-full w-full max-w-4xl flex-col rounded-xl border border-line bg-panel p-4 shadow-xl">
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <h3 className="font-semibold">Run on batch</h3>
          <span className="text-xs text-muted">
            “{pipeline.name}” · {train ? 'train' : 'inference'} mode
          </span>
          <button onClick={onClose} className={`${btn} ml-auto`} aria-label="Close">
            ✕
          </button>
        </div>
        <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
          <button className={btn} disabled={running} onClick={() => start('folder')}>
            Choose folder…
          </button>
          <button className={btn} disabled={running} onClick={() => start('files')}>
            Choose images…
          </button>
          <button className={btn} disabled={running} onClick={() => start('zip')}>
            Choose .zip…
          </button>
          <label className="ml-2 flex items-center gap-1">
            Engine
            <select value={engine} onChange={(e) => setEngine(e.target.value as 'browser' | 'server')} className="rounded border border-line bg-panel-2 px-1 py-0.5" disabled={running}>
              <option value="browser" disabled={needsServer}>
                Browser (OpenCV.js)
              </option>
              <option value="server">Python server (exact exported code)</option>
            </select>
          </label>
          {running && (
            <button className={`${btn} border-danger/50 text-danger`} onClick={() => cancel.current?.abort()}>
              Cancel
            </button>
          )}
          <button className={`${btn} ml-auto bg-accent font-medium text-accent-fg`} disabled={running || !items.some((i) => i.raw)} onClick={downloadAll}>
            Download results (.zip)
          </button>
        </div>
        {items.length > 0 && (
          <div className="mb-2">
            <div className="h-1.5 overflow-hidden rounded bg-panel-2" role="progressbar" aria-valuemin={0} aria-valuemax={items.length} aria-valuenow={done}>
              <div className="h-full bg-accent transition-all" style={{ width: `${(100 * done) / items.length}%` }} />
            </div>
            <p className="mt-1 text-[11px] text-muted">
              {done}/{items.length} processed{items.some((i) => i.error) ? ` · ${items.filter((i) => i.error).length} failed` : ''}
            </p>
          </div>
        )}
        <div className="scroll-thin grid min-h-0 flex-1 grid-cols-2 gap-2 overflow-y-auto sm:grid-cols-4">
          {items.map((it, i) => (
            <figure key={i} className="overflow-hidden rounded-md border border-line">
              <div className="flex aspect-[4/3] items-center justify-center bg-panel-2">
                {it.display ? <Thumb display={it.display} label={it.name} /> : it.error ? <p className="p-2 text-[11px] text-danger">{it.error}</p> : <span className="text-xs text-muted">…</span>}
              </div>
              <figcaption className="truncate px-2 py-1 text-[11px]">{it.name}</figcaption>
            </figure>
          ))}
          {!items.length && <p className="col-span-full p-6 text-center text-sm text-muted">Pick a folder, several images or a .zip. Outputs are saved as PNG (8-bit images) or .npy (float / CHW tensors).</p>}
        </div>
      </div>
    </div>
  );
}
