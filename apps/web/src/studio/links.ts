// Lightweight view of the Preprocessing Studio pipelines for the model builder (main bundle): which pipeline
// feeds which Image input. The studio store (lazy-loaded) pushes its state here and owns the edits while loaded;
// otherwise edits go straight to local storage.
import { getBlock, buildIR, type Project } from '@modelith/blocks';
import { analyzePipeline, defaultInputMeta, generatePipelineFiles, parsePipeline, trainHook, type Pipeline } from '@modelith/preprocess';
import { create } from 'zustand';

export const PIPELINES_KEY = 'modelith:pipelines';

interface Links {
  pipelines: Pipeline[];
}

function load(): Pipeline[] {
  try {
    const raw = localStorage.getItem(PIPELINES_KEY);
    return raw ? (JSON.parse(raw).pipelines as unknown[]).map(parsePipeline) : [];
  } catch {
    return [];
  }
}

export const useLinks = create<Links>(() => ({ pipelines: load() }));

let studioLink: ((nodeId: string | undefined, pipelineId: string) => void) | undefined;

/** Called by the studio store when it loads, so edits go through its undo history. */
export function registerStudio(link: typeof studioLink, pipelines: Pipeline[]): void {
  studioLink = link;
  useLinks.setState({ pipelines });
}

export function publishPipelines(pipelines: Pipeline[]): void {
  useLinks.setState({ pipelines });
}

/** Link pipeline `pipelineId` to Image input `nodeId` (or unlink when nodeId is undefined). */
export function linkPipeline(pipelineId: string, nodeId: string | undefined): void {
  if (studioLink) {
    studioLink(nodeId, pipelineId);
    return;
  }
  const pipelines = useLinks.getState().pipelines.map((p) => {
    if (p.id === pipelineId) return nodeId ? { ...p, linkedModelId: nodeId } : { ...p, linkedModelId: undefined };
    if (nodeId && p.linkedModelId === nodeId) return { ...p, linkedModelId: undefined };
    return p;
  });
  useLinks.setState({ pipelines });
  try {
    const raw = localStorage.getItem(PIPELINES_KEY);
    const activeId = raw ? JSON.parse(raw).activeId : pipelines[0]?.id;
    localStorage.setItem(PIPELINES_KEY, JSON.stringify({ pipelines, activeId }));
  } catch {
    /* storage unavailable */
  }
}

export function linkedPipeline(pipelines: Pipeline[], nodeId: string): Pipeline | undefined {
  return pipelines.find((p) => p.linkedModelId === nodeId);
}

/** Output of a pipeline at inference (for the badge), e.g. `3×224×224`. */
export function pipelineOutput(p: Pipeline): string {
  const m = analyzePipeline(p, defaultInputMeta(p), { train: false }).final;
  const dims = m.layout === 'CHW' ? [m.c, m.h, m.w] : [m.h, m.w, m.c];
  return `${dims.join('×')} ${m.dtype}`;
}

/** Preprocessing files + train.py hook for the first linked Image input of a project (PyTorch). */
export function projectPreprocessing(project: Project, pipelines: Pipeline[]) {
  const node = project.nodes.find((n) => n.type === 'image_input' && linkedPipeline(pipelines, n.id));
  if (!node) return undefined;
  const p = linkedPipeline(pipelines, node.id)!;
  let heads: { name: string; task: string }[] = [];
  try {
    heads = buildIR(project).heads.map((h) => ({ name: h.name, task: h.head!.task }));
  } catch {
    /* invalid graph: generateProject reports the errors */
  }
  const pretrained = project.nodes.some((n) => getBlock(n.type)?.pretrained?.(n.params));
  const files = generatePipelineFiles(p, {
    framework: project.framework,
    model: { c: node.params.channels, h: node.params.height, w: node.params.width, framework: project.framework, pretrained },
  });
  return { files, train: { ...trainHook(p, { input: node.params.name, heads }), pipelineName: p.name } };
}

/** Merge pipelines from an imported .modelith.json (same id → replaced). */
export function importPipelines(raw: unknown[]): void {
  const incoming = raw.map(parsePipeline);
  const pipelines = [...useLinks.getState().pipelines.filter((p) => !incoming.some((x) => x.id === p.id)), ...incoming];
  useLinks.setState({ pipelines });
  try {
    localStorage.setItem(PIPELINES_KEY, JSON.stringify({ pipelines, activeId: incoming[0].id }));
  } catch {
    /* storage unavailable */
  }
  importHook?.(incoming);
}

let importHook: ((p: Pipeline[]) => void) | undefined;
/** The studio store (when loaded) adds imported pipelines to its own state. */
export function onImportPipelines(fn: (p: Pipeline[]) => void): void {
  importHook = fn;
}
