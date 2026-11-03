// Meta propagation and validation: works out shape / dtype / colour space after every step without running
// any pixels, and reports problems with one-click fixes (§6).
import { channelsOf, formatMeta, isFloat } from './meta';
import { getStep } from './registry';
import type { ImageMeta, Issue, Pipeline, PipelineStep, StepDef } from './types';

/** What the linked Modelith image input expects. */
export interface ModelInput {
  c: number;
  h: number;
  w: number;
  framework: 'pytorch' | 'tensorflow';
  /** The model uses a pretrained backbone (suggest ImageNet mean/std). */
  pretrained: boolean;
  /** Input normalisation chosen on the model's Image block. */
  norm?: string;
  label?: string;
}

export interface StepInfo {
  uid: string;
  index: number;
  def?: StepDef;
  in: ImageMeta;
  out: ImageMeta;
  /** Runs in the current mode (enabled and train/inference matches). */
  active: boolean;
  issues: Issue[];
}

export interface PipelineAnalysis {
  steps: StepInfo[];
  byUid: Record<string, StepInfo>;
  input: ImageMeta;
  final: ImageMeta;
  /** Pipeline-level and step issues, errors first. */
  issues: Issue[];
  errors: number;
}

export function stepActive(s: PipelineStep, train: boolean): boolean {
  if (!s.enabled) return false;
  return s.mode === 'both' || (s.mode === 'train' ? train : !train);
}

const DTYPE_WORD: Record<string, string> = { uint8: 'uint8', uint16: 'uint16', float32: 'float32', float16: 'float16' };

function channelWord(c: number): string {
  return c === 1 ? 'a single-channel' : c === 3 ? 'a 3-channel' : c === 4 ? 'a 4-channel' : `a ${c}-channel`;
}

/** Generic `accepts` check with fixes. */
function acceptIssues(def: StepDef, m: ImageMeta, s: PipelineStep, steps: StepInfo[]): Issue[] {
  const a = def.accepts;
  const out: Issue[] = [];
  const layouts = a?.layout ?? ['HWC'];
  if (!layouts.includes(m.layout)) {
    const conv = [...steps].reverse().find((x) => x.active && x.def?.category === 'Normalise & tensor' && x.out.layout === 'CHW');
    out.push({
      level: 'error',
      uid: s.uid,
      message: `${def.label} needs an HWC image, but the image is already ${m.layout}${conv ? ` after '${conv.def!.label}'` : ''}. Tensor conversion belongs at the end of the pipeline.`,
      fixes: conv ? [{ label: `Move '${conv.def!.label}' to the end`, kind: 'moveToEnd' }] : [],
    });
    return out;
  }
  if (a?.c && !a.c.includes(m.c)) {
    const want = a.c.map((c) => channelWord(c).replace(/^an? /, '')).join(' or ');
    const fixes: Issue['fixes'] = [];
    if (a.c.includes(1)) fixes.push({ label: "Add 'Convert colour → GRAY' before it", kind: 'insertBefore', def: 'convert_color', params: { to: 'GRAY' } });
    if (a.c.includes(3) && m.c === 1) fixes.push({ label: "Add 'Gray → 3 channels' before it", kind: 'insertBefore', def: 'convert_color', params: { to: 'BGR' } });
    if (a.c.includes(3) && m.c === 4) fixes.push({ label: "Add 'Drop alpha' before it", kind: 'insertBefore', def: 'drop_alpha' });
    out.push({ level: 'error', uid: s.uid, message: `${def.label} needs ${want} image; this image has ${m.c} channel${m.c === 1 ? '' : 's'}.`, fixes });
  }
  if (a?.dtype && !a.dtype.includes(m.dtype)) {
    const want = a.dtype.map((d) => DTYPE_WORD[d]).join(' or ');
    // Find the step that made it float, to suggest moving it to the end.
    const culprit = [...steps].reverse().find((x) => x.active && x.in.dtype !== x.out.dtype && x.out.dtype === m.dtype);
    const fixes: Issue['fixes'] = [];
    if (culprit && culprit.def?.category === 'Normalise & tensor')
      fixes.push({ label: `Move '${culprit.def.label}' to the end`, kind: 'moveToEnd' });
    if (a.dtype.includes('uint8'))
      fixes.push({ label: "Add 'Change dtype → uint8' before it", kind: 'insertBefore', def: 'convert_dtype', params: { to: 'uint8', scale: isFloat(m) ? 'minmax' : 'clip' } });
    out.push({
      level: 'error',
      uid: s.uid,
      message: `${def.label} needs ${want}; this image is ${m.dtype}${culprit ? ` after '${culprit.def?.label}'` : ''}.${culprit?.def?.category === 'Normalise & tensor' ? ` Move ${culprit.def.label} to the end.` : ''}`,
      fixes,
    });
  }
  if (a?.space && !a.space.includes(m.space)) {
    out.push({
      level: 'error',
      uid: s.uid,
      message: `${def.label} expects ${a.space.join(' or ')} input; the image is ${m.space}.`,
      fixes: [{ label: `Add 'Convert colour → ${a.space[0]}' before it`, kind: 'insertBefore', def: 'convert_color', params: { to: a.space[0] } }],
    });
  }
  return out;
}

const TENSOR_IDS = new Set(['scale_01', 'standardize', 'to_chw', 'add_batch', 'cast_float']);
const HEAVY_IDS = new Set(['bilateral', 'nlm_denoise', 'median_blur', 'guided_filter', 'edge_preserving', 'tv_denoise', 'wavelet_denoise', 'detail_enhance', 'retinex', 'kmeans_quantize', 'grabcut']);

export function analyzePipeline(pipeline: Pipeline, input: ImageMeta, opts: { train: boolean; model?: ModelInput } = { train: false }): PipelineAnalysis {
  const infos: StepInfo[] = [];
  let m = input;
  pipeline.steps.forEach((s, index) => {
    const def = getStep(s.def);
    const active = !!def && stepActive(s, opts.train);
    const info: StepInfo = { uid: s.uid, index, def, in: m, out: m, active, issues: [] };
    if (!def) {
      info.issues.push({ level: 'error', uid: s.uid, message: `Unknown step '${s.def}' (from a newer version?). It is skipped.`, fixes: [{ label: 'Remove it', kind: 'remove' }] });
    } else if (active) {
      const acc = acceptIssues(def, m, s, infos);
      info.issues.push(...acc);
      if (!acc.some((i) => i.level === 'error')) {
        for (const i of def.check?.(m, s.params) ?? []) info.issues.push({ ...i, uid: s.uid });
        try {
          info.out = def.outMeta(m, s.params);
        } catch (e) {
          info.issues.push({ level: 'error', uid: s.uid, message: (e as Error).message });
        }
      }
      if (def.kind === 'random' && (s.p ?? 1) <= 0) info.issues.push({ level: 'info', uid: s.uid, message: 'Probability is 0, so this augmentation never applies.' });
      if (def.kind === 'random' && s.mode !== 'train')
        info.issues.push({
          level: 'warning',
          uid: s.uid,
          message: 'Random augmentation runs at inference too; inference pipelines should be deterministic.',
          fixes: [{ label: 'Make it train-only', kind: 'setMode', mode: 'train' }],
        });
      if (def.visualOnly && !s.exportVisual) info.issues.push({ level: 'info', uid: s.uid, message: 'Visualisation only: left out of the exported code.' });
    }
    if (info.out.c !== channelsOf(info.out.space) && info.out.layout === 'HWC' && !['GRAY'].includes(info.out.space) && info.out.c <= 4 && def?.id !== 'select_channel') {
      // Bookkeeping guard: steps must keep c and space consistent.
      info.issues.push({ level: 'warning', uid: s.uid, message: `Internal: ${info.out.c} channels tagged as ${info.out.space}.` });
    }
    infos.push(info);
    m = info.out;
  });

  orderAdvice(infos);
  const issues: Issue[] = infos.flatMap((i) => i.issues);
  if (opts.model) issues.push(...modelIssues(pipeline, infos, m, opts.model));
  issues.sort((a, b) => rank(a) - rank(b));
  return {
    steps: infos,
    byUid: Object.fromEntries(infos.map((i) => [i.uid, i])),
    input,
    final: m,
    issues,
    errors: issues.filter((i) => i.level === 'error').length,
  };
}

const rank = (i: Issue) => (i.level === 'error' ? 0 : i.level === 'warning' ? 1 : 2);

/** Order warnings (not errors): tensor steps not at the end, double resizes, undone conversions, slow orderings. */
function orderAdvice(infos: StepInfo[]): void {
  const act = infos.filter((i) => i.active && i.def);
  act.forEach((info, k) => {
    const def = info.def!;
    const next = act[k + 1];
    if (TENSOR_IDS.has(def.id)) {
      const later = act.slice(k + 1).find((x) => !TENSOR_IDS.has(x.def!.id) && !x.def!.visualOnly);
      if (later && !info.issues.length && !later.issues.some((i) => i.level === 'error'))
        info.issues.push({
          level: 'warning',
          uid: info.uid,
          message: `${def.label} runs before '${later.def!.label}'. Normalisation and tensor conversion usually go at the end of the pipeline.`,
          fixes: [{ label: 'Move to the end', kind: 'moveToEnd' }],
        });
    }
    if (next && (def.id === 'resize' || def.id === 'letterbox') && (next.def!.id === 'resize' || next.def!.id === 'letterbox'))
      info.issues.push({
        level: 'warning',
        uid: info.uid,
        message: `Followed directly by '${next.def!.label}'. Two resizes in a row cost time and detail; the second alone gives the same size.`,
        fixes: [{ label: 'Remove this step', kind: 'remove' }],
      });
    if (next && def.id === 'convert_color' && next.def!.id === 'convert_color' && next.out.space === info.in.space && next.out.dtype === info.in.dtype)
      next.issues.push({ level: 'warning', uid: next.uid, message: `Converts back to ${next.out.space} right after converting to ${info.out.space}; both steps can be removed.`, fixes: [{ label: 'Remove this step', kind: 'remove' }] });
    if (HEAVY_IDS.has(def.id) && Math.max(info.in.h, info.in.w) > 1024) {
      const shrink = act.slice(k + 1).find((x) => (x.def!.id === 'resize' || x.def!.id === 'letterbox') && x.out.h * x.out.w < info.in.h * info.in.w);
      if (shrink)
        info.issues.push({
          level: 'warning',
          uid: info.uid,
          message: `${def.label} runs on a ${info.in.w}×${info.in.h} image before '${shrink.def!.label}' shrinks it to ${shrink.out.w}×${shrink.out.h}. Resizing first is ~${Math.round((info.in.h * info.in.w) / (shrink.out.h * shrink.out.w))}× faster.`,
        });
    }
  });
}

function modelIssues(pipeline: Pipeline, infos: StepInfo[], final: ImageMeta, model: ModelInput): Issue[] {
  const out: Issue[] = [];
  const name = model.label ?? 'the linked model';
  if (final.c !== model.c || final.h !== model.h || final.w !== model.w) {
    out.push({
      level: 'error',
      message: `Output is ${formatMeta(final)} but ${name} expects ${model.c}×${model.h}×${model.w}.`,
      fixes: [
        ...(final.h !== model.h || final.w !== model.w
          ? [
              { label: `Add Resize to ${model.w}×${model.h}`, kind: 'append' as const, def: 'resize', params: { mode: 'exact', size: [model.w, model.h] } },
              { label: `Add Letterbox to ${model.w}×${model.h}`, kind: 'append' as const, def: 'letterbox', params: { size: [model.w, model.h] } },
            ]
          : []),
        ...(final.c !== model.c && model.c === 1 ? [{ label: 'Add Convert colour → GRAY', kind: 'append' as const, def: 'convert_color', params: { to: 'GRAY' } }] : []),
        { label: `Update model input to ${final.c}×${final.h}×${final.w}`, kind: 'updateModelInput', c: final.c, h: final.h, w: final.w },
      ],
    });
  }
  if (!isFloat(final))
    out.push({
      level: 'warning',
      message: `The model receives ${final.dtype}; networks expect float32. Add 'Standardise' or 'Scale to [0,1]' at the end.`,
      fixes: [{ label: 'Add Standardise (ImageNet)', kind: 'append', def: 'standardize', params: { preset: 'imagenet' } }],
    });
  const std = infos.find((i) => i.active && i.def?.id === 'standardize');
  if (model.pretrained && (!std || pipeline.steps.find((s) => s.uid === std.uid)?.params.preset !== 'imagenet'))
    out.push({
      level: 'info',
      uid: std?.uid,
      message: `${name} uses a pretrained backbone; it was trained on ImageNet-normalised RGB input (mean 0.485, 0.456, 0.406; std 0.229, 0.224, 0.225).`,
      fixes: std
        ? [{ label: 'Use ImageNet mean/std', kind: 'setParams', params: { preset: 'imagenet' } }]
        : [{ label: 'Add Standardise (ImageNet)', kind: 'append', def: 'standardize', params: { preset: 'imagenet' } }],
    });
  if (model.pretrained && final.space === 'BGR')
    out.push({ level: 'warning', message: 'Pretrained backbones expect RGB, but the output is BGR.', fixes: [{ label: 'Add BGR → RGB', kind: 'append', def: 'convert_color', params: { to: 'RGB' } }] });
  return out;
}
