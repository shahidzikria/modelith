// Pipeline → preprocess.py (pure cv2 + NumPy). Deterministic: the same pipeline always gives the same file.
import { analyzePipeline, type PipelineAnalysis } from '../analyze';
import { describeMeta, imageMeta } from '../meta';
import { ident, py, pyFloat } from '../py';
export { pyRaw } from '../py';
import { getStep } from '../registry';
import type { ImageMeta, Pipeline, PipelineStep, PyCtx, ReadFlag, StepDef } from '../types';
import { pyHelperCode, resolvePyHelpers, TARGETS_PY } from './helpers';
import { Source } from './source';

export interface ExportStep {
  step: PipelineStep;
  def: StepDef;
  /** 1-based position among exported steps. */
  index: number;
  fn: string;
  name: string;
  in: ImageMeta;
  out: ImageMeta;
}

export interface ExportOptions {
  /** Include visualisation-only steps even when not ticked (server preview). */
  includeVisual?: boolean;
  /** Input meta used for shape comments (defaults to a 480×640 image as read). */
  input?: ImageMeta;
}



type ConstValue = number | string | boolean | unknown[] | { __raw: string } | null;

function renderConst(v: ConstValue): string {
  if (v && typeof v === 'object' && '__raw' in (v as object)) return (v as { __raw: string }).__raw;
  return py(v);
}

export const READ_FLAGS: Record<ReadFlag, { py: string; doc: string }> = {
  COLOR: { py: 'cv2.IMREAD_COLOR', doc: 'BGR uint8 (H, W, 3)' },
  GRAYSCALE: { py: 'cv2.IMREAD_GRAYSCALE', doc: 'GRAY uint8 (H, W)' },
  UNCHANGED: { py: 'cv2.IMREAD_UNCHANGED', doc: 'as stored: alpha and 16-bit kept' },
  ANYDEPTH: { py: 'cv2.IMREAD_ANYDEPTH', doc: 'GRAY at the stored bit depth (8 or 16-bit)' },
};

export function defaultInputMeta(p: Pipeline): ImageMeta {
  if (p.read.flag === 'GRAYSCALE') return imageMeta(480, 640, 1);
  if (p.read.flag === 'ANYDEPTH') return imageMeta(480, 640, 1, 'uint16');
  return imageMeta(480, 640, 3);
}

/** Exported steps with their function names and the meta flowing in and out (train meta for train steps). */
export function exportSteps(p: Pipeline, opts: ExportOptions = {}): ExportStep[] {
  const input = opts.input ?? defaultInputMeta(p);
  const inf = analyzePipeline(p, input, { train: false });
  const tr = analyzePipeline(p, input, { train: true });
  const chosen = p.steps.filter((s) => {
    const def = getStep(s.def);
    return s.enabled && def && !def.batch && (!def.visualOnly || s.exportVisual || opts.includeVisual);
  });
  return chosen.map((s, i) => {
    const def = getStep(s.def)!;
    const a: PipelineAnalysis = s.mode === 'inference' ? inf : tr;
    const info = a.byUid[s.uid];
    const name = `${String(i + 1).padStart(2, '0')}_${ident(def.id)}`;
    return { step: s, def, index: i + 1, fn: `step_${name}`, name, in: info.in, out: info.out };
  });
}

interface EmittedStep {
  x: ExportStep;
  consts: { name: string; value: string; comment?: string }[];
  body: string;
  doc?: string;
}

function emitSteps(steps: ExportStep[], helpers: Set<string>): EmittedStep[] {
  const counts = new Map<string, number>();
  for (const x of steps) counts.set(x.def.id, (counts.get(x.def.id) ?? 0) + 1);
  return steps.map((x) => {
    const prefix = (x.def.constPrefix ?? ident(x.def.id)).toUpperCase() + ((counts.get(x.def.id) ?? 0) > 1 ? `_${x.index}` : '');
    const consts: EmittedStep['consts'] = [];
    const ctx: PyCtx = {
      in: x.in,
      out: x.out,
      index: x.index,
      c: (name, value, comment) => {
        const full = `${prefix}_${name.toUpperCase()}`;
        if (!consts.some((c) => c.name === full)) consts.push({ name: full, value: renderConst(value as ConstValue), comment });
        return full;
      },
      helper: (n) => helpers.add(n),
    };
    const em = x.def.emitPython(x.step.params, ctx);
    let body = em.body.replace(/\s+$/, '');
    if (x.def.kind === 'random' && (x.step.p ?? 1) < 1) {
      const p = ctx.c('p', x.step.p, 'probability of applying');
      body = `if rng.random() >= ${p}:\n    return img\n${body}`;
    }
    if (x.in.space !== x.out.space || x.in.dtype !== x.out.dtype || x.in.layout !== x.out.layout || x.in.c !== x.out.c) body += `\n# img is now ${describeMeta(x.out)}`;
    return { x, consts, body, doc: em.doc };
  });
}

function signature(x: ExportStep): string {
  const args = ['img'];
  if (x.def.affectsGeometry || x.def.usesTargets) args.push('t');
  if (x.def.kind === 'random') args.push('rng');
  return args.join(', ');
}

function firstSentence(s: string): string {
  const m = /^(.*?[.!?])(\s|$)/.exec(s);
  return (m ? m[1] : s).replace(/"/g, "'");
}

/** Greedy word wrap (docstrings stay under ~100 columns once indented). */
export function wrapText(s: string, width = 92): string[] {
  const out: string[] = [];
  let line = '';
  for (const word of s.split(/\s+/)) {
    if (line && line.length + 1 + word.length > width) {
      out.push(line);
      line = word;
    } else line = line ? `${line} ${word}` : word;
  }
  if (line) out.push(line);
  return out;
}

export function generatePreprocessPy(p: Pipeline, opts: ExportOptions = {}): { file: ReturnType<Source['file']>; steps: ExportStep[] } {
  const steps = exportSteps(p, opts);
  const helpers = new Set<string>();
  const emitted = emitSteps(steps, helpers);
  const input = opts.input ?? defaultInputMeta(p);
  const last = steps.length ? steps[steps.length - 1].out : input;
  const finalInf = analyzePipeline(p, input, { train: false }).final;
  const read = READ_FLAGS[p.read.flag];
  const s = new Source();

  // ---- module docstring
  const width = Math.max(10, ...steps.map((x) => x.def.label.length + (x.step.mode === 'both' ? 0 : x.step.mode.length + 3)));
  s.add(`"""Image preprocessing pipeline "${p.name.replace(/"/g, "'")}" (generated by Modelith Preprocessing Studio).`);
  s.blank();
  s.add(`Steps, with shapes for a ${input.w}×${input.h} input:`);
  s.blank();
  s.add(`    input  ${read.doc} from load_image()`);
  for (const x of steps) {
    const tag = x.step.mode === 'both' ? '' : ` [${x.step.mode}]`;
    s.add(`    ${String(x.index).padStart(2)}. ${(x.def.label + tag).padEnd(width)}  → ${describeMeta(x.out)}`, 0, x.step.uid);
  }
  s.blank();
  s.add(`Output (inference): ${describeMeta(finalInf)}`);
  s.blank();
  s.add('Usage:');
  s.add('    img = load_image("photo.jpg")');
  s.add('    x, t = preprocess(img)                                   # inference (deterministic)');
  s.add('    x, t = preprocess(img, mask=m, boxes=b, train=True)      # training (augmentations on)');
  s.add('    boxes = postprocess_boxes(pred_boxes, t.history)         # back to original pixels');
  s.blank();
  s.add('    python preprocess.py input_dir output_dir [--train] [--seed N] [--save-steps] [--npy]');
  s.add('"""');
  s.blank();
  s.add('from __future__ import annotations');
  s.blank();
  s.add('import argparse\nfrom pathlib import Path');
  s.blank();
  s.add('import cv2\nimport numpy as np');
  s.blank(2);

  // ---- constants
  s.add(`SEED = ${p.seed}  # np.random.Generator(np.random.PCG64(SEED)) drives every random step`);
  s.add(`READ_FLAG = ${read.py}${p.read.exif || p.read.flag === 'UNCHANGED' || p.read.flag === 'ANYDEPTH' ? '' : ' | cv2.IMREAD_IGNORE_ORIENTATION'}`);
  for (const e of emitted) {
    if (!e.consts.length) continue;
    s.blank();
    s.add(`# ${e.x.index}. ${e.x.def.label}`, 0, e.x.step.uid);
    for (const c of e.consts) s.add(`${c.name} = ${c.value}${c.comment ? `  # ${c.comment}` : ''}`, 0, e.x.step.uid);
  }
  s.blank(2);

  // ---- helpers
  s.add(TARGETS_PY);
  for (const h of resolvePyHelpers(helpers)) {
    s.blank(2);
    s.add(pyHelperCode(h));
  }
  s.blank(2);

  s.add(`def load_image(path):
    """Read an image with cv2 (works with non-ASCII paths): ${read.doc}."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, READ_FLAG)
    if img is None:
        raise ValueError(f"cannot decode image: {path}")
    return img`);
  s.blank(2);

  // ---- steps
  for (const e of emitted) {
    const x = e.x;
    s.add(`def ${x.fn}(${signature(x)}):`, 0, x.step.uid);
    const summary = wrapText(firstSentence(x.def.doc));
    const doc = [`"""${summary[0]}`, ...summary.slice(1),...(e.doc ? ['', ...e.doc.split('\n')] : []), '', `in:  ${describeMeta(x.in)}`, `out: ${describeMeta(x.out)}`, `cv2: ${x.def.cv2}`, '"""'];
    s.add(doc.join('\n'), 1, x.step.uid);
    s.add(e.body, 1, x.step.uid);
    s.add('return img', 1, x.step.uid);
    s.blank(2);
  }

  // ---- runner
  s.add(`def run_steps(img, t, rng, train=False, start=0, on_step=None):
    """Apply steps[start:] in order. on_step(index, name, colour_space, img) is called after each step that runs."""
    steps = [`);
  s.add('# (name, applies in, colour space after, function)', 2);
  for (const x of steps) {
    const args = signature(x).split(', ').slice(1);
    const fn = args.length ? `lambda x: ${x.fn}(x, ${args.join(', ')})` : x.fn;
    s.add(`(${py(x.name)}, ${py(x.step.mode)}, ${py(x.out.space)}, ${fn}),`, 2, x.step.uid);
  }
  s.add(`    ]
    for i, (name, mode, space, fn) in enumerate(steps):
        if i < start or (mode == "train" and not train) or (mode == "inference" and train):
            continue
        img = fn(img)
        if on_step is not None:
            on_step(i, name, space, img)
    return img`);
  s.blank(2);

  s.add(`_DEFAULT_RNG = np.random.Generator(np.random.PCG64(SEED))


def preprocess(img, mask=None, boxes=None, keypoints=None, train=False, rng=None, on_step=None):
    """Run the pipeline on one image from load_image() (${read.doc}).

    Returns (image, targets). The image is ${describeMeta(finalInf)} at inference.
    targets.mask / .boxes / .keypoints follow every geometric step; pass targets.history to
    postprocess_boxes / postprocess_mask / postprocess_keypoints to map predictions back.
    train=True applies train-only steps (augmentations) and skips inference-only ones. Pass one rng
    for a whole run (default: a module-level Generator seeded with SEED).
    """
    rng = _DEFAULT_RNG if rng is None else rng
    t = Targets((img.shape[1], img.shape[0]), mask, boxes, keypoints)
    img = run_steps(img, t, rng, train=train, on_step=on_step)
    return img, t`);
  s.blank(2);

  // ---- CLI
  s.add(`IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def save_array(path, img, space, npy=False):
    """Save a step result: .png for 8/16-bit HWC images (converted to BGR for viewing), else / also .npy."""
    path = Path(path)
    viewable = img.ndim in (2, 3) and img.dtype in (np.uint8, np.uint16) and (img.ndim == 2 or img.shape[2] in (1, 3, 4))
    if viewable:
        out = img
        if space in ("RGB", "RGBA"):
            out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if img.shape[2] == 3 else cv2.COLOR_RGBA2BGRA)
        cv2.imencode(".png", out)[1].tofile(str(path.with_suffix(".png")))
    if npy or not viewable:
        np.save(path.with_suffix(".npy"), img)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preprocessing pipeline on a folder of images.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train", action="store_true", help="apply train-only steps (augmentations)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--save-steps", action="store_true", help="also save every intermediate step")
    parser.add_argument("--npy", action="store_true", help="always save exact .npy arrays too")
    args = parser.parse_args()
    rng = np.random.Generator(np.random.PCG64(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for path in files:
        img = load_image(path)
        on_step = None
        if args.save_steps:
            step_dir = args.output_dir / f"{path.stem}_steps"
            step_dir.mkdir(exist_ok=True)

            def on_step(i, name, space, x, step_dir=step_dir):
                save_array(step_dir / name, x, space, args.npy)

        out, _ = preprocess(img, train=args.train, rng=rng, on_step=on_step)
        save_array(args.output_dir / path.stem, out, ${py(last.space)}, args.npy)
    print(f"processed {len(files)} image(s) -> {args.output_dir}")


if __name__ == "__main__":
    main()`);
  return { file: s.file('preprocess.py'), steps };
}

export { pyFloat };
