// Runs every step's parity cases (and every preset) through the browser engine (OpenCV.js in Node), writing
// inputs, outputs and the generated preprocess.py for tests/preprocess/test_parity.py to compare against.
// Usage: npx tsx tests/preprocess/gen-parity.ts [--only id1,id2]
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  allSteps,
  defaultParams,
  generatePipelineFiles,
  generatePreprocessPy,
  getStep,
  runSteps,
  missingInBrowser,
  Targets,
  type ImageBuf,
  type Pipeline,
  type PipelineStep,
  type TestCase,
} from '../../packages/preprocess/src/index';
import { presets } from '../../packages/preprocess/src/presets';
import { loadCv } from './cv-node';
import { BOXES, INPUTS, KEYPOINTS, maskImage } from './images';
import { writeNpy } from './npy';

const root = join(dirname(fileURLToPath(import.meta.url)), 'generated', 'parity');
const onlyArg = process.argv.indexOf('--only');
const only = onlyArg > 0 ? new Set(process.argv[onlyArg + 1].split(',')) : undefined;

const shapeOf = (b: ImageBuf) => (b.meta.c === 1 ? [b.meta.h, b.meta.w] : [b.meta.h, b.meta.w, b.meta.c]);

function saveBuf(path: string, b: ImageBuf) {
  writeNpy(path, b.data, shapeOf(b));
}

interface Entry {
  name: string;
  step?: string;
  preset?: string;
  train: boolean;
  seed: number;
  tolerance: number;
  targets: boolean;
  serverOnly: boolean;
  error?: string;
  layout: string;
  batch: boolean;
  dtype: string;
  /** The tested step moves pixels (for target round-trip tests). */
  geometry?: 'axis' | 'warp' | 'remap' | 'none';
  random?: boolean;
  readFlag: string;
}

/** Geometric steps that keep boxes axis-aligned (box round trips must be exact). */
const AXIS = new Set(['resize', 'letterbox', 'pad', 'crop', 'flip', 'rotate90', 'pyramid', 'random_flip', 'random_rotate90', 'random_resized_crop', 'dct_view']);

const cv = await loadCv();
rmSync(root, { recursive: true, force: true });
mkdirSync(root, { recursive: true });
const manifest: Entry[] = [];
let uid = 0;

function run(name: string, pipeline: Pipeline, input: ImageBuf, opts: { train: boolean; tolerance: number; targets: boolean; step?: string; preset?: string; geometry?: Entry['geometry']; random?: boolean }) {
  const dir = join(root, name);
  mkdirSync(dir, { recursive: true });
  saveBuf(join(dir, 'input.npy'), input);
  const { file } = generatePreprocessPy(pipeline, { input: input.meta, includeVisual: true });
  writeFileSync(join(dir, 'preprocess.py'), file.content);
  writeFileSync(join(dir, 'pipeline.json'), JSON.stringify(pipeline, null, 2));
  if (opts.preset) for (const f of generatePipelineFiles(pipeline, { input: input.meta })) if (f.path !== 'preprocess.py') writeFileSync(join(dir, f.path), f.content);
  const mask = opts.targets ? maskImage() : undefined;
  if (mask) saveBuf(join(dir, 'mask.npy'), mask);
  const t0 = opts.targets ? new Targets(cv, { mask, boxes: BOXES, keypoints: KEYPOINTS, history: [], size: [input.meta.w, input.meta.h] }).data() : Targets.empty(cv, input.meta.w, input.meta.h).data();
  const res = runSteps(cv, input, t0, pipeline.steps, { train: opts.train, seed: pipeline.seed, rootKey: name });
  const entry: Entry = {
    name,
    step: opts.step,
    preset: opts.preset,
    train: opts.train,
    seed: pipeline.seed,
    tolerance: opts.tolerance,
    targets: opts.targets,
    serverOnly: res.serverFrom !== undefined,
    layout: 'HWC',
    batch: false,
    dtype: '',
    geometry: opts.geometry,
    random: opts.random,
    readFlag: pipeline.read.flag,
  };
  if (res.error) entry.error = `step ${res.errorAt}: ${res.error}`;
  else if (res.serverFrom === undefined) {
    const last = res.outputs[res.outputs.length - 1];
    saveBuf(join(dir, 'browser.npy'), last.buf);
    entry.layout = last.buf.meta.layout;
    entry.batch = last.buf.meta.batch;
    entry.dtype = last.buf.meta.dtype;
    if (opts.targets) {
      if (last.targets.mask) saveBuf(join(dir, 'browser_mask.npy'), last.targets.mask);
      writeFileSync(join(dir, 'browser_targets.json'), JSON.stringify({ boxes: last.targets.boxes, keypoints: last.targets.keypoints }));
    }
  }
  manifest.push(entry);
}

for (const def of allSteps()) {
  if (only && !only.has(def.id)) continue;
  const cases: TestCase[] = def.testCases ?? [{}];
  cases.forEach((tc, i) => {
    const inputName = tc.input ?? (def.accepts?.c && !def.accepts.c.includes(3) ? 'gray' : 'color');
    const input = INPUTS[inputName]();
    const mk = (d: string, params?: Record<string, unknown>, extra: Partial<PipelineStep> = {}): PipelineStep => ({
      uid: `s${uid++}`,
      def: d,
      params: { ...defaultParams(d), ...(params ?? {}) },
      enabled: true,
      mode: 'both',
      ...extra,
    });
    const steps = [...(tc.before ?? []).map((b) => mk(b.def, b.params)), mk(def.id, tc.params, { p: tc.p ?? (def.kind === 'random' ? 1 : undefined) })];
    const pipeline: Pipeline = {
      version: 1,
      id: 'test',
      name: `${def.id} case ${i}`,
      seed: tc.seed ?? 7,
      read: { flag: inputName === 'u16' || inputName === 'rgba' || inputName === 'float' ? 'UNCHANGED' : inputName === 'color' ? 'COLOR' : 'GRAYSCALE', exif: true },
      steps,
    };
    const train = tc.train ?? def.kind === 'random';
    const geometry = !def.affectsGeometry ? 'none' : /remap|undistort|elastic|grid_distortion|optical/.test(def.id + def.cv2) ? 'remap' : AXIS.has(def.id) ? 'axis' : 'warp';
    run(`${def.id}__${i}`, pipeline, input, { train, tolerance: tc.tolerance ?? def.tolerance ?? 0, targets: tc.targets ?? def.affectsGeometry, step: def.id, geometry, random: def.kind === 'random' });
  });
}

if (!only) {
  for (const preset of presets) {
    for (const train of [false, true]) {
      const pipeline = preset.build();
      const input = INPUTS[preset.testInput ?? 'color']();
      const tol = Math.max(0, ...pipeline.steps.map((s) => getStep(s.def)?.tolerance ?? 0));
      run(`preset_${preset.id}__${train ? 'train' : 'inference'}`, pipeline, input, { train, tolerance: Math.max(tol, preset.tolerance ?? 0), targets: true, preset: preset.id });
    }
  }
}

writeFileSync(join(root, 'manifest.json'), JSON.stringify(manifest, null, 2));
const bad = manifest.filter((m) => m.error);
const server = manifest.filter((m) => m.serverOnly);
console.log(`${manifest.length} cases (${server.length} server-only) → ${root}`);
for (const b of bad) console.log(`  BROWSER ERROR ${b.name}: ${b.error}`);
void missingInBrowser;
void getStep;
