// Ready-made pipelines (§9). Each loads as an ordinary, editable pipeline.
import { defaultParams, getStep } from './registry';
import type { Params, Pipeline, PipelineStep, ReadSpec, StepMode } from './types';

export interface Preset {
  id: string;
  label: string;
  description: string;
  build(): Pipeline;
  /** Test image used by the parity / export tests. */
  testInput?: 'color' | 'gray' | 'u16' | 'doc';
  /** Extra tolerance for the preset as a whole. */
  tolerance?: number;
}

let counter = 0;
export const newUid = (prefix = 's') => `${prefix}${Date.now().toString(36)}${(counter++).toString(36)}`;

type StepSpec = [def: string, params?: Params, extra?: { mode?: StepMode; p?: number; enabled?: boolean; note?: string }];

/** Build a pipeline step with defaults filled in (augmentations default to train-only, p = 0.5). */
export function makeStep(def: string, params: Params = {}, extra: StepSpec[2] = {}): PipelineStep {
  const d = getStep(def);
  if (!d) throw new Error(`Unknown step: ${def}`);
  const random = d.kind === 'random';
  return {
    uid: newUid(),
    def,
    params: { ...defaultParams(def), ...structuredClone(params) },
    enabled: extra.enabled ?? true,
    mode: extra.mode ?? d.defaultMode ?? (random ? 'train' : 'both'),
    ...(random ? { p: extra.p ?? 0.5 } : {}),
    ...(extra.note ? { note: extra.note } : {}),
  };
}

export function makePipeline(name: string, steps: StepSpec[], opts: { read?: Partial<ReadSpec>; seed?: number; id?: string } = {}): Pipeline {
  return {
    version: 1,
    id: opts.id ?? newUid('p'),
    name,
    seed: opts.seed ?? 0,
    read: { flag: 'COLOR', exif: true, ...opts.read },
    steps: steps.map(([d, p, e]) => makeStep(d, p, e)),
  };
}

const TRAIN = { mode: 'train' as const };
const INFER = { mode: 'inference' as const };

export const presets: Preset[] = [
  {
    id: 'imagenet_cls',
    label: 'ImageNet classification',
    description: 'Resize shorter side to 256 → centre crop 224 → RGB → ImageNet mean/std → CHW (the standard evaluation transform).',
    build: () =>
      makePipeline('ImageNet classification', [
        ['resize', { mode: 'shorter', side: 256, interp: 'LINEAR' }],
        ['crop', { mode: 'center', size: [224, 224] }],
        ['convert_color', { to: 'RGB' }],
        ['standardize', { preset: 'imagenet' }],
        ['to_chw'],
      ]),
  },
  {
    id: 'imagenet_train',
    label: 'ImageNet training augmentation',
    description: 'Train: random resized crop 224, horizontal flip, colour jitter. Inference: resize 256 + centre crop 224. Both: RGB, ImageNet normalisation, CHW.',
    build: () =>
      makePipeline('ImageNet training', [
        ['random_resized_crop', { size: [224, 224] }, { ...TRAIN, p: 1 }],
        ['resize', { mode: 'shorter', side: 256, interp: 'LINEAR' }, INFER],
        ['crop', { mode: 'center', size: [224, 224] }, INFER],
        ['random_flip', { dir: 'h' }, { ...TRAIN, p: 0.5 }],
        ['color_jitter', { brightness: 0.4, contrast: 0.4, saturation: 0.4, hue: 0.1 }, { ...TRAIN, p: 0.8 }],
        ['convert_color', { to: 'RGB' }],
        ['standardize', { preset: 'imagenet' }],
        ['to_chw'],
      ]),
  },
  {
    id: 'yolo_det',
    label: 'YOLO detection',
    description: 'Letterbox 640 (grey 114 padding, inverse transform for boxes), HSV / colour jitter, flip, random affine; Mosaic per batch in dataset.py. RGB, [0, 1], CHW.',
    build: () =>
      makePipeline('YOLO detection', [
        ['letterbox', { size: [640, 640], color: '#727272' }],
        ['color_jitter', { brightness: 0.4, contrast: 0, saturation: 0.7, hue: 0.015 }, { ...TRAIN, p: 1 }],
        ['random_flip', { dir: 'h' }, { ...TRAIN, p: 0.5 }],
        ['random_affine', { degrees: 0, scale: [0.5, 1.5], translate: 0.1, shear: 0, color: '#727272' }, { ...TRAIN, p: 1 }],
        ['mosaic', { size: [640, 640] }, { ...TRAIN, p: 1, note: 'Applied per batch (4 images) in dataset.py; turn off for the last ~10 epochs.' }],
        ['convert_color', { to: 'RGB' }],
        ['scale_01'],
        ['to_chw'],
      ]),
  },
  {
    id: 'semseg',
    label: 'Semantic segmentation',
    description: 'Resize 512×512 (masks with NEAREST), flip, scale-rotate, RGB, ImageNet normalisation, CHW.',
    build: () =>
      makePipeline('Semantic segmentation', [
        ['resize', { mode: 'exact', size: [512, 512], interp: 'LINEAR' }],
        ['random_flip', { dir: 'h' }, { ...TRAIN, p: 0.5 }],
        ['random_affine', { degrees: 15, scale: [0.75, 1.25], translate: 0.05, shear: 0, border: 'REFLECT_101' }, { ...TRAIN, p: 0.7 }],
        ['convert_color', { to: 'RGB' }],
        ['standardize', { preset: 'imagenet' }],
        ['to_chw'],
      ]),
  },
  {
    id: 'doc_ocr',
    label: 'Document OCR cleanup',
    description: 'Grayscale read → deskew → shadow removal → salt-and-pepper denoise → Sauvola binarisation.',
    testInput: 'doc',
    build: () =>
      makePipeline(
        'Document OCR cleanup',
        [
          ['deskew', { method: 'minarearect', maxAngle: 20 }],
          ['shadow_removal', { dilate: 7, median: 21 }],
          ['denoise_preset', { preset: 'sp' }],
          ['sauvola', { method: 'sauvola', window: 25, k: 0.2 }],
        ],
        { read: { flag: 'GRAYSCALE' } },
      ),
  },
  {
    id: 'forensics',
    label: 'Document tamper forensics',
    description: 'Train: JPEG re-compression augmentation. Both: SRM residual filter bank (3-channel noise stream) → CHW. ELA is included but disabled: enable it instead of SRM to compare.',
    build: () =>
      makePipeline('Tamper forensics', [
        ['jpeg_compression', { quality: [60, 95] }, { ...TRAIN, p: 0.5 }],
        ['ela', { quality: 90, scale: 15 }, { enabled: false, note: 'Alternative input: enable this and disable the SRM filter.' }],
        ['srm_filter', { truncate: true, T: 3 }],
        ['to_chw'],
      ]),
  },
  {
    id: 'depth',
    label: 'Monocular depth',
    description: 'Resize shorter side to 384 keeping the aspect ratio, colour jitter, horizontal flip that also flips the depth map (attach it as the mask), RGB, ImageNet normalisation, CHW.',
    build: () =>
      makePipeline('Monocular depth', [
        ['resize', { mode: 'shorter', side: 384, interp: 'LINEAR' }],
        ['color_jitter', { brightness: 0.2, contrast: 0.2, saturation: 0.2, hue: 0.02 }, { ...TRAIN, p: 0.8 }],
        ['random_flip', { dir: 'h' }, { ...TRAIN, p: 0.5, note: 'The depth map is passed as the mask, so it flips with the image.' }],
        ['convert_color', { to: 'RGB' }],
        ['standardize', { preset: 'imagenet' }],
        ['to_chw'],
      ]),
  },
  {
    id: 'medical_ct',
    label: 'Medical CT / X-ray',
    description: '16-bit read (ANYDEPTH) → windowing (level / width) → CLAHE → scale to [0, 1] → (1, H, W).',
    testInput: 'u16',
    build: () =>
      makePipeline(
        'Medical CT / X-ray',
        [
          ['ct_window', { preset: 'custom', level: 10000, width: 20000, intercept: 0 }],
          ['clahe', { clip: 2, tiles: 8 }],
          ['scale_01'],
          ['to_chw'],
        ],
        { read: { flag: 'ANYDEPTH' } },
      ),
  },
  {
    id: 'low_light',
    label: 'Low-light enhancement',
    description: 'Gamma 1.8 lifts the shadows, CLAHE on the L channel adds local contrast, a bilateral filter removes the amplified noise.',
    build: () =>
      makePipeline('Low-light enhancement', [
        ['gamma', { gamma: 1.8 }],
        ['clahe', { clip: 2.5, tiles: 8 }],
        ['bilateral', { d: 9, sigmaColor: 50, sigmaSpace: 50 }],
      ]),
  },
  {
    id: 'satellite',
    label: 'Satellite / aerial',
    description: 'Percentile stretch (p2–p98), random 90° rotations and flips (orientation is arbitrary from above), RGB, [0, 1], CHW.',
    build: () =>
      makePipeline('Satellite / aerial', [
        ['percentile_stretch', { pLo: 2, pHi: 98 }],
        ['random_rotate90', {}, { ...TRAIN, p: 1 }],
        ['random_flip', { dir: 'h' }, { ...TRAIN, p: 0.5 }],
        ['random_flip', { dir: 'v' }, { ...TRAIN, p: 0.5 }],
        ['convert_color', { to: 'RGB' }],
        ['scale_01'],
        ['to_chw'],
      ]),
  },
];

export function getPreset(id: string): Preset | undefined {
  return presets.find((p) => p.id === id);
}
