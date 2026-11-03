// Messages between the UI and the OpenCV.js worker.
import type { Display, DisplayOptions, ImageBuf, ImageMeta, RunStep, StepOutput, TargetsData } from '@modelith/preprocess';

export type WorkerIn =
  | { type: 'setInput'; inputKey: string; buf: ImageBuf; targets: TargetsData }
  | {
      type: 'run';
      id: number;
      inputKey: string;
      steps: RunStep[];
      seed: number;
      train: boolean;
      /** Longest-side limit for the low-resolution proxy used while dragging sliders. */
      maxSide?: number;
      display: DisplayOptions;
    }
  | {
      type: 'variants';
      id: number;
      inputKey: string;
      variants: { steps: RunStep[]; seed: number }[];
      /** Also return the final raw buffer (batch export). */
      returnRaw?: boolean;
      train: boolean;
      maxSide?: number;
      display: DisplayOptions;
    }
  | { type: 'display'; id: number; buf: ImageBuf; display: DisplayOptions }
  | { type: 'dropInput'; inputKey: string };

export type WorkerOut =
  | { type: 'ready'; missing: Record<string, string[]> }
  | { type: 'input'; id: number; raw: ImageBuf; display: Display; targets: TargetsData }
  | { type: 'step'; id: number; index: number; out: StepOutput; display: Display }
  | { type: 'done'; id: number; error?: string; errorAt?: number; serverFrom?: number; totalMs: number }
  | { type: 'variant'; id: number; index: number; display?: Display; targets?: TargetsData; meta?: ImageMeta; raw?: ImageBuf; error?: string }
  | { type: 'displayed'; id: number; display: Display };
