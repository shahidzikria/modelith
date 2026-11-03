// Registry of extra exported files (Albumentations, torchvision, tf.data, dataset.py). Kept separate from
// codegen/index.ts so emitter modules can register without a circular import.
import type { ModelInput } from '../analyze';
import type { GeneratedFile, ImageMeta, Pipeline } from '../types';

export interface PipelineExportOptions {
  /** Framework of the linked model (decides which transforms file is primary). */
  framework?: 'pytorch' | 'tensorflow';
  model?: ModelInput;
  input?: ImageMeta;
  /** Include visualisation-only steps that were not ticked (server preview). */
  includeVisual?: boolean;
}

type Extra = (p: Pipeline, opts: PipelineExportOptions) => GeneratedFile | undefined;
export const extraFiles: Extra[] = [];

export function registerExportFile(fn: Extra): void {
  extraFiles.push(fn);
}
