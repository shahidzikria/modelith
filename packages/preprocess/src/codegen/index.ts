// Pipeline → all exported files.
import { serializePipeline } from '../schema';
import type { GeneratedFile, Pipeline } from '../types';
import { extraFiles, type PipelineExportOptions } from './exports';
import { generatePreprocessPy } from './python';
import './transforms';
import './dataset';

export { registerExportFile, type PipelineExportOptions } from './exports';

export function generatePipelineFiles(p: Pipeline, opts: PipelineExportOptions = {}): GeneratedFile[] {
  const files: GeneratedFile[] = [generatePreprocessPy(p, { input: opts.input, includeVisual: opts.includeVisual }).file];
  for (const fn of extraFiles) {
    const f = fn(p, opts);
    if (f) files.push(f);
  }
  files.push({ path: 'pipeline.json', content: serializePipeline(p), language: 'json' });
  return files;
}
