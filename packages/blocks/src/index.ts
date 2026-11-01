import './blocks/index';

export * from './types';
export * from './shapes';
export * from './registry';
export * from './py';
export { analyze, incoming, topoSort, trainingMemoryBytes, type Analysis, type Issue, type NodeInfo } from './graph/analyze';
export {
  previewAppends,
  previewBlock,
  requiredMultiple,
  suggestNext,
  suggestParamValues,
  type AppendPreview,
  type ParamSuggestion,
  type Suggestion,
} from './graph/recommend';
export { buildIR, CodegenError, type IR, type IRNode } from './codegen/ir';
export { generateProject, PINNED, type GeneratedProject } from './codegen/project';
export type { GeneratedFile } from './codegen/source';
export { templates, getTemplate, type Template } from './templates/index';
export { GraphBuilder } from './templates/builder';
