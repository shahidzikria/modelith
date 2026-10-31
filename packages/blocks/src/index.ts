import './blocks/index';

export * from './types';
export * from './shapes';
export * from './registry';
export * from './py';
export { analyze, incoming, topoSort, trainingMemoryBytes, type Analysis, type Issue, type NodeInfo } from './graph/analyze';
export { buildIR, CodegenError, type IR, type IRNode } from './codegen/ir';
export { generateProject, PINNED, type GeneratedProject } from './codegen/project';
export type { GeneratedFile } from './codegen/source';
