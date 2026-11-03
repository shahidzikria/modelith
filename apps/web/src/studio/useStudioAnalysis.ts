import { getBlock } from '@modelith/blocks';
import { analyzePipeline, imageMeta, type ImageMeta, type ModelInput, type Pipeline, type PipelineAnalysis } from '@modelith/preprocess';
import { useMemo } from 'react';
import { useStore as useModelStore } from '../store';
import { useRuns } from './engine';
import { useStudio } from './store';

/** Image-input nodes of the current Modelith model (targets for "Link to model"). */
export function useModelInputs() {
  const project = useModelStore((s) => s.project);
  return useMemo(() => {
    const pretrained = project.nodes.some((n) => getBlock(n.type)?.pretrained?.(n.params));
    return project.nodes
      .filter((n) => n.type === 'image_input')
      .map((n) => ({
        id: n.id,
        label: `${project.name} · ${n.params.name} (${n.params.channels}×${n.params.height}×${n.params.width})`,
        input: {
          c: n.params.channels,
          h: n.params.height,
          w: n.params.width,
          framework: project.framework,
          pretrained,
          norm: n.params.norm,
          label: `${project.name} (${n.params.name})`,
        } satisfies ModelInput,
      }));
  }, [project]);
}

/** Meta of the current sample as read by the pipeline's read flag (before any step runs). */
export function useInputMeta(pipeline: Pipeline): ImageMeta {
  const sample = useStudio((s) => s.samples[s.sampleIndex]);
  const runInput = useRuns((s) => (s.A?.pipelineId === pipeline.id ? s.A?.input?.raw.meta : s.B?.pipelineId === pipeline.id ? s.B?.input?.raw.meta : undefined));
  if (runInput && !runInput.batch && runInput.w === sample.w) return runInput;
  const f = pipeline.read.flag;
  const ex = sample.exact?.meta;
  if (f === 'GRAYSCALE') return imageMeta(sample.h, sample.w, 1);
  if (f === 'ANYDEPTH') return imageMeta(sample.h, sample.w, 1, ex?.dtype === 'uint16' ? 'uint16' : 'uint8');
  if (f === 'UNCHANGED') return ex ?? imageMeta(sample.h, sample.w, sample.hasAlpha ? 4 : 3);
  return imageMeta(sample.h, sample.w, 3);
}

export function usePipelineAnalysis(pipeline: Pipeline): PipelineAnalysis {
  const train = useStudio((s) => s.train);
  const input = useInputMeta(pipeline);
  const inputs = useModelInputs();
  const model = pipeline.linkedModelId ? inputs.find((i) => i.id === pipeline.linkedModelId)?.input : undefined;
  return useMemo(() => analyzePipeline(pipeline, input, { train, model }), [pipeline, input, train, model]);
}
