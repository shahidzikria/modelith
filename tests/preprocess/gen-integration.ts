// Generates Modelith projects whose Image input is linked to a Preprocessing Studio pipeline, using the same
// projectPreprocessing() the web app uses. tests/preprocess/test_integration.py then trains them for 2 steps.
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { generateProject, templates } from '../../packages/blocks/src/index';
import { makePipeline } from '../../packages/preprocess/src/index';
import { projectPreprocessing } from '../../apps/web/src/studio/links';

const root = join(dirname(fileURLToPath(import.meta.url)), 'generated', 'integration');
rmSync(root, { recursive: true, force: true });

const cases = [
  {
    id: 'simple-cnn',
    task: 'classification',
    pipeline: () =>
      makePipeline('Small classifier', [
        ['resize', { mode: 'exact', size: [64, 64], interp: 'AREA' }],
        ['random_flip', { dir: 'h' }, { mode: 'train', p: 0.5 }],
        ['color_jitter', {}, { mode: 'train', p: 0.8 }],
        ['convert_color', { to: 'RGB' }],
        ['standardize', { preset: 'imagenet' }],
        ['to_chw'],
      ]),
  },
  {
    id: 'unet',
    task: 'segmentation',
    pipeline: () =>
      makePipeline('Segmentation', [
        ['resize', { mode: 'exact', size: [128, 128], interp: 'LINEAR' }],
        ['random_flip', { dir: 'h' }, { mode: 'train', p: 0.5 }],
        ['random_affine', { degrees: 10 }, { mode: 'train', p: 0.5 }],
        ['convert_color', { to: 'RGB' }],
        ['standardize', { preset: 'imagenet' }],
        ['to_chw'],
      ]),
  },
];

for (const c of cases) {
  const project = templates.find((t) => t.id === c.id)!.build();
  for (const n of project.nodes) if ('pretrained' in n.params) n.params.pretrained = false;
  const input = project.nodes.find((n) => n.type === 'image_input')!;
  const pipeline = { ...c.pipeline(), linkedModelId: input.id };
  const pre = projectPreprocessing(project, [pipeline])!;
  const { files } = generateProject(project, { preprocessing: pre });
  const dir = join(root, c.id);
  mkdirSync(dir, { recursive: true });
  for (const f of files) writeFileSync(join(dir, f.path), f.content);
  writeFileSync(join(dir, 'case.json'), JSON.stringify({ task: c.task, size: [input.params.width, input.params.height] }));
  console.log(`${c.id}: ${files.map((f) => f.path).join(', ')}`);
}
