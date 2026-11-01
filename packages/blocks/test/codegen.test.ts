import { describe, expect, it } from 'vitest';
import { CodegenError, generateProject, templates, type Framework, type GeneratedFile } from '../src/index';

describe('codegen snapshots', () => {
  for (const t of templates) {
    for (const fw of ['pytorch', 'tensorflow'] as Framework[]) {
      it(`${t.id} / ${fw}`, () => {
        let files: GeneratedFile[];
        try {
          files = generateProject({ ...t.build(), framework: fw }).files;
        } catch (err) {
          // Unsupported framework combinations must fail with a readable reason, never crash.
          expect(err).toBeInstanceOf(CodegenError);
          expect((err as CodegenError).issues.join(' ')).toMatch(/not available in TensorFlow/);
          return;
        }
        expect(files.find((f) => f.path === 'model.py')!.content).toMatchSnapshot('model.py');
        expect(files.find((f) => f.path === 'train.py')!.content).toMatchSnapshot('train.py');
      });
    }
  }

  it('is deterministic', () => {
    const t = templates.find((x) => x.id === 'yolov8s')!;
    const a = generateProject(t.build()).files.map((f) => f.content);
    const b = generateProject(t.build()).files.map((f) => f.content);
    expect(a).toEqual(b);
  });

  it('links generated lines back to graph nodes', () => {
    const t = templates.find((x) => x.id === 'resnet18')!;
    const model = generateProject(t.build()).files[0];
    const lines = model.content.split('\n');
    const linked = Object.keys(model.lineMap!).map(Number);
    expect(linked.length).toBeGreaterThan(10);
    expect(lines[linked[0] - 1]).toMatch(/# node:/);
  });
});
