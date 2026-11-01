// Generate code for every template × framework into tests/generated/<template>/<framework>/.
// Usage: npx tsx tests/generate-fixtures.ts [--no-pretrained] [--only id1,id2]
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CodegenError, generateProject, templates, type Framework, type Project } from '../packages/blocks/src/index';

const root = join(dirname(fileURLToPath(import.meta.url)), 'generated');
const noPretrained = process.argv.includes('--no-pretrained');
const onlyArg = process.argv.indexOf('--only');
const only = onlyArg > 0 ? new Set(process.argv[onlyArg + 1].split(',')) : undefined;

const summary: Record<string, Record<string, string>> = {};
rmSync(root, { recursive: true, force: true });
for (const t of templates) {
  if (only && !only.has(t.id)) continue;
  summary[t.id] = {};
  for (const fw of ['pytorch', 'tensorflow'] as Framework[]) {
    const project: Project = { ...t.build(), framework: fw };
    if (noPretrained) for (const n of project.nodes) if ('pretrained' in n.params) n.params.pretrained = false;
    const dir = join(root, t.id, fw);
    try {
      const { files, ir } = generateProject(project);
      mkdirSync(dir, { recursive: true });
      for (const f of files) writeFileSync(join(dir, f.path), f.content);
      const expected = {
        template: t.id,
        framework: fw,
        params: ir.analysis.totalParams,
        wrapped: !!ir.wrapper,
        heads: Object.fromEntries(ir.heads.map((h) => [h.name, { task: h.head!.task, dims: h.out.kind === 'output' ? h.out.dims : [] }])),
      };
      writeFileSync(join(dir, 'expected.json'), JSON.stringify(expected, null, 2));
      summary[t.id][fw] = 'ok';
    } catch (err) {
      summary[t.id][fw] = err instanceof CodegenError ? `unsupported: ${err.issues.join('; ')}` : `ERROR: ${(err as Error).stack}`;
    }
  }
}
mkdirSync(root, { recursive: true });
writeFileSync(join(root, 'summary.json'), JSON.stringify(summary, null, 2));
for (const [id, r] of Object.entries(summary)) console.log(`${id.padEnd(22)} torch: ${r.pytorch?.slice(0, 60)}  |  tf: ${r.tensorflow?.slice(0, 90)}`);
if (Object.values(summary).some((r) => Object.values(r).some((v) => v.startsWith('ERROR')))) process.exit(1);
