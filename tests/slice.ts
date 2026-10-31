import { generateProject } from '../packages/blocks/src/index';
import { GraphBuilder } from '../packages/blocks/src/templates/builder';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const g = new GraphBuilder();
const x = g.add('image_input', { height: 64, width: 64 });
const c = g.chain(x, ['conv_bn_act', { out: 32, stride: 2 }], ['maxpool'], ['conv2d', { out: 64 }], ['gap'], ['dropout', { p: 0.2 }]);
g.add('cls_head', { classes: 5 }, c);
const out = process.argv[2];
const { files } = generateProject(g.project('Slice'));
mkdirSync(out, { recursive: true });
for (const f of files) writeFileSync(join(out, f.path), f.content);
console.log(files.map((f) => f.path).join(' '));
