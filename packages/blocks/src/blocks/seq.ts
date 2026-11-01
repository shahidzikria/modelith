import { call, py } from '../py';
import { defineBlock } from '../registry';
import { img, seq, tok } from '../shapes';
import { ShapeError, type SequenceShape } from '../types';
import { bool, float, int, lstmParams, select, transformerLayerParams } from './common';

defineBlock({
  type: 'embedding',
  category: 'Seq & attn',
  label: 'Embedding',
  doc: 'Look-up table mapping token ids to dense vectors: L tokens → L×D.',
  varName: 'embed',
  inputs: [{ name: 'tokens', accepts: ['tokens'] }],
  params: [int('dim', 'Embedding size', 256, { min: 1, max: 16384 }), bool('padIdx', 'Token 0 is padding', true)],
  inferShape: (ins, p) => {
    const t = tok(ins[0]);
    return { kind: 'sequence', l: t.l, d: p.dim };
  },
  paramCount: (ins, p) => (ins[0] as { vocab: number }).vocab * p.dim,
  macs: () => 0,
  emitTorch: ({ ins, params: p }) => ({ init: call('nn.Embedding', [(ins[0] as { vocab: number }).vocab, p.dim], { padding_idx: p.padIdx ? 0 : undefined }) }),
  emitKeras: ({ ins, params: p, name }) => ({ expr: (a) => `${call('layers.Embedding', [(ins[0] as { vocab: number }).vocab, p.dim], { name })}(${a[0]})` }),
});

defineBlock({
  type: 'pos_encoding',
  category: 'Seq & attn',
  label: 'Positional encoding',
  doc: 'Adds absolute position information: learned (BERT/ViT) or fixed sinusoidal (Transformer). For RoPE, enable it on the Transformer encoder.',
  paper: 'Vaswani et al. 2017',
  varName: 'pos',
  inputs: [{ name: 'x', accepts: ['sequence'] }],
  params: [select('mode', 'Type', ['learned', 'sinusoidal'], 'learned')],
  inferShape: (ins) => seq(ins[0]),
  paramCount: (ins, p) => (p.mode === 'learned' ? (ins[0] as SequenceShape).l * (ins[0] as SequenceShape).d : 0),
  macs: () => 0,
  emitTorch: ({ ins, params: p }) => {
    const s = ins[0] as SequenceShape;
    const cls = p.mode === 'learned' ? 'LearnedPositionalEmbedding' : 'SinusoidalPositionalEncoding';
    return { init: call(cls, [s.l, s.d]), helpers: [cls] };
  },
  emitKeras: ({ params: p, name }) => ({ expr: (a) => `PositionEmbedding(${py(p.mode)}, name=${py(name)})(${a[0]})`, helpers: ['PositionEmbedding'] }),
});

defineBlock({
  type: 'patch_embed',
  category: 'Seq & attn',
  label: 'Patch embed',
  doc: 'Splits the image into p×p patches and embeds each one (ViT). Optional [CLS] token.',
  paper: 'Dosovitskiy et al. 2021 (ViT)',
  varName: 'patch_embed',
  inputs: [{ name: 'image', accepts: ['image'] }],
  params: [int('patch', 'Patch size', 16, { min: 1, max: 64 }), int('dim', 'Embedding size', 192, { min: 8, max: 4096 }), bool('cls', 'Prepend [CLS] token', true)],
  inferShape: (ins, p) => {
    const s = img(ins[0]);
    if (s.h % p.patch || s.w % p.patch) {
      throw new ShapeError(`Image ${s.h}×${s.w} is not divisible by the patch size ${p.patch}.`, `Use an input size that is a multiple of ${p.patch}.`);
    }
    const gh = s.h / p.patch;
    const gw = s.w / p.patch;
    return { kind: 'sequence', l: gh * gw + (p.cls ? 1 : 0), d: p.dim, grid: { h: gh, w: gw, stride: s.stride * p.patch }, prefix: p.cls ? 1 : 0 };
  },
  paramCount: (ins, p) => (ins[0] as { c: number }).c * p.dim * p.patch * p.patch + p.dim + (p.cls ? p.dim : 0),
  emitTorch: ({ ins, params: p }) => ({ init: call('PatchEmbed', [(ins[0] as { c: number }).c, p.dim, p.patch], { cls_token: p.cls }), helpers: ['PatchEmbed'] }),
  emitKeras: ({ params: p, name }) => ({ expr: (a) => call('patch_embed', [`§${a[0]}`, p.dim, p.patch], { cls_token: p.cls, name }), helpers: ['patch_embed'] }),
});

const heads = (def = 3) => int('heads', 'Attention heads', def, { min: 1, max: 64, help: 'The model dimension must be divisible by the number of heads.' });

defineBlock({
  type: 'transformer_encoder',
  category: 'Seq & attn',
  label: 'Transformer encoder',
  doc: 'depth × pre-norm block (multi-head self-attention + MLP) with a final LayerNorm. Optional RoPE.',
  paper: 'Vaswani et al. 2017; Dosovitskiy et al. 2021',
  varName: 'encoder',
  inputs: [{ name: 'x', accepts: ['sequence'] }],
  params: [
    int('depth', 'Depth', 12, { min: 1, max: 48 }),
    heads(3),
    float('mlpRatio', 'MLP ratio', 4, { min: 1, max: 8, step: 0.5 }),
    float('dropout', 'Dropout', 0.0, { min: 0, max: 0.5, step: 0.05 }),
    float('dropPath', 'Drop path (max)', 0.1, { min: 0, max: 0.5, step: 0.05, help: 'Stochastic depth, increasing linearly with depth.' }),
    bool('rope', 'Rotary embeddings (RoPE)', false, { help: 'Su et al. 2021. PyTorch only.' }),
  ],
  inferShape: (ins, p) => {
    const s = seq(ins[0]);
    if (s.d % p.heads) throw new ShapeError(`Model dimension ${s.d} is not divisible by ${p.heads} heads.`, `Use ${largestDivisor(s.d, p.heads)} heads or change the embedding size.`);
    return s;
  },
  paramCount: (ins, p) => {
    const d = (ins[0] as SequenceShape).d;
    return p.depth * transformerLayerParams(d, Math.floor(d * p.mlpRatio)) + 2 * d;
  },
  macs: (ins, p, _o, params) => {
    const s = ins[0] as SequenceShape;
    return params * s.l + p.depth * 2 * s.l * s.l * s.d;
  },
  emitTorch: ({ ins, params: p }) => ({
    init: call('TransformerEncoder', [(ins[0] as SequenceShape).d], {
      depth: p.depth,
      heads: p.heads,
      mlp_ratio: p.mlpRatio === 4 ? undefined : p.mlpRatio,
      dropout: p.dropout || undefined,
      drop_path: p.dropPath || undefined,
      rope: p.rope || undefined,
    }),
    helpers: ['TransformerEncoder'],
  }),
  emitKeras: ({ params: p, name }) =>
    p.rope
      ? { unsupported: 'RoPE is only emitted for PyTorch.' }
      : { expr: (a) => call('transformer_encoder', [`§${a[0]}`], { depth: p.depth, heads: p.heads, mlp_ratio: p.mlpRatio, dropout: p.dropout, name }), helpers: ['transformer_encoder'] },
});

defineBlock({
  type: 'transformer_decoder',
  category: 'Seq & attn',
  label: 'Transformer decoder',
  doc: 'depth × pre-norm block with self-attention, cross-attention to a memory sequence, and MLP.',
  paper: 'Vaswani et al. 2017',
  varName: 'decoder',
  inputs: [
    { name: 'queries', accepts: ['sequence'] },
    { name: 'memory', accepts: ['sequence'] },
  ],
  params: [int('depth', 'Depth', 6, { min: 1, max: 48 }), heads(8), float('mlpRatio', 'MLP ratio', 4, { min: 1, max: 8, step: 0.5 }), float('dropout', 'Dropout', 0.1, { min: 0, max: 0.5, step: 0.05 })],
  inferShape: (ins, p, ctx) => {
    const q = seq(ins[0], 'Queries');
    const m = seq(ins[1], 'Memory');
    if (q.d % p.heads) throw new ShapeError(`Query dimension ${q.d} is not divisible by ${p.heads} heads.`);
    if (ctx.framework === 'tensorflow' && m.d !== q.d) throw new ShapeError(`Keras cross-attention here needs equal dims (queries ${q.d}, memory ${m.d}).`, 'Add a Linear layer on the memory.');
    return q;
  },
  paramCount: (ins, p) => {
    const d = (ins[0] as SequenceShape).d;
    const m = (ins[1] as SequenceShape).d;
    const h = Math.floor(d * p.mlpRatio);
    const self = 4 * d * d + 4 * d;
    const cross = 2 * d * d + 2 * d * m + 4 * d;
    return p.depth * (self + cross + 2 * d * h + h + d + 6 * d) + 2 * d;
  },
  emitTorch: ({ ins, params: p }) => ({
    init: call('TransformerDecoder', [(ins[0] as SequenceShape).d, (ins[1] as SequenceShape).d], { depth: p.depth, heads: p.heads, mlp_ratio: p.mlpRatio === 4 ? undefined : p.mlpRatio, dropout: p.dropout }),
    helpers: ['TransformerDecoder'],
  }),
  emitKeras: ({ params: p, name }) => ({
    expr: (a) => call('transformer_decoder', [`§${a[0]}`, `§${a[1]}`], { depth: p.depth, heads: p.heads, mlp_ratio: p.mlpRatio, dropout: p.dropout, name }),
    helpers: ['transformer_decoder'],
  }),
});

defineBlock({
  type: 'rnn',
  category: 'Seq & attn',
  label: 'LSTM / GRU',
  doc: 'Recurrent layers returning every hidden state; bidirectional doubles the feature size.',
  paper: 'Hochreiter & Schmidhuber 1997; Cho et al. 2014',
  varName: 'rnn',
  inputs: [{ name: 'x', accepts: ['sequence'] }],
  params: [
    select('cell', 'Cell', ['lstm', 'gru'], 'lstm'),
    int('hidden', 'Hidden size', 256, { min: 1, max: 8192 }),
    int('layers', 'Layers', 2, { min: 1, max: 8 }),
    bool('bidirectional', 'Bidirectional', true),
    float('dropout', 'Dropout between layers', 0.1, { min: 0, max: 0.5, step: 0.05 }),
  ],
  inferShape: (ins, p) => {
    const s = seq(ins[0]);
    return { kind: 'sequence', l: s.l, d: p.hidden * (p.bidirectional ? 2 : 1) };
  },
  paramCount: (ins, p) => {
    const dirs = p.bidirectional ? 2 : 1;
    const gates = p.cell === 'lstm' ? 4 : 3;
    let total = 0;
    for (let i = 0; i < p.layers; i++) total += dirs * lstmParams(i === 0 ? (ins[0] as SequenceShape).d : p.hidden * dirs, p.hidden, gates);
    return total;
  },
  emitTorch: ({ ins, params: p }) => ({
    init: call('RecurrentLayer', [p.cell, (ins[0] as SequenceShape).d, p.hidden], { layers: p.layers, bidirectional: p.bidirectional, dropout: p.dropout }),
    helpers: ['RecurrentLayer'],
  }),
  emitKeras: ({ params: p, name }) => ({
    expr: (a) => call('recurrent', [`§${a[0]}`, p.cell, p.hidden], { num_layers: p.layers, bidirectional: p.bidirectional, dropout: p.dropout, name }),
    helpers: ['recurrent'],
  }),
});

defineBlock({
  type: 'seq_pool',
  category: 'Seq & attn',
  label: 'Sequence pooling',
  doc: 'Reduce a sequence to one vector: mean, max, the [CLS] token (first) or the last token.',
  varName: 'pooled',
  inputs: [{ name: 'x', accepts: ['sequence'] }],
  params: [select('mode', 'Mode', ['mean', 'max', 'cls', 'last'], 'mean')],
  inferShape: (ins, p) => {
    const s = seq(ins[0]);
    if (p.mode === 'cls' && s.prefix === 0) throw new ShapeError('This sequence has no [CLS] token.', 'Enable "Prepend [CLS] token" on Patch embed, or use mean pooling.');
    return { kind: 'vector', d: s.d };
  },
  paramCount: () => 0,
  macs: () => 0,
  emitTorch: ({ params: p }) => ({
    forward: (a) => ({ mean: `${a[0]}.mean(dim=1)`, max: `${a[0]}.amax(dim=1)`, cls: `${a[0]}[:, 0]`, last: `${a[0]}[:, -1]` })[p.mode as string]!,
  }),
  emitKeras: ({ params: p, name }) => ({
    expr: (a) =>
      ({
        mean: `layers.GlobalAveragePooling1D(name=${py(name)})(${a[0]})`,
        max: `layers.GlobalMaxPooling1D(name=${py(name)})(${a[0]})`,
        cls: `${a[0]}[:, 0]`,
        last: `${a[0]}[:, -1]`,
      })[p.mode as string]!,
  }),
});

function largestDivisor(d: number, near: number): number {
  for (let h = near; h >= 1; h--) if (d % h === 0) return h;
  return 1;
}
