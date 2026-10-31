import { py } from '../py';
import { defineBlock } from '../registry';
import { ShapeError } from '../types';
import { int, select, str } from './common';

const nameParam = (def: string) =>
  str('name', 'Name', def, { help: 'Python argument name of this input in forward() / keras.Input.' });

defineBlock({
  type: 'image_input',
  category: 'Inputs',
  label: 'Image',
  doc: 'Image or feature-map input (C × H × W). PyTorch uses NCHW, Keras uses NHWC.',
  varName: 'image',
  isInput: true,
  inputs: [],
  params: [
    nameParam('image'),
    int('channels', 'Channels', 3, { min: 1, max: 1024, help: '3 for RGB, 1 for grayscale. Pretrained backbones need 3.' }),
    int('height', 'Height', 224, { min: 8, max: 4096 }),
    int('width', 'Width', 224, { min: 8, max: 4096 }),
    select('norm', 'Normalisation', ['imagenet', '0-1', '-1-1'], 'imagenet', {
      help: 'Pixel normalisation applied by the data pipeline (ImageNet mean/std, [0,1] or [-1,1]).',
    }),
  ],
  inferShape: (_ins, p) => {
    if (p.channels < 1 || p.height < 1 || p.width < 1) throw new ShapeError('Channels, height and width must be positive.');
    return { kind: 'image', c: p.channels, h: p.height, w: p.width, stride: 1 };
  },
  emitTorch: () => ({}),
  emitKeras: (ctx) => ({
    expr: () => `keras.Input(shape=(${ctx.params.height}, ${ctx.params.width}, ${ctx.params.channels}), name=${py(ctx.name)})`,
  }),
});

defineBlock({
  type: 'text_input',
  category: 'Inputs',
  label: 'Text',
  doc: 'Token-id sequence produced by a tokenizer (BPE / WordPiece).',
  varName: 'tokens',
  isInput: true,
  inputs: [],
  params: [
    nameParam('tokens'),
    int('seqLen', 'Sequence length', 128, { min: 1, max: 32768 }),
    int('vocab', 'Vocabulary size', 30522, { min: 2, max: 1000000, help: '30522 for bert-base-uncased.' }),
    str('tokenizer', 'Tokenizer (HF name)', 'bert-base-uncased', { help: 'Hugging Face tokenizer used in train.py data loading.' }),
  ],
  inferShape: (_ins, p) => ({ kind: 'tokens', l: p.seqLen, vocab: p.vocab }),
  emitTorch: () => ({}),
  emitKeras: (ctx) => ({
    expr: () => `keras.Input(shape=(${ctx.params.seqLen},), dtype="int32", name=${py(ctx.name)})`,
  }),
});

defineBlock({
  type: 'tabular_input',
  category: 'Inputs',
  label: 'Tabular',
  doc: 'Fixed-length vector of numeric features (standardise them in the data pipeline).',
  varName: 'features',
  isInput: true,
  inputs: [],
  params: [nameParam('features'), int('features', 'Features', 32, { min: 1, max: 100000 })],
  inferShape: (_ins, p) => ({ kind: 'vector', d: p.features }),
  emitTorch: () => ({}),
  emitKeras: (ctx) => ({ expr: () => `keras.Input(shape=(${ctx.params.features},), name=${py(ctx.name)})` }),
});
