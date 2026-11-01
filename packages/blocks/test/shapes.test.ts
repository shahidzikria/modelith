import { describe, expect, it } from 'vitest';
import { analyze, convOut, convTransposeOut, GraphBuilder, type Framework, type Shape } from '../src/index';

/** Build input → block and return the block's analysis info. */
function infer(type: string, params: Record<string, unknown>, input: Record<string, unknown> = {}, fw: Framework = 'pytorch') {
  const g = new GraphBuilder();
  const x = g.add('image_input', input);
  const id = g.add(type, params, x);
  return analyze(g.project('t', fw)).nodes[id];
}

describe('formulas', () => {
  it('PyTorch conv output', () => {
    expect(convOut(224, 7, 2, 'same', 1, 'pytorch')).toBe(112);
    expect(convOut(225, 3, 2, 'same', 1, 'pytorch')).toBe(113);
    expect(convOut(224, 3, 1, 'valid', 1, 'pytorch')).toBe(222);
    expect(convOut(64, 3, 1, 'same', 6, 'pytorch')).toBe(64);
  });
  it('even kernels differ between frameworks', () => {
    // PyTorch padding = (k-1)//2 = 1 for k=4 loses one pixel; Keras "same" keeps ceil(h/s).
    expect(convOut(32, 4, 1, 'same', 1, 'pytorch')).toBe(31);
    expect(convOut(32, 4, 1, 'same', 1, 'tensorflow')).toBe(32);
    expect(convOut(33, 4, 2, 'same', 1, 'tensorflow')).toBe(17);
  });
  it('transposed conv', () => {
    expect(convTransposeOut(16, 2, 2, 0, 0)).toBe(32);
    expect(convTransposeOut(16, 3, 2, 1, 1)).toBe(32);
    expect(convTransposeOut(7, 4, 2, 1, 0)).toBe(14);
  });
});

describe('block inference', () => {
  it('conv with stride tracks stride relative to input', () => {
    const n = infer('conv2d', { out: 32, stride: 2 }, { height: 65, width: 64 });
    expect(n.shape).toEqual({ kind: 'image', c: 32, h: 33, w: 32, stride: 2 });
    expect(n.params).toBe(3 * 32 * 9 + 32);
  });
  it('groups must divide channels', () => {
    expect(infer('conv2d', { out: 32, groups: 2 }).error?.message).toMatch(/Groups/);
  });
  it('patch embed needs divisible sizes', () => {
    expect(infer('patch_embed', { patch: 16 }, { height: 230, width: 224 }).error?.message).toMatch(/not divisible/);
    const ok = infer('patch_embed', { patch: 16, dim: 192 }).shape as Extract<Shape, { kind: 'sequence' }>;
    expect([ok.l, ok.d, ok.prefix]).toEqual([197, 192, 1]);
  });
  it('pretrained backbone requires 3 channels', () => {
    expect(infer('timm_backbone', { model: 'resnet50' }, { channels: 1 }).error?.message).toMatch(/3 input channels/);
  });
  it('backbone pyramid uses exact timm channels and ceil sizes', () => {
    const s = infer('timm_backbone', { model: 'convnext_tiny', output: 'pyramid' }, { height: 250, width: 224 }).shape;
    expect(s).toEqual({
      kind: 'pyramid',
      levels: [
        { c: 192, h: 32, w: 28, stride: 8 },
        { c: 384, h: 16, w: 14, stride: 16 },
        { c: 768, h: 8, w: 7, stride: 32 },
      ],
    });
  });
  it('max pool halves the size', () => {
    expect(infer('maxpool', { kernel: 2, stride: 2 }, { height: 9, width: 8 }).shape).toMatchObject({ h: 4, w: 4 });
  });
});

describe('graph validation', () => {
  it('concat with mismatched sizes suggests an upsample', () => {
    const g = new GraphBuilder();
    const x = g.add('image_input', { height: 64, width: 64 });
    const a = g.add('conv2d', { out: 8 }, x);
    const b = g.add('conv2d', { out: 8, stride: 2 }, x);
    const c = g.add('concat', {}, [a, b]);
    const r = analyze(g.project('t'));
    expect(r.nodes[c].error?.message).toMatch(/64×64 vs 32×32/);
    expect(r.nodes[c].error?.fix).toMatch(/Upsample ×2/);
  });
  it('attention heads must divide the model dim', () => {
    const g = new GraphBuilder();
    const x = g.add('text_input');
    const e = g.add('embedding', { dim: 100 }, x);
    const t = g.add('transformer_encoder', { heads: 3 }, e);
    expect(analyze(g.project('t')).nodes[t].error?.message).toMatch(/not divisible by 3 heads/);
  });
  it('errors propagate downstream as waiting', () => {
    const g = new GraphBuilder();
    const x = g.add('image_input', { height: 64, width: 64 });
    const bad = g.add('patch_embed', { patch: 48 }, x);
    const enc = g.add('transformer_encoder', {}, bad);
    const r = analyze(g.project('t'));
    expect(r.nodes[bad].error).toBeTruthy();
    expect(r.nodes[enc].waiting).toBe(true);
  });
  it('a graph without a head is an error', () => {
    const g = new GraphBuilder();
    g.add('image_input');
    expect(analyze(g.project('t')).issues.some((i) => /task head/.test(i.message))).toBe(true);
  });
  it('torchvision detectors need equal-channel pyramids', () => {
    const g = new GraphBuilder();
    const x = g.add('image_input', { height: 256, width: 256 });
    const b = g.add('timm_backbone', { model: 'resnet50', output: 'pyramid', pretrained: false }, x);
    const d = g.add('faster_rcnn', {}, b);
    expect(analyze(g.project('t')).nodes[d].error?.fix).toMatch(/FPN/);
  });
  it('cycles are detected', () => {
    const g = new GraphBuilder();
    const x = g.add('image_input');
    const a = g.add('conv2d', {}, x);
    const b = g.add('conv2d', {}, a);
    const p = g.project('t');
    p.edges.push({ id: 'loop', source: b, target: a, targetPort: 0 });
    expect(analyze(p).issues.some((i) => /cycle/.test(i.message))).toBe(true);
  });
});
