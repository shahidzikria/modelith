import { describe, expect, it } from 'vitest';
import { analyze, GraphBuilder, previewAppends, requiredMultiple, suggestNext, suggestParamValues } from '../src/index';

describe('append previews', () => {
  const g = new GraphBuilder();
  const x = g.add('image_input');
  const conv = g.add('conv_bn_act', { out: 64, stride: 2 }, x);
  const p = g.project('t');
  const a = analyze(p);

  it('predicts the output of compatible blocks', () => {
    const prev = previewAppends(p, a, conv);
    expect(prev.gap).toMatchObject({ ok: true, shape: { kind: 'vector', d: 64 } });
    expect(prev.maxpool.shape).toMatchObject({ kind: 'image', c: 64, h: 56, w: 56 });
  });
  it('explains incompatible blocks', () => {
    const prev = previewAppends(p, a, conv);
    expect(prev.embedding.ok).toBe(false);
    expect(prev.embedding.reason).toMatch(/Needs token ids/i);
    expect(prev.linear).toMatchObject({ ok: false, fix: expect.stringMatching(/Global average pool/) });
    expect(prev.add).toMatchObject({ ok: true, needsMore: true });
  });
  it('suggests valid next blocks', () => {
    const next = suggestNext(p, a, conv);
    expect(next.map((s) => s.type)).toContain('gap');
    expect(next.every((s) => s.ok && s.shape)).toBe(true);
  });
});

describe('parameter suggestions', () => {
  it('input sizes respect patch size and network stride', () => {
    const g = new GraphBuilder();
    const x = g.add('image_input');
    const pe = g.add('patch_embed', { patch: 16 }, x);
    g.add('transformer_encoder', {}, pe);
    const p = g.project('t');
    expect(requiredMultiple(analyze(p))).toBe(16);
    const sizes = suggestParamValues(p, x, 'height', 20)!.map((s) => s.value);
    expect(sizes).toContain(224);
    expect(sizes.every((v) => v % 16 === 0)).toBe(true);
    expect(sizes).not.toContain(96 + 8);
  });
  it('attention heads divide the model dimension', () => {
    const g = new GraphBuilder();
    const x = g.add('image_input');
    const pe = g.add('patch_embed', { dim: 100 }, x);
    const enc = g.add('transformer_encoder', { heads: 2 }, pe);
    const heads = suggestParamValues(g.project('t'), enc, 'heads')!;
    expect(heads.map((s) => s.value)).toEqual([1, 2, 4]);
    expect(heads.find((s) => s.current)?.value).toBe(2);
  });
  it('no suggestions for params without candidates', () => {
    const g = new GraphBuilder();
    const x = g.add('image_input');
    const h = g.add('cls_head', {}, g.add('gap', {}, x));
    expect(suggestParamValues(g.project('t'), h, 'classes')).toBeUndefined();
  });
});
