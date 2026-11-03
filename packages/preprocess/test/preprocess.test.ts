import { describe, expect, it } from 'vitest';
import {
  allSteps,
  analyzePipeline,
  CATEGORIES,
  defaultParams,
  generatePipelineFiles,
  generatePreprocessPy,
  imageMeta,
  makePipeline,
  makeStep,
  parsePipeline,
  presets,
  Rng,
  serializePipeline,
  type Pipeline,
} from '../src/index';

const color = imageMeta(480, 640, 3);

describe('step registry', () => {
  const steps = allSteps();

  it('covers every category with documented steps', () => {
    expect(steps.length).toBeGreaterThanOrEqual(110);
    for (const c of CATEGORIES) expect(steps.some((s) => s.category === c)).toBe(true);
    for (const s of steps) {
      expect(s.doc.length, s.id).toBeGreaterThan(40);
      expect(s.cv2, s.id).toBeTruthy();
      expect(typeof s.emitPython, s.id).toBe('function');
      // Either a browser implementation, or explicitly routed to the server.
      expect(!!s.applyBrowser || s.engine === 'server', s.id).toBe(true);
    }
  });

  it('has unique ids and valid defaults that pass outMeta', () => {
    expect(new Set(steps.map((s) => s.id)).size).toBe(steps.length);
    for (const s of steps) {
      const p = defaultParams(s.id);
      for (const spec of s.params) {
        if (spec.type === 'select') expect(spec.options!.map((o) => o.value), `${s.id}.${spec.key}`).toContain(p[spec.key]);
        if (spec.type === 'oddInt') expect(p[spec.key] % 2, `${s.id}.${spec.key}`).toBe(1);
      }
    }
  });

  it('marks geometric steps as moving targets', () => {
    for (const s of steps.filter((x) => x.category === 'Geometric')) expect(s.affectsGeometry, s.id).toBe(true);
  });
});

describe('RNG matches NumPy', () => {
  it('Generator(PCG64(seed)).random() for several seeds', () => {
    // Values from numpy 2.2: np.random.Generator(np.random.PCG64(seed)).random(3)
    const expected: Record<number, number[]> = {
      0: [0.6369616873214543, 0.2697867137638703, 0.04097352393619469],
      42: [0.7739560485559633, 0.4388784397520523, 0.8585979199113825],
      12345: [0.22733602246716966, 0.31675833970975287, 0.7973654573327341],
    };
    for (const [seed, vals] of Object.entries(expected)) {
      const r = new Rng(Number(seed));
      expect([r.random(), r.random(), r.random()]).toEqual(vals);
    }
  });

  it('round-trips its state', () => {
    const a = new Rng(7);
    a.random();
    const b = new Rng(a.getState());
    expect(b.random()).toBe(a.random());
  });
});

describe('pipeline.json', () => {
  it('round-trips every preset', () => {
    for (const p of presets) {
      const pl = p.build();
      expect(parsePipeline(JSON.parse(serializePipeline(pl)))).toEqual(JSON.parse(serializePipeline(pl)));
    }
  });

  it('migrates version 0 (trainOnly list, readFlag) and fills missing params', () => {
    const v0 = {
      name: 'old',
      readFlag: 'GRAYSCALE',
      trainOnly: ['b'],
      steps: [
        { uid: 'a', def: 'clahe', params: { clip: 3 } },
        { uid: 'b', def: 'random_flip', params: {} },
        { uid: 'c', def: 'gaussian_blur', params: { k: 7, bogus: 1 } },
      ],
    };
    const p = parsePipeline(v0);
    expect(p.version).toBe(1);
    expect(p.read.flag).toBe('GRAYSCALE');
    expect(p.steps.map((s) => s.mode)).toEqual(['both', 'train', 'both']);
    expect(p.steps[0].params).toEqual({ clip: 3, tiles: 8 });
    expect(p.steps[2].params).not.toHaveProperty('bogus');
    expect(p.steps[1].p).toBe(0.5);
  });

  it('rejects garbage with a readable error', () => {
    expect(() => parsePipeline({ version: 9, steps: [] })).toThrow(/version/);
    expect(() => parsePipeline({ steps: [{}] })).toThrow(/def/);
  });
});

describe('validation (§6)', () => {
  const issues = (p: Pipeline, input = color, model?: Parameters<typeof analyzePipeline>[2]['model']) => analyzePipeline(p, input, { train: false, model }).issues;

  it('wrong channels: equalise needs 1 or 3 channels, Otsu needs single-channel', () => {
    const p = makePipeline('x', [['threshold', { method: 'otsu' }]]);
    const i = issues(p).find((x) => x.level === 'error')!;
    expect(i.message).toMatch(/Otsu.*uint8 single-channel/);
    expect(i.fixes?.[0]).toMatchObject({ kind: 'insertBefore', def: 'convert_color', params: { to: 'GRAY' } });
  });

  it('wrong dtype after normalisation suggests moving Normalise to the end', () => {
    const p = makePipeline('x', [['standardize'], ['clahe']]);
    const i = issues(p).find((x) => x.level === 'error')!;
    expect(i.message).toMatch(/CLAHE needs uint8 or uint16; this image is float32 after 'Standardise/);
    expect(i.fixes?.map((f) => f.kind)).toContain('moveToEnd');
  });

  it('even kernels, Canny low > high, adaptive blockSize', () => {
    expect(issues(makePipeline('x', [['gaussian_blur', { k: 4 }]]))[0].message).toMatch(/must be odd/);
    expect(issues(makePipeline('x', [['canny', { low: 200, high: 100 }]]))[0].message).toMatch(/low threshold .* above the high/);
    const ad = makePipeline('x', [['convert_color', { to: 'GRAY' }], ['adaptive_threshold', { block: 10 }]]);
    expect(issues(ad).some((i) => /blockSize must be odd/.test(i.message))).toBe(true);
    expect(issues(makePipeline('x', [['median_blur', { k: 61 }]]), imageMeta(40, 40, 3)).some((i) => /larger than/.test(i.message))).toBe(true);
  });

  it('order advice: tensor steps not at the end, double resize, undone conversion', () => {
    expect(issues(makePipeline('x', [['scale_01'], ['gaussian_blur']])).some((i) => /usually go at the end/.test(i.message))).toBe(true);
    expect(issues(makePipeline('x', [['resize'], ['resize']])).some((i) => /Two resizes/.test(i.message))).toBe(true);
    expect(issues(makePipeline('x', [['convert_color', { to: 'HSV' }], ['convert_color', { to: 'BGR' }]])).some((i) => /Converts back/.test(i.message))).toBe(true);
  });

  it('heavy filter before a shrinking resize on a large image', () => {
    const p = makePipeline('x', [['bilateral'], ['resize', { size: [256, 256] }]]);
    expect(issues(p, imageMeta(2000, 3000, 3)).some((i) => /Resizing first is ~\d+× faster/.test(i.message))).toBe(true);
  });

  it('random steps at inference are flagged', () => {
    const p: Pipeline = { ...makePipeline('x', []), steps: [makeStep('random_flip', {}, { mode: 'both' })] };
    expect(issues(p).some((i) => /inference pipelines should be deterministic/.test(i.message))).toBe(true);
  });

  it('model compatibility: size mismatch offers Resize / Letterbox / update model; pretrained wants ImageNet RGB', () => {
    const p = makePipeline('x', [['resize', { size: [256, 256] }], ['scale_01'], ['to_chw']]);
    const all = issues(p, color, { c: 3, h: 224, w: 224, framework: 'pytorch', pretrained: true });
    const size = all.find((i) => /expects 3×224×224/.test(i.message))!;
    expect(size.fixes?.map((f) => f.kind)).toEqual(['append', 'append', 'updateModelInput']);
    expect(all.some((i) => /ImageNet-normalised/.test(i.message))).toBe(true);
    expect(all.some((i) => /expect RGB, but the output is BGR/.test(i.message))).toBe(true);
  });

  it('every preset is valid in both modes', () => {
    for (const pr of presets) {
      const input = pr.testInput === 'u16' ? imageMeta(480, 640, 1, 'uint16') : pr.testInput === 'doc' ? imageMeta(480, 640, 1) : color;
      for (const train of [false, true]) {
        const errs = analyzePipeline(pr.build(), input, { train }).issues.filter((i) => i.level === 'error');
        expect(errs, `${pr.id} ${train}`).toEqual([]);
      }
    }
  });
});

describe('code generation', () => {
  it('is deterministic and has one function per enabled step', () => {
    const p = presets.find((x) => x.id === 'imagenet_train')!.build();
    const a = generatePipelineFiles(p).map((f) => f.content);
    const b = generatePipelineFiles(structuredClone(p)).map((f) => f.content);
    expect(a).toEqual(b);
    const py = a[0];
    expect(py.match(/^def step_\d\d_/gm)).toHaveLength(p.steps.length);
    expect(py).toContain('def preprocess(img, mask=None, boxes=None, keypoints=None, train=False, rng=None');
    expect(py).toContain('def postprocess_boxes(boxes, history)');
    expect(py).toContain('if __name__ == "__main__":');
    expect(py).toMatch(/RANDOM_RESIZED_CROP_SIZE = \(224, 224\)/);
    expect(py).not.toMatch(/np\.random\.(rand|seed|randint)\(/);
  });

  it('leaves out disabled, batch-level and visualisation-only steps', () => {
    const p = makePipeline('x', [['resize'], ['gaussian_blur', {}, { enabled: false }], ['mixup'], ['apply_colormap']]);
    const py = generatePreprocessPy(p).file.content;
    expect(py).toContain('def step_01_resize');
    expect(py).not.toMatch(/gaussian_blur|apply_colormap|def step_\d\d_mixup/);
    const files = generatePipelineFiles(p);
    expect(files.find((f) => f.path === 'dataset.py')!.content).toContain('def mixup(images, labels, rng');
  });

  it('links lines back to steps', () => {
    const p = makePipeline('x', [['resize'], ['clahe']]);
    const f = generatePreprocessPy(p).file;
    const owners = new Set(Object.values(f.lineMap!));
    expect(owners).toEqual(new Set(p.steps.map((s) => s.uid)));
  });

  it('preset snapshots', () => {
    for (const pr of presets) expect(generatePreprocessPy(pr.build()).file.content.replace(/"[sp][0-9a-z]{8,}"/g, '"<uid>"')).toMatchSnapshot(pr.id);
  });
});

describe('extras', () => {
  it('diffs versions with minimal moves', async () => {
    const { diffPipelines, describeDiff } = await import('../src/index');
    const a = makePipeline('x', [['resize'], ['clahe'], ['gaussian_blur'], ['standardize']]);
    const b = structuredClone(a);
    b.steps.push(b.steps.splice(0, 1)[0]); // resize to the end
    b.steps[1].params.k = 7;
    b.steps.splice(2, 1); // remove standardize
    const d = diffPipelines(a, b).map(describeDiff);
    expect(d).toContain('− Standardise (mean / std) (was step 4)');
    expect(d.filter((x) => x.startsWith('↕'))).toEqual(['↕ Resize: step 1 → 3']);
    expect(d).toContain('✎ Gaussian blur: k 5 → 7');
  });

  it('quality metrics and suggestions on synthetic images', async () => {
    const { quality, suggestSteps } = await import('../src/index');
    const n = 64 * 64;
    const flat = { meta: imageMeta(64, 64, 1), data: new Uint8Array(n).fill(30) };
    const q = quality(flat, flat);
    expect(q.psnr).toBe(Infinity);
    expect(q.ssim).toBeCloseTo(1, 6);
    expect(q.entropy).toBe(0);
    const s = suggestSteps(flat).suggestions.map((x) => x.def);
    expect(s).toContain('gamma'); // dark
    expect(s).toContain('clahe'); // low contrast
  });
});
