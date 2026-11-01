# Netloom

Drag-and-drop deep-learning model builder. Wire inputs, layers, CNN / attention blocks, pretrained
backbones, necks and task heads on a canvas; get live shape inference, parameter / FLOP / memory
estimates and validation; then export runnable **PyTorch** (or **Keras 3**) code: `model.py`,
`train.py`, `config.yaml`, `requirements.txt`, `README.md` and `graph.json`.

## Quick start

```bash
npm install
npm run dev                 # http://localhost:5173
npm test                    # Vitest: shape inference + codegen snapshots
```

Optional verify service (real forward pass, shape / param diff in the Summary tab):

```bash
cd apps/verify && pip install -r requirements.txt && uvicorn main:app --port 8000
```

Execute every template's generated code (what CI runs):

```bash
npx tsx tests/generate-fixtures.ts --no-pretrained
python -m pytest tests/python -q      # smoke test, UI shape/param check, 2 train steps, ONNX export
npx playwright test                   # UI flows incl. a 390 px viewport (set PW_CHROME to reuse a local Chromium)
```

## Architecture

```
packages/blocks   single source of truth (pure TypeScript, no React)
  src/types.ts          tensor kinds (image, sequence, vector, tokens, pyramid, video, output), BlockDef
  src/shapes.ts         conv / pool / transposed-conv formulas (PyTorch and Keras "same" semantics)
  src/blocks/*.ts       block registry: params, inferShape, paramCount, emitTorch, emitKeras, head spec
  src/helpers/*.ts      Python helper classes emitted only when used (dependency-ordered)
  src/graph/analyze.ts  topo sort, shape inference, validation, params / FLOPs / memory
  src/codegen/          IR → model.py / train.py (PyTorch, Keras) + config, requirements, README
  src/templates/        19 + 2 ready-made graphs
apps/web          React 18 + Vite + React Flow + Zustand (zundo undo/redo) + Monaco + Tailwind
apps/verify       FastAPI sandbox: POST /verify runs the generated model and returns real shapes
tests/            fixture generator, pytest (executes generated code), Playwright e2e
```

Pipeline: `graph → analyze (validate, topo sort, infer) → prune unused → IR → emitter → files`.
Every generated `__init__` / `forward` line is mapped to its node, so moving the cursor in the
code view highlights the node and selecting a node reveals its code.

## Adding a block

Create (or extend) a file in `packages/blocks/src/blocks/` and call `defineBlock({...})`:

```ts
defineBlock({
  type: 'my_block', category: 'CNN blocks', label: 'My block', doc: '…', varName: 'my',
  inputs: [{ name: 'x', accepts: ['image'] }],
  params: [int('out', 'Output channels', 64)],
  inferShape: (ins, p, ctx) => ({ ...img(ins[0]), c: p.out }),     // throw ShapeError(msg, fix)
  paramCount: (ins, p) => …,
  emitTorch: ({ ins, params }) => ({ init: `MyBlock(${ins[0].c}, ${params.out})`, helpers: ['MyBlock'] }),
  emitKeras: () => ({ unsupported: 'reason shown in the UI' }),
});
```

Put Python helper classes in `src/helpers/` with `defineTorchHelpers` / `defineKerasHelpers`, import the
file from `src/blocks/index.ts`, then add a template and run the pytest suite.

## Status

See [PLAN.md](PLAN.md) for milestones. Notable limits:

- TensorFlow export covers classification, segmentation, dense regression, text and tabular graphs;
  detection heads, CRF, ArcFace, CLIP and some backbones are flagged unsupported in the UI with the reason.
  The Keras emitter has not yet been executed in CI (TensorFlow isn't installed in the dev environment).
- Grouping nodes into reusable custom blocks, the AI copilot, model import and the other §8 extras are not built yet.
- FLOPs are estimates (YOLOv8-s: ~42 GFLOPs predicted vs 28.6 reported); parameter counts are checked
  against the real models in CI (±3%).
