# Modelith — build done done done (120 steps, 273 parity cases) done done (tf.data generated, not executed in CI) done done except the AI assistant (§12.4) |||||||plan

Drag-and-drop deep-learning model builder that exports runnable PyTorch / Keras 3 code.

## Architecture

```
packages/blocks      single source of truth: tensor types, block registry, shape inference,
                     IR, PyTorch + Keras emitters, templates  (pure TS, no React)
apps/web             React 18 + Vite + React Flow UI (palette · canvas · inspector · code view)
apps/verify          FastAPI service: instantiate generated code, return real shapes/params
tests/               fixture generator (TS) + pytest that executes every generated template
```

Pipeline: `graph → validate → topo sort → prune → IR → emitter → files (model.py, train.py,
config.yaml, requirements.txt, README.md, graph.json)`.

## Milestones

| # | Milestone | Status |
|---|-----------|--------|
| M1 | Core types, registry, shape inference, IR, PyTorch emitter for the vertical slice (Image → Conv → GAP → Classification head), pytest harness | done |
| M2 | Web UI: palette, canvas, ports/edges with shape labels, inspector, live inference, code view, ZIP download, autosave, undo/redo | done |
| M3 | Block library wave 1: layers, CNN blocks, sequence & attention, necks/merges, dense heads | done |
| M4 | Pretrained backbones (timm) with exact per-stage channels; pyramids | done |
| M5 | Detection heads: YOLOv8 (TAL + DFL + GIoU + NMS), torchvision two-stage/one-stage wrappers, SSD, DETR | done (PyTorch) |
| M6 | train.py generator: synthetic data, AdamW groups, cosine warm-up, AMP, EMA, eval metrics, export | done (PyTorch) |
| M7 | Templates gallery (all §6 templates) + CI-style execution of every template | done |
| M8 | Keras 3 emitter (functional API, NHWC) + unsupported flags with reasons | emitter written, not yet executed (no TF locally) |
| M9 | Custom blocks (group → reusable block, repeat ×N), ELK auto-layout, code ↔ graph linking | ELK + code↔graph done; custom blocks todo |
| M10 | Verify service, memory/latency estimates, extras (§8) | verify + memory done; extras todo |

Rule: never ship a template whose generated code has not been executed by `tests/python`.

---

# Preprocessing Studio — build plan

A second module inside Modelith: build an ordered pipeline of OpenCV / NumPy preprocessing steps by drag and drop,
see every intermediate live, and export runnable `preprocess.py` (+ Albumentations / torchvision / tf.data / dataset
loaders) that a Modelith model's `train.py` can use.

## Architecture

```
packages/preprocess   single source of truth (pure TS, no React)
  src/types.ts          ImageMeta, ParamSpec, StepDef, Pipeline
  src/steps/<category>/<step>.ts
                        one file per step: params, docs, outMeta, validation, browser impl (cv passed in),
                        Python emitter, optional Albumentations / torchvision / tf emitters
  src/analyze.ts        meta propagation + validation + order advice + model compatibility (no pixels needed)
  src/runtime/          browser engine: Mat scope (frees every cv.Mat), Targets (mask/boxes/keypoints +
                        history), NumPy-compatible PCG64 RNG, step runner with cache keys
  src/codegen/          pipeline → preprocess.py, transforms_albumentations.py, transforms_torchvision.py,
                        transforms_tf.py, dataset.py, pipeline.json
  src/presets.ts        the 10 shipped pipelines
apps/web/src/studio   React UI: library · viewer · filmstrip · analysis · sortable pipeline · inspector · export;
                      OpenCV.js runs in a Web Worker
apps/studio-server    FastAPI: POST /preprocess/run executes the generated preprocess.py (the reference)
                      from step k onward and returns every intermediate; /preprocess/decode for 16-bit/TIFF/EXIF
tests/preprocess      Node (OpenCV.js) vs Python (cv2 4.12) parity, export, target round-trip, server tests
```

### Decisions

- **The backend runs the exported code.** `/preprocess/run` executes the generated `preprocess.py`, so the
  server preview *is* the exported code by construction. Parity tests therefore compare the browser engine
  against generated Python, which covers both "browser vs server" and "export vs preview".
- **Versions are pinned on both sides**: OpenCV.js `4.12.0` (`@techstark/opencv-js@4.12.0-release.1`) and
  `opencv-contrib-python-headless==4.12.0.88`, so deterministic uint8 steps can match exactly.
- **Bit-identical randomness.** Generated code uses one `np.random.Generator(np.random.PCG64(seed))` and only
  draws from `rng.random()` (uniform, integers and normals are derived from it in small helpers). The browser
  ports NumPy's `SeedSequence` + `PCG64`, so a random step previews exactly what the exported code does.
- **Geometric steps go through `Targets`** (Python class emitted into `preprocess.py`, mirrored in TS): masks
  are warped with `INTER_NEAREST`, boxes transform all 4 corners → enclosing box → clip → drop below 10 %
  visibility, and every transform is recorded in `t.history` so `postprocess_boxes / _mask / _keypoints`
  can undo resize, letterbox, crop, pad, flip, rot90 and warps.
- **Step mode lives on the step** (`mode: 'both' | 'train' | 'inference'`) rather than a separate
  `trainOnly` list, so there is one source of truth; augmentations default to `train`.
- **Pipelines are stored next to the model** (`modelith:pipelines`), included in `.modelith.json` exports, and a
  pipeline links to an Image input node (`linkedModelId = <node id>`). `generateProject()` stays independent of
  this package: the web app passes the preprocessing files and the `build_datasets()` body in as options.
- **Read image** is the fixed first row of every pipeline (flags + EXIF), emitted as `load_image(path)`.

## Milestones

| # | Milestone | Status |
|---|-----------|--------|
| P1 | Vertical slice: types, registry, meta/validation, PCG64, Targets, runtime; Read → Resize → CLAHE → Normalise; preprocess.py; worker preview; sortable list; parity test | done |
| P2 | Studio UI complete: library drag-in, inspector controls, viewer modes, overlays, pixel inspector, filmstrip, histograms, timing, undo/redo, autosave, variant B, samples, 9-sample grid | done |
| P3 | Step library §4.1–4.12, one category at a time, each with parity cases | done (120 steps, 273 parity cases) |
| P4 | Server fallback (`apps/studio-server`), batch run, 16-bit decode | done |
| P5 | Albumentations / torchvision / tf / dataset emitters; model linking + train.py integration | done (tf.data generated, not executed in CI) |
| P6 | Presets (10), export tests, target round-trip tests, Playwright | done |
| P7 | Extras §12 (auto-suggest, parameter sweep, dataset stats, quality metrics, version history) | done, except the AI assistant (§12.4) |
