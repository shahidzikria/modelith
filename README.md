# Modelith

**Design deep-learning models visually, then export runnable PyTorch or TensorFlow / Keras code.**

Modelith is a drag-and-drop model builder that runs in your browser. You place inputs, layers, CNN and attention
blocks, pretrained backbones, necks and task heads on a canvas and wire them together. As you build, Modelith
works out the tensor shape at every step, counts parameters and FLOPs, estimates training memory and flags
mistakes. When the graph is ready, it generates a complete, runnable project: `model.py`, `train.py`,
`config.yaml`, `requirements.txt`, `README.md` and `graph.json`.

- **Live shape inference:** every block shows its input and output size, e.g. `in 3×224×224 → out 64×112×112 ↓2×`.
- **Guidance while you build:** when you select a block, the palette previews what each other block would output
  after it, and Properties lists sensible next layers and parameter values that keep the whole model valid.
- **Validation with fixes:** incompatible connections are refused with an explanation, and errors suggest a
  fix ("Add Global average pool or Flatten first").
- **21 ready-made templates:** classification, segmentation, depth, detection (YOLOv8, Faster/Mask R-CNN,
  RetinaNet, SSD, DETR), text, multimodal (CLIP) and tabular.
- **Tested code generation:** every template's generated PyTorch code is executed in the test suite (forward pass,
  shape and parameter check, two training steps, ONNX export).

---

## Contents

1. [Requirements](#requirements)
2. [Install and run](#install-and-run)
3. [Using Modelith](#using-modelith)
4. [Running the generated code](#running-the-generated-code)
5. [Verify service (optional)](#verify-service-optional)
6. [Tests](#tests)
7. [Project structure](#project-structure)
8. [Adding a block](#adding-a-block)
9. [Status and limitations](#status-and-limitations)
10. [License](#license)

---

## Requirements

| To do this | You need |
|---|---|
| Run the web app | **Node.js 20+** and **npm 10+** |
| Run generated PyTorch code | **Python 3.10+**; `pip install -r requirements.txt` from the generated project (PyTorch 2.x, timm, torchvision) |
| Run generated Keras code | Python 3.10+ with TensorFlow 2.16+ / Keras 3 |
| Run the verify service and Python tests | Python 3.10+ with `torch`, `torchvision`, `timm`, `scipy`, `fastapi`, `uvicorn`, `pytest`, `onnx` |

Everything in the browser works offline; no account or server is needed.

## Install and run

```bash
git clone https://github.com/shahidzikria/modelith.git
cd modelith
npm install
npm run dev
```

Open **http://localhost:5173**.

To build an optimised version and serve it locally:

```bash
npm run build -w apps/web
npm run preview -w apps/web      # http://localhost:4173
```

The build output in `apps/web/dist/` is a static site; you can host it on any static host (GitHub Pages,
Netlify, Vercel, S3, nginx).

---

## Using Modelith

### The screen

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Top bar: name · Templates · Import · Export · Undo/Redo · Layout · Fit ·     │
│          Reset · PyTorch/TensorFlow · Validate · Generate code · Theme       │
├────────────┬─────────────────────────────────────────┬───────────────────────┤
│ Palette    │ Canvas                                  │ Inspector             │
│ (blocks by │ (your model graph; connections are      │ Properties · Summary  │
│  category) │  labelled with the shape they carry)    │ Code · Notes          │
└────────────┴─────────────────────────────────────────┴───────────────────────┘
```

On a phone, the palette opens from **☰** (top left) and the inspector from **⚙** (top right).

### 1. Start a model

- **From a template:** click **Templates** and pick one (e.g. *ResNet-18 classifier*, *U-Net*, *YOLOv8-s style*).
  Templates are normal graphs you can edit freely.
- **From scratch:** choose **Empty canvas**, then add an **Input** block first: **Image** (C × H × W),
  **Text** (token ids) or **Tabular** (feature vector).

### 2. Add blocks

There are two ways to add a block from the palette:

- **Click** it: it's added to the right of the selected block and **connected automatically** if the shapes are
  compatible. This is the fastest way to build a chain.
- **Drag** it onto the canvas: it's placed where you drop it, unconnected. Wire it up in step 3.

Use the search box to filter blocks by name or description.

**Shape previews:** select a block on the canvas, and every palette entry shows what it would output if added
next (e.g. `→ 64×56×56`). Blocks that can't follow are greyed out; hover them to see why and how to fix it.

### 3. Connect blocks

Drag from a block's **output port** (right edge) to another block's **input port** (left edge). Every connection
is labelled with the tensor shape it carries. Modelith refuses connections that can't work (e.g. a feature map
into an Embedding layer) or that would create a cycle, and tells you why. Dropping onto an occupied port
replaces the old connection. Merge blocks (**Add**, **Concat**, **Multiply**) grow a new port each time you
connect one.

### 4. Edit a block

Click a block to open it in **Properties**:

- **Shapes:** input(s), output, parameter count and FLOPs for this block.
- **Hyper-parameters:** channels, kernel, stride, activation and so on. Changes update the whole graph instantly.
- **Suggested values:** clickable chips under fields like channels, kernel, stride, attention heads and patch
  size. Only values that keep the entire graph valid are offered. On the Image input, the suggested sizes are
  multiples of the network's total downsampling (e.g. 32 for a ResNet, 16 for ViT/16).
- **Name in code:** the variable name used for this block in `model.py`.
- **Next layer:** common next blocks with their predicted output; click one to add and connect it.
- **Duplicate**, **Reset parameters** (back to defaults) and **Delete block**.

### 5. Delete, undo and reset

| Action | How |
|---|---|
| Delete a block | Select it and click **Delete** above it, **Delete block** in Properties, or press <kbd>Del</kbd> |
| Delete several blocks | Shift/Ctrl-click or drag a box to select them, then **Delete N blocks** or <kbd>Del</kbd> |
| Delete a connection | Click it, then click the **×** on it, or press <kbd>Del</kbd> |
| Undo / redo | **↶ / ↷** in the top bar, <kbd>Ctrl</kbd>+<kbd>Z</kbd> / <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd> |
| Clear everything | **Reset** in the top bar, then confirm (undoable with <kbd>Ctrl</kbd>+<kbd>Z</kbd>) |

### 6. Check the model

Open the **Summary** tab:

- Total **parameters**, **FLOPs** per sample, **weight size**, number of heads.
- **Training memory** estimate for a batch size you choose (weights, gradients, AdamW state and activations).
- **Validation:** every error and warning; click one to jump to the block. **Validate** in the top bar shows
  the count.
- **Layers:** a table of every block with its output shape and parameters.

Blocks that don't feed any head are faded and left out of the generated code.

### 7. Choose a framework and generate code

1. Pick **PyTorch** or **TensorFlow** in the top bar. Blocks that the chosen framework can't export are marked
   with the reason.
2. Click **Generate code**. The **Code** tab shows each file in its own tab.
3. **Copy file** copies the current file; **Download ZIP** downloads the whole project.
4. Click a block to jump to its code; move the cursor in the code to highlight the block it came from.

### 8. Save and share

- Your work is **saved automatically** in the browser (local storage).
- **Export** downloads the project as a `.modelith.json` file; **Import** loads one back.
- The **Notes** tab keeps free-text notes with the project (they're also written to `graph.json`).

### Keyboard shortcuts

| Key | Action |
|---|---|
| <kbd>Ctrl</kbd>+<kbd>Z</kbd> / <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd> (or <kbd>Ctrl</kbd>+<kbd>Y</kbd>) | Undo / redo |
| <kbd>Del</kbd> / <kbd>Backspace</kbd> | Delete selected blocks or connections |
| <kbd>Ctrl</kbd>+<kbd>C</kbd> / <kbd>Ctrl</kbd>+<kbd>V</kbd> | Copy / paste selected blocks |
| <kbd>Ctrl</kbd>+<kbd>D</kbd> | Duplicate selection |
| <kbd>L</kbd> | Auto-layout the graph |
| <kbd>F</kbd> | Fit the graph in view |
| <kbd>Esc</kbd> | Clear selection |

### Block categories

| Category | Blocks |
|---|---|
| Inputs | Image, Text, Tabular |
| Layers | Conv2d, Conv·BN·Act, depthwise-separable conv, ConvTranspose2d, BatchNorm, GroupNorm, LayerNorm, RMSNorm, activation, max/avg pool, global average pool, flatten, linear, dropout, upsample, image ↔ sequence |
| CNN blocks | ResNet Basic/Bottleneck, MBConv, inverted residual, Fused-MBConv, RepVGG, DoubleConv, ConvNeXt, C2f, SPPF, Squeeze-Excite, CBAM, ASPP |
| Seq & attn | Embedding, positional encoding, patch embed, transformer encoder/decoder, LSTM/GRU, sequence pooling |
| Backbones | Pretrained timm models: ResNet-18/50, EfficientNet-B0–B3, ConvNeXt-T/S, MobileNetV3, RegNetY, Swin-T, ViT-S/B, DINOv2 ViT-S |
| Necks/merge | FPN, PAN, BiFPN, make pyramid, select level, add, concat, multiply, resize-like |
| Heads | Classification, token classification (CRF option), segmentation, dense regression (e.g. depth), embedding (ArcFace), regression, contrastive (CLIP) |
| Detectors | YOLOv8 head, SSD, Faster R-CNN, Mask R-CNN, RetinaNet, FCOS, DETR |

Hover any palette entry for a description and the paper it comes from.

### Templates

| Template | Task |
|---|---|
| ResNet-18 classifier, ConvNeXt-T fine-tune, ViT-Tiny/16, EfficientNet-B0 transfer, Simple CNN | Image classification |
| U-Net, DeepLabV3 (ResNet-50 + ASPP), SegFormer-lite, Tampering localisation | Segmentation |
| Monocular depth | Depth estimation |
| Multi-task (cls + seg) | Classification + segmentation |
| YOLOv8-s style, Faster R-CNN R50-FPN, SSD-MobileNetV3, RetinaNet, DETR-R50 | Object detection |
| Mask R-CNN | Instance segmentation |
| BERT-style text classifier, BiLSTM-CRF NER tagger | Text |
| CLIP-style dual encoder | Image–text |
| Tabular MLP | Tabular regression |

---

## Running the generated code

Unzip the download and, from inside the folder:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python model.py                      # builds the model, prints a summary, runs a test forward pass
python train.py --max-steps 2        # quick smoke test on synthetic data
python train.py                      # full training run
```

### The generated files

| File | Contents |
|---|---|
| `model.py` | The model (`build_model()`), a `CONFIG` dict, `example_inputs()` and a self-test when run directly |
| `train.py` | Complete training loop: data loading, losses and metrics per head, AdamW with weight-decay groups, cosine schedule with warm-up, mixed precision, EMA, checkpoints/resume, evaluation and export |
| `config.yaml` | Hyper-parameters (epochs, batch size, learning rate, image size, …) |
| `requirements.txt` | Pinned dependencies |
| `README.md` | Instructions specific to your model, including the expected data format |
| `graph.json` | The Modelith graph, so you can re-import or diff it |

### Training on your own data

`train.py` starts with a `SyntheticDataset` so it runs out of the box. Replace it in `build_datasets()` with your
data; the function's docstring describes the exact format each head expects (e.g. class indices, masks, boxes
in `xyxy` pixels with labels).

### `train.py` options

| Flag | Meaning |
|---|---|
| `--epochs N` | Number of epochs (default from `config.yaml`) |
| `--batch-size N` | Batch size |
| `--lr X` | Peak learning rate |
| `--max-steps N` | Stop after N optimiser steps (useful for smoke tests) |
| `--resume PATH` | Resume from a checkpoint |
| `--out DIR` | Output directory for checkpoints and exports |
| `--no-export` | Skip the export at the end |

After training, PyTorch projects export **TorchScript** and **ONNX**; Keras projects export a **SavedModel**
and **TFLite**.

---

## Verify service (optional)

A small FastAPI service runs the generated `model.py` for real and reports the true output shapes and
parameter count. The **Verify** button in the Summary tab then compares them with Modelith's prediction.

```bash
cd apps/verify
pip install -r requirements.txt
uvicorn main:app --port 8000
```

The app uses `http://localhost:8000` by default; set `VITE_VERIFY_URL` (e.g. `VITE_VERIFY_URL=http://myhost:8000`)
when running or building the web app to use another address. The service executes submitted code, so run it
only on your own machine or inside the provided container (`apps/verify/Dockerfile`), never exposed to the
internet.

---

## Tests

| Command | What it checks |
|---|---|
| `npm test` | Vitest: shape formulas, validation, recommendations, generated-code snapshots for every template × framework |
| `npm run gen:fixtures -- --no-pretrained` then `npm run test:py` | Executes every template's generated PyTorch code: forward pass, shapes and parameters match the UI (±3%), two training steps, checkpoint and ONNX export |
| `python -m pytest apps/verify` | Verify service |
| `npx playwright install chromium` then `npx playwright test` | Browser tests: templates, click-to-append, drag-and-connect, refused connections, parameter edits, suggestions, delete, reset, undo/redo, code copy, 390 px phone layout |

To reuse an existing Chromium for Playwright, set `PW_CHROME` to its executable path. CI
(`.github/workflows/ci.yml`) runs all of the above on every push.

---

## Project structure

```
packages/blocks        Single source of truth (pure TypeScript, no React)
  src/types.ts           Tensor kinds (image, sequence, vector, tokens, pyramid, output) and BlockDef
  src/shapes.ts          Conv / pool / transposed-conv formulas (PyTorch and Keras semantics)
  src/blocks/*.ts        Block registry: params, shape inference, param counts, PyTorch + Keras emitters
  src/helpers/*.ts       Python helper classes, emitted only when a model uses them
  src/graph/analyze.ts   Topological sort, shape inference, validation, params / FLOPs / memory
  src/graph/recommend.ts Shape previews, next-layer and parameter suggestions
  src/codegen/           Graph → model.py / train.py (PyTorch, Keras), config, requirements, README
  src/templates/         The 21 ready-made graphs
apps/web               React 18 + Vite + React Flow + Zustand + Monaco + Tailwind UI
apps/verify            FastAPI service that runs generated code and reports real shapes
tests/                 Fixture generator, pytest (executes generated code), Playwright tests
```

Pipeline: `graph → analyze (validate, sort, infer shapes) → drop unused blocks → IR → emitter → files`.

## Adding a block

Create (or extend) a file in `packages/blocks/src/blocks/` and call `defineBlock`:

```ts
defineBlock({
  type: 'my_block', category: 'CNN blocks', label: 'My block', doc: 'What it does.', varName: 'my',
  inputs: [{ name: 'x', accepts: ['image'] }],
  params: [int('out', 'Output channels', 64)],
  inferShape: (ins, p) => ({ ...img(ins[0]), c: p.out }),        // throw ShapeError(message, fix) on bad input
  paramCount: (ins, p) => convParams(img(ins[0]).c, p.out, 3),
  emitTorch: ({ ins, params }) => ({ init: `MyBlock(${ins[0].c}, ${params.out})`, helpers: ['MyBlock'] }),
  emitKeras: () => ({ unsupported: 'Reason shown in the UI.' }),
});
```

Put Python helper classes in `src/helpers/` with `defineTorchHelpers` / `defineKerasHelpers`, import the file
from `src/blocks/index.ts`, then add the block to a template and run the Python tests so its generated code
is executed.

## Status and limitations

See [PLAN.md](PLAN.md) for milestones.

- **TensorFlow export is not yet tested.** The Keras generator covers classification, segmentation, dense
  regression, text and tabular graphs, but its output hasn't been executed in CI yet. Detection heads, CRF,
  ArcFace, CLIP, BiFPN and some backbones are PyTorch-only and are marked as such in the UI.
- **FLOPs are estimates** (YOLOv8-s: ~42 GFLOPs predicted vs 28.6 reported). Parameter counts are checked
  against the real models (±3%).
- Not built yet: grouping blocks into reusable custom blocks, an AI assistant, importing existing PyTorch/ONNX
  models, hyper-parameter sweep export, version history and share links.

## License

[MIT](LICENSE) © 2026 Shahid Zikria
