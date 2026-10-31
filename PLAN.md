# Netloom — build plan

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
| M1 | Core types, registry, shape inference, IR, PyTorch emitter for the vertical slice (Image → Conv → GAP → Classification head), pytest harness | |
| M2 | Web UI: palette, canvas, ports/edges with shape labels, inspector, live inference, code view, ZIP download, autosave, undo/redo | |
| M3 | Block library wave 1: layers, CNN blocks, sequence & attention, necks/merges, dense heads | |
| M4 | Pretrained backbones (timm) with exact per-stage channels; pyramids | |
| M5 | Detection heads: YOLOv8 (TAL + DFL + GIoU + NMS), torchvision two-stage/one-stage wrappers, SSD, DETR | |
| M6 | train.py generator: synthetic data, AdamW groups, cosine warm-up, AMP, EMA, eval metrics, export | |
| M7 | Templates gallery (all §6 templates) + CI-style execution of every template | |
| M8 | Keras 3 emitter (functional API, NHWC) + unsupported flags with reasons | |
| M9 | Custom blocks (group → reusable block, repeat ×N), ELK auto-layout, code ↔ graph linking | |
| M10 | Verify service, memory/latency estimates, extras (§8) | |

Rule: never ship a template whose generated code has not been executed by `tests/python`.
