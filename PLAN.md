# Modelith — build plan

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
