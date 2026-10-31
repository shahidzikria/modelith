import { py } from "../py";
import { pyShape } from '../shapes';
import type { IR, IRNode } from './ir';
import { Source, type GeneratedFile } from './source';
import { helperRefs, pyDict } from "./torch";

export interface TrainHyper {
  epochs: number;
  batch: number;
  lr: number;
  weightDecay: number;
  warmupEpochs: number;
}

/** Sensible defaults per task: smaller LR for pretrained / transformer models, smaller batches for dense tasks. */
export function defaultHyper(ir: IR): TrainHyper {
  const tasks = new Set(ir.heads.map((h) => h.head!.task));
  const dense = tasks.has('segmentation') || tasks.has('dense') || tasks.has('detection') || tasks.has('instance');
  const lr = ir.pretrained ? 1e-4 : ir.hasTransformer ? 3e-4 : 1e-3;
  return {
    epochs: tasks.has('detection') || tasks.has('instance') ? 100 : 30,
    batch: dense ? 8 : 32,
    lr,
    weightDecay: 0.05,
    warmupEpochs: ir.hasTransformer || tasks.has('detection') ? 3 : 1,
  };
}

function sampleInput(n: IRNode): string {
  const s = n.out;
  if (s.kind === 'tokens') return `torch.randint(0, ${s.vocab}, ${pyShape(s, 'pytorch', '').replace('(, ', '(')})`;
  if (s.kind === 'image' && n.params.norm !== 'imagenet') return `torch.rand(${s.c}, ${s.h}, ${s.w})`;
  return `torch.randn${pyShape(s, 'pytorch', '').replace('(, ', '(')}`;
}

function inputDoc(n: IRNode): string {
  const s = n.out;
  if (s.kind === 'image') {
    const norm = n.params.norm === 'imagenet' ? 'ImageNet mean/std normalised' : n.params.norm === '0-1' ? 'scaled to [0, 1]' : 'scaled to [-1, 1]';
    return `float32 (${s.c}, ${s.h}, ${s.w}), ${norm}`;
  }
  if (s.kind === 'tokens') return `int64 (${s.l},) token ids from the "${n.params.tokenizer}" tokenizer, padded / truncated`;
  if (s.kind === 'vector') return `float32 (${s.d},) standardised features`;
  return s.kind;
}

const METRICS_CODE: Record<string, string> = {
  accuracy: `class Accuracy:
    """Top-1 / top-5 accuracy for (B, C) logits, token accuracy for (B, L, C) logits (-100 ignored)."""

    def __init__(self) -> None:
        self.correct = self.correct5 = self.total = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        if logits.ndim == 3:
            mask = target != -100
            self.correct += (logits.argmax(-1)[mask] == target[mask]).sum().item()
            self.total += mask.sum().item()
            return
        k = min(5, logits.shape[1])
        top = logits.topk(k, dim=1).indices
        self.correct += (top[:, 0] == target).sum().item()
        self.correct5 += (top == target[:, None]).any(dim=1).sum().item()
        self.total += target.numel()

    def compute(self) -> dict[str, float]:
        out = {"acc": self.correct / max(1, self.total)}
        if self.correct5:
            out["top5"] = self.correct5 / max(1, self.total)
        return out`,
  multilabel: `class MultiLabelAccuracy:
    """Mean per-label accuracy at a 0.5 probability threshold."""

    def __init__(self) -> None:
        self.correct = self.total = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        self.correct += ((logits > 0).float() == target).sum().item()
        self.total += target.numel()

    def compute(self) -> dict[str, float]:
        return {"label_acc": self.correct / max(1, self.total)}`,
  miou: `class MeanIoU:
    """Mean intersection-over-union from a running confusion matrix."""

    def __init__(self, num_classes: int) -> None:
        self.n = max(2, num_classes)
        self.confusion = torch.zeros(self.n, self.n, dtype=torch.long)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        if logits.shape[1] == 1:
            pred, target = (logits[:, 0] > 0).long(), target[:, 0].long()
        else:
            pred = logits.argmax(1)
        idx = target.flatten().cpu() * self.n + pred.flatten().cpu()
        self.confusion += torch.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n)

    def compute(self) -> dict[str, float]:
        inter = self.confusion.diag().float()
        union = self.confusion.sum(0).float() + self.confusion.sum(1).float() - inter
        iou = inter / union.clamp_min(1)
        return {"miou": iou[union > 0].mean().item() if (union > 0).any() else 0.0}`,
  rmse: `class RMSE:
    """Root-mean-square error."""

    def __init__(self) -> None:
        self.sq = self.total = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        self.sq += ((pred.float() - target.float()) ** 2).sum().item()
        self.total += target.numel()

    def compute(self) -> dict[str, float]:
        return {"rmse": math.sqrt(self.sq / max(1, self.total))}`,
  depth: `class DepthMetrics:
    """Standard monocular-depth metrics over valid pixels: RMSE, AbsRel and δ < 1.25."""

    def __init__(self) -> None:
        self.sq = self.abs_rel = self.delta1 = self.total = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        valid = target > 0
        p, t = pred[valid].float().clamp_min(1e-6), target[valid].float()
        self.sq += ((p - t) ** 2).sum().item()
        self.abs_rel += ((p - t).abs() / t).sum().item()
        self.delta1 += (torch.maximum(p / t, t / p) < 1.25).sum().item()
        self.total += t.numel()

    def compute(self) -> dict[str, float]:
        n = max(1, self.total)
        return {"rmse": math.sqrt(self.sq / n), "abs_rel": self.abs_rel / n, "delta1": self.delta1 / n}`,
  map: `class DetectionMAP:
    """COCO-style mAP via torchmetrics (pip install torchmetrics pycocotools); skipped if unavailable."""

    def __init__(self) -> None:
        try:
            from torchmetrics.detection import MeanAveragePrecision

            self.metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        except Exception:  # torchmetrics / pycocotools missing
            self.metric = None

    def update(self, preds: list[dict[str, torch.Tensor]], targets: list[dict[str, torch.Tensor]]) -> None:
        if self.metric is not None:
            self.metric.update(
                [{k: v.detach().cpu() for k, v in p.items() if k in ("boxes", "scores", "labels")} for p in preds],
                [{k: v.detach().cpu() for k, v in t.items() if k in ("boxes", "labels")} for t in targets],
            )

    def compute(self) -> dict[str, float]:
        if self.metric is None:
            return {}
        result = self.metric.compute()
        return {"mAP": float(result["map"]), "mAP50": float(result["map_50"])}`,
};

function metricCtor(h: IRNode): string | undefined {
  switch (h.head!.metric) {
    case 'accuracy':
      return 'Accuracy()';
    case 'multilabel':
      return 'MultiLabelAccuracy()';
    case 'miou':
      return `MeanIoU(${h.params.classes})`;
    case 'rmse':
      return 'RMSE()';
    case 'depth':
      return 'DepthMetrics()';
    case 'map':
      return 'DetectionMAP()';
    default:
      return undefined;
  }
}

function realDataHint(ir: IR): string[] {
  const tasks = new Set(ir.heads.map((h) => h.head!.task));
  const img = ir.inputs.find((i) => i.type === 'image_input');
  const txt = ir.inputs.find((i) => i.type === 'text_input');
  const head = ir.heads[0];
  const lines: string[] = [];
  if (img && (tasks.has('classification') || tasks.has('multilabel'))) {
    const s = img.out as { h: number; w: number };
    lines.push(
      'Image classification from class folders (data/train/<class>/*.jpg):',
      '',
      '    from torchvision import datasets, transforms',
      '    tf = transforms.Compose([',
      `        transforms.RandomResizedCrop((${s.h}, ${s.w})),`,
      '        transforms.RandomHorizontalFlip(),',
      '        transforms.ToTensor(),',
      '        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),',
      '    ])',
      `    train = DictDataset(datasets.ImageFolder("data/train", tf), ${py(img.name)}, ${py(head.name)})`,
    );
  }
  if (img && tasks.has('segmentation')) {
    lines.push(
      'Segmentation: pair each image with an integer mask PNG of the same size',
      '(pixel value = class id; for a binary mask use 0/1 and return it as float (1, H, W)),',
      'e.g. torchvision.datasets.VOCSegmentation("data", image_set="train", download=True).',
    );
  }
  if (img && (tasks.has('detection') || tasks.has('instance'))) {
    lines.push(
      'Detection from COCO json (pip install pycocotools):',
      '',
      '    from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2',
      '    coco = wrap_dataset_for_transforms_v2(CocoDetection("images/", "instances.json"))',
      '    # convert each sample to ({"image": img}, {head: {"boxes": xyxy, "labels": 0..C-1}})',
      '    # and resize images to the declared input size, scaling boxes accordingly.',
    );
  }
  if (txt) {
    lines.push(
      'Text with Hugging Face datasets + tokenizers:',
      '',
      '    from datasets import load_dataset',
      '    from transformers import AutoTokenizer',
      `    tok = AutoTokenizer.from_pretrained(${py(txt.params.tokenizer)})`,
      `    enc = lambda b: tok(b["text"], padding="max_length", truncation=True, max_length=${txt.params.seqLen})`,
      '    ds = load_dataset("imdb").map(enc, batched=True)',
    );
  }
  if (tasks.has('dense')) {
    lines.push('Dense regression: pair each image with a float target map of the same size (e.g. depth in metres, 0 = invalid).');
  }
  return lines;
}

export function generateTorchTrain(ir: IR, hyper: TrainHyper = defaultHyper(ir)): GeneratedFile {
  const src = new Source();
  const internal = !!ir.wrapper;
  const heads = ir.heads;
  const metricKinds = [...new Set(heads.map((h) => h.head!.metric))].filter((m) => METRICS_CODE[m]);
  const lossImports = [...new Set(heads.flatMap((h) => helperRefs(h.head!.torchLoss)))];
  const postImports = [...new Set(heads.map((h) => h.head!.postprocess).filter((x): x is string => !!x))];
  const hasDetection = heads.some((h) => h.head!.metric === 'map');
  const modelImports = ['CONFIG', 'build_model', 'example_inputs', ...lossImports, ...postImports].sort((a, b) => a.localeCompare(b));

  src.add(`"""Train ${ir.name} (generated by Netloom).

    python train.py                              # synthetic data: runs immediately
    python train.py --epochs 50 --batch-size 64
    python train.py --resume checkpoints/last.pt
    python train.py --epochs 1 --max-steps 2     # quick check

Replace SyntheticDataset in build_datasets() with real data; the docstrings give the exact format.
"""`);
  src.blank();
  src.add('from __future__ import annotations');
  src.blank();
  src.add('import argparse');
  src.add('import copy');
  src.add("import math");
  src.add('import os');
  src.add('import time');
  src.blank();
  src.add('import torch');
  src.add('from torch import nn');
  src.add('from torch.utils.data import DataLoader, Dataset, default_collate');
  src.blank();
  src.add(`from model import ${modelImports.join(', ')}`);
  src.blank(2);

  // ------------------------------------------------------------- hyper-parameters
  src.add('# ' + '-'.repeat(75));
  src.add('# Hyper-parameters');
  src.add('# ' + '-'.repeat(75));
  src.add(`EPOCHS = ${hyper.epochs}`);
  src.add(`BATCH_SIZE = ${hyper.batch}`);
  src.add(`LR = ${hyper.lr}${ir.pretrained ? '  # low: the backbone is pretrained' : ir.hasTransformer ? '  # transformers prefer a smaller LR' : ''}`);
  src.add(`WEIGHT_DECAY = ${hyper.weightDecay}`);
  src.add(`WARMUP_EPOCHS = ${hyper.warmupEpochs}`);
  src.add('GRAD_CLIP = 1.0  # max gradient norm, 0 disables');
  src.add('USE_EMA = True  # exponential moving average of the weights for evaluation');
  src.add('EMA_DECAY = 0.999');
  src.add('NUM_WORKERS = 0');
  src.add('OUT_DIR = "checkpoints"');
  if (!internal) src.add(`LOSS_WEIGHTS = ${pyDict(Object.fromEntries(heads.map((h) => [h.name, 1.0])), 0).replace(/: 1,/g, ': 1.0,')}`);
  src.add('DEVICE = "cuda" if torch.cuda.is_available() else "cpu"');
  src.add('USE_AMP = DEVICE == "cuda"');
  src.blank(2);

  // ------------------------------------------------------------- data
  src.add('# ' + '-'.repeat(75));
  src.add('# Data');
  src.add('# ' + '-'.repeat(75));
  src.blank(2);
  if (hasDetection) {
    src.add(`def random_detection_target(height: int, width: int, num_classes: int) -> dict[str, torch.Tensor]:
    """1–5 random boxes (xyxy, pixels) with labels in [0, num_classes - 1]."""
    n = int(torch.randint(1, 6, ()).item())
    xy = torch.rand(n, 2) * torch.tensor([width * 0.7, height * 0.7])
    wh = torch.rand(n, 2) * torch.tensor([width * 0.25, height * 0.25]) + 8
    boxes = torch.cat([xy, (xy + wh).minimum(torch.tensor([width - 1.0, height - 1.0]))], dim=1)
    return {"boxes": boxes, "labels": torch.randint(0, num_classes, (n,))}`);
    src.blank(2);
  }

  const docLines = [
    ...ir.inputs.map((i) => `    inputs[${py(i.name)}]: ${inputDoc(i)}`),
    ...heads.map((h) => `    targets[${py(h.name)}]: ${h.head!.targetDoc}`),
  ].join('\n');
  src.add(`class SyntheticDataset(Dataset):
    """Random samples with exactly the shapes the model and losses expect.

    Real data must yield the same (inputs, targets) structure:
${docLines}
    """

    def __init__(self, length: int = 256) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[dict, dict]:`);
  src.add('inputs = {', 2);
  for (const i of ir.inputs) src.add(`${py(i.name)}: ${sampleInput(i)},`, 3);
  src.add('}', 2);
  src.add('targets = {', 2);
  for (const h of heads) src.add(`${py(h.name)}: ${h.head!.torchTarget},`, 3);
  src.add('}', 2);
  src.add('return inputs, targets', 2);
  src.blank(2);

  src.add(`class DictDataset(Dataset):
    """Adapt a classic (x, y) dataset to the ({input: x}, {head: y}) format used here."""

    def __init__(self, base: Dataset, input_name: str, head_name: str) -> None:
        self.base, self.input_name, self.head_name = base, input_name, head_name

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[dict, dict]:
        x, y = self.base[index]
        return {self.input_name: x}, {self.head_name: torch.as_tensor(y)}`);
  src.blank(2);

  src.add(`def collate(batch: list[tuple[dict, dict]]) -> tuple[dict, dict]:
    """Stack inputs; stack equal-shape targets, keep variable-size ones (e.g. boxes) as lists."""
    inputs = default_collate([b[0] for b in batch])
    targets = {}
    for key in batch[0][1]:
        values = [b[1][key] for b in batch]
        same = all(isinstance(v, torch.Tensor) for v in values) and len({tuple(v.shape) for v in values}) == 1
        targets[key] = torch.stack(values) if same else values
    return inputs, targets`);
  src.blank(2);

  const hint = realDataHint(ir);
  src.add(`def build_datasets() -> tuple[Dataset, Dataset]:
    """Return (train, val) datasets yielding (inputs, targets) dicts.${hint.length ? '\n\n    ' + hint.join('\n    ').replace(/\n {4}\n/g, '\n\n') : ''}
    """
    return SyntheticDataset(256), SyntheticDataset(64)`);
  src.blank(2);

  // ------------------------------------------------------------- losses / metrics
  src.add('# ' + '-'.repeat(75));
  src.add('# Losses, metrics, optimisation');
  src.add('# ' + '-'.repeat(75));
  src.blank(2);
  if (!internal) {
    src.add('def build_criteria() -> nn.ModuleDict:');
    src.add('"""One loss per head. Losses with parameters (CRF, ArcFace) are trained too."""', 1);
    src.add('return nn.ModuleDict(', 1);
    src.add('{', 2);
    for (const h of heads) src.add(`${py(h.name)}: ${h.head!.torchLoss},`, 3);
    src.add('}', 2);
    src.add(')', 1);
    src.blank(2);
  }

  for (const m of metricKinds) {
    src.add(METRICS_CODE[m]);
    src.blank(2);
  }
  src.add('def build_metrics() -> dict:');
  const metricEntries = heads.map((h) => [h.name, metricCtor(h)] as const).filter(([, c]) => c);
  if (metricEntries.length) {
    src.add('return {', 1);
    for (const [n, c] of metricEntries) src.add(`${py(n)}: ${c},`, 2);
    src.add('}', 1);
  } else src.add('return {}', 1);
  src.blank(2);

  src.add(`def param_groups(module: nn.Module, weight_decay: float) -> list[dict]:
    """No weight decay on biases and normalisation / scale parameters."""
    decay, no_decay = [], []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return [g for g in groups if g["params"]]


def cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    """Linear warm-up then cosine decay to zero (LR multiplier)."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class ModelEMA:
    """Exponential moving average of model weights (evaluated instead of the raw model)."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_v, v in zip(self.module.state_dict().values(), model.state_dict().values()):
            if v.dtype.is_floating_point:
                ema_v.lerp_(v, 1.0 - self.decay)
            else:
                ema_v.copy_(v)


def to_device(x, device: str):
    """Move tensors inside nested dicts / lists to a device."""
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    return x`);
  src.blank(2);

  if (internal) {
    const hn = py(ir.wrapper!.name);
    src.add(`def compute_loss(model: nn.Module, inputs: dict, targets: dict) -> tuple[torch.Tensor, dict]:
    """torchvision detectors return their own loss dict in train mode."""
    losses = model(**inputs, targets=targets[${hn}])
    losses = {k: v.float() for k, v in losses.items()}
    return sum(losses.values()), losses`);
  } else {
    src.add(`def compute_loss(model: nn.Module, inputs: dict, targets: dict, criteria: nn.ModuleDict) -> tuple[torch.Tensor, dict]:
    """Weighted sum of per-head losses (computed in float32)."""
    outputs = model(**inputs)
    if not isinstance(outputs, dict):
        outputs = {${py(heads[0].name)}: outputs}
    losses = {name: criteria[name](outputs[name], targets[name]) for name in criteria}
    return sum(LOSS_WEIGHTS[name] * loss for name, loss in losses.items()), losses`);
  }
  src.blank(2);

  // ------------------------------------------------------------- train / eval
  const crit = internal ? '' : ', criteria';
  src.add(`def train_one_epoch(model, loader, optimizer, scheduler, scaler, ema${internal ? '' : ', criteria'}, max_steps: int, step: int) -> int:
    model.train()
    start = time.time()
    for inputs, targets in loader:
        inputs, targets = to_device(inputs, DEVICE), to_device(targets, DEVICE)
        with torch.autocast(device_type=DEVICE, enabled=USE_AMP):
            loss, parts = compute_loss(model, inputs, targets${crit})
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if GRAD_CLIP > 0:
            scaler.unscale_(optimizer)
            params = [p for g in optimizer.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        step += 1
        if step % 20 == 0 or step == 1:
            detail = " ".join(f"{k}={v.item():.4f}" for k, v in parts.items())
            print(f"  step {step:>6}  loss={loss.item():.4f}  {detail}  ({time.time() - start:.1f}s)")
        if max_steps and step >= max_steps:
            break
    return step`);
  src.blank(2);

  src.add('@torch.no_grad()');
  if (internal) {
    const hn = py(ir.wrapper!.name);
    src.add(`def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    """Detection mAP (losses are only available in train mode for torchvision detectors)."""
    model.eval()
    metrics = build_metrics()
    for inputs, targets in loader:
        inputs, targets = to_device(inputs, DEVICE), to_device(targets, DEVICE)
        preds = model(**inputs)
        for metric in metrics.values():
            metric.update(preds, targets[${hn}])
    results: dict[str, float] = {}
    for name, metric in metrics.items():
        results.update({f"{name}/{k}": v for k, v in metric.compute().items()})
    return results`);
  } else {
    const post = heads.filter((h) => h.head!.postprocess);
    src.add(`def evaluate(model: nn.Module, loader: DataLoader, criteria: nn.ModuleDict) -> dict[str, float]:
    model.eval()
    metrics = build_metrics()
    total, batches = 0.0, 0
    for inputs, targets in loader:
        inputs, targets = to_device(inputs, DEVICE), to_device(targets, DEVICE)
        outputs = model(**inputs)
        if not isinstance(outputs, dict):
            outputs = {${py(heads[0].name)}: outputs}
        total += sum(LOSS_WEIGHTS[n] * criteria[n](outputs[n], targets[n]).item() for n in criteria)
        batches += 1`);
    for (const h of post) {
      const fn = h.head!.postprocess!;
      if (fn === 'crf_decode') src.add(`outputs[${py(h.name)}] = crf_decode(outputs[${py(h.name)}], criteria[${py(h.name)}])`, 2);
      else src.add(`outputs[${py(h.name)}] = ${fn}(outputs[${py(h.name)}])`, 2);
    }
    src.addCont(`for name, metric in metrics.items():
            metric.update(outputs[name], targets[name])
    results = {"val_loss": total / max(1, batches)}
    for name, metric in metrics.items():
        results.update({f"{name}/{k}": v for k, v in metric.compute().items()})
    return results`, 2);
  }
  src.blank(2);

  // ------------------------------------------------------------- export
  src.add(`def export(model: nn.Module, out_dir: str) -> None:
    """Export TorchScript and ONNX (opset 17, dynamic batch)."""
    model = model.eval().cpu()
    example = example_inputs(batch_size=2)
    names = list(example)
    try:
        traced = torch.jit.trace(model, example_kwarg_inputs=example, strict=False)
        traced.save(os.path.join(out_dir, "model.torchscript.pt"))
        print(f"saved {out_dir}/model.torchscript.pt")
    except Exception as exc:  # data-dependent control flow (e.g. NMS) can't be traced
        print(f"TorchScript export skipped: {exc}")
    try:
        torch.onnx.export(
            model,
            (),
            os.path.join(out_dir, "model.onnx"),
            kwargs=example,
            input_names=names,
            dynamic_axes={name: {0: "batch"} for name in names},
            opset_version=17,
            dynamo=False,
        )
        print(f"saved {out_dir}/model.onnx")
    except Exception as exc:
        print(f"ONNX export skipped: {exc}")`);
  src.blank(2);

  // ------------------------------------------------------------- main
  const optim = internal
    ? '    optimizer = torch.optim.AdamW(param_groups(model, WEIGHT_DECAY), lr=args.lr)'
    : `    criteria = build_criteria().to(DEVICE)
    groups = param_groups(model, WEIGHT_DECAY) + param_groups(criteria, WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(groups, lr=args.lr)`;
  const loadCrit = internal ? '' : '\n        criteria.load_state_dict(ckpt["criteria"])';
  const saveCrit = internal ? '' : '\n            "criteria": criteria.state_dict(),';
  src.add(`def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (0 = full run)")
    parser.add_argument("--resume", default="", help="checkpoint to resume from")
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    train_set, val_set = build_datasets()
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate, drop_last=True
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=NUM_WORKERS, collate_fn=collate)

    model = build_model().to(DEVICE)
${optim}
    total_steps = args.epochs * len(train_loader)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = min(WARMUP_EPOCHS * len(train_loader), max(1, total_steps // 10))
    schedule = lambda s: cosine_with_warmup(s, total_steps, warmup)  # noqa: E731
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    ema = ModelEMA(model, EMA_DECAY) if USE_EMA else None

    start_epoch, step, best = 0, 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])${loadCrit}
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if ema is not None and ckpt.get("ema"):
            ema.module.load_state_dict(ckpt["ema"])
        start_epoch, step, best = ckpt["epoch"] + 1, ckpt["step"], ckpt["best"]
        print(f"resumed from {args.resume} (epoch {start_epoch})")

    print(f"training on {DEVICE} for {args.epochs} epochs ({len(train_loader)} steps/epoch)")
    for epoch in range(start_epoch, args.epochs):
        print(f"epoch {epoch + 1}/{args.epochs}")
        args_ = (model, train_loader, optimizer, scheduler, scaler, ema${crit})
        step = train_one_epoch(*args_, max_steps=args.max_steps, step=step)
        eval_model = ema.module if ema is not None else model
        results = evaluate(eval_model, val_loader${crit})
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in results.items()))
        score = results.get("val_loss", -sum(results.values()))
        best = min(best, score)
        state = {
            "model": model.state_dict(),${saveCrit}
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema": ema.module.state_dict() if ema is not None else None,
            "epoch": epoch,
            "step": step,
            "best": best,
            "config": CONFIG,
        }
        torch.save(state, os.path.join(args.out, "last.pt"))
        if score <= best:
            torch.save(state, os.path.join(args.out, "best.pt"))
            print(f"  new best checkpoint ({score:.4f})")
        if args.max_steps and step >= args.max_steps:
            break

    if not args.no_export:
        export(ema.module if ema is not None else model, args.out)


if __name__ == "__main__":
    main()`);
  return src.file('train.py');
}
