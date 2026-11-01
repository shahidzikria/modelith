import { dedent } from '../py';
import { defineTorchHelpers } from './registry';

defineTorchHelpers({
  // ------------------------------------------------------------------ YOLO
  make_anchors: {
    code: dedent(`
      def make_anchors(
          feat_sizes: list[tuple[int, int]], strides: list[int], device: torch.device | None = None, offset: float = 0.5
      ) -> tuple[torch.Tensor, torch.Tensor]:
          """Anchor points at cell centres (grid units, (N, 2) as x, y) and the stride of each point (N, 1)."""
          points, stride_t = [], []
          for (h, w), s in zip(feat_sizes, strides):
              ys = torch.arange(h, device=device, dtype=torch.float32) + offset
              xs = torch.arange(w, device=device, dtype=torch.float32) + offset
              yy, xx = torch.meshgrid(ys, xs, indexing="ij")
              points.append(torch.stack([xx, yy], dim=-1).reshape(-1, 2))
              stride_t.append(torch.full((h * w, 1), float(s), device=device))
          return torch.cat(points), torch.cat(stride_t)
    `),
  },

  YOLOHead: {
    deps: ['ConvBNAct'],
    code: dedent(`
      class YOLOHead(nn.Module):
          """Anchor-free decoupled YOLOv8 head with distribution-focal-loss box regression.

          forward(feats) returns raw predictions (B, 4·reg_max + num_classes, N) concatenated over levels
          (N = Σ h·w). Decode with yolo_postprocess(); train with YOLOLoss. Reference: Ultralytics YOLOv8.
          """

          def __init__(self, in_channels: list[int], num_classes: int, strides: list[int], reg_max: int = 16) -> None:
              super().__init__()
              self.nc, self.reg_max, self.strides = num_classes, reg_max, list(strides)
              c2 = max(16, in_channels[0] // 4, 4 * reg_max)
              c3 = max(in_channels[0], min(num_classes, 100))
              self.box = nn.ModuleList(
                  nn.Sequential(ConvBNAct(c, c2, 3), ConvBNAct(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1)) for c in in_channels
              )
              self.cls = nn.ModuleList(
                  nn.Sequential(ConvBNAct(c, c3, 3), ConvBNAct(c3, c3, 3), nn.Conv2d(c3, num_classes, 1)) for c in in_channels
              )
              self.init_bias()

          def init_bias(self) -> None:
              """Box bias 1.0; class bias log(5 / nc / (640 / stride)²) — a low initial objectness prior."""
              for box, cls, s in zip(self.box, self.cls, self.strides):
                  box[-1].bias.data[:] = 1.0
                  cls[-1].bias.data[:] = math.log(5 / self.nc / (640 / s) ** 2)

          def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
              outs = [torch.cat([box(f), cls(f)], dim=1).flatten(2) for f, box, cls in zip(feats, self.box, self.cls)]
              return torch.cat(outs, dim=2)
    `),
  },

  yolo_postprocess: {
    section: 'post',
    deps: ['make_anchors'],
    imports: ['import torchvision'],
    code: dedent(`
      @torch.no_grad()
      def yolo_postprocess(
          preds: torch.Tensor,
          num_classes: int,
          strides: list[int],
          feat_sizes: list[tuple[int, int]],
          reg_max: int = 16,
          conf: float = 0.25,
          iou: float = 0.7,
          max_det: int = 300,
      ) -> list[dict[str, torch.Tensor]]:
          """Decode raw YOLO predictions: DFL expectation → ltrb distances → xyxy pixels, then class-wise NMS."""
          preds = preds.float().transpose(1, 2)
          box, cls = preds.split([4 * reg_max, num_classes], dim=-1)
          points, stride = make_anchors(feat_sizes, strides, preds.device)
          proj = torch.arange(reg_max, device=preds.device, dtype=torch.float32)
          ltrb = (box.reshape(*box.shape[:2], 4, reg_max).softmax(-1) * proj).sum(-1)
          boxes = torch.cat([points - ltrb[..., :2], points + ltrb[..., 2:]], dim=-1) * stride
          scores = cls.sigmoid()
          results = []
          for b in range(preds.shape[0]):
              score, label = scores[b].max(dim=-1)
              keep = score > conf
              bx, score, label = boxes[b][keep], score[keep], label[keep]
              idx = torchvision.ops.batched_nms(bx, score, label, iou)[:max_det]
              results.append({"boxes": bx[idx], "scores": score[idx], "labels": label[idx]})
          return results
    `),
  },

  YOLOLoss: {
    section: 'loss',
    deps: ['make_anchors'],
    imports: ['import torchvision'],
    code: dedent(`
      class YOLOLoss(nn.Module):
          """YOLOv8 loss: task-aligned assignment, BCE on IoU-aware soft targets, GIoU and distribution focal loss.

          Assignment: top-k (10) anchors inside each box ranked by score^0.5 · IoU^6; anchors claimed by several
          boxes keep the highest-IoU box. Weights: 7.5 box, 0.5 cls, 1.5 dfl.
          targets: list (per image) of {"boxes": (M, 4) xyxy pixels, "labels": (M,) int64 in [0, C-1]}.
          """

          def __init__(
              self,
              num_classes: int,
              strides: list[int],
              feat_sizes: list[tuple[int, int]],
              reg_max: int = 16,
              topk: int = 10,
              alpha: float = 0.5,
              beta: float = 6.0,
          ) -> None:
              super().__init__()
              self.nc, self.reg_max, self.topk, self.alpha, self.beta = num_classes, reg_max, topk, alpha, beta
              points, stride = make_anchors(feat_sizes, strides)
              self.register_buffer("points", points, persistent=False)
              self.register_buffer("stride", stride, persistent=False)
              self.register_buffer("proj", torch.arange(reg_max, dtype=torch.float32), persistent=False)

          def forward(self, preds: torch.Tensor, targets: list[dict[str, torch.Tensor]]) -> torch.Tensor:
              preds = preds.float().transpose(1, 2)
              box_logits, cls_logits = preds.split([4 * self.reg_max, self.nc], dim=-1)
              b, n, _ = preds.shape
              dist = box_logits.reshape(b, n, 4, self.reg_max)
              ltrb = (dist.softmax(-1) * self.proj).sum(-1)
              pred_boxes = torch.cat([self.points - ltrb[..., :2], self.points + ltrb[..., 2:]], dim=-1) * self.stride
              anchors = self.points * self.stride
              loss_box = loss_cls = loss_dfl = preds.new_zeros(())
              for i, t in enumerate(targets):
                  scores_t, boxes_t, fg = self.assign(
                      cls_logits[i].detach().sigmoid(), pred_boxes[i].detach(), anchors, t["labels"], t["boxes"].float()
                  )
                  norm = scores_t.sum().clamp_min(1.0)
                  loss_cls = loss_cls + F.binary_cross_entropy_with_logits(cls_logits[i], scores_t, reduction="sum") / norm
                  if fg.any():
                      weight = scores_t[fg].sum(-1)
                      giou = torchvision.ops.generalized_box_iou_loss(pred_boxes[i][fg], boxes_t[fg], reduction="none")
                      loss_box = loss_box + (giou * weight).sum() / norm
                      s, p = self.stride[fg], self.points[fg]
                      target_ltrb = torch.cat([p - boxes_t[fg, :2] / s, boxes_t[fg, 2:] / s - p], dim=-1)
                      target_ltrb = target_ltrb.clamp(0, self.reg_max - 1.01)
                      loss_dfl = loss_dfl + (self.dfl(dist[i][fg], target_ltrb) * weight).sum() / norm
              return (7.5 * loss_box + 0.5 * loss_cls + 1.5 * loss_dfl) / len(targets)

          def dfl(self, dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
              """Distribution focal loss: CE to the two bins around each continuous target distance."""
              left = target.long()
              right = left + 1
              w_left = right.float() - target
              logits = dist.reshape(-1, self.reg_max)
              ce_left = F.cross_entropy(logits, left.reshape(-1), reduction="none").view(left.shape)
              ce_right = F.cross_entropy(logits, right.reshape(-1), reduction="none").view(left.shape)
              return (ce_left * w_left + ce_right * (1 - w_left)).mean(-1)

          @torch.no_grad()
          def assign(
              self, scores: torch.Tensor, boxes: torch.Tensor, anchors: torch.Tensor, gt_labels: torch.Tensor, gt_boxes: torch.Tensor
          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
              """Task-aligned assigner → soft class targets (N, C), target boxes (N, 4), foreground mask (N,)."""
              n = anchors.shape[0]
              if gt_boxes.numel() == 0:
                  return scores.new_zeros(n, self.nc), boxes.new_zeros(n, 4), torch.zeros(n, dtype=torch.bool, device=scores.device)
              lt = anchors[None] - gt_boxes[:, None, :2]
              rb = gt_boxes[:, None, 2:] - anchors[None]
              inside = torch.cat([lt, rb], dim=-1).amin(-1) > 1e-9
              iou = torchvision.ops.box_iou(gt_boxes, boxes).clamp_min(0)
              metric = scores[:, gt_labels].t().pow(self.alpha) * iou.pow(self.beta) * inside
              top = metric.topk(min(self.topk, n), dim=1).indices
              candidate = torch.zeros_like(inside).scatter_(1, top, True) & inside
              multi = candidate.sum(0) > 1
              if multi.any():
                  best = torch.zeros_like(candidate).scatter_(0, iou.argmax(0, keepdim=True), True)
                  candidate = torch.where(multi[None], best, candidate)
              fg = candidate.any(0)
              gt_idx = candidate.float().argmax(0)
              metric = metric * candidate
              norm = metric * (iou * candidate).amax(1, keepdim=True) / (metric.amax(1, keepdim=True) + 1e-9)
              target_scores = F.one_hot(gt_labels[gt_idx], self.nc).float() * (norm.amax(0) * fg)[:, None]
              return target_scores, gt_boxes[gt_idx], fg
    `),
  },

  // ------------------------------------------------------------------- SSD
  SSDHead: {
    code: dedent(`
      class SSDHead(nn.Module):
          """SSD multibox head: per level 3×3 convs for A·(C + 1) class logits and A·4 box offsets.

          Returns (B, N, 4 + C + 1) with offsets first; class 0 is background inside the head.
          Paper: SSD, Liu et al. 2016 (arXiv:1512.02325).
          """

          def __init__(self, in_channels: list[int], num_classes: int, num_anchors: int = 4) -> None:
              super().__init__()
              self.nc = num_classes
              self.cls = nn.ModuleList(nn.Conv2d(c, num_anchors * (num_classes + 1), 3, padding=1) for c in in_channels)
              self.reg = nn.ModuleList(nn.Conv2d(c, num_anchors * 4, 3, padding=1) for c in in_channels)

          def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
              cls, reg = [], []
              for f, c, r in zip(feats, self.cls, self.reg):
                  b = f.shape[0]
                  cls.append(c(f).permute(0, 2, 3, 1).reshape(b, -1, self.nc + 1))
                  reg.append(r(f).permute(0, 2, 3, 1).reshape(b, -1, 4))
              return torch.cat([torch.cat(reg, dim=1), torch.cat(cls, dim=1)], dim=-1)
    `),
  },

  ssd_boxes: {
    code: dedent(`
      def ssd_default_boxes(
          feat_sizes: list[tuple[int, int]],
          image_size: tuple[int, int],
          ratios: tuple[float, ...] = (1.0, 2.0, 0.5),
          smin: float = 0.1,
          smax: float = 0.9,
      ) -> torch.Tensor:
          """Default boxes (cx, cy, w, h in pixels), ordered like SSDHead outputs: level, y, x, anchor."""
          height, width = image_size
          levels = len(feat_sizes)
          scales = [smin + (smax - smin) * k / max(1, levels - 1) for k in range(levels)] + [1.0]
          boxes = []
          for k, (h, w) in enumerate(feat_sizes):
              s, s_next = scales[k], scales[k + 1]
              whs = [(s * math.sqrt(r), s / math.sqrt(r)) for r in ratios] + [(math.sqrt(s * s_next),) * 2]
              yy, xx = torch.meshgrid((torch.arange(h) + 0.5) / h, (torch.arange(w) + 0.5) / w, indexing="ij")
              centers = torch.stack([xx, yy], dim=-1).reshape(-1, 1, 2).expand(-1, len(whs), 2)
              sizes = torch.tensor(whs).expand(h * w, -1, -1)
              boxes.append(torch.cat([centers, sizes], dim=-1).reshape(-1, 4))
          return torch.cat(boxes) * torch.tensor([width, height, width, height], dtype=torch.float32)


      def ssd_encode(boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
          """cxcywh boxes → offsets relative to anchors (variances 0.1 / 0.2)."""
          xy = (boxes[:, :2] - anchors[:, :2]) / anchors[:, 2:] / 0.1
          wh = torch.log(boxes[:, 2:] / anchors[:, 2:]) / 0.2
          return torch.cat([xy, wh], dim=-1)


      def ssd_decode(offsets: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
          """Offsets → xyxy boxes in pixels."""
          xy = offsets[..., :2] * 0.1 * anchors[:, 2:] + anchors[:, :2]
          wh = torch.exp(offsets[..., 2:] * 0.2) * anchors[:, 2:]
          return torch.cat([xy - wh / 2, xy + wh / 2], dim=-1)
    `),
  },

  ssd_postprocess: {
    section: 'post',
    deps: ['ssd_boxes'],
    imports: ['import torchvision'],
    code: dedent(`
      @torch.no_grad()
      def ssd_postprocess(
          preds: torch.Tensor,
          feat_sizes: list[tuple[int, int]],
          image_size: tuple[int, int],
          conf: float = 0.01,
          iou: float = 0.45,
          max_det: int = 200,
      ) -> list[dict[str, torch.Tensor]]:
          """Softmax scores (background dropped), decode offsets, class-wise NMS; labels are 0-based."""
          preds = preds.float()
          anchors = ssd_default_boxes(feat_sizes, image_size).to(preds.device)
          boxes = ssd_decode(preds[..., :4], anchors)
          scores = preds[..., 4:].softmax(-1)[..., 1:]
          results = []
          for b in range(preds.shape[0]):
              anchor_idx, label = (scores[b] > conf).nonzero(as_tuple=True)
              bx, score = boxes[b][anchor_idx], scores[b][anchor_idx, label]
              keep = torchvision.ops.batched_nms(bx, score, label, iou)[:max_det]
              results.append({"boxes": bx[keep], "scores": score[keep], "labels": label[keep]})
          return results
    `),
  },

  SSDLoss: {
    section: 'loss',
    deps: ['ssd_boxes'],
    imports: ['import torchvision'],
    code: dedent(`
      class SSDLoss(nn.Module):
          """Multibox loss: IoU ≥ 0.5 matching (plus the best anchor of every box), smooth-L1 on offsets,
          cross-entropy with 3:1 hard-negative mining. Labels shift +1 internally (0 = background)."""

          def __init__(self, feat_sizes: list[tuple[int, int]], image_size: tuple[int, int], neg_ratio: float = 3.0, iou: float = 0.5) -> None:
              super().__init__()
              anchors = ssd_default_boxes(feat_sizes, image_size)
              self.register_buffer("anchors", anchors, persistent=False)
              self.register_buffer("anchors_xyxy", torchvision.ops.box_convert(anchors, "cxcywh", "xyxy"), persistent=False)
              self.neg_ratio, self.iou = neg_ratio, iou

          def match(self, boxes: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
              n = self.anchors.shape[0]
              if boxes.numel() == 0:
                  return torch.zeros(n, dtype=torch.long, device=boxes.device), self.anchors.new_zeros(n, 4)
              iou = torchvision.ops.box_iou(boxes, self.anchors_xyxy)
              best_iou, best_gt = iou.max(dim=0)
              best_anchor = iou.argmax(dim=1)
              best_gt[best_anchor] = torch.arange(len(boxes), device=boxes.device)
              best_iou[best_anchor] = 1.0
              target_labels = labels[best_gt] + 1
              target_labels[best_iou < self.iou] = 0
              matched = torchvision.ops.box_convert(boxes[best_gt], "xyxy", "cxcywh")
              return target_labels, ssd_encode(matched, self.anchors)

          def forward(self, preds: torch.Tensor, targets: list[dict[str, torch.Tensor]]) -> torch.Tensor:
              preds = preds.float()
              total = preds.new_zeros(())
              for i, t in enumerate(targets):
                  labels, offsets = self.match(t["boxes"].float(), t["labels"])
                  pos = labels > 0
                  num_pos = pos.sum().clamp_min(1)
                  ce = F.cross_entropy(preds[i, :, 4:], labels, reduction="none")
                  neg_ce = ce.masked_fill(pos, 0.0)
                  k = int(min(self.neg_ratio * num_pos.item(), (~pos).sum().item()))
                  hard = neg_ce.topk(k).indices
                  loss_cls = (ce[pos].sum() + ce[hard].sum()) / num_pos
                  loss_box = F.smooth_l1_loss(preds[i, pos, :4], offsets[pos], reduction="sum") / num_pos
                  total = total + loss_cls + loss_box
              return total / len(targets)
    `),
  },

  // ----------------------------------------------------------- torchvision
  TorchvisionDetector: {
    imports: ['import torchvision', 'collections'],
    code: dedent(`
      class TorchvisionDetector(nn.Module):
          """Adapts a torchvision detector to batched image tensors and 0-based labels.

          torchvision reserves label 0 for background (num_classes = C + 1), so labels are shifted +1 for
          training and −1 in the predictions. Train mode returns a loss dict, eval mode a list of detections.
          """

          def __init__(self, detector: nn.Module) -> None:
              super().__init__()
              self.detector = detector

          def forward(self, images: torch.Tensor, targets: list[dict[str, torch.Tensor]] | None = None):
              if targets is not None:
                  targets = [{**t, "labels": t["labels"] + 1} for t in targets]
              out = self.detector(list(images), targets)
              if self.training:
                  return out
              return [{**o, "labels": o["labels"] - 1} for o in out]
    `),
  },

  // ------------------------------------------------------------------ DETR
  sine_position_2d: {
    code: dedent(`
      def sine_position_2d(h: int, w: int, dim: int, device: torch.device, temperature: float = 10000.0) -> torch.Tensor:
          """DETR-style 2-D sine / cosine position encoding, (h·w, dim)."""
          d = dim // 4
          omega = 1.0 / temperature ** (torch.arange(d, device=device, dtype=torch.float32) / d)
          yy, xx = torch.meshgrid(
              torch.arange(h, device=device, dtype=torch.float32), torch.arange(w, device=device, dtype=torch.float32), indexing="ij"
          )
          y = yy.flatten()[:, None] * omega[None]
          x = xx.flatten()[:, None] * omega[None]
          pos = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)
          return F.pad(pos, (0, dim - pos.shape[1]))
    `),
  },

  DETRHead: {
    deps: ['sine_position_2d'],
    code: dedent(`
      class DETRHead(nn.Module):
          """DETR head: 1×1 projection, 2-D sine positions, transformer encoder–decoder over learned queries,
          then class (C + 1, "no object" last) and box (normalised cx, cy, w, h) predictions.

          Returns (B, queries, C + 1 + 4). Paper: End-to-End Object Detection with Transformers,
          Carion et al. 2020 (arXiv:2005.12872). Simplification: positions are added once at the input.
          """

          def __init__(
              self,
              c_in: int,
              num_classes: int,
              dim: int = 256,
              heads: int = 8,
              enc_layers: int = 6,
              dec_layers: int = 6,
              queries: int = 100,
              ffn: int = 2048,
              dropout: float = 0.1,
          ) -> None:
              super().__init__()
              self.proj = nn.Conv2d(c_in, dim, 1)
              self.transformer = nn.Transformer(dim, heads, enc_layers, dec_layers, ffn, dropout, batch_first=True)
              self.query = nn.Embedding(queries, dim)
              self.cls = nn.Linear(dim, num_classes + 1)
              self.box = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 4))

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              if isinstance(x, (list, tuple)):
                  x = x[-1]
              x = self.proj(x)
              b, d, h, w = x.shape
              src = x.flatten(2).transpose(1, 2) + sine_position_2d(h, w, d, x.device)
              tgt = self.query.weight.unsqueeze(0).expand(b, -1, -1)
              hs = self.transformer(src, tgt)
              return torch.cat([self.cls(hs), self.box(hs).sigmoid()], dim=-1)
    `),
  },

  detr_postprocess: {
    section: 'post',
    imports: ['import torchvision'],
    code: dedent(`
      @torch.no_grad()
      def detr_postprocess(preds: torch.Tensor, image_size: tuple[int, int], conf: float = 0.5) -> list[dict[str, torch.Tensor]]:
          """Softmax over classes (dropping "no object"), cxcywh → xyxy pixels, confidence threshold."""
          preds = preds.float()
          h, w = image_size
          prob = preds[..., :-4].softmax(-1)[..., :-1]
          scores, labels = prob.max(-1)
          boxes = torchvision.ops.box_convert(preds[..., -4:], "cxcywh", "xyxy") * preds.new_tensor([w, h, w, h])
          return [
              {"boxes": boxes[b][scores[b] > conf], "scores": scores[b][scores[b] > conf], "labels": labels[b][scores[b] > conf]}
              for b in range(preds.shape[0])
          ]
    `),
  },

  DETRLoss: {
    section: 'loss',
    imports: ['import torchvision', 'from scipy.optimize import linear_sum_assignment'],
    code: dedent(`
      class DETRLoss(nn.Module):
          """Hungarian matching (cost: 1·class + 5·L1 + 2·GIoU), then CE ("no object" weight 0.1)
          + 5·L1 + 2·GIoU on matched boxes. Targets give xyxy pixels; they are normalised by the image size."""

          def __init__(self, num_classes: int, image_size: tuple[int, int], eos_coef: float = 0.1) -> None:
              super().__init__()
              self.nc, self.image_size = num_classes, image_size
              weight = torch.ones(num_classes + 1)
              weight[-1] = eos_coef
              self.register_buffer("empty_weight", weight, persistent=False)

          def forward(self, preds: torch.Tensor, targets: list[dict[str, torch.Tensor]]) -> torch.Tensor:
              preds = preds.float()
              logits, boxes = preds[..., :-4], preds[..., -4:]
              h, w = self.image_size
              scale = boxes.new_tensor([w, h, w, h])
              xyxy = lambda b: torchvision.ops.box_convert(b, "cxcywh", "xyxy")  # noqa: E731
              total = preds.new_zeros(())
              for i, t in enumerate(targets):
                  labels = t["labels"]
                  target_cls = torch.full((logits.shape[1],), self.nc, dtype=torch.long, device=preds.device)
                  if len(labels):
                      tgt = torchvision.ops.box_convert(t["boxes"].float() / scale, "xyxy", "cxcywh")
                      with torch.no_grad():
                          cost = (
                              -logits[i].softmax(-1)[:, labels]
                              + 5 * torch.cdist(boxes[i], tgt, p=1)
                              - 2 * torchvision.ops.generalized_box_iou(xyxy(boxes[i]), xyxy(tgt))
                          )
                          rows, cols = linear_sum_assignment(cost.float().cpu().numpy())
                      q = torch.as_tensor(rows, device=preds.device)
                      g = torch.as_tensor(cols, device=preds.device)
                      target_cls[q] = labels[g]
                      n = len(labels)
                      total = total + 5 * F.l1_loss(boxes[i][q], tgt[g], reduction="sum") / n
                      giou = torchvision.ops.generalized_box_iou(xyxy(boxes[i][q]), xyxy(tgt[g])).diag()
                      total = total + 2 * (1 - giou).sum() / n
                  total = total + F.cross_entropy(logits[i], target_cls, self.empty_weight)
              return total / len(targets)
    `),
  },
});
