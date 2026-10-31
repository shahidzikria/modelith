import { dedent } from '../py';
import { defineTorchHelpers } from './registry';

defineTorchHelpers({
  ClassificationHead: {
    code: dedent(`
      class ClassificationHead(nn.Module):
          """Pool (if needed) → optional hidden MLP → dropout → linear class logits."""

          def __init__(self, c_in: int, num_classes: int, hidden: int = 0, dropout: float = 0.0, pool: str = "none") -> None:
              super().__init__()
              self.pool = pool
              layers: list[nn.Module] = []
              if hidden > 0:
                  layers += [nn.Linear(c_in, hidden), nn.GELU()]
                  c_in = hidden
              layers += [nn.Dropout(dropout), nn.Linear(c_in, num_classes)]
              self.mlp = nn.Sequential(*layers)

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              if self.pool == "spatial":
                  x = x.mean(dim=(2, 3))
              elif self.pool == "tokens":
                  x = x.mean(dim=1)
              return self.mlp(x)
    `),
  },

  SegmentationHead: {
    deps: ['ConvBNAct'],
    code: dedent(`
      class SegmentationHead(nn.Module):
          """3×3 Conv·BN·ReLU → 1×1 classifier → bilinear upsample to the input resolution."""

          def __init__(self, c_in: int, num_classes: int, hidden: int, size: tuple[int, int]) -> None:
              super().__init__()
              self.size = size
              self.conv = ConvBNAct(c_in, hidden, 3, act="relu") if hidden > 0 else nn.Identity()
              self.cls = nn.Conv2d(hidden if hidden > 0 else c_in, num_classes, 1)

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              x = self.cls(self.conv(x))
              return F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
    `),
  },

  DenseHead: {
    deps: ['ConvBNAct'],
    code: dedent(`
      class DenseHead(nn.Module):
          """Dense regression (depth, restoration, reconstruction): conv → 1×1 → upsample → output activation."""

          def __init__(self, c_in: int, c_out: int, hidden: int, size: tuple[int, int], out_act: str = "none") -> None:
              super().__init__()
              self.size = size
              self.conv = ConvBNAct(c_in, hidden, 3, act="relu") if hidden > 0 else nn.Identity()
              self.proj = nn.Conv2d(hidden if hidden > 0 else c_in, c_out, 1)
              self.out_act = out_act

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              x = self.proj(self.conv(x))
              if tuple(x.shape[-2:]) != self.size:
                  x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
              if self.out_act == "sigmoid":
                  return x.sigmoid()
              if self.out_act == "softplus":
                  return F.softplus(x)
              if self.out_act == "relu":
                  return F.relu(x)
              return x
    `),
  },

  EmbeddingHead: {
    code: dedent(`
      class EmbeddingHead(nn.Module):
          """Pool (if needed) → linear projection → BatchNorm → L2-normalised embedding."""

          def __init__(self, c_in: int, dim: int, pool: str = "none") -> None:
              super().__init__()
              self.pool = pool
              self.proj = nn.Linear(c_in, dim, bias=False)
              self.bn = nn.BatchNorm1d(dim)

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              if self.pool == "spatial":
                  x = x.mean(dim=(2, 3))
              elif self.pool == "tokens":
                  x = x.mean(dim=1)
              return F.normalize(self.bn(self.proj(x)), dim=-1)
    `),
  },

  ContrastiveHead: {
    code: dedent(`
      class ContrastiveHead(nn.Module):
          """Project two modalities into a shared space and return the scaled similarity matrix.

          Paper: CLIP, Radford et al. 2021 (arXiv:2103.00020). Row i is sample i of the first input.
          """

          def __init__(self, c_a: int, c_b: int, dim: int) -> None:
              super().__init__()
              self.proj_a = nn.Linear(c_a, dim, bias=False)
              self.proj_b = nn.Linear(c_b, dim, bias=False)
              self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

          def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
              a = F.normalize(self.proj_a(a), dim=-1)
              b = F.normalize(self.proj_b(b), dim=-1)
              return self.logit_scale.exp().clamp(max=100) * a @ b.t()
    `),
  },

  // ----------------------------------------------------------------- losses

  TokenCrossEntropy: {
    section: 'loss',
    code: dedent(`
      class TokenCrossEntropy(nn.Module):
          """Per-token cross-entropy; positions labelled -100 (padding / sub-words) are ignored."""

          def __init__(self, label_smoothing: float = 0.0) -> None:
              super().__init__()
              self.label_smoothing = label_smoothing

          def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
              return F.cross_entropy(
                  logits.float().reshape(-1, logits.shape[-1]),
                  target.reshape(-1),
                  ignore_index=-100,
                  label_smoothing=self.label_smoothing,
              )
    `),
  },

  CRF: {
    section: 'loss',
    code: dedent(`
      class CRF(nn.Module):
          """Linear-chain CRF used as the loss of a token-classification head.

          forward(emissions, tags) returns the mean negative log-likelihood; decode() runs Viterbi.
          Tags equal to -100 are padding and must form a suffix of each sequence.
          Paper: Lafferty et al. 2001; BiLSTM-CRF, Huang et al. 2015 (arXiv:1508.01991).
          """

          def __init__(self, num_tags: int) -> None:
              super().__init__()
              self.start = nn.Parameter(torch.empty(num_tags).uniform_(-0.1, 0.1))
              self.end = nn.Parameter(torch.empty(num_tags).uniform_(-0.1, 0.1))
              self.trans = nn.Parameter(torch.empty(num_tags, num_tags).uniform_(-0.1, 0.1))  # trans[i, j]: i → j

          def forward(self, emissions: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
              emissions = emissions.float()
              mask = tags != -100
              tags = tags.clamp_min(0)
              return (self._log_partition(emissions, mask) - self._score(emissions, tags, mask)).mean()

          def _score(self, e: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
              score = self.start[tags[:, 0]] + e[:, 0].gather(1, tags[:, :1]).squeeze(1)
              for i in range(1, e.shape[1]):
                  step = self.trans[tags[:, i - 1], tags[:, i]] + e[:, i].gather(1, tags[:, i : i + 1]).squeeze(1)
                  score = score + step * mask[:, i].float()
              last = tags.gather(1, (mask.sum(1) - 1).clamp_min(0).unsqueeze(1)).squeeze(1)
              return score + self.end[last]

          def _log_partition(self, e: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
              alpha = self.start + e[:, 0]
              for i in range(1, e.shape[1]):
                  nxt = torch.logsumexp(alpha.unsqueeze(2) + self.trans + e[:, i].unsqueeze(1), dim=1)
                  alpha = torch.where(mask[:, i : i + 1], nxt, alpha)
              return torch.logsumexp(alpha + self.end, dim=1)

          @torch.no_grad()
          def decode(self, emissions: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
              """Viterbi decoding: (B, L, T) emissions → (B, L) most likely tags."""
              e = emissions.float()
              b, length, t = e.shape
              if mask is None:
                  mask = torch.ones(b, length, dtype=torch.bool, device=e.device)
              keep = torch.arange(t, device=e.device).expand(b, t)
              score = self.start + e[:, 0]
              history = []
              for i in range(1, length):
                  best, idx = (score.unsqueeze(2) + self.trans).max(dim=1)
                  score = torch.where(mask[:, i : i + 1], best + e[:, i], score)
                  history.append(torch.where(mask[:, i : i + 1], idx, keep))
              best_last = (score + self.end).argmax(1)
              path = [best_last]
              for idx in reversed(history):
                  best_last = idx.gather(1, best_last.unsqueeze(1)).squeeze(1)
                  path.append(best_last)
              return torch.stack(path[::-1], dim=1)
    `),
  },

  SegmentationLoss: {
    section: 'loss',
    code: dedent(`
      class SegmentationLoss(nn.Module):
          """Cross-entropy (BCE for a single class) + soft Dice.

          Paper: V-Net Dice loss, Milletari et al. 2016 (arXiv:1606.04797).
          """

          def __init__(self, num_classes: int, dice_weight: float = 1.0) -> None:
              super().__init__()
              self.num_classes = num_classes
              self.dice_weight = dice_weight

          def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
              logits = logits.float()
              if self.num_classes == 1:
                  target = target.float()
                  base = F.binary_cross_entropy_with_logits(logits, target)
                  prob, onehot, dims = logits.sigmoid(), target, (0, 2, 3)
              else:
                  base = F.cross_entropy(logits, target)
                  prob = logits.softmax(1)
                  onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
                  dims = (0, 2, 3)
              inter = (prob * onehot).sum(dims)
              union = prob.sum(dims) + onehot.sum(dims)
              dice = 1.0 - ((2.0 * inter + 1.0) / (union + 1.0)).mean()
              return base + self.dice_weight * dice
    `),
  },

  SILogLoss: {
    section: 'loss',
    code: dedent(`
      class SILogLoss(nn.Module):
          """Scale-invariant log loss for monocular depth (pixels with target ≤ 0 are ignored).

          Paper: Eigen et al. 2014 (arXiv:1406.2283); λ = 0.85 as in BTS / AdaBins.
          """

          def __init__(self, lambd: float = 0.85) -> None:
              super().__init__()
              self.lambd = lambd

          def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
              pred, target = pred.float(), target.float()
              valid = target > 0
              d = torch.log(pred[valid].clamp_min(1e-6)) - torch.log(target[valid])
              return torch.sqrt((d**2).mean() - self.lambd * d.mean() ** 2 + 1e-8) * 10.0
    `),
  },

  ArcFaceLoss: {
    section: 'loss',
    code: dedent(`
      class ArcFaceLoss(nn.Module):
          """Additive angular margin softmax over L2-normalised embeddings.

          Paper: ArcFace, Deng et al. 2019 (arXiv:1801.07698).
          """

          def __init__(self, dim: int, num_classes: int, s: float = 30.0, m: float = 0.5) -> None:
              super().__init__()
              self.weight = nn.Parameter(torch.empty(num_classes, dim))
              nn.init.xavier_uniform_(self.weight)
              self.s, self.m = s, m

          def forward(self, emb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
              cos = F.linear(F.normalize(emb.float()), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
              theta = torch.acos(cos)
              target_logit = torch.cos(theta + self.m)
              onehot = F.one_hot(target, cos.shape[1]).bool()
              logits = torch.where(onehot, target_logit, cos) * self.s
              return F.cross_entropy(logits, target)
    `),
  },

  CLIPLoss: {
    section: 'loss',
    code: dedent(`
      class CLIPLoss(nn.Module):
          """Symmetric InfoNCE over a (B, B) similarity matrix; matching pairs are on the diagonal."""

          def forward(self, logits: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
              logits = logits.float()
              labels = torch.arange(logits.shape[0], device=logits.device)
              return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    `),
  },
});
