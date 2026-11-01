import { dedent } from '../py';
import { defineTorchHelpers } from './registry';

defineTorchHelpers({
  make_stage: {
    code: dedent(`
      def make_stage(block: type[nn.Module], c_in: int, c_out: int, n: int, stride: int = 1, **kwargs) -> nn.Sequential:
          """Stack n blocks; only the first one changes the channel count / stride."""
          return nn.Sequential(*[block(c_in if i == 0 else c_out, c_out, stride if i == 0 else 1, **kwargs) for i in range(n)])
    `),
  },

  DoubleConv: {
    deps: ['ConvBNAct'],
    code: dedent(`
      class DoubleConv(nn.Module):
          """(3×3 Conv → BN → ReLU) × 2, the U-Net building block.

          Paper: U-Net, Ronneberger et al. 2015 (arXiv:1505.04597).
          """

          def __init__(self, c_in: int, c_out: int, c_mid: int | None = None) -> None:
              super().__init__()
              c_mid = c_mid or c_out
              self.conv1 = ConvBNAct(c_in, c_mid, 3, act="relu")
              self.conv2 = ConvBNAct(c_mid, c_out, 3, act="relu")

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              return self.conv2(self.conv1(x))
    `),
  },

  BasicBlock: {
    deps: ['ConvBNAct'],
    code: dedent(`
      class BasicBlock(nn.Module):
          """ResNet basic block: two 3×3 convs plus an identity or 1×1 projection shortcut.

          Paper: Deep Residual Learning, He et al. 2016 (arXiv:1512.03385).
          """

          def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
              super().__init__()
              self.conv1 = ConvBNAct(c_in, c_out, 3, stride, act="relu")
              self.conv2 = ConvBNAct(c_out, c_out, 3, act="none")
              self.shortcut = (
                  nn.Identity() if stride == 1 and c_in == c_out else ConvBNAct(c_in, c_out, 1, stride, act="none")
              )
              self.act = nn.ReLU(inplace=True)

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))
    `),
  },

  Bottleneck: {
    deps: ['ConvBNAct'],
    code: dedent(`
      class Bottleneck(nn.Module):
          """ResNet bottleneck: 1×1 reduce → 3×3 (stride) → 1×1 expand, plus a shortcut.

          Paper: Deep Residual Learning, He et al. 2016 (arXiv:1512.03385). Width = c_out / expansion.
          """

          def __init__(self, c_in: int, c_out: int, stride: int = 1, expansion: int = 4) -> None:
              super().__init__()
              width = c_out // expansion
              self.conv1 = ConvBNAct(c_in, width, 1, act="relu")
              self.conv2 = ConvBNAct(width, width, 3, stride, act="relu")
              self.conv3 = ConvBNAct(width, c_out, 1, act="none")
              self.shortcut = (
                  nn.Identity() if stride == 1 and c_in == c_out else ConvBNAct(c_in, c_out, 1, stride, act="none")
              )
              self.act = nn.ReLU(inplace=True)

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              return self.act(self.conv3(self.conv2(self.conv1(x))) + self.shortcut(x))
    `),
  },

  SqueezeExcite: {
    deps: ['make_act'],
    code: dedent(`
      class SqueezeExcite(nn.Module):
          """Channel attention: global pool → 1×1 reduce → activation → 1×1 expand → sigmoid gate.

          Paper: Squeeze-and-Excitation Networks, Hu et al. 2018 (arXiv:1709.01507).
          """

          def __init__(self, c: int, hidden: int, act: str = "silu") -> None:
              super().__init__()
              self.reduce = nn.Conv2d(c, hidden, 1)
              self.act = make_act(act)
              self.expand = nn.Conv2d(hidden, c, 1)

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              s = x.mean(dim=(2, 3), keepdim=True)
              return x * torch.sigmoid(self.expand(self.act(self.reduce(s))))
    `),
  },

  MBConv: {
    deps: ['ConvBNAct', 'SqueezeExcite', 'DropPath'],
    code: dedent(`
      class MBConv(nn.Module):
          """Inverted residual: 1×1 expand → k×k depthwise → squeeze-excite → 1×1 linear projection.

          Papers: MobileNetV2 (arXiv:1801.04381), EfficientNet (arXiv:1905.11946).
          """

          def __init__(
              self,
              c_in: int,
              c_out: int,
              stride: int = 1,
              k: int = 3,
              expand: int = 4,
              se: float = 0.25,
              act: str = "silu",
              drop_path: float = 0.0,
          ) -> None:
              super().__init__()
              mid = c_in * expand
              self.expand = ConvBNAct(c_in, mid, 1, act=act) if expand != 1 else nn.Identity()
              self.dw = ConvBNAct(mid, mid, k, stride, g=mid, act=act)
              self.se = SqueezeExcite(mid, max(1, int(c_in * se)), act=act) if se > 0 else nn.Identity()
              self.project = ConvBNAct(mid, c_out, 1, act="none")
              self.use_residual = stride == 1 and c_in == c_out
              self.drop_path = DropPath(drop_path)

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              y = self.project(self.se(self.dw(self.expand(x))))
              return x + self.drop_path(y) if self.use_residual else y
    `),
  },

  FusedMBConv: {
    deps: ['ConvBNAct', 'DropPath'],
    code: dedent(`
      class FusedMBConv(nn.Module):
          """Fused inverted residual: one k×k expand conv (with stride) → 1×1 projection.

          Paper: EfficientNetV2, Tan & Le 2021 (arXiv:2104.00298).
          """

          def __init__(
              self, c_in: int, c_out: int, stride: int = 1, k: int = 3, expand: int = 4, act: str = "silu", drop_path: float = 0.0
          ) -> None:
              super().__init__()
              if expand == 1:
                  self.expand = ConvBNAct(c_in, c_out, k, stride, act=act)
                  self.project = nn.Identity()
              else:
                  self.expand = ConvBNAct(c_in, c_in * expand, k, stride, act=act)
                  self.project = ConvBNAct(c_in * expand, c_out, 1, act="none")
              self.use_residual = stride == 1 and c_in == c_out
              self.drop_path = DropPath(drop_path)

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              y = self.project(self.expand(x))
              return x + self.drop_path(y) if self.use_residual else y
    `),
  },

  ConvNeXtBlock: {
    deps: ['DropPath'],
    code: dedent(`
      class ConvNeXtBlock(nn.Module):
          """7×7 depthwise conv → LayerNorm → Linear(4d) → GELU → Linear(d) → LayerScale, residual.

          Paper: A ConvNet for the 2020s, Liu et al. 2022 (arXiv:2201.03545).
          """

          def __init__(self, dim: int, layer_scale: float = 1e-6, drop_path: float = 0.0) -> None:
              super().__init__()
              self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
              self.norm = nn.LayerNorm(dim, eps=1e-6)
              self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
              self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
              self.drop_path = DropPath(drop_path)

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              y = self.dw(x).permute(0, 2, 3, 1)
              y = self.gamma * self.mlp(self.norm(y))
              return x + self.drop_path(y.permute(0, 3, 1, 2))
    `),
  },

  YoloBottleneck: {
    deps: ['ConvBNAct'],
    code: dedent(`
      class YoloBottleneck(nn.Module):
          """Two 3×3 Conv·BN·SiLU with an optional residual (YOLOv5 / YOLOv8)."""

          def __init__(self, c: int, shortcut: bool = True) -> None:
              super().__init__()
              self.cv1 = ConvBNAct(c, c, 3)
              self.cv2 = ConvBNAct(c, c, 3)
              self.add = shortcut

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              y = self.cv2(self.cv1(x))
              return x + y if self.add else y
    `),
  },

  C2f: {
    deps: ['ConvBNAct', 'YoloBottleneck'],
    code: dedent(`
      class C2f(nn.Module):
          """CSP bottleneck with two convolutions: split, n bottlenecks, concatenate every partial output.

          Reference: Ultralytics YOLOv8 (2023).
          """

          def __init__(self, c_in: int, c_out: int, n: int = 1, shortcut: bool = True, e: float = 0.5) -> None:
              super().__init__()
              self.c = int(c_out * e)
              self.cv1 = ConvBNAct(c_in, 2 * self.c, 1)
              self.cv2 = ConvBNAct((2 + n) * self.c, c_out, 1)
              self.m = nn.ModuleList(YoloBottleneck(self.c, shortcut) for _ in range(n))

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              y = list(self.cv1(x).chunk(2, dim=1))
              for m in self.m:
                  y.append(m(y[-1]))
              return self.cv2(torch.cat(y, dim=1))
    `),
  },

  SPPF: {
    deps: ['ConvBNAct'],
    code: dedent(`
      class SPPF(nn.Module):
          """Spatial pyramid pooling (fast): three chained k×k max-pools, concatenated (YOLOv5 / YOLOv8)."""

          def __init__(self, c_in: int, c_out: int, k: int = 5) -> None:
              super().__init__()
              c = c_in // 2
              self.cv1 = ConvBNAct(c_in, c, 1)
              self.cv2 = ConvBNAct(4 * c, c_out, 1)
              self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              y = [self.cv1(x)]
              for _ in range(3):
                  y.append(self.pool(y[-1]))
              return self.cv2(torch.cat(y, dim=1))
    `),
  },

  CBAM: {
    code: dedent(`
      class CBAM(nn.Module):
          """Channel attention (shared MLP on avg + max pooling) followed by k×k spatial attention.

          Paper: CBAM, Woo et al. 2018 (arXiv:1807.06521).
          """

          def __init__(self, c: int, reduction: int = 16, k: int = 7) -> None:
              super().__init__()
              hidden = max(8, c // reduction)
              self.mlp = nn.Sequential(nn.Conv2d(c, hidden, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden, c, 1))
              self.spatial = nn.Conv2d(2, 1, k, padding=k // 2)

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              avg = self.mlp(x.mean(dim=(2, 3), keepdim=True))
              mx = self.mlp(x.amax(dim=(2, 3), keepdim=True))
              x = x * torch.sigmoid(avg + mx)
              s = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
              return x * torch.sigmoid(self.spatial(s))
    `),
  },

  ASPP: {
    deps: ['ConvBNAct'],
    code: dedent(`
      class ASPP(nn.Module):
          """Atrous spatial pyramid pooling: 1×1, dilated 3×3 branches and an image-pooling branch.

          Paper: DeepLabV3, Chen et al. 2017 (arXiv:1706.05587).
          Note: BatchNorm on the pooled 1×1 branch needs batch size ≥ 2 in train mode.
          """

          def __init__(self, c_in: int, c_out: int = 256, rates: tuple[int, ...] = (6, 12, 18), dropout: float = 0.1) -> None:
              super().__init__()
              self.branches = nn.ModuleList(
                  [ConvBNAct(c_in, c_out, 1, act="relu")] + [ConvBNAct(c_in, c_out, 3, d=r, act="relu") for r in rates]
              )
              self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBNAct(c_in, c_out, 1, act="relu"))
              self.project = nn.Sequential(ConvBNAct(c_out * (len(rates) + 2), c_out, 1, act="relu"), nn.Dropout(dropout))

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              ys = [branch(x) for branch in self.branches]
              ys.append(F.interpolate(self.pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False))
              return self.project(torch.cat(ys, dim=1))
    `),
  },

  RepVGGBlock: {
    deps: ['ConvBNAct'],
    code: dedent(`
      class RepVGGBlock(nn.Module):
          """Training-time RepVGG block: 3×3 conv-BN + 1×1 conv-BN (+ identity BN), summed, then ReLU.

          The branches can be fused into a single 3×3 conv for inference.
          Paper: RepVGG, Ding et al. 2021 (arXiv:2101.03697).
          """

          def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
              super().__init__()
              self.dense = ConvBNAct(c_in, c_out, 3, stride, act="none")
              self.one = ConvBNAct(c_in, c_out, 1, stride, act="none")
              self.identity = nn.BatchNorm2d(c_in) if c_in == c_out and stride == 1 else None
              self.act = nn.ReLU(inplace=True)

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              y = self.dense(x) + self.one(x)
              if self.identity is not None:
                  y = y + self.identity(x)
              return self.act(y)
    `),
  },
});
