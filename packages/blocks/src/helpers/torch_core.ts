import { dedent } from '../py';
import { defineTorchHelpers } from './registry';

defineTorchHelpers({
  make_act: {
    code: dedent(`
      def make_act(name: str) -> nn.Module:
          """Build an activation module by name ("none" gives an identity)."""
          if name == "relu":
              return nn.ReLU(inplace=True)
          if name == "relu6":
              return nn.ReLU6(inplace=True)
          if name == "silu":
              return nn.SiLU(inplace=True)
          if name == "gelu":
              return nn.GELU()
          if name == "leaky_relu":
              return nn.LeakyReLU(0.1, inplace=True)
          if name == "hardswish":
              return nn.Hardswish(inplace=True)
          if name == "mish":
              return nn.Mish(inplace=True)
          if name == "sigmoid":
              return nn.Sigmoid()
          if name == "tanh":
              return nn.Tanh()
          return nn.Identity()
    `),
  },

  ConvBNAct: {
    deps: ['make_act'],
    code: dedent(`
      class ConvBNAct(nn.Module):
          """Conv2d → BatchNorm2d → activation, with "same" padding by default."""

          def __init__(
              self,
              c_in: int,
              c_out: int,
              k: int = 3,
              s: int = 1,
              d: int = 1,
              g: int = 1,
              act: str = "silu",
          ) -> None:
              super().__init__()
              self.conv = nn.Conv2d(c_in, c_out, k, s, d * (k - 1) // 2, dilation=d, groups=g, bias=False)
              self.bn = nn.BatchNorm2d(c_out)
              self.act = make_act(act)

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              return self.act(self.bn(self.conv(x)))
    `),
  },

  DWSepConv: {
    deps: ['ConvBNAct'],
    code: dedent(`
      class DWSepConv(nn.Module):
          """Depthwise k×k conv + pointwise 1×1 conv, each with BN and activation.

          Paper: MobileNets, Howard et al. 2017 (arXiv:1704.04861).
          """

          def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, act: str = "relu") -> None:
              super().__init__()
              self.dw = ConvBNAct(c_in, c_in, k, s, g=c_in, act=act)
              self.pw = ConvBNAct(c_in, c_out, 1, act=act)

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              return self.pw(self.dw(x))
    `),
  },

  LayerNorm2d: {
    code: dedent(`
      class LayerNorm2d(nn.LayerNorm):
          """LayerNorm over the channel axis of an NCHW feature map."""

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              x = x.permute(0, 2, 3, 1)
              x = super().forward(x)
              return x.permute(0, 3, 1, 2)
    `),
  },

  seq_to_image: {
    code: dedent(`
      def seq_to_image(x: torch.Tensor, h: int, w: int, prefix: int = 0) -> torch.Tensor:
          """(B, prefix + h·w, D) token sequence → (B, D, h, w) feature map (drops prefix tokens)."""
          x = x[:, prefix:]
          return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], h, w)
    `),
  },

  DropPath: {
    code: dedent(`
      class DropPath(nn.Module):
          """Stochastic depth: randomly drops the whole residual branch per sample.

          Paper: Deep Networks with Stochastic Depth, Huang et al. 2016 (arXiv:1603.09382).
          """

          def __init__(self, p: float = 0.0) -> None:
              super().__init__()
              self.p = p

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              if self.p == 0.0 or not self.training:
                  return x
              keep = 1.0 - self.p
              mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
              return x * mask / keep
    `),
  },
});
