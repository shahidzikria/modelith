import { dedent } from '../py';
import { defineTorchHelpers } from './registry';

defineTorchHelpers({
  PatchEmbed: {
    code: dedent(`
      class PatchEmbed(nn.Module):
          """Split the image into p×p patches, embed each linearly, optionally prepend a [CLS] token.

          Paper: An Image is Worth 16x16 Words (ViT), Dosovitskiy et al. 2021 (arXiv:2010.11929).
          """

          def __init__(self, c_in: int, dim: int, patch: int, cls_token: bool = True) -> None:
              super().__init__()
              self.proj = nn.Conv2d(c_in, dim, patch, stride=patch)
              self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) if cls_token else None

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              x = self.proj(x).flatten(2).transpose(1, 2)
              if self.cls_token is not None:
                  x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
              return x
    `),
  },

  LearnedPositionalEmbedding: {
    code: dedent(`
      class LearnedPositionalEmbedding(nn.Module):
          """Trainable absolute position embedding added to a (B, L, D) sequence (BERT / ViT)."""

          def __init__(self, length: int, dim: int) -> None:
              super().__init__()
              self.pos = nn.Parameter(torch.zeros(1, length, dim))
              nn.init.trunc_normal_(self.pos, std=0.02)

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              return x + self.pos
    `),
  },

  SinusoidalPositionalEncoding: {
    code: dedent(`
      class SinusoidalPositionalEncoding(nn.Module):
          """Fixed sin / cos position encoding (Vaswani et al. 2017, arXiv:1706.03762)."""

          def __init__(self, length: int, dim: int) -> None:
              super().__init__()
              position = torch.arange(length).unsqueeze(1)
              div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
              pe = torch.zeros(1, length, dim)
              pe[0, :, 0::2] = torch.sin(position * div)
              pe[0, :, 1::2] = torch.cos(position * div[: dim // 2])
              self.register_buffer("pe", pe, persistent=False)

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              return x + self.pe
    `),
  },

  apply_rope: {
    code: dedent(`
      def apply_rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
          """Rotary position embedding on the last axis of (B, H, N, Dh) queries / keys.

          Paper: RoFormer, Su et al. 2021 (arXiv:2104.09864).
          """
          n, dh = x.shape[-2], x.shape[-1]
          half = dh // 2
          freqs = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
          angles = torch.arange(n, device=x.device, dtype=torch.float32)[:, None] * freqs[None]
          cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
          x1, x2 = x[..., :half], x[..., half : 2 * half]
          rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
          return torch.cat([rotated, x[..., 2 * half :]], dim=-1) if dh % 2 else rotated
    `),
  },

  Attention: {
    deps: ['apply_rope'],
    code: dedent(`
      class Attention(nn.Module):
          """Multi-head self-attention with fused scaled-dot-product attention and optional RoPE."""

          def __init__(self, dim: int, heads: int, dropout: float = 0.0, rope: bool = False) -> None:
              super().__init__()
              self.heads, self.dropout, self.rope = heads, dropout, rope
              self.qkv = nn.Linear(dim, 3 * dim)
              self.proj = nn.Linear(dim, dim)

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              b, n, d = x.shape
              q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
              if self.rope:
                  q, k = apply_rope(q), apply_rope(k)
              x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
              return self.proj(x.transpose(1, 2).reshape(b, n, d))
    `),
  },

  TransformerBlock: {
    deps: ['Attention', 'DropPath'],
    code: dedent(`
      class TransformerBlock(nn.Module):
          """Pre-norm encoder block: x + Attention(LN(x)), then x + MLP(LN(x)).

          Papers: Attention Is All You Need (arXiv:1706.03762); ViT (arXiv:2010.11929).
          """

          def __init__(
              self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, drop_path: float = 0.0, rope: bool = False
          ) -> None:
              super().__init__()
              hidden = int(dim * mlp_ratio)
              self.norm1 = nn.LayerNorm(dim, eps=1e-6)
              self.attn = Attention(dim, heads, dropout, rope)
              self.norm2 = nn.LayerNorm(dim, eps=1e-6)
              self.mlp = nn.Sequential(
                  nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout)
              )
              self.drop_path = DropPath(drop_path)

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              x = x + self.drop_path(self.attn(self.norm1(x)))
              return x + self.drop_path(self.mlp(self.norm2(x)))
    `),
  },

  TransformerEncoder: {
    deps: ['TransformerBlock'],
    code: dedent(`
      class TransformerEncoder(nn.Module):
          """depth × pre-norm TransformerBlock (linearly increasing drop path) + final LayerNorm."""

          def __init__(
              self,
              dim: int,
              depth: int,
              heads: int,
              mlp_ratio: float = 4.0,
              dropout: float = 0.0,
              drop_path: float = 0.0,
              rope: bool = False,
          ) -> None:
              super().__init__()
              rates = [drop_path * i / max(1, depth - 1) for i in range(depth)]
              self.blocks = nn.ModuleList(TransformerBlock(dim, heads, mlp_ratio, dropout, r, rope) for r in rates)
              self.norm = nn.LayerNorm(dim, eps=1e-6)

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              for block in self.blocks:
                  x = block(x)
              return self.norm(x)
    `),
  },

  TransformerDecoder: {
    code: dedent(`
      class TransformerDecoderBlock(nn.Module):
          """Pre-norm decoder block: self-attention, cross-attention to the memory, MLP."""

          def __init__(self, dim: int, memory_dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
              super().__init__()
              self.norm1 = nn.LayerNorm(dim, eps=1e-6)
              self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
              self.norm2 = nn.LayerNorm(dim, eps=1e-6)
              self.cross_attn = nn.MultiheadAttention(
                  dim, heads, dropout=dropout, kdim=memory_dim, vdim=memory_dim, batch_first=True
              )
              self.norm3 = nn.LayerNorm(dim, eps=1e-6)
              hidden = int(dim * mlp_ratio)
              self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

          def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
              h = self.norm1(x)
              x = x + self.self_attn(h, h, h, need_weights=False)[0]
              h = self.norm2(x)
              x = x + self.cross_attn(h, memory, memory, need_weights=False)[0]
              return x + self.mlp(self.norm3(x))


      class TransformerDecoder(nn.Module):
          """depth × TransformerDecoderBlock + final LayerNorm; forward(x, memory)."""

          def __init__(self, dim: int, memory_dim: int, depth: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
              super().__init__()
              self.blocks = nn.ModuleList(
                  TransformerDecoderBlock(dim, memory_dim, heads, mlp_ratio, dropout) for _ in range(depth)
              )
              self.norm = nn.LayerNorm(dim, eps=1e-6)

          def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
              for block in self.blocks:
                  x = block(x, memory)
              return self.norm(x)
    `),
  },

  RecurrentLayer: {
    code: dedent(`
      class RecurrentLayer(nn.Module):
          """LSTM / GRU over a (B, L, D) sequence, returning every hidden state (B, L, H × directions)."""

          def __init__(
              self, cell: str, d_in: int, hidden: int, layers: int = 1, bidirectional: bool = True, dropout: float = 0.0
          ) -> None:
              super().__init__()
              rnn = nn.LSTM if cell == "lstm" else nn.GRU
              self.rnn = rnn(
                  d_in,
                  hidden,
                  num_layers=layers,
                  batch_first=True,
                  bidirectional=bidirectional,
                  dropout=dropout if layers > 1 else 0.0,
              )

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              return self.rnn(x)[0]
    `),
  },
});
