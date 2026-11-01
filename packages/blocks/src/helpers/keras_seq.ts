import { dedent } from '../py';
import { defineKerasHelpers } from './registry';

defineKerasHelpers({
  ClassToken: {
    code: dedent(`
      class ClassToken(layers.Layer):
          """Prepend a learnable [CLS] token to a (B, N, D) sequence (ViT)."""

          def build(self, input_shape) -> None:
              self.cls = self.add_weight(shape=(1, 1, input_shape[-1]), initializer="zeros", name="cls")

          def call(self, x):
              cls = ops.broadcast_to(self.cls, (ops.shape(x)[0], 1, ops.shape(x)[-1]))
              return ops.concatenate([cls, x], axis=1)
    `),
  },

  patch_embed: {
    deps: ['ClassToken'],
    code: dedent(`
      def patch_embed(x, dim: int, patch: int, cls_token: bool = True, name: str = "patch_embed"):
          """Non-overlapping p×p patch embedding (ViT)."""
          x = layers.Conv2D(dim, patch, strides=patch, name=f"{name}_proj")(x)
          x = layers.Reshape((x.shape[1] * x.shape[2], dim), name=f"{name}_flatten")(x)
          return ClassToken(name=f"{name}_cls")(x) if cls_token else x
    `),
  },

  PositionEmbedding: {
    code: dedent(`
      class PositionEmbedding(layers.Layer):
          """Add learned or fixed sinusoidal position encodings to a (B, L, D) sequence."""

          def __init__(self, mode: str = "learned", **kwargs) -> None:
              super().__init__(**kwargs)
              self.mode = mode

          def build(self, input_shape) -> None:
              length, dim = input_shape[1], input_shape[2]
              if self.mode == "learned":
                  init = keras.initializers.TruncatedNormal(stddev=0.02)
                  self.pos = self.add_weight(shape=(1, length, dim), initializer=init, name="pos")
                  return
              position = np.arange(length)[:, None]
              div = np.exp(np.arange(0, dim, 2) * (-np.log(10000.0) / dim))
              pe = np.zeros((1, length, dim), dtype="float32")
              pe[0, :, 0::2] = np.sin(position * div)
              pe[0, :, 1::2] = np.cos(position * div[: dim // 2])
              self.pos = self.add_weight(shape=pe.shape, initializer=keras.initializers.Constant(pe), trainable=False, name="pos")

          def call(self, x):
              return x + self.pos

          def get_config(self) -> dict:
              return {**super().get_config(), "mode": self.mode}
    `),
  },

  transformer_encoder: {
    code: dedent(`
      def transformer_encoder(x, depth: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, name: str = "encoder"):
          """depth × pre-norm transformer block + final LayerNorm (Vaswani 2017; ViT 2021)."""
          dim = x.shape[-1]
          for i in range(depth):
              p = f"{name}_{i}"
              h = layers.LayerNormalization(epsilon=1e-6, name=f"{p}_ln1")(x)
              h = layers.MultiHeadAttention(heads, dim // heads, dropout=dropout, name=f"{p}_attn")(h, h)
              x = layers.Add(name=f"{p}_add1")([x, h])
              h = layers.LayerNormalization(epsilon=1e-6, name=f"{p}_ln2")(x)
              h = layers.Dense(int(dim * mlp_ratio), activation="gelu", name=f"{p}_fc1")(h)
              h = layers.Dropout(dropout, name=f"{p}_drop")(h)
              h = layers.Dense(dim, name=f"{p}_fc2")(h)
              x = layers.Add(name=f"{p}_add2")([x, h])
          return layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln")(x)
    `),
  },

  transformer_decoder: {
    code: dedent(`
      def transformer_decoder(x, memory, depth: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, name: str = "decoder"):
          """depth × pre-norm decoder block (self-attention, cross-attention, MLP) + final LayerNorm."""
          dim = x.shape[-1]
          for i in range(depth):
              p = f"{name}_{i}"
              h = layers.LayerNormalization(epsilon=1e-6, name=f"{p}_ln1")(x)
              x = layers.Add(name=f"{p}_add1")([x, layers.MultiHeadAttention(heads, dim // heads, dropout=dropout, name=f"{p}_self")(h, h)])
              h = layers.LayerNormalization(epsilon=1e-6, name=f"{p}_ln2")(x)
              x = layers.Add(name=f"{p}_add2")([x, layers.MultiHeadAttention(heads, dim // heads, dropout=dropout, name=f"{p}_cross")(h, memory)])
              h = layers.LayerNormalization(epsilon=1e-6, name=f"{p}_ln3")(x)
              h = layers.Dense(int(dim * mlp_ratio), activation="gelu", name=f"{p}_fc1")(h)
              x = layers.Add(name=f"{p}_add3")([x, layers.Dense(dim, name=f"{p}_fc2")(h)])
          return layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln")(x)
    `),
  },

  recurrent: {
    code: dedent(`
      def recurrent(x, cell: str, hidden: int, num_layers: int = 1, bidirectional: bool = True, dropout: float = 0.0, name: str = "rnn"):
          """Stacked LSTM / GRU returning all hidden states."""
          rnn = layers.LSTM if cell == "lstm" else layers.GRU
          for i in range(num_layers):
              layer = rnn(hidden, return_sequences=True, dropout=dropout if i > 0 else 0.0, name=f"{name}_{i}")
              x = layers.Bidirectional(layer, name=f"{name}_{i}_bi")(x) if bidirectional else layer(x)
          return x
    `),
  },
});
