import { dedent } from '../py';
import { defineKerasHelpers } from './registry';

defineKerasHelpers({
  activation: {
    code: dedent(`
      def activation(name: str, layer_name: str) -> layers.Layer:
          """Activation layer by name ("none" gives an identity)."""
          if name == "leaky_relu":
              return layers.LeakyReLU(negative_slope=0.1, name=layer_name)
          if name == "relu6":
              return layers.ReLU(max_value=6.0, name=layer_name)
          if name == "hardswish":
              return layers.Activation("hard_swish", name=layer_name)
          if name == "none":
              return layers.Identity(name=layer_name)
          return layers.Activation(name, name=layer_name)
    `),
  },

  conv_bn_act: {
    deps: ['activation'],
    code: dedent(`
      def conv_bn_act(x, filters: int, k: int = 3, s: int = 1, d: int = 1, groups: int = 1, act: str = "silu", name: str = "cba"):
          """Conv2D → BatchNormalization → activation with "same" padding (NHWC)."""
          if groups > 1 and groups == x.shape[-1] == filters:
              x = layers.DepthwiseConv2D(k, strides=s, padding="same", dilation_rate=d, use_bias=False, name=f"{name}_dw")(x)
          else:
              x = layers.Conv2D(
                  filters, k, strides=s, padding="same", dilation_rate=d, groups=groups, use_bias=False, name=f"{name}_conv"
              )(x)
          x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"{name}_bn")(x)
          return activation(act, f"{name}_act")(x)
    `),
  },

  dw_sep_conv: {
    deps: ['conv_bn_act'],
    code: dedent(`
      def dw_sep_conv(x, filters: int, k: int = 3, s: int = 1, act: str = "relu", name: str = "dws"):
          """Depthwise k×k conv + pointwise 1×1 conv, each with BN and activation (MobileNets)."""
          x = conv_bn_act(x, x.shape[-1], k, s, groups=x.shape[-1], act=act, name=f"{name}_dw")
          return conv_bn_act(x, filters, 1, act=act, name=f"{name}_pw")
    `),
  },

  LayerScale: {
    code: dedent(`
      class LayerScale(layers.Layer):
          """Learnable per-channel scale initialised to a small value (CaiT / ConvNeXt)."""

          def __init__(self, init_value: float = 1e-6, **kwargs) -> None:
              super().__init__(**kwargs)
              self.init_value = init_value

          def build(self, input_shape) -> None:
              self.gamma = self.add_weight(
                  shape=(input_shape[-1],), initializer=keras.initializers.Constant(self.init_value), name="gamma"
              )

          def call(self, x):
              return x * self.gamma

          def get_config(self) -> dict:
              return {**super().get_config(), "init_value": self.init_value}
    `),
  },
});
