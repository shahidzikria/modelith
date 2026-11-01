import { dedent } from '../py';
import { defineKerasHelpers } from './registry';

defineKerasHelpers({
  repeat_block: {
    code: dedent(`
      def repeat_block(x, block, n: int, filters: int, stride: int, name: str, **kwargs):
          """Apply block n times; only the first one changes channels / stride."""
          for i in range(n):
              x = block(x, filters, stride if i == 0 else 1, name=f"{name}_{i}", **kwargs)
          return x
    `),
  },

  double_conv: {
    deps: ['conv_bn_act'],
    code: dedent(`
      def double_conv(x, filters: int, mid: int | None = None, name: str = "double_conv"):
          """(3×3 Conv → BN → ReLU) × 2 (U-Net)."""
          x = conv_bn_act(x, mid or filters, 3, act="relu", name=f"{name}_1")
          return conv_bn_act(x, filters, 3, act="relu", name=f"{name}_2")
    `),
  },

  basic_block: {
    deps: ['conv_bn_act'],
    code: dedent(`
      def basic_block(x, filters: int, stride: int = 1, name: str = "basic"):
          """ResNet basic block (He et al. 2016)."""
          shortcut = x
          if stride != 1 or x.shape[-1] != filters:
              shortcut = conv_bn_act(x, filters, 1, stride, act="none", name=f"{name}_proj")
          y = conv_bn_act(x, filters, 3, stride, act="relu", name=f"{name}_1")
          y = conv_bn_act(y, filters, 3, act="none", name=f"{name}_2")
          return layers.ReLU(name=f"{name}_out")(layers.Add(name=f"{name}_add")([y, shortcut]))
    `),
  },

  bottleneck_block: {
    deps: ['conv_bn_act'],
    code: dedent(`
      def bottleneck_block(x, filters: int, stride: int = 1, expansion: int = 4, name: str = "bottleneck"):
          """ResNet bottleneck: 1×1 reduce → 3×3 → 1×1 expand (He et al. 2016)."""
          width = filters // expansion
          shortcut = x
          if stride != 1 or x.shape[-1] != filters:
              shortcut = conv_bn_act(x, filters, 1, stride, act="none", name=f"{name}_proj")
          y = conv_bn_act(x, width, 1, act="relu", name=f"{name}_1")
          y = conv_bn_act(y, width, 3, stride, act="relu", name=f"{name}_2")
          y = conv_bn_act(y, filters, 1, act="none", name=f"{name}_3")
          return layers.ReLU(name=f"{name}_out")(layers.Add(name=f"{name}_add")([y, shortcut]))
    `),
  },

  squeeze_excite: {
    code: dedent(`
      def squeeze_excite(x, hidden: int, act: str = "silu", name: str = "se"):
          """Squeeze-and-excitation channel gate (Hu et al. 2018)."""
          c = x.shape[-1]
          s = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_pool")(x)
          s = layers.Conv2D(hidden, 1, activation=act, name=f"{name}_reduce")(s)
          s = layers.Conv2D(c, 1, activation="sigmoid", name=f"{name}_expand")(s)
          return layers.Multiply(name=f"{name}_scale")([x, s])
    `),
  },

  mbconv: {
    deps: ['conv_bn_act', 'squeeze_excite'],
    code: dedent(`
      def mbconv(x, filters: int, stride: int = 1, k: int = 3, expand: int = 4, se: float = 0.25, act: str = "silu", name: str = "mbconv"):
          """Inverted residual with squeeze-excite (MobileNetV2 / EfficientNet)."""
          c_in = x.shape[-1]
          mid = c_in * expand
          y = conv_bn_act(x, mid, 1, act=act, name=f"{name}_expand") if expand != 1 else x
          y = conv_bn_act(y, mid, k, stride, groups=mid, act=act, name=f"{name}_dw")
          if se > 0:
              y = squeeze_excite(y, max(1, int(c_in * se)), act=act, name=f"{name}_se")
          y = conv_bn_act(y, filters, 1, act="none", name=f"{name}_project")
          if stride == 1 and c_in == filters:
              y = layers.Add(name=f"{name}_add")([x, y])
          return y
    `),
  },

  fused_mbconv: {
    deps: ['conv_bn_act'],
    code: dedent(`
      def fused_mbconv(x, filters: int, stride: int = 1, k: int = 3, expand: int = 4, act: str = "silu", name: str = "fused"):
          """Fused inverted residual (EfficientNetV2)."""
          c_in = x.shape[-1]
          if expand == 1:
              y = conv_bn_act(x, filters, k, stride, act=act, name=f"{name}_conv")
          else:
              y = conv_bn_act(x, c_in * expand, k, stride, act=act, name=f"{name}_expand")
              y = conv_bn_act(y, filters, 1, act="none", name=f"{name}_project")
          if stride == 1 and c_in == filters:
              y = layers.Add(name=f"{name}_add")([x, y])
          return y
    `),
  },

  convnext_block: {
    deps: ['LayerScale'],
    code: dedent(`
      def convnext_block(x, layer_scale: float = 1e-6, name: str = "convnext"):
          """7×7 depthwise → LayerNorm → MLP ×4 → LayerScale, residual (Liu et al. 2022)."""
          dim = x.shape[-1]
          y = layers.DepthwiseConv2D(7, padding="same", name=f"{name}_dw")(x)
          y = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln")(y)
          y = layers.Dense(4 * dim, activation="gelu", name=f"{name}_fc1")(y)
          y = layers.Dense(dim, name=f"{name}_fc2")(y)
          y = LayerScale(layer_scale, name=f"{name}_gamma")(y)
          return layers.Add(name=f"{name}_add")([x, y])
    `),
  },

  c2f: {
    deps: ['conv_bn_act'],
    code: dedent(`
      def c2f(x, filters: int, n: int = 1, shortcut: bool = True, e: float = 0.5, name: str = "c2f"):
          """CSP bottleneck with two convolutions (YOLOv8)."""
          c = int(filters * e)
          y = ops.split(conv_bn_act(x, 2 * c, 1, name=f"{name}_cv1"), 2, axis=-1)
          for i in range(n):
              z = conv_bn_act(y[-1], c, 3, name=f"{name}_m{i}_1")
              z = conv_bn_act(z, c, 3, name=f"{name}_m{i}_2")
              y.append(layers.Add(name=f"{name}_m{i}_add")([y[-1], z]) if shortcut else z)
          return conv_bn_act(layers.Concatenate(name=f"{name}_cat")(y), filters, 1, name=f"{name}_cv2")
    `),
  },

  sppf: {
    deps: ['conv_bn_act'],
    code: dedent(`
      def sppf(x, filters: int, k: int = 5, name: str = "sppf"):
          """Spatial pyramid pooling - fast (YOLOv5 / YOLOv8)."""
          y = [conv_bn_act(x, x.shape[-1] // 2, 1, name=f"{name}_cv1")]
          for i in range(3):
              y.append(layers.MaxPooling2D(k, strides=1, padding="same", name=f"{name}_pool{i}")(y[-1]))
          return conv_bn_act(layers.Concatenate(name=f"{name}_cat")(y), filters, 1, name=f"{name}_cv2")
    `),
  },

  cbam: {
    code: dedent(`
      def cbam(x, reduction: int = 16, k: int = 7, name: str = "cbam"):
          """Channel then spatial attention (Woo et al. 2018)."""
          c = x.shape[-1]
          fc1 = layers.Dense(max(8, c // reduction), activation="relu", name=f"{name}_fc1")
          fc2 = layers.Dense(c, name=f"{name}_fc2")
          avg = fc2(fc1(layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_avg")(x)))
          mx = fc2(fc1(layers.GlobalMaxPooling2D(keepdims=True, name=f"{name}_max")(x)))
          gate = layers.Activation("sigmoid", name=f"{name}_cgate")(layers.Add(name=f"{name}_cadd")([avg, mx]))
          x = layers.Multiply(name=f"{name}_cmul")([x, gate])
          s = layers.Concatenate(name=f"{name}_scat")([ops.mean(x, axis=-1, keepdims=True), ops.max(x, axis=-1, keepdims=True)])
          s = layers.Conv2D(1, k, padding="same", activation="sigmoid", name=f"{name}_spatial")(s)
          return layers.Multiply(name=f"{name}_smul")([x, s])
    `),
  },

  aspp: {
    deps: ['conv_bn_act'],
    code: dedent(`
      def aspp(x, filters: int = 256, rates: tuple[int, ...] = (6, 12, 18), dropout: float = 0.1, name: str = "aspp"):
          """Atrous spatial pyramid pooling (DeepLabV3)."""
          h, w = x.shape[1], x.shape[2]
          ys = [conv_bn_act(x, filters, 1, act="relu", name=f"{name}_b0")]
          ys += [conv_bn_act(x, filters, 3, d=r, act="relu", name=f"{name}_b{i + 1}") for i, r in enumerate(rates)]
          pooled = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_pool")(x)
          pooled = conv_bn_act(pooled, filters, 1, act="relu", name=f"{name}_pool_conv")
          ys.append(layers.Resizing(h, w, interpolation="bilinear", name=f"{name}_pool_up")(pooled))
          y = conv_bn_act(layers.Concatenate(name=f"{name}_cat")(ys), filters, 1, act="relu", name=f"{name}_project")
          return layers.Dropout(dropout, name=f"{name}_drop")(y)
    `),
  },

  repvgg_block: {
    deps: ['conv_bn_act'],
    code: dedent(`
      def repvgg_block(x, filters: int, stride: int = 1, name: str = "repvgg"):
          """Training-time RepVGG block (Ding et al. 2021)."""
          ys = [
              conv_bn_act(x, filters, 3, stride, act="none", name=f"{name}_3x3"),
              conv_bn_act(x, filters, 1, stride, act="none", name=f"{name}_1x1"),
          ]
          if stride == 1 and x.shape[-1] == filters:
              ys.append(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"{name}_id")(x))
          return layers.ReLU(name=f"{name}_out")(layers.Add(name=f"{name}_add")(ys))
    `),
  },
});
