# filepath: /Users/arkajyotisaha/Desktop/My-Thesis/code/src/model/transformer_tf.py
#!/usr/bin/env python3
"""
Transformer (Keras 3–safe) with optional Conv1D stem and a TCN‑style conv head.
• No Lambda layers → safe model save/load.
• MultiHeadAttention uses causal mask at call time (Keras 3 requirement).
• Optional Squeeze‑and‑Excitation in conv head.
• Pooling options: GAP | Attention | GAP+Attention.
"""
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, models

# ───────────────────────── utilities ─────────────────────────
def _act(name: str):
    name = (name or "relu").lower()
    if name == "gelu":  return tf.nn.gelu
    if name == "swish": return tf.nn.swish
    return tf.nn.relu

# ───────────────── custom, Keras‑3 safe layers ───────────────

@tf.keras.utils.register_keras_serializable(package="hft")
class PositionalEncoding1D(layers.Layer):
    def __init__(self, seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len, self.d_model = int(seq_len), int(d_model)
    def build(self, input_shape):
        import numpy as np
        L, D = self.seq_len, self.d_model
        pos = np.arange(L, dtype=np.float32)[:, None]
        i   = np.arange(D, dtype=np.float32)[None, :]
        angle = pos / np.power(10000.0, (2.0 * (i // 2)) / float(D))
        enc = np.zeros((L, D), dtype=np.float32)
        enc[:, 0::2] = np.sin(angle[:, 0::2])
        enc[:, 1::2] = np.cos(angle[:, 1::2])
        # Keras 3: shape is positional; name is keyword
        self.pe = self.add_weight(
            (L, D),
            initializer=tf.keras.initializers.Constant(enc),
            trainable=False,
            name="pos_encoding",
        )
        super().build(input_shape)
    def call(self, x):  # (B,L,D)
        return x + tf.expand_dims(self.pe, 0)
    def compute_output_shape(self, s): return s
    def get_config(self): return {"seq_len": self.seq_len, "d_model": self.d_model, **super().get_config()}

@tf.keras.utils.register_keras_serializable(package="hft")
class AttentionPool1D(layers.Layer):
    """Additive attention pooling over time; returns (B, D)."""
    def __init__(self, units: int, activation: str = "tanh", **kwargs):
        super().__init__(**kwargs)
        self.units, self.activation = int(units), activation.lower()
    def build(self, input_shape):
        D = int(input_shape[-1])
        self.W = self.add_weight((D, self.units), initializer="glorot_uniform", name="W")
        self.b = self.add_weight((self.units,), initializer="zeros", name="b")
        self.u = self.add_weight((self.units, 1), initializer="glorot_uniform", name="u")
        super().build(input_shape)
    def call(self, x):              # x: (B, L, D)
        h = tf.tensordot(x, self.W, axes=[[2],[0]]) + self.b   # (B,L,U)
        h = tf.nn.tanh(h) if self.activation == "tanh" else tf.nn.relu(h)
        e = tf.tensordot(h, self.u, axes=[[2],[0]])            # (B,L,1)
        a = tf.nn.softmax(e, axis=1)                           # (B,L,1)
        return tf.reduce_sum(x * a, axis=1)                    # (B,D)
    def compute_output_shape(self, s): return (s[0], s[2])
    def get_config(self): return {"units": self.units, "activation": self.activation, **super().get_config()}

@tf.keras.utils.register_keras_serializable(package="hft")
class SqueezeExcite1D(layers.Layer):
    def __init__(self, se_ratio: float = 0.25, **kwargs):
        super().__init__(**kwargs); self.se_ratio = float(se_ratio)
    def build(self, input_shape):
        C = int(input_shape[-1]); r = max(1, int(C * self.se_ratio))
        self.fc1 = layers.Dense(r, activation="relu"); self.fc2 = layers.Dense(C, activation="sigmoid")
        super().build(input_shape)
    def call(self, x):
        s = tf.reduce_mean(x, axis=1, keepdims=True)  # (B,1,C)
        z = self.fc2(self.fc1(s))
        return x * z
    def get_config(self): return {"se_ratio": self.se_ratio, **super().get_config()}

# ───────────────── building blocks ───────────────────────────

def _encoder_block(x, i: int, d_model: int, n_heads: int, ff_dim: int,
                   dropout: float, norm_eps: float = 1e-5, attn_dropout: float = 0.0):
    attn = layers.MultiHeadAttention(
        num_heads=int(n_heads),
        key_dim=int(d_model // n_heads),
        dropout=float(attn_dropout),
        name=f"mha_{i}",
    )
    attn_out = attn(query=x, value=x, key=x, use_causal_mask=True)  # causal at call time (Keras 3)
    x = layers.Add(name=f"res_attn_{i}")([x, attn_out])
    x = layers.LayerNormalization(epsilon=norm_eps, name=f"ln_attn_{i}")(x)

    y = layers.Dense(ff_dim, activation="relu", name=f"ffn1_{i}")(x)
    if dropout and dropout > 0: y = layers.Dropout(dropout, name=f"drop_ffn_{i}")(y)
    y = layers.Dense(d_model, name=f"ffn2_{i}")(y)

    x = layers.Add(name=f"res_ffn_{i}")([x, y])
    x = layers.LayerNormalization(epsilon=norm_eps, name=f"ln_ffn_{i}")(x)
    return x

def _causal_pad(x, kernel: int, dilation: int):
    pad = int((kernel - 1) * dilation)
    if pad <= 0: return x
    return layers.ZeroPadding1D(padding=(pad, 0))(x)

def _residual_conv_block(x, i: int, filters: int, kernel: int, stride: int,
                         dilation: int = 1, separable: bool = True, activation: str = "relu",
                         dropout: float = 0.0, padding: str = "causal", use_se: bool = False, se_ratio: float = 0.25):
    act = _act(activation)
    z = x
    pad = "valid" if padding.lower() == "causal" else "same"
    if padding.lower() == "causal":
        z = _causal_pad(z, kernel, dilation)

    # Keras 3: SeparableConv1D initializers are split
    Conv = layers.SeparableConv1D if separable else layers.Conv1D
    kwargs = dict(
        filters=int(filters),
        kernel_size=int(kernel),
        strides=int(stride),
        dilation_rate=int(dilation),
        padding=pad,
        activation=None,
        use_bias=True,
        name=f"conv_{i}",
    )
    if separable:
        kwargs.update(
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
        )
    else:
        kwargs.update(kernel_initializer="he_normal")

    y = Conv(**kwargs)(z)
    y = layers.BatchNormalization(name=f"bn_{i}")(y)
    y = layers.Activation(act, name=f"act_{i}")(y)
    if dropout and dropout > 0: y = layers.Dropout(float(dropout), name=f"drop_{i}")(y)

    # match channels if needed
    if int(x.shape[-1]) != int(filters):
        skip = layers.Conv1D(filters, 1, padding="same", name=f"skip_{i}")(x)
    else:
        skip = x

    out = layers.Add(name=f"res_add_{i}")([skip, y])
    if use_se:
        out = SqueezeExcite1D(se_ratio=se_ratio, name=f"se_{i}")(out)
    return out

# ───────────────── model factory ─────────────────────────────

def create_transformer_model(
    *,
    seq_len: int,
    n_numeric: int,
    n_classes: int,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    ff_dim: int = 128,
    dropout: float = 0.1,
    # conv stem
    conv_layers: int = 0,
    conv_filters: int = 64,
    conv_kernel: int = 5,
    conv_stride: int = 1,
    conv_padding: str = "causal",
    conv_activation: str = "relu",
    # conv head
    conv_head_layers: int = 0,
    conv_head_filters: int = 64,
    conv_head_kernel: int = 5,
    conv_head_dilations: list[int] | None = None,
    conv_head_separable: bool = True,
    conv_head_activation: str = "relu",
    conv_head_dropout: float = 0.05,
    conv_head_padding: str = "causal",
    use_se: bool = True,
    se_ratio: float = 0.25,
    pool_type: str = "gap_attn",   # gap|attn|gap_attn
    head_hidden: int = 0,
    head_dropout: float = 0.05,
) -> tf.keras.Model:

    inp = layers.Input(shape=(seq_len, n_numeric), name="inp")

    # linear embed to d_model
    x = layers.Dense(d_model, name="embed")(inp)

    # optional conv stem (causal)
    for i in range(int(conv_layers or 0)):
        x = _residual_conv_block(
            x, i=i,
            filters=conv_filters, kernel=conv_kernel, stride=conv_stride,
            dilation=1, separable=False, activation=conv_activation,
            dropout=0.0, padding=conv_padding, use_se=False,
        )

    # add positional encoding
    x = PositionalEncoding1D(seq_len=seq_len, d_model=d_model, name="pos_enc")(x)

    # transformer encoder blocks
    for i in range(int(n_layers)):
        x = _encoder_block(
            x, i=i, d_model=d_model, n_heads=n_heads,
            ff_dim=ff_dim, dropout=dropout, norm_eps=1e-5, attn_dropout=0.0
        )

    # optional TCN‑style conv head (dilations)
    if conv_head_layers and int(conv_head_layers) > 0:
        dils = list(conv_head_dilations) if conv_head_dilations else [1, 2, 4]
        for i in range(int(conv_head_layers)):
            d = dils[i % len(dils)]
            x = _residual_conv_block(
                x, i=100+i,  # separate name scope
                filters=conv_head_filters, kernel=conv_head_kernel, stride=1,
                dilation=int(d), separable=bool(conv_head_separable),
                activation=conv_head_activation, dropout=conv_head_dropout,
                padding=conv_head_padding, use_se=use_se, se_ratio=se_ratio
            )

    # pooling
    p = pool_type.lower()
    heads = []
    if p in ("gap", "gap_attn"):
        heads.append(layers.GlobalAveragePooling1D(name="gap")(x))
    if p in ("attn", "gap_attn"):
        heads.append(AttentionPool1D(units=d_model, name="attn_pool")(x))
    h = heads[0] if len(heads) == 1 else layers.Concatenate(name="concat_pool")(heads)

    if head_hidden and int(head_hidden) > 0:
        h = layers.Dense(int(head_hidden), activation="relu", name="head_fc")(h)
        if head_dropout and head_dropout > 0: h = layers.Dropout(head_dropout, name="head_drop")(h)

    out = layers.Dense(n_classes, activation="softmax", name="out")(h)

    model = models.Model(inputs=inp, outputs=out, name="TransformerHFT")
    return model