# """
# Transformer helpers – V2 (Lambda‑free pooling).
# """
# from __future__ import annotations
# import tensorflow as tf
# from tensorflow.keras import layers, models

# def _positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
#     pos  = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
#     idx  = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
#     angle = 1.0 / tf.pow(10_000.0, (2 * (idx // 2)) / d_model)
#     rad   = pos * angle
#     sines = tf.sin(rad[:, 0::2]);  cos = tf.cos(rad[:, 1::2])
#     enc   = tf.concat([sines, cos], axis=-1)
#     return tf.expand_dims(enc, 0)  # (1, L, D)

# def create_transformer_model(
#     *,
#     seq_len: int,
#     n_numeric: int,
#     n_classes: int = 3,
#     d_model: int   = 64,
#     n_heads: int   = 4,
#     n_layers: int  = 2,
#     ff_dim: int    = 128,
#     dropout: float = 0.1,
# ) -> tf.keras.Model:
#     assert d_model % n_heads == 0, "d_model must be divisible by num_heads"
#     inp = layers.Input(shape=(seq_len, n_numeric), dtype="float32", name="num")
#     x = inp if n_numeric == d_model else layers.Dense(d_model, name="proj")(inp)

#     pe = _positional_encoding(seq_len, d_model)
#     x  = layers.Add(name="add_pos")([x, pe])
#     if dropout: x = layers.Dropout(dropout)(x)

#     for i in range(n_layers):
#         attn = layers.MultiHeadAttention(n_heads, key_dim=d_model // n_heads,
#                                          dropout=dropout, name=f"mha_{i}")(x, x)
#         x = layers.LayerNormalization(epsilon=1e-6)(x + attn)
#         ffn = layers.Dense(ff_dim, activation="relu")(x)
#         if dropout: ffn = layers.Dropout(dropout)(ffn)
#         ffn = layers.Dense(d_model)(ffn)
#         x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)

#     x = layers.GlobalAveragePooling1D(name="gap")(x)
#     out = layers.Dense(n_classes, activation="softmax")(x)

#     model = models.Model(inp, out, name="TransformerTF")
#     model.compile(optimizer=tf.keras.optimizers.Adam(),
#                   loss="sparse_categorical_crossentropy",
#                   metrics=["accuracy"])
#     return model



"""
Transformer helpers – V3 (class-weight friendly, stable, no Lambda).
Adds: optional dropout stack and configurable norm-eps.
"""
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, models

def _positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    pos  = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
    idx  = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
    angle = 1.0 / tf.pow(10_000.0, (2 * (idx // 2)) / d_model)
    rad   = pos * angle
    sines = tf.sin(rad[:, 0::2]);  cos = tf.cos(rad[:, 1::2])
    enc   = tf.concat([sines, cos], axis=-1)
    return tf.expand_dims(enc, 0)  # (1, L, D)

def create_transformer_model(
    *,
    seq_len: int,
    n_numeric: int,
    n_classes: int = 3,
    d_model: int   = 64,
    n_heads: int   = 4,
    n_layers: int  = 2,
    ff_dim: int    = 128,
    dropout: float = 0.1,
    norm_eps: float = 1e-6,
) -> tf.keras.Model:
    assert d_model % n_heads == 0, "d_model must be divisible by num_heads"
    inp = layers.Input(shape=(seq_len, n_numeric), dtype="float32", name="num")
    x = inp if n_numeric == d_model else layers.Dense(d_model, name="proj")(inp)

    pe = _positional_encoding(seq_len, d_model)
    x  = layers.Add(name="add_pos")([x, pe])
    if dropout: x = layers.Dropout(dropout)(x)

    for i in range(n_layers):
        attn = layers.MultiHeadAttention(
            n_heads, key_dim=d_model // n_heads, dropout=dropout, name=f"mha_{i}"
        )(x, x)
        x = layers.LayerNormalization(epsilon=norm_eps, name=f"ln_attn_{i}")(x + attn)
        ffn = layers.Dense(ff_dim, activation="relu", name=f"ffn1_{i}")(x)
        if dropout: ffn = layers.Dropout(dropout, name=f"drop_ffn_{i}")(ffn)
        ffn = layers.Dense(d_model, name=f"ffn2_{i}")(ffn)
        x = layers.LayerNormalization(epsilon=norm_eps, name=f"ln_ffn_{i}")(x + ffn)

    # simple CLS surrogate via average pooling
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    if dropout: x = layers.Dropout(dropout, name="head_drop")(x)
    logits = layers.Dense(n_classes, name="logits")(x)
    out = layers.Softmax(name="softmax")(logits)

    model = models.Model(inp, out, name="TransformerTF")
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model