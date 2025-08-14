# filepath: /Users/arkajyotisaha/Desktop/My-Thesis/code/train_tf.py
#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, yaml
import numpy as np, pandas as pd
import tensorflow as tf
tf.keras.backend.set_floatx("float32")
try:
    tf.config.experimental.enable_tensor_float_32_execution(True)
except Exception:
    pass

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from src.utils import data_utils
from src.model.transformer_tf import create_transformer_model

def _smart_save(model: tf.keras.Model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".keras":
        model.save(path)
    elif ext in {".h5", ".hdf5"}:
        model.save(path, save_format="h5")
    else:
        raise ValueError("Filename must end with .keras or .h5/.hdf5")

def _load_cfg(cfg_fp: str) -> dict:
    with open(cfg_fp, "r") as f:
        cfg = yaml.safe_load(f)
    for k in ("data","model","training","model_type"):
        if k not in cfg: sys.exit(f"[config] missing top‑level key: {k}")
    return cfg

def _macro_f1_sparse(y_true, y_pred):
    # y_true is integer (sparse), y_pred is prob
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred = tf.argmax(y_pred, axis=1, output_type=tf.int32)
    C = tf.shape(y_pred)[0]
    n_classes = tf.shape(tf.unique(y_true)[0])[0]  # fallback, unused below
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=tf.shape(y_pred)[-1] if y_pred.shape.rank > 0 else 3, dtype=tf.float32)
    tp = tf.linalg.diag_part(cm)
    fp = tf.reduce_sum(cm, axis=0) - tp
    fn = tf.reduce_sum(cm, axis=1) - tp
    denom = 2.0*tp + fp + fn + 1e-7
    f1 = tf.where(denom > 0, 2.0*tp/denom, 0.0)
    return tf.reduce_mean(f1)

def _compute_class_weights(y: np.ndarray, cap: float, floor: float, power: float) -> dict[int, float]:
    counts = np.bincount(y.astype(int))
    K = (counts > 0).sum()
    N = counts.sum()
    inv = np.zeros_like(counts, dtype=np.float64)
    nz  = counts > 0
    inv[nz] = N / (K * counts[nz])
    inv = np.power(inv, float(power))
    inv = np.clip(inv, float(floor), float(cap))
    return {int(i): float(w) for i, w in enumerate(inv) if counts[i] > 0}

def main(cfg_fp: str) -> None:
    cfg = _load_cfg(cfg_fp)
    assert cfg["model_type"].lower() == "transformer", "Only 'transformer' supported here."

    # ── load data
    tr = pd.read_parquet(cfg["data"]["train_file"])
    va = pd.read_parquet(cfg["data"]["val_file"])
    te = pd.read_parquet(cfg["data"]["test_file"]) if cfg["data"].get("test_file") else None

    label_col = cfg["data"].get("label_col", "y")
    time_col  = cfg["data"].get("time_col", "timestamp")

    def split(df: pd.DataFrame):
        if df is None: return None, None
        if time_col in df.columns:
            df = df.sort_values(time_col)
        y  = df[label_col].to_numpy()
        X  = df.drop(columns=[c for c in [label_col, time_col] if c in df.columns])
        return X, y

    Xtr_df, ytr_raw = split(tr); Xva_df, yva_raw = split(va); Xte_df, yte_raw = split(te)

    # print label stats (raw)
    def _cnt(y): from collections import Counter; return dict(Counter(y.tolist()))
    print("train label counts (raw):", _cnt(ytr_raw))
    print("val   label counts (raw):", _cnt(yva_raw))
    if yte_raw is not None:
        print("test  label counts (raw):", _cnt(yte_raw))

    # Determine feature columns from TRAIN and align val/test
    feat_cols = Xtr_df.select_dtypes(include=["number","bool"]).columns.tolist()
    Xtr_df = Xtr_df[feat_cols].copy()
    Xva_df = Xva_df[feat_cols].copy()
    if Xte_df is not None: Xte_df = Xte_df[feat_cols].copy()

    # Labels → 0‑based contiguous ints (SAVE THE SHIFT)
    shift = int(-ytr_raw.min()) if ytr_raw.min() < 0 else 0
    ytr = ytr_raw + shift
    yva = yva_raw + shift
    yte = (yte_raw + shift) if yte_raw is not None else None

    print("train label distribution (shifted):", _cnt(ytr))
    print("val   label distribution (shifted):", _cnt(yva))

    n_classes = int(max(ytr.max(), yva.max()) + 1)

    # Scale
    Xtr_np, Xva_np, Xte_np, scaler = data_utils.scale_features(
        Xtr_df.to_numpy(dtype=np.float32),
        Xva_df.to_numpy(dtype=np.float32),
        Xte_df.to_numpy(dtype=np.float32) if Xte_df is not None else None,
    )
    scaler["feature_names"] = feat_cols
    scaler["label_shift"]   = int(shift)
    # keep explicit order for reporting (harmless if n_classes!=3)
    scaler["label_order"]   = [-1, 0, 1][:n_classes]

    # Sequences
    L = int(cfg["model"].get("seq_length") or cfg["model"].get("seq_length".replace("seq_length","seq_length")))
    Xtr_seq, ytr_seq = data_utils.create_sequences(Xtr_np, ytr, L)
    Xva_seq, yva_seq = data_utils.create_sequences(Xva_np, yva, L)

    # Model (pass optional conv stem/head params if present)
    mcfg = cfg["model"]
    model = create_transformer_model(
        seq_len=L,
        n_numeric=Xtr_seq.shape[2],
        n_classes=n_classes,
        d_model=int(mcfg.get("d_model", 64)),
        n_heads=int(mcfg.get("num_heads", 4)),
        n_layers=int(mcfg.get("num_layers", 2)),
        ff_dim=int(mcfg.get("ff_dim", 128)),
        dropout=float(mcfg.get("dropout_rate", 0.1)),
        conv_layers=int(mcfg.get("conv_layers", 0) or 0),
        conv_filters=int(mcfg.get("conv_filters", 64)),
        conv_kernel=int(mcfg.get("conv_kernel", 5)),
        conv_stride=int(mcfg.get("conv_stride", 1)),
        conv_padding=str(mcfg.get("conv_padding", "causal")),
        conv_activation=str(mcfg.get("conv_activation", "relu")),
        conv_head_layers=int(mcfg.get("conv_head_layers", 0) or 0),
        conv_head_filters=int(mcfg.get("conv_head_filters", 64)),
        conv_head_kernel=int(mcfg.get("conv_head_kernel", 5)),
        conv_head_dilations=mcfg.get("conv_head_dilations", [1,2,4]),
        conv_head_separable=bool(mcfg.get("conv_head_separable", True)),
        conv_head_activation=str(mcfg.get("conv_head_activation", "relu")),
        conv_head_dropout=float(mcfg.get("conv_head_dropout", 0.0)),
        conv_head_padding=str(mcfg.get("conv_head_padding", "causal")),
        use_se=bool(mcfg.get("use_se", True)),
        se_ratio=float(mcfg.get("se_ratio", 0.25)),
        pool_type=str(mcfg.get("pool_type", "gap_attn")),
        head_dropout=float(mcfg.get("head_dropout", 0.0)),
        head_hidden=int(mcfg.get("head_hidden", 0)),
    )

    # Compile
    tcfg = cfg["training"]
    lr = float(tcfg.get("learning_rate", 5e-4))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", _macro_f1_sparse],
    )

    # Callbacks
    cbs = [
        ModelCheckpoint(
            filepath=tcfg["output_model_file"],
            monitor=str(tcfg.get("monitor", "val_accuracy")),
            save_best_only=True,
            save_weights_only=False,
            mode="max" if "acc" in str(tcfg.get("monitor", "val_accuracy")) else "auto",
            verbose=1,
        ),
        EarlyStopping(
            monitor=str(tcfg.get("monitor", "val_accuracy")),
            patience=int(tcfg.get("patience", 40)),
            restore_best_weights=True,
            mode="max" if "acc" in str(tcfg.get("monitor", "val_accuracy")) else "auto",
        ),
    ]
    if bool(tcfg.get("use_reduce_lr", True)):
        cbs.append(ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5, patience=10, min_lr=1e-6, verbose=1
        ))

    # Class weights (capped/floored/powered)
    class_w = None
    if bool(tcfg.get("class_weight", True)):
        class_w = _compute_class_weights(
            ytr_seq,
            cap=float(tcfg.get("max_class_weight", 5.0)),
            floor=float(tcfg.get("min_class_weight", 1.0)),
            power=float(tcfg.get("class_weight_power", 1.0)),
        )
        print("[info] class_weight used (after cap):", class_w)

    # Fit
    history = model.fit(
        Xtr_seq, ytr_seq,
        validation_data=(Xva_seq, yva_seq),
        epochs=int(tcfg.get("epochs", 120)),
        batch_size=int(tcfg.get("batch_size", 128)),
        shuffle=True,
        class_weight=class_w,
        callbacks=cbs,
        verbose=2,
    )

    # Save scaler JSON
    os.makedirs(os.path.dirname(tcfg["output_scaler_file"]), exist_ok=True)
    with open(tcfg["output_scaler_file"], "w") as f:
        json.dump(scaler, f)
    print("✓ model saved at", tcfg["output_model_file"])
    print("✓ scaler saved at", tcfg["output_scaler_file"])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)