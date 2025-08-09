# #!/usr/bin/env python3
# # src/train_tf.py   –  v3.4 (save label_shift used during training)
# from __future__ import annotations
# import argparse, yaml, json, os, sys
# import numpy as np, pandas as pd
# import tensorflow as tf
# tf.keras.backend.set_floatx("float32")
# try:
#     tf.config.experimental.enable_tensor_float_32_execution(True)
# except Exception:
#     pass

# from tensorflow.keras.callbacks import EarlyStopping
# from src.utils import data_utils
# from src.model.transformer_tf import create_transformer_model

# def _smart_save(model: tf.keras.Model, path: str) -> None:
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     ext = os.path.splitext(path)[1].lower()
#     if ext == ".keras":
#         model.save(path)  # Keras 3 infers from extension
#     elif ext in {".h5",".hdf5"}:
#         model.save(path, save_format="h5")
#     else:
#         raise ValueError("Filename must end with .keras or .h5/.hdf5")

# def _load_cfg(cfg_fp: str) -> dict:
#     with open(cfg_fp, "r") as f: cfg = yaml.safe_load(f)
#     for k in ("data","model","training","model_type"):
#         if k not in cfg: sys.exit(f"[config] missing top‑level key: {k}")
#     return cfg

# def main(cfg_fp: str) -> None:
#     cfg = _load_cfg(cfg_fp)
#     tr = pd.read_parquet(cfg["data"]["train_file"])
#     va = pd.read_parquet(cfg["data"]["val_file"])
#     te = pd.read_parquet(cfg["data"]["test_file"]) if cfg["data"].get("test_file") else None

#     label_col = cfg["data"]["label_col"]
#     time_col  = cfg["data"].get("time_col", "timestamp")

#     def split(df: pd.DataFrame):
#         if time_col in df.columns:
#             df = df.sort_values(time_col)
#         y  = df[label_col].to_numpy()
#         X  = df.drop(columns=[c for c in [label_col, time_col] if c in df.columns])
#         return X, y

#     Xtr_df, ytr = split(tr); Xva_df, yva = split(va); Xte_df, yte = split(te) if te is not None else (None, None)

#     # Determine feature columns from TRAIN and align val/test
#     feat_cols = Xtr_df.select_dtypes(include=["number","bool"]).columns.tolist()
#     Xtr_df = Xtr_df[feat_cols].copy()
#     Xva_df = Xva_df[feat_cols].copy()
#     if Xte_df is not None: Xte_df = Xte_df[feat_cols].copy()

#     # Labels → 0‑based contiguous ints (SAVE THE SHIFT)
#     shift = int(-ytr.min()) if ytr.min() < 0 else 0
#     ytr = ytr + shift
#     yva = yva + shift
#     yte = (yte + shift) if yte is not None else None
#     n_classes = int(max(ytr.max(), yva.max()) + 1)

#     # Scale
#     Xtr_np, Xva_np, Xte_np, scaler = data_utils.scale_features(
#         Xtr_df.to_numpy(dtype=np.float32),
#         Xva_df.to_numpy(dtype=np.float32),
#         Xte_df.to_numpy(dtype=np.float32) if Xte_df is not None else None,
#     )
#     scaler["feature_names"] = feat_cols
#     scaler["label_shift"]   = shift        # ✅ persist label mapping used to train
#     scaler["label_order"]   = [-1, 0, 1]   # optional: for consistent reporting

#     # Sequences
#     L = int(cfg["model"]["seq_length"])
#     Xtr_seq, ytr_seq = data_utils.create_sequences(Xtr_np, ytr, L)
#     Xva_seq, yva_seq = data_utils.create_sequences(Xva_np, yva, L)

#     # Model
#     if cfg["model_type"].lower() != "transformer":
#         sys.exit("Only 'transformer' configured here.")
#     model = create_transformer_model(
#         seq_len   = L,
#         n_numeric = Xtr_seq.shape[2],
#         n_classes = n_classes,
#         d_model   = cfg["model"].get("d_model", 64),
#         n_heads   = cfg["model"].get("num_heads", 4),
#         n_layers  = cfg["model"].get("num_layers", 2),
#         ff_dim    = cfg["model"].get("ff_dim", 128),
#         dropout   = cfg["model"].get("dropout_rate", 0.1),
#     )

#     # LR override
#     lr = cfg["training"].get("learning_rate")
#     if lr is not None:
#         lr = float(lr)
#         try:
#             tf.keras.backend.set_value(model.optimizer.learning_rate, lr)
#         except Exception:
#             model.optimizer.learning_rate = lr

#     # Callbacks & class weights
#     callbacks = [EarlyStopping(monitor="val_loss",
#                                patience=int(cfg["training"].get("patience", 10)),
#                                restore_best_weights=True)]
#     if cfg["training"].get("tensorboard_dir"):
#         callbacks.append(tf.keras.callbacks.TensorBoard(cfg["training"]["tensorboard_dir"]))

#     class_w = data_utils.compute_class_weights(ytr_seq) if cfg["training"].get("class_weight") else None

#     # Train
#     model.fit(
#         Xtr_seq, ytr_seq,
#         validation_data=(Xva_seq, yva_seq),
#         epochs     = int(cfg["training"]["epochs"]),
#         batch_size = int(cfg["training"]["batch_size"]),
#         shuffle=True,
#         class_weight=class_w,
#         callbacks=callbacks,
#         verbose=2,
#     )

#     # Save
#     _smart_save(model, cfg["training"]["output_model_file"])
#     print("✓ model saved at", cfg["training"]["output_model_file"])
#     with open(cfg["training"]["output_scaler_file"], "w") as f:
#         json.dump(scaler, f)

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--config", required=True)
#     main(p.parse_args().config)



# #!/usr/bin/env python3
# # src/train_tf.py   –  v3.4 (save label_shift used during training)
# from __future__ import annotations
# import argparse, yaml, json, os, sys
# import numpy as np, pandas as pd
# import tensorflow as tf
# tf.keras.backend.set_floatx("float32")
# try:
#     tf.config.experimental.enable_tensor_float_32_execution(True)
# except Exception:
#     pass

# from tensorflow.keras.callbacks import EarlyStopping
# from src.utils import data_utils
# from src.model.transformer_tf import create_transformer_model

# def _smart_save(model: tf.keras.Model, path: str) -> None:
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     ext = os.path.splitext(path)[1].lower()
#     if ext == ".keras":
#         model.save(path)  # Keras 3 infers from extension
#     elif ext in {".h5",".hdf5"}:
#         model.save(path, save_format="h5")
#     else:
#         raise ValueError("Filename must end with .keras or .h5/.hdf5")

# def _load_cfg(cfg_fp: str) -> dict:
#     with open(cfg_fp, "r") as f: cfg = yaml.safe_load(f)
#     for k in ("data","model","training","model_type"):
#         if k not in cfg: sys.exit(f"[config] missing top‑level key: {k}")
#     return cfg

# def main(cfg_fp: str) -> None:
#     cfg = _load_cfg(cfg_fp)
#     tr = pd.read_parquet(cfg["data"]["train_file"])
#     va = pd.read_parquet(cfg["data"]["val_file"])
#     te = pd.read_parquet(cfg["data"]["test_file"]) if cfg["data"].get("test_file") else None

#     label_col = cfg["data"]["label_col"]
#     time_col  = cfg["data"].get("time_col", "timestamp")

#     def split(df: pd.DataFrame):
#         if df is None: return None, None
#         if time_col in df.columns: df = df.sort_values(time_col)
#         y  = df[label_col].to_numpy()
#         X  = df.drop(columns=[c for c in [label_col, time_col] if c in df.columns])
#         return X, y

#     Xtr_df, ytr = split(tr); Xva_df, yva = split(va); Xte_df, yte = split(te)

#     # Determine feature columns from TRAIN and align val/test
#     feat_cols = Xtr_df.select_dtypes(include=["number","bool"]).columns.tolist()
#     Xtr_df = Xtr_df[feat_cols].copy()
#     Xva_df = Xva_df[feat_cols].copy()
#     if Xte_df is not None: Xte_df = Xte_df[feat_cols].copy()

#     # Labels → 0‑based contiguous ints (SAVE THE SHIFT)
#     shift = int(-ytr.min()) if ytr.min() < 0 else 0
#     ytr = ytr + shift
#     yva = yva + shift
#     yte = (yte + shift) if yte is not None else None
#     n_classes = int(max(ytr.max(), yva.max()) + 1)

#     # Scale
#     Xtr_np, Xva_np, Xte_np, scaler = data_utils.scale_features(
#         Xtr_df.to_numpy(dtype=np.float32),
#         Xva_df.to_numpy(dtype=np.float32),
#         Xte_df.to_numpy(dtype=np.float32) if Xte_df is not None else None,
#     )
#     scaler["feature_names"] = feat_cols
#     scaler["label_shift"]   = shift        # ✅ persist label mapping used to train
#     scaler["label_order"]   = [-1, 0, 1]   # optional: for consistent reporting

#     # Sequences
#     L = int(cfg["model"]["seq_length"])
#     Xtr_seq, ytr_seq = data_utils.create_sequences(Xtr_np, ytr, L)
#     Xva_seq, yva_seq = data_utils.create_sequences(Xva_np, yva, L)

#     # Model
#     if cfg["model_type"].lower() != "transformer":
#         sys.exit("Only 'transformer' configured here.")
#     model = create_transformer_model(
#         seq_len   = L,
#         n_numeric = Xtr_seq.shape[2],
#         n_classes = n_classes,
#         d_model   = cfg["model"].get("d_model", 64),
#         n_heads   = cfg["model"].get("num_heads", 4),
#         n_layers  = cfg["model"].get("num_layers", 2),
#         ff_dim    = cfg["model"].get("ff_dim", 128),
#         dropout   = cfg["model"].get("dropout_rate", 0.1),
#     )

#     # LR override
#     lr = cfg["training"].get("learning_rate")
#     if lr is not None:
#         lr = float(lr)
#         try:
#             tf.keras.backend.set_value(model.optimizer.learning_rate, lr)
#         except Exception:
#             model.optimizer.learning_rate = lr

#     # Callbacks & class weights
#     callbacks = [EarlyStopping(monitor="val_loss",
#                                patience=int(cfg["training"].get("patience", 10)),
#                                restore_best_weights=True)]
#     if cfg["training"].get("tensorboard_dir"):
#         callbacks.append(tf.keras.callbacks.TensorBoard(cfg["training"]["tensorboard_dir"]))

#     class_w = data_utils.compute_class_weights(ytr_seq) if cfg["training"].get("class_weight") else None

#     # Train
#     model.fit(
#         Xtr_seq, ytr_seq,
#         validation_data=(Xva_seq, yva_seq),
#         epochs     = int(cfg["training"]["epochs"]),
#         batch_size = int(cfg["training"]["batch_size"]),
#         shuffle=True,
#         class_weight=class_w,
#         callbacks=callbacks,
#         verbose=2,
#     )

#     # Save
#     _smart_save(model, cfg["training"]["output_model_file"])
#     print("✓ model saved at", cfg["training"]["output_model_file"])
#     with open(cfg["training"]["output_scaler_file"], "w") as f:
#         json.dump(scaler, f)

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--config", required=True)
#     main(p.parse_args().config)


#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from src.model.transformer_tf import create_transformer_model
from src.utils import data_utils

# deterministic-ish
tf.random.set_seed(42)
np.random.seed(42)

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(cfg_path: str):
    cfg = load_yaml(cfg_path)

    train_fp = cfg["data"]["train_file"]
    val_fp   = cfg["data"]["val_file"]
    test_fp  = cfg["data"]["test_file"]
    label_col = cfg["data"].get("label_col", "y")
    time_col  = cfg["data"].get("time_col", "timestamp")

    # Load features/labels (raw labels in {-1,0,1})
    Xtr_df, ytr_raw = data_utils.load_features_labels(train_fp, label_col, drop_cols=time_col)
    Xva_df, yva_raw = data_utils.load_features_labels(val_fp,   label_col, drop_cols=time_col)
    Xte_df, yte_raw = data_utils.load_features_labels(test_fp,  label_col, drop_cols=time_col)

    # Scale on train; apply to others
    Xtr = Xtr_df.to_numpy(dtype=np.float32, copy=False)
    Xva = Xva_df.to_numpy(dtype=np.float32, copy=False)
    Xte = Xte_df.to_numpy(dtype=np.float32, copy=False)
    Xtr_s, Xva_s, Xte_s, scaler = data_utils.scale_features(Xtr, Xva, Xte)

    # Persist meta for eval alignment
    scaler["feature_names"] = list(Xtr_df.columns)
    scaler["label_order"]   = [-1, 0, 1]
    scaler["label_shift"]   = 1  # we will map {-1,0,1} -> {0,1,2}

    # Show BOTH raw and shifted label distributions (clarity)
    def _dist(arr: np.ndarray) -> dict:
        u, c = np.unique(arr, return_counts=True)
        return {int(k): int(v) for k, v in zip(u, c)}

    print(f"train label counts (raw): {_dist(ytr_raw)}")
    print(f"val   label counts (raw): {_dist(yva_raw)}")
    print(f"test  label counts (raw): {_dist(yte_raw)}")

    SHIFT = scaler["label_shift"]
    ytr = (ytr_raw + SHIFT).astype(np.int32)
    yva = (yva_raw + SHIFT).astype(np.int32)
    yte = (yte_raw + SHIFT).astype(np.int32)

    print(f"train label distribution (shifted for Keras): {_dist(ytr)}")
    print(f"val   label distribution (shifted for Keras): {_dist(yva)}")

    # Build rolling sequences
    L = int(cfg["model"]["seq_length"])
    Xtr_seq, ytr_seq = data_utils.create_sequences(Xtr_s, ytr, L)
    Xva_seq, yva_seq = data_utils.create_sequences(Xva_s, yva, L)
    # test is unused for training; we just verify dims here
    _ = data_utils.create_sequences(Xte_s, yte, L)

    # Class weights (shifted space {0,1,2})
    cw = data_utils.compute_class_weights(ytr_seq) if cfg["training"].get("class_weight", True) else None

    # Model
    n_features = Xtr_seq.shape[-1]
    n_classes  = len(scaler["label_order"])  # 3
    model = create_transformer_model(
        seq_len=L,
        n_numeric=n_features,
        n_classes=n_classes,
        d_model=int(cfg["model"]["d_model"]),
        n_heads=int(cfg["model"]["num_heads"]),
        n_layers=int(cfg["model"]["num_layers"]),
        ff_dim=int(cfg["model"]["ff_dim"]),
        dropout=float(cfg["model"]["dropout_rate"]),
    )

    # Optimizer with LR from config
    lr = float(cfg["training"].get("learning_rate", 5e-4))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Training
    epochs     = int(cfg["training"]["epochs"])
    batch_size = int(cfg["training"]["batch_size"])
    patience   = int(cfg["training"].get("patience", 100))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(3, patience//4), verbose=1, min_lr=1e-6
        ),
    ]

    history = model.fit(
        Xtr_seq, ytr_seq,
        validation_data=(Xva_seq, yva_seq),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw,
        verbose=2,
        callbacks=callbacks,
    )

    # Save model and scaler
    out_model = cfg["training"]["output_model_file"]
    out_scaler = cfg["training"]["output_scaler_file"]
    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    Path(out_scaler).parent.mkdir(parents=True, exist_ok=True)

    model.save(out_model)  # .keras
    with open(out_scaler, "w") as f:
        json.dump(scaler, f, indent=2)

    print(f"✓ model saved at {out_model}")
    print(f"✓ scaler saved at {out_scaler}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)