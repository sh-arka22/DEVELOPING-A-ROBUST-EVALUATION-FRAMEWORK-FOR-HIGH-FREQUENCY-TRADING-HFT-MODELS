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


# #!/usr/bin/env python3
# from __future__ import annotations
# import argparse, os, json
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import yaml

# from src.model.transformer_tf import create_transformer_model
# from src.utils import data_utils

# # deterministic-ish
# tf.random.set_seed(42)
# np.random.seed(42)

# def load_yaml(path: str) -> dict:
#     with open(path, "r") as f:
#         return yaml.safe_load(f)

# def main(cfg_path: str):
#     cfg = load_yaml(cfg_path)

#     train_fp = cfg["data"]["train_file"]
#     val_fp   = cfg["data"]["val_file"]
#     test_fp  = cfg["data"]["test_file"]
#     label_col = cfg["data"].get("label_col", "y")
#     time_col  = cfg["data"].get("time_col", "timestamp")

#     # Load features/labels (raw labels in {-1,0,1})
#     Xtr_df, ytr_raw = data_utils.load_features_labels(train_fp, label_col, drop_cols=time_col)
#     Xva_df, yva_raw = data_utils.load_features_labels(val_fp,   label_col, drop_cols=time_col)
#     Xte_df, yte_raw = data_utils.load_features_labels(test_fp,  label_col, drop_cols=time_col)

#     # Scale on train; apply to others
#     Xtr = Xtr_df.to_numpy(dtype=np.float32, copy=False)
#     Xva = Xva_df.to_numpy(dtype=np.float32, copy=False)
#     Xte = Xte_df.to_numpy(dtype=np.float32, copy=False)
#     Xtr_s, Xva_s, Xte_s, scaler = data_utils.scale_features(Xtr, Xva, Xte)

#     # Persist meta for eval alignment
#     scaler["feature_names"] = list(Xtr_df.columns)
#     scaler["label_order"]   = [-1, 0, 1]
#     scaler["label_shift"]   = 1  # we will map {-1,0,1} -> {0,1,2}

#     # Show BOTH raw and shifted label distributions (clarity)
#     def _dist(arr: np.ndarray) -> dict:
#         u, c = np.unique(arr, return_counts=True)
#         return {int(k): int(v) for k, v in zip(u, c)}

#     print(f"train label counts (raw): {_dist(ytr_raw)}")
#     print(f"val   label counts (raw): {_dist(yva_raw)}")
#     print(f"test  label counts (raw): {_dist(yte_raw)}")

#     SHIFT = scaler["label_shift"]
#     ytr = (ytr_raw + SHIFT).astype(np.int32)
#     yva = (yva_raw + SHIFT).astype(np.int32)
#     yte = (yte_raw + SHIFT).astype(np.int32)

#     print(f"train label distribution (shifted for Keras): {_dist(ytr)}")
#     print(f"val   label distribution (shifted for Keras): {_dist(yva)}")

#     # Build rolling sequences
#     L = int(cfg["model"]["seq_length"])
#     Xtr_seq, ytr_seq = data_utils.create_sequences(Xtr_s, ytr, L)
#     Xva_seq, yva_seq = data_utils.create_sequences(Xva_s, yva, L)
#     # test is unused for training; we just verify dims here
#     _ = data_utils.create_sequences(Xte_s, yte, L)

#     # Class weights (shifted space {0,1,2})
#     cw = data_utils.compute_class_weights(ytr_seq) if cfg["training"].get("class_weight", True) else None

#     # Model
#     n_features = Xtr_seq.shape[-1]
#     n_classes  = len(scaler["label_order"])  # 3
#     model = create_transformer_model(
#         seq_len=L,
#         n_numeric=n_features,
#         n_classes=n_classes,
#         d_model=int(cfg["model"]["d_model"]),
#         n_heads=int(cfg["model"]["num_heads"]),
#         n_layers=int(cfg["model"]["num_layers"]),
#         ff_dim=int(cfg["model"]["ff_dim"]),
#         dropout=float(cfg["model"]["dropout_rate"]),
#     )

#     # Optimizer with LR from config
#     lr = float(cfg["training"].get("learning_rate", 5e-4))
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"],
#     )

#     # Training
#     epochs     = int(cfg["training"]["epochs"])
#     batch_size = int(cfg["training"]["batch_size"])
#     patience   = int(cfg["training"].get("patience", 100))

#     callbacks = [
#         tf.keras.callbacks.EarlyStopping(
#             monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
#         ),
#         tf.keras.callbacks.ReduceLROnPlateau(
#             monitor="val_loss", factor=0.5, patience=max(3, patience//4), verbose=1, min_lr=1e-6
#         ),
#     ]

#     history = model.fit(
#         Xtr_seq, ytr_seq,
#         validation_data=(Xva_seq, yva_seq),
#         epochs=epochs,
#         batch_size=batch_size,
#         class_weight=cw,
#         verbose=2,
#         callbacks=callbacks,
#     )

#     # Save model and scaler
#     out_model = cfg["training"]["output_model_file"]
#     out_scaler = cfg["training"]["output_scaler_file"]
#     Path(out_model).parent.mkdir(parents=True, exist_ok=True)
#     Path(out_scaler).parent.mkdir(parents=True, exist_ok=True)

#     model.save(out_model)  # .keras
#     with open(out_scaler, "w") as f:
#         json.dump(scaler, f, indent=2)

#     print(f"✓ model saved at {out_model}")
#     print(f"✓ scaler saved at {out_scaler}")

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", required=True)
#     args = ap.parse_args()
#     main(args.config)


#!/usr/bin/env python3
# Train Transformer on {-1,0,1} labels produced by build_features.py
# - Drops time column from inputs, but uses it (when present) only for ordering upstream.
# - Persists scaler stats + feature order + label_shift for eval alignment.

# from __future__ import annotations
# import argparse, json
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import yaml

# from src.model.transformer_tf import create_transformer_model
# from src.utils import data_utils

# # deterministic-ish
# tf.random.set_seed(42)
# np.random.seed(42)

# def load_yaml(path: str) -> dict:
#     with open(path, "r") as f:
#         return yaml.safe_load(f)

# def main(cfg_path: str):
#     cfg = load_yaml(cfg_path)

#     train_fp = cfg["data"]["train_file"]
#     val_fp   = cfg["data"]["val_file"]
#     test_fp  = cfg["data"].get("test_file") or val_fp  # safe fallback
#     label_col = cfg["data"].get("label_col", "y")
#     time_col  = cfg["data"].get("time_col", "timestamp")

#     # Load features/labels (labels in {-1,0,1})
#     Xtr_df, ytr_raw = data_utils.load_features_labels(train_fp, label_col, drop_cols=time_col)
#     Xva_df, yva_raw = data_utils.load_features_labels(val_fp,   label_col, drop_cols=time_col)
#     Xte_df, yte_raw = data_utils.load_features_labels(test_fp,  label_col, drop_cols=time_col)

#     # Scale using TRAIN stats; apply to others
#     Xtr = Xtr_df.to_numpy(dtype=np.float32, copy=False)
#     Xva = Xva_df.to_numpy(dtype=np.float32, copy=False)
#     Xte = Xte_df.to_numpy(dtype=np.float32, copy=False)
#     Xtr_s, Xva_s, Xte_s, scaler = data_utils.scale_features(Xtr, Xva, Xte)

#     # Persist meta for eval alignment
#     scaler["feature_names"] = list(Xtr_df.columns)
#     scaler["label_order"]   = [-1, 0, 1]
#     scaler["label_shift"]   = 1  # map {-1,0,1} -> {0,1,2}

#     # Show distributions
#     def _dist(arr: np.ndarray) -> dict:
#         u, c = np.unique(arr, return_counts=True)
#         return {int(k): int(v) for k, v in zip(u, c)}
#     print(f"train label counts (raw): {_dist(ytr_raw)}")
#     print(f"val   label counts (raw): {_dist(yva_raw)}")
#     print(f"test  label counts (raw): {_dist(yte_raw)}")

#     SHIFT = scaler["label_shift"]
#     ytr = (ytr_raw + SHIFT).astype(np.int32)
#     yva = (yva_raw + SHIFT).astype(np.int32)
#     yte = (yte_raw + SHIFT).astype(np.int32)

#     print(f"train label distribution (shifted): {_dist(ytr)}")
#     print(f"val   label distribution (shifted): {_dist(yva)}")

#     # Sequences
#     L = int(cfg["model"]["seq_length"])
#     Xtr_seq, ytr_seq = data_utils.create_sequences(Xtr_s, ytr, L)
#     Xva_seq, yva_seq = data_utils.create_sequences(Xva_s, yva, L)
#     # (Optional) ensure test also sequences OK
#     _ = data_utils.create_sequences(Xte_s, yte, L)

#     # Class weights (in shifted space {0,1,2})
#     class_weights = data_utils.compute_class_weights(ytr_seq) if cfg["training"].get("class_weight", True) else None

#     # Model
#     n_features = Xtr_seq.shape[-1]
#     n_classes  = len(scaler["label_order"])  # 3
#     model = create_transformer_model(
#         seq_len=L,
#         n_numeric=n_features,
#         n_classes=n_classes,
#         d_model=int(cfg["model"].get("d_model", 64)),
#         n_heads=int(cfg["model"].get("num_heads", 4)),
#         n_layers=int(cfg["model"].get("num_layers", 2)),
#         ff_dim=int(cfg["model"].get("ff_dim", 128)),
#         dropout=float(cfg["model"].get("dropout_rate", 0.1)),
#     )

#     # Optimizer
#     lr = float(cfg["training"].get("learning_rate", 5e-4))
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"],
#     )

#     # Callbacks
#     epochs     = int(cfg["training"]["epochs"])
#     batch_size = int(cfg["training"]["batch_size"])
#     patience   = int(cfg["training"].get("patience", 20))
#     callbacks = [
#         tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
#                                          restore_best_weights=True, verbose=1),
#         tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
#                                              patience=max(3, patience//4), verbose=1, min_lr=1e-6),
#     ]

#     # Train
#     model.fit(
#         Xtr_seq, ytr_seq,
#         validation_data=(Xva_seq, yva_seq),
#         epochs=epochs,
#         batch_size=batch_size,
#         class_weight=class_weights,
#         verbose=2,
#         callbacks=callbacks,
#         shuffle=True,
#     )

#     # Save model and scaler
#     out_model  = cfg["training"]["output_model_file"]
#     out_scaler = cfg["training"]["output_scaler_file"]
#     Path(out_model).parent.mkdir(parents=True, exist_ok=True)
#     Path(out_scaler).parent.mkdir(parents=True, exist_ok=True)

#     model.save(out_model)  # .keras preferred
#     with open(out_scaler, "w") as f:
#         json.dump(scaler, f, indent=2)

#     print(f"✓ model saved at {out_model}")
#     print(f"✓ scaler saved at {out_scaler}")

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", required=True)
#     args = ap.parse_args()
#     main(args.config)

#!/usr/bin/env python3
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


# #!/usr/bin/env python3
# from __future__ import annotations
# import argparse, os, json
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import yaml

# from src.model.transformer_tf import create_transformer_model
# from src.utils import data_utils

# # deterministic-ish
# tf.random.set_seed(42)
# np.random.seed(42)

# def load_yaml(path: str) -> dict:
#     with open(path, "r") as f:
#         return yaml.safe_load(f)

# def main(cfg_path: str):
#     cfg = load_yaml(cfg_path)

#     train_fp = cfg["data"]["train_file"]
#     val_fp   = cfg["data"]["val_file"]
#     test_fp  = cfg["data"]["test_file"]
#     label_col = cfg["data"].get("label_col", "y")
#     time_col  = cfg["data"].get("time_col", "timestamp")

#     # Load features/labels (raw labels in {-1,0,1})
#     Xtr_df, ytr_raw = data_utils.load_features_labels(train_fp, label_col, drop_c<truncated__content/>
#!/usr/bin/env python3
# # #!/usr/bin/env python3
# #!/usr/bin/env python3
# """
# train_tf.py – Robust trainer with balanced-accuracy checkpointing

# Fixes collapse-to-one-class by:
#   • Computing val balanced accuracy each epoch via a callback
#   • Checkpointing the best model on val_bal_acc (mode='max')
#   • Capping class weights (default 5.0) to avoid extreme gradients
#   • Disabling focal loss unless explicitly requested (config flag default False)
#   • Saving scaler metadata: feature_names, label_shift, label_order
# """
# from __future__ import annotations
# import argparse, json, sys
# from pathlib import Path
# from typing import Dict, Tuple

# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import yaml
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# from src.model.transformer_tf import create_transformer_model
# from src.utils import data_utils

# tf.random.set_seed(42)
# np.random.seed(42)

# def _load_yaml(path: str) -> dict:
#     with open(path, "r") as f:
#         return yaml.safe_load(f)

# def _dist(y: np.ndarray) -> Dict[int, int]:
#     u, c = np.unique(y, return_counts=True)
#     return {int(k): int(v) for k, v in zip(u, c)}

# # ---- robust LR override (handles Variable / schedule / float / stray string)
# def _set_learning_rate(model: tf.keras.Model, new_lr) -> float:
#     new_lr = float(new_lr)
#     opt = model.optimizer
#     try:
#         lr_obj = getattr(opt, "learning_rate", getattr(opt, "lr", None))
#         if hasattr(lr_obj, "assign"):
#             lr_obj.assign(new_lr)
#         elif callable(lr_obj):
#             if hasattr(opt, "learning_rate"):
#                 opt.learning_rate = new_lr
#             else:
#                 opt.lr = new_lr
#         else:
#             if hasattr(opt, "learning_rate"):
#                 opt.learning_rate = new_lr
#             else:
#                 opt.lr = new_lr
#     except Exception:
#         try:
#             tf.keras.backend.set_value(getattr(opt, "learning_rate", getattr(opt, "lr")), new_lr)
#         except Exception:
#             if hasattr(opt, "learning_rate"):
#                 opt.learning_rate = new_lr
#             else:
#                 opt.lr = new_lr
#     try:
#         return float(tf.keras.backend.get_value(getattr(opt, "learning_rate", getattr(opt, "lr"))))
#     except Exception:
#         obj = getattr(opt, "learning_rate", getattr(opt, "lr", None))
#         return float(obj) if isinstance(obj, (int, float)) else new_lr

# # ---- balanced accuracy (macro recall) callback
# class ValBalancedAccuracy(tf.keras.callbacks.Callback):
#     def __init__(self, val_data, name: str = "val_bal_acc"):
#         super().__init__()
#         self.val_data = val_data
#         self.name = name

#     @staticmethod
#     def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> float:
#         eps = 1e-12
#         recalls = []
#         for cls in labels:
#             idx = (y_true == cls)
#             if idx.sum() == 0:
#                 recalls.append(0.0)
#             else:
#                 recalls.append(float((y_pred[idx] == cls).sum() / (idx.sum() + eps)))
#         return float(np.mean(recalls))

#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         Xva, yva = self.val_data
#         y_pred = np.argmax(self.model.predict(Xva, verbose=0), axis=1)
#         labels = np.array(sorted(np.unique(yva)))
#         bal = self._balanced_accuracy(yva, y_pred, labels)
#         logs[self.name] = bal
#         print(f"\n[epoch {epoch+1}] {self.name}: {bal:.4f}")

# def main(cfg_path: str):
#     cfg = _load_yaml(cfg_path)

#     # data paths
#     train_fp = cfg["data"]["train_file"]
#     val_fp   = cfg["data"]["val_file"]
#     test_fp  = cfg["data"].get("test_file")
#     label_col = cfg["data"].get("label_col", "y")
#     time_col  = cfg["data"].get("time_col", "timestamp")

#     # load split helper
#     def load_split(fp: str | None) -> Tuple[pd.DataFrame | None, np.ndarray | None]:
#         if not fp:
#             return None, None
#         df = pd.read_parquet(fp)
#         if time_col in df.columns:
#             df = df.sort_values(time_col)
#         if label_col not in df.columns:
#             raise KeyError(f"Label column '{label_col}' not found in {fp}")
#         y = df[label_col].to_numpy()
#         X = df.drop(columns=[c for c in [label_col, time_col] if c in df.columns])
#         X = X.select_dtypes(include=["number", "bool"]).copy()
#         return X, y

#     Xtr_df, ytr_raw = load_split(train_fp)
#     Xva_df, yva_raw = load_split(val_fp)
#     Xte_df, yte_raw = load_split(test_fp) if test_fp else (None, None)

#     # align features to train order
#     feat_cols = Xtr_df.columns.tolist()
#     Xtr_df = Xtr_df[feat_cols].copy()
#     Xva_df = Xva_df[feat_cols].copy()
#     if Xte_df is not None:
#         Xte_df = Xte_df[feat_cols].copy()

#     # log raw dists
#     print(f"train label counts (raw): {_dist(ytr_raw)}")
#     print(f"val   label counts (raw): {_dist(yva_raw)}")
#     if yte_raw is not None:
#         print(f"test  label counts (raw): {_dist(yte_raw)}")

#     # shift labels to 0..K-1
#     shift = int(-ytr_raw.min()) if ytr_raw.min() < 0 else 0
#     ytr = (ytr_raw + shift).astype(int)
#     yva = (yva_raw + shift).astype(int)
#     n_classes = int(max(ytr.max(), yva.max()) + 1)
#     print(f"train label distribution (shifted): {_dist(ytr)}")
#     print(f"val   label distribution (shifted): {_dist(yva)}")

#     # scale with train stats
#     Xtr_np, Xva_np, Xte_np, scaler = data_utils.scale_features(
#         Xtr_df.to_numpy(dtype=np.float32),
#         Xva_df.to_numpy(dtype=np.float32),
#         Xte_df.to_numpy(dtype=np.float32) if Xte_df is not None else None,
#     )
#     scaler["feature_names"] = feat_cols
#     scaler["label_shift"]   = shift
#     scaler["label_order"]   = [-1, 0, 1]

#     # sequences
#     L = int(cfg["model"]["seq_length"])
#     Xtr_seq, ytr_seq = data_utils.create_sequences(Xtr_np, ytr, L)
#     Xva_seq, yva_seq = data_utils.create_sequences(Xva_np, yva, L)

#     # class weights with cap
#     use_cw = bool(cfg["training"].get("class_weight", True))
#     max_w  = float(cfg["training"].get("max_class_weight", 5.0))
#     class_w = data_utils.compute_class_weights(ytr_seq, max_weight=max_w) if use_cw else None
#     if class_w:
#         print(f"[info] class_weights (capped at {max_w}): {class_w}")

#     # model
#     model = create_transformer_model(
#         seq_len=L,
#         n_numeric=Xtr_seq.shape[2],
#         n_classes=n_classes,
#         d_model=cfg["model"].get("d_model", 64),
#         n_heads=cfg["model"].get("num_heads", 4),
#         n_layers=cfg["model"].get("num_layers", 2),
#         ff_dim=cfg["model"].get("ff_dim", 128),
#         dropout=cfg["model"].get("dropout_rate", 0.1),
#     )

#     # optimizer (+ optional focal loss)
#     lr = float(cfg["training"].get("learning_rate", 5e-4))
#     use_focal = bool(cfg["training"].get("use_focal_loss", False))
#     loss = "sparse_categorical_crossentropy"
#     if use_focal:
#         gamma = float(cfg["training"].get("focal_gamma", 2.0))
#         def focal_loss(y_true, y_pred):
#             y_true = tf.cast(y_true, tf.int32)
#             y_true_oh = tf.one_hot(y_true, depth=n_classes)
#             y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
#             ce = -tf.reduce_sum(y_true_oh * tf.math.log(y_pred), axis=-1)
#             pt = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
#             fl = tf.pow(1.0 - pt, gamma) * ce
#             return tf.reduce_mean(fl)
#         loss = focal_loss

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
#         loss=loss,
#         metrics=["accuracy"],
#     )

#     # callbacks: compute val_bal_acc + checkpoint on it
#     patience = int(cfg["training"].get("patience", 40))
#     val_bal_cb = ValBalancedAccuracy((Xva_seq, yva_seq), name="val_bal_acc")

#     ckpt_path = cfg["training"]["output_model_file"]
#     mcp = ModelCheckpoint(
#         filepath=ckpt_path,
#         monitor="val_bal_acc",
#         mode="max",
#         save_best_only=True,
#         save_weights_only=False,
#         verbose=1,
#     )

#     callbacks = [
#         val_bal_cb,
#         mcp,
#         EarlyStopping(monitor="val_bal_acc", mode="max", patience=patience, restore_best_weights=True, verbose=1),
#     ]
#     if bool(cfg["training"].get("use_reduce_lr", True)):
#         callbacks.append(
#             ReduceLROnPlateau(monitor="val_bal_acc", mode="max", factor=0.5, patience=max(3, patience // 3), min_lr=1e-6, verbose=1)
#         )
#     if cfg["training"].get("tensorboard_dir"):
#         callbacks.append(tf.keras.callbacks.TensorBoard(cfg["training"]["tensorboard_dir"]))

#     # train
#     model.fit(
#         Xtr_seq, ytr_seq,
#         validation_data=(Xva_seq, yva_seq),
#         epochs=int(cfg["training"]["epochs"]),
#         batch_size=int(cfg["training"]["batch_size"]),
#         class_weight=class_w,
#         shuffle=True,
#         callbacks=callbacks,
#         verbose=2,
#     )

#     # save scaler (model already checkpointed best)
#     scaler_path = cfg["training"]["output_scaler_file"]
#     Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
#     with open(scaler_path, "w") as f:
#         json.dump(scaler, f, indent=2)
#     print(f"✓ best model (by val_bal_acc) saved at {ckpt_path}")
#     print(f"✓ scaler saved at {scaler_path}")

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", required=True)
#     args = ap.parse_args()
#     main(args.config)


# #!/usr/bin/env python3
# """
# train_tf.py – Robust trainer for imbalanced {-1,0,1} labels

# What’s new:
#   • Balanced Accuracy (macro recall) + Macro‑F1 Keras metrics
#   • EarlyStopping + ModelCheckpoint on val_bal_acc (mode='max')
#   • Capped class weights to prevent collapse (configurable)
#   • Optional focal loss (default OFF)
#   • Saves feature_names, label_shift, label_order in scaler JSON
# """
# from __future__ import annotations
# import argparse, json, os, sys
# from pathlib import Path
# from typing import Dict, Tuple

# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import yaml

# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from src.model.transformer_tf import create_transformer_model
# from src.utils import data_utils

# # Deterministic-ish
# tf.random.set_seed(42)
# np.random.seed(42)

# # ──────────────────────────── metrics ────────────────────────────
# class BalancedAccuracy(tf.keras.metrics.Metric):
#     """Macro recall (balanced accuracy) computed from a running confusion matrix."""
#     def __init__(self, n_classes: int, name: str = "bal_acc", **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.n_classes = int(n_classes)
#         self.cm = self.add_weight(
#             name="cm", shape=(self.n_classes, self.n_classes),
#             initializer="zeros", dtype=tf.float32
#         )

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
#         y_pred = tf.reshape(tf.cast(tf.argmax(y_pred, axis=-1), tf.int32), [-1])
#         cm = tf.math.confusion_matrix(
#             y_true, y_pred, num_classes=self.n_classes, dtype=tf.float32
#         )
#         self.cm.assign_add(cm)

#     def result(self):
#         tp = tf.linalg.diag_part(self.cm)
#         support = tf.reduce_sum(self.cm, axis=1)
#         recall = tf.math.divide_no_nan(tp, support)
#         return tf.reduce_mean(recall)

#     def reset_state(self):
#         self.cm.assign(tf.zeros_like(self.cm))


# class MacroF1(tf.keras.metrics.Metric):
#     """Macro F1 computed from a running confusion matrix."""
#     def __init__(self, n_classes: int, name: str = "macro_f1", **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.n_classes = int(n_classes)
#         self.cm = self.add_weight(
#             name="cm", shape=(self.n_classes, self.n_classes),
#             initializer="zeros", dtype=tf.float32
#         )

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
#         y_pred = tf.reshape(tf.cast(tf.argmax(y_pred, axis=-1), tf.int32), [-1])
#         cm = tf.math.confusion_matrix(
#             y_true, y_pred, num_classes=self.n_classes, dtype=tf.float32
#         )
#         self.cm.assign_add(cm)

#     def result(self):
#         tp = tf.linalg.diag_part(self.cm)
#         pred_sum = tf.reduce_sum(self.cm, axis=0)  # predicted positives per class
#         true_sum = tf.reduce_sum(self.cm, axis=1)  # actual positives per class
#         precision = tf.math.divide_no_nan(tp, pred_sum)
#         recall    = tf.math.divide_no_nan(tp, true_sum)
#         f1 = tf.math.divide_no_nan(2.0 * precision * recall, precision + recall)
#         # average only over classes that appear (true_sum > 0)
#         mask = tf.cast(true_sum > 0, tf.float32)
#         denom = tf.reduce_sum(mask)
#         return tf.math.divide_no_nan(tf.reduce_sum(f1 * mask), denom)

#     def reset_state(self):
#         self.cm.assign(tf.zeros_like(self.cm))

# # ─────────────────────── learning‑rate override ───────────────────────
# def _set_learning_rate(model: tf.keras.Model, new_lr) -> float:
#     """Robustly set LR (works for Variable, schedule, float, or stray string)."""
#     new_lr = float(new_lr)
#     opt = model.optimizer
#     try:
#         lr_obj = getattr(opt, "learning_rate", getattr(opt, "lr", None))
#         if hasattr(lr_obj, "assign"):          # tf.Variable
#             lr_obj.assign(new_lr)
#         elif callable(lr_obj):                  # schedule
#             if hasattr(opt, "learning_rate"):
#                 opt.learning_rate = new_lr
#             else:
#                 opt.lr = new_lr
#         else:                                   # number or bad string
#             if hasattr(opt, "learning_rate"):
#                 opt.learning_rate = new_lr
#             else:
#                 opt.lr = new_lr
#     except Exception:
#         try:
#             tf.keras.backend.set_value(getattr(opt, "learning_rate", getattr(opt, "lr")), new_lr)
#         except Exception:
#             if hasattr(opt, "learning_rate"):
#                 opt.learning_rate = new_lr
#             else:
#                 opt.lr = new_lr
#     try:
#         return float(tf.keras.backend.get_value(getattr(opt, "learning_rate", getattr(opt, "lr"))))
#     except Exception:
#         obj = getattr(opt, "learning_rate", getattr(opt, "lr", None))
#         return float(obj) if isinstance(obj, (int, float)) else new_lr

# # ───────────────────────────── losses ─────────────────────────────
# def make_ce_loss(from_logits: bool, n_classes: int, label_smoothing: float = 0.0):
#     ce = tf.keras.losses.CategoricalCrossentropy(
#         from_logits=from_logits, label_smoothing=float(label_smoothing)
#     )
#     def _loss(y_true, y_pred):
#         y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), depth=n_classes)
#         return ce(y_true_oh, y_pred)
#     return _loss

# def make_focal_loss(from_logits: bool, n_classes: int, gamma: float = 2.0, label_smoothing: float = 0.0, eps: float = 1e-7):
#     gamma = float(gamma)
#     def _loss(y_true, y_pred):
#         if from_logits:
#             y_pred = tf.nn.softmax(y_pred, axis=-1)
#         y_true = tf.cast(y_true, tf.int32)
#         y_true_oh = tf.one_hot(y_true, depth=n_classes)
#         if label_smoothing > 0.0:
#             y_true_oh = (1.0 - label_smoothing) * y_true_oh + label_smoothing / float(n_classes)
#         p_t = tf.reduce_sum(y_true_oh * tf.clip_by_value(y_pred, eps, 1.0), axis=-1)
#         loss = - tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
#         return loss  # (batch,) so class_weight/sample_weight can apply
#     return _loss

# # ──────────────────────────── helpers ────────────────────────────
# def _load_yaml(path: str) -> dict:
#     with open(path, "r") as f:
#         return yaml.safe_load(f)

# def _dist(y: np.ndarray) -> Dict[int, int]:
#     u, c = np.unique(y, return_counts=True)
#     return {int(k): int(v) for k, v in zip(u.tolist(), c.tolist())}

# def main(cfg_path: str):
#     cfg = _load_yaml(cfg_path)

#     # ── paths & columns
#     train_fp = cfg["data"]["train_file"]
#     val_fp   = cfg["data"]["val_file"]
#     test_fp  = cfg["data"].get("test_file")
#     label_col = cfg["data"].get("label_col", "y")
#     time_col  = cfg["data"].get("time_col", "timestamp")

#     # ── load frames → (X_df, y)
#     def load_split(fp: str | None) -> Tuple[pd.DataFrame | None, np.ndarray | None]:
#         if not fp: return None, None
#         Xdf, y = data_utils.load_features_labels(fp, label_col, drop_cols=time_col)
#         return Xdf, y

#     Xtr_df, ytr_raw = load_split(train_fp)
#     Xva_df, yva_raw = load_split(val_fp)
#     Xte_df, yte_raw = load_split(test_fp) if test_fp else (None, None)

#     # ── align feature order to train
#     feat_cols = Xtr_df.columns.tolist()
#     Xtr_df = Xtr_df[feat_cols].copy()
#     Xva_df = Xva_df[feat_cols].copy()
#     if Xte_df is not None:
#         Xte_df = Xte_df[feat_cols].copy()

#     # ── log raw distributions
#     print(f"train label counts (raw): {_dist(ytr_raw)}")
#     print(f"val   label counts (raw): {_dist(yva_raw)}")
#     if yte_raw is not None:
#         print(f"test  label counts (raw): {_dist(yte_raw)}")

#     # ── map labels to 0..K-1 (save shift)
#     shift = int(-ytr_raw.min()) if ytr_raw.min() < 0 else 0
#     ytr = (ytr_raw + shift).astype(np.int32)
#     yva = (yva_raw + shift).astype(np.int32)
#     yte = (yte_raw + shift).astype(np.int32) if yte_raw is not None else None
#     n_classes = int(max(ytr.max(), yva.max()) + 1)
#     print(f"train label distribution (shifted): {_dist(ytr)}")
#     print(f"val   label distribution (shifted): {_dist(yva)}")

#     # ── scale using train stats
#     Xtr, Xva, Xte, scaler = data_utils.scale_features(
#         X_train=Xtr_df.to_numpy(dtype=np.float32),
#         X_val=Xva_df.to_numpy(dtype=np.float32),
#         X_test=Xte_df.to_numpy(dtype=np.float32) if Xte_df is not None else None,
#     )
#     scaler["feature_names"] = feat_cols
#     scaler["label_shift"]   = shift
#     scaler["label_order"]   = [-1, 0, 1]

#     # ── create sequences
#     L = int(cfg["model"]["seq_length"])
#     Xtr_seq, ytr_seq = data_utils.create_sequences(Xtr, ytr, L)
#     Xva_seq, yva_seq = data_utils.create_sequences(Xva, yva, L)

#     # ── class weights (capped)
#     cw_cfg  = cfg["training"].get("class_weight", True)
#     cw_cap  = float(cfg["training"].get("max_class_weight", 5.0))
#     cw_floor = float(cfg["training"].get("min_class_weight", 1.0))
#     cw_power = float(cfg["training"].get("class_weight_power", 1.0))
#     class_w = None
#     if cw_cfg:
#         class_w = data_utils.compute_class_weights(
#             ytr_seq, cap=cw_cap, floor=cw_floor, power=cw_power, normalize=True
#         )
#         print("[info] class_weight used (after cap):", {int(k): round(float(v), 3) for k, v in class_w.items()})

#     # ── model
#     if cfg.get("model_type", "transformer").lower() != "transformer":
#         sys.exit("[train_tf] Only 'transformer' is supported.")
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

#     # try to infer if outputs are logits or probabilities
#     last = model.layers[-1]
#     from_logits = True
#     if hasattr(last, "activation"):
#         if last.activation == tf.keras.activations.softmax:
#             from_logits = False

#     # optimizer & loss
#     lr = float(cfg["training"].get("learning_rate", 5e-4))
#     opt = tf.keras.optimizers.Adam(learning_rate=lr)

#     use_focal = bool(cfg["training"].get("use_focal_loss", False))
#     focal_gamma = float(cfg["training"].get("focal_gamma", 2.0))
#     label_smoothing = float(cfg["training"].get("label_smoothing", 0.0))

#     if use_focal:
#         loss_fn = make_focal_loss(from_logits=from_logits, n_classes=n_classes,
#                                   gamma=focal_gamma, label_smoothing=label_smoothing)
#     else:
#         loss_fn = make_ce_loss(from_logits=from_logits, n_classes=n_classes,
#                                label_smoothing=label_smoothing)

#     metrics = [
#         "accuracy",
#         BalancedAccuracy(n_classes, name="bal_acc"),
#         MacroF1(n_classes, name="macro_f1"),
#     ]
#     model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

#     # allow LR override even if optimizer had a schedule/string
#     eff_lr = _set_learning_rate(model, lr)
#     print(f"[info] Using learning_rate={eff_lr}")

#     # ── callbacks: monitor val_bal_acc (mode='max')
#     monitor_metric = cfg["training"].get("monitor", "val_bal_acc")
#     patience = int(cfg["training"].get("patience", 40))
#     out_model = cfg["training"]["output_model_file"]
#     out_scaler = cfg["training"]["output_scaler_file"]
#     Path(out_model).parent.mkdir(parents=True, exist_ok=True)
#     Path(out_scaler).parent.mkdir(parents=True, exist_ok=True)

#     callbacks = [
#         ModelCheckpoint(
#             filepath=out_model, monitor=monitor_metric, mode="max",
#             save_best_only=True, save_weights_only=False, verbose=1
#         ),
#         EarlyStopping(
#             monitor=monitor_metric, mode="max",
#             patience=patience, restore_best_weights=True, verbose=1
#         ),
#     ]
#     if bool(cfg["training"].get("use_reduce_lr", True)):
#         callbacks.append(
#             ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(5, patience // 4),
#                               min_lr=1e-6, verbose=1)
#         )
#     if cfg["training"].get("tensorboard_dir"):
#         callbacks.append(tf.keras.callbacks.TensorBoard(cfg["training"]["tensorboard_dir"]))

#     # ── train
#     history = model.fit(
#         Xtr_seq, ytr_seq,
#         validation_data=(Xva_seq, yva_seq),
#         epochs=int(cfg["training"].get("epochs", 100)),
#         batch_size=int(cfg["training"].get("batch_size", 128)),
#         class_weight=class_w,
#         shuffle=True,
#         callbacks=callbacks,
#         verbose=2,
#     )

#     # Save scaler/meta (model already checkpointed to best via ModelCheckpoint)
#     scaler["class_counts_train"] = _dist(ytr_seq)
#     scaler["class_weight_used"]  = {int(k): float(v) for k, v in (class_w or {}).items()}
#     with open(out_scaler, "w") as f:
#         json.dump(scaler, f, indent=2)

#     print(f"\n✓ best model saved at {out_model} (monitor={monitor_metric})")
#     print(f"✓ scaler saved at {out_scaler}")


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", required=True)
#     args = ap.parse_args()
#     main(args.config)


#!/usr/bin/env python3
# src/train_tf.py – v4.1 (canonical)
# Training pipeline for Transformer on {-1,0,1} labels
# • Drops time column (kept only for ordering upstream)
# • Saves scaler stats + feature order + label_shift for eval alignment
# • Class-imbalance controls: inverse-freq weights with cap/floor/power
# • Optional focal loss and label smoothing
# • Optional ReduceLROnPlateau; EarlyStopping on configurable monitor

from __future__ import annotations
import argparse, json, os, sys
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
tf.keras.backend.set_floatx("float32")
try:
    tf.config.experimental.enable_tensor_float_32_execution(True)
except Exception:
    pass


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_sparse_focal_loss(gamma: float = 2.0) -> tf.keras.losses.Loss:
    """
    Sparse categorical focal loss on softmax probabilities (no logits assumption).
    FL = - (1 - p_t)^gamma * log(p_t), averaged over batch.
    Class weighting (if any) is applied via Keras `class_weight` in model.fit.
    """
    gamma = float(gamma)

    def loss(y_true, y_pred):
        # y_true: (B,), int; y_pred: (B, C), probabilities
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        # numerical safety
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1)  # (B,)
        return tf.reduce_mean(tf.pow(1.0 - p_t, gamma) * (-tf.math.log(p_t)))

    return loss


def make_sparse_xent_with_smoothing(n_classes: int, label_smoothing: float) -> tf.keras.losses.Loss:
    """
    Implements label smoothing with sparse targets by internally one-hotting and
    using categorical crossentropy with smoothing.
    """
    label_smoothing = float(label_smoothing)
    cce = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=n_classes)
        return cce(y_true_oh, y_pred)

    return loss


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)

    # ---------------- Data paths / columns ----------------
    train_fp = cfg["data"]["train_file"]
    val_fp   = cfg["data"]["val_file"]
    test_fp  = cfg["data"].get("test_file", None)
    label_col = cfg["data"].get("label_col", "y")
    time_col  = cfg["data"].get("time_col", "timestamp")

    # ---------------- Load features/labels ----------------
    Xtr_df, ytr_raw = data_utils.load_features_labels(train_fp, label_col, drop_cols=time_col)
    Xva_df, yva_raw = data_utils.load_features_labels(val_fp,   label_col, drop_cols=time_col)
    Xte_df, yte_raw = (data_utils.load_features_labels(test_fp, label_col, drop_cols=time_col)[0:2]
                       if test_fp else (None, None))

    # ensure numeric-only feature frames (val/test aligned to train columns)
    feat_cols = Xtr_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    Xtr_df = Xtr_df[feat_cols].copy()
    Xva_df = Xva_df[feat_cols].copy()
    if Xte_df is not None:
        Xte_df = Xte_df[feat_cols].copy()

    # ---------------- Label mapping & scaler ----------------
    # map {-1,0,1} -> {0,1,2}
    shift = int(-ytr_raw.min()) if ytr_raw.min() < 0 else 0
    ytr = (ytr_raw + shift).astype(np.int32)
    yva = (yva_raw + shift).astype(np.int32)
    yte = (yte_raw + shift).astype(np.int32) if yte_raw is not None else None
    n_classes = int(max(ytr.max(), yva.max()) + 1)

    Xtr_np, Xva_np, Xte_np, scaler = data_utils.scale_features(
        Xtr_df.to_numpy(dtype=np.float32, copy=False),
        Xva_df.to_numpy(dtype=np.float32, copy=False),
        Xte_df.to_numpy(dtype=np.float32, copy=False) if Xte_df is not None else None,
    )
    # persist meta for eval
    scaler["feature_names"] = feat_cols
    scaler["label_order"]   = [-1, 0, 1]
    scaler["label_shift"]   = shift

    # ---------------- Sequences ----------------
    L = int(cfg["model"]["seq_length"])
    Xtr_seq, ytr_seq = data_utils.create_sequences(Xtr_np, ytr, L)
    Xva_seq, yva_seq = data_utils.create_sequences(Xva_np, yva, L)
    if Xte_np is not None and yte is not None:
        _ = data_utils.create_sequences(Xte_np, yte, L)  # smoke check

    # ---------------- Class imbalance controls ----------------
    class_weight_cfg = cfg["training"].get("class_weight", True)
    cw_cap   = float(cfg["training"].get("max_class_weight", 5.0))
    cw_floor = float(cfg["training"].get("min_class_weight", 1.0))
    cw_pow   = float(cfg["training"].get("class_weight_power", 1.0))

    class_w = None
    if class_weight_cfg:
        # weights are computed in the shifted space {0,1,2}
        class_w = data_utils.compute_class_weights(
            ytr_seq,
            cap=cw_cap,
            floor=cw_floor,
            power=cw_pow,
            normalize=True,
        )

    # ---------------- Model ----------------
    model = create_transformer_model(
        seq_len   = L,
        n_numeric = Xtr_seq.shape[2],
        n_classes = n_classes,
        d_model   = int(cfg["model"].get("d_model", 64)),
        n_heads   = int(cfg["model"].get("num_heads", 4)),
        n_layers  = int(cfg["model"].get("num_layers", 2)),
        ff_dim    = int(cfg["model"].get("ff_dim", 128)),
        dropout   = float(cfg["model"].get("dropout_rate", 0.1)),
    )

    # loss selection
    use_focal       = bool(cfg["training"].get("use_focal_loss", False))
    focal_gamma     = float(cfg["training"].get("focal_gamma", 2.0))
    label_smoothing = float(cfg["training"].get("label_smoothing", 0.0))

    if use_focal:
        loss_fn = make_sparse_focal_loss(gamma=focal_gamma)
    elif label_smoothing > 0.0:
        loss_fn = make_sparse_xent_with_smoothing(n_classes, label_smoothing)
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    lr = float(cfg["training"].get("learning_rate", 5e-4))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    # ---------------- Callbacks ----------------
    monitor = cfg["training"].get("monitor", "val_loss")
    patience = int(cfg["training"].get("patience", 40))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience, restore_best_weights=True, verbose=1
        )
    ]
    if bool(cfg["training"].get("use_reduce_lr", True)):
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor, factor=0.5, patience=max(3, patience // 4),
                verbose=1, min_lr=1e-6
            )
        )

    # ---------------- Train ----------------
    epochs     = int(cfg["training"].get("epochs", 120))
    batch_size = int(cfg["training"].get("batch_size", 128))

    # simple label diagnostics
    def _dist(y: np.ndarray) -> dict:
        u, c = np.unique(y, return_counts=True)
        return {int(k): int(v) for k, v in zip(u, c)}
    print("train label counts (shifted):", _dist(ytr_seq))
    print("val   label counts (shifted):", _dist(yva_seq))
    if len(_dist(ytr_seq)) <= 1:
        print("⚠️  training labels have a single class — consider rebuilding features with quantile alpha.")

    model.fit(
        Xtr_seq, ytr_seq,
        validation_data=(Xva_seq, yva_seq),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_w,
        verbose=2,
        callbacks=callbacks,
        shuffle=True,
    )

    # ---------------- Save ----------------
    out_model  = cfg["training"]["output_model_file"]
    out_scaler = cfg["training"]["output_scaler_file"]
    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    Path(out_scaler).parent.mkdir(parents=True, exist_ok=True)

    model.save(out_model)  # .keras preferred
    with open(out_scaler, "w") as f:
        json.dump(scaler, f, indent=2)

    print(f"✓ model saved at {out_model}")
    print(f"✓ scaler saved at {out_scaler}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)