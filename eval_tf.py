# #!/usr/bin/env python3
# """
# eval_tf.py – V4
# * Safe‑load .keras/.h5
# * Infers seq_len from model unless --seq is given
# * Aligns test features to the training feature order stored in scaler JSON
# * Uses the SAME label shift as training (saved in scaler JSON)
# """
# from __future__ import annotations
# import argparse, json, os
# import numpy as np, pandas as pd, tensorflow as tf
# from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
# from src.utils import data_utils

# def _smart_load(path: str) -> tf.keras.Model:
#     ext = os.path.splitext(path)[1].lower()
#     if ext in {".keras", ".h5", ".hdf5"}:
#         return tf.keras.models.load_model(path, compile=False)
#     raise ValueError("model must end with .keras or .h5/.hdf5")

# def run(model_fp: str, test_fp: str, scaler_json: str,
#         seq_len: int | None, label_col: str = "y", time_col: str | None = "timestamp"):

#     model = _smart_load(model_fp)

#     # infer L from model unless overridden
#     if seq_len is None:
#         in_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
#         seq_len  = int(in_shape[1])

#     # load test features/labels
#     Xdf, y_raw = data_utils.load_features_labels(test_fp, label_col, drop_cols=time_col)

#     # load scaler & align columns
#     scaler = json.load(open(scaler_json))
#     mean = np.asarray(scaler["mean"], dtype=np.float32)
#     std  = np.asarray(scaler["std"],  dtype=np.float32)
#     std[std == 0] = 1.0

#     feat_names   = scaler.get("feature_names", None)
#     label_shift  = int(scaler.get("label_shift", 0))   # ✅ use training shift
#     label_order  = scaler.get("label_order", None)     # optional

#     if feat_names:
#         missing = [c for c in feat_names if c not in Xdf.columns]
#         if missing:
#             raise ValueError(f"Test set missing expected feature columns: {missing[:8]}...")
#         Xdf = Xdf[feat_names]

#     if Xdf.shape[1] != mean.shape[0]:
#         raise ValueError(f"Feature dimension mismatch: X has {Xdf.shape[1]}, scaler has {mean.shape[0]}.")

#     # scale
#     X = (Xdf.to_numpy(dtype=np.float32, copy=False) - mean) / std

#     # labels: APPLY THE TRAINING SHIFT
#     y_raw = np.asarray(y_raw)
#     y_int = (y_raw + label_shift).astype(np.int32, copy=False)

#     # sequences
#     Xseq, yseq = data_utils.create_sequences(X, y_int, seq_len)

#     # predict
#     y_pred_int = np.argmax(model.predict(Xseq, verbose=0), axis=1)

#     # map back to original label space
#     y_true = yseq - label_shift
#     y_pred = y_pred_int - label_shift

#     # choose reporting order (fixed order helps when some classes absent)
#     if label_order is None:
#         labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
#     else:
#         labels = [int(c) for c in label_order if c in set(np.concatenate([y_true, y_pred])) or True]

#     print("\nConfusion Matrix\n", confusion_matrix(y_true, y_pred, labels=labels))
#     print("\nClassification Report\n",
#           classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0))
#     print("Balanced Accuracy:", round(balanced_accuracy_score(y_true, y_pred), 4))

# if __name__ == "__main__":
#     pa = argparse.ArgumentParser()
#     pa.add_argument("model");  pa.add_argument("test");  pa.add_argument("scaler")
#     pa.add_argument("--seq", type=int, help="Override sequence length (default: read from model)")
#     pa.add_argument("--label", default="y"); pa.add_argument("--time_col", default="timestamp")
#     args = pa.parse_args()
#     run(args.model, args.test, args.scaler, args.seq, args.label, args.time_col)



# #!/usr/bin/env python3
# """
# eval_tf.py – V5
# * Safe‑load .keras/.h5
# * Infers seq_len from model unless --seq is given
# * Aligns test features to the training feature order stored in scaler JSON
# * Uses the SAME label shift as training (saved in scaler JSON)
# * Robust balanced‑accuracy when some classes are absent
# """
# from __future__ import annotations
# import argparse, json, os
# import numpy as np, pandas as pd, tensorflow as tf
# from sklearn.metrics import confusion_matrix, classification_report, recall_score
# from src.utils import data_utils

# def _smart_load(path: str) -> tf.keras.Model:
#     ext = os.path.splitext(path)[1].lower()
#     if ext in {".keras", ".h5", ".hdf5"}:
#         return tf.keras.models.load_model(path, compile=False)
#     raise ValueError("model must end with .keras or .h5/.hdf5")

# def run(model_fp: str, test_fp: str, scaler_json: str,
#         seq_len: int | None, label_col: str = "y", time_col: str | None = "timestamp"):

#     model = _smart_load(model_fp)

#     # infer L from model unless overridden
#     if seq_len is None:
#         in_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
#         seq_len  = int(in_shape[1])

#     # load test features/labels
#     Xdf, y_raw = data_utils.load_features_labels(test_fp, label_col, drop_cols=time_col)

#     # load scaler & align columns
#     scaler = json.load(open(scaler_json))
#     mean = np.asarray(scaler["mean"], dtype=np.float32)
#     std  = np.asarray(scaler["std"],  dtype=np.float32)
#     std[std == 0] = 1.0

#     feat_names   = scaler.get("feature_names", None)
#     label_shift  = int(scaler.get("label_shift", 0))   # ✅ use training shift
#     label_order  = scaler.get("label_order", [-1,0,1])

#     if feat_names:
#         missing = [c for c in feat_names if c not in Xdf.columns]
#         if missing:
#             raise ValueError(f"Test set missing expected feature columns: {missing[:8]}...")
#         Xdf = Xdf[feat_names]

#     if Xdf.shape[1] != mean.shape[0]:
#         raise ValueError(f"Feature dimension mismatch: X has {Xdf.shape[1]}, scaler has {mean.shape[0]}.")

#     # scale
#     X = (Xdf.to_numpy(dtype=np.float32, copy=False) - mean) / std

#     # labels: APPLY THE TRAINING SHIFT
#     y_raw = np.asarray(y_raw)
#     y_int = (y_raw + label_shift).astype(np.int32, copy=False)

#     # sequences
#     Xseq, yseq = data_utils.create_sequences(X, y_int, seq_len)

#     # predict
#     y_pred_int = np.argmax(model.predict(Xseq, verbose=0), axis=1)

#     # map back to original label space
#     y_true = (yseq - label_shift).astype(int)
#     y_pred = (y_pred_int - label_shift).astype(int)

#     labels_fixed = [int(c) for c in label_order]  # always report on full set

#     # diagnostics
#     uniq_true, cnt_true = np.unique(y_true, return_counts=True)
#     uniq_pred, cnt_pred = np.unique(y_pred, return_counts=True)
#     print("y_true distribution:", dict(zip(uniq_true.tolist(), cnt_true.tolist())))
#     print("y_pred distribution:", dict(zip(uniq_pred.tolist(), cnt_pred.tolist())))

#     # metrics
#     cm = confusion_matrix(y_true, y_pred, labels=labels_fixed)
#     print("\nConfusion Matrix\n", cm)
#     print("\nClassification Report\n",
#           classification_report(y_true, y_pred, labels=labels_fixed, digits=4, zero_division=0))
#     bal_acc = recall_score(y_true, y_pred, labels=labels_fixed, average="macro", zero_division=0)
#     print("Balanced Accuracy:", round(float(bal_acc), 4))

# if __name__ == "__main__":
#     pa = argparse.ArgumentParser()
#     pa.add_argument("model");  pa.add_argument("test");  pa.add_argument("scaler")
#     pa.add_argument("--seq", type=int, help="Override sequence length (default: read from model)")
#     pa.add_argument("--label", default="y"); pa.add_argument("--time_col", default="timestamp")
#     args = pa.parse_args()
#     run(args.model, args.test, args.scaler, args.seq, args.label, args.time_col)


#!/usr/bin/env python3
"""
eval_tf.py – V5.1
* Safe‑load .keras/.h5
* Infers seq_len from model unless --seq is given
* Aligns test features to the training feature order stored in scaler JSON
* Uses the SAME label shift as training (saved in scaler JSON)
* Warns if only one class in y_true (reduce --alpha and rebuild features)
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, recall_score
from src.utils import data_utils

def _smart_load(path: str) -> tf.keras.Model:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".keras", ".h5", ".hdf5"}:
        return tf.keras.models.load_model(path, compile=False)
    raise ValueError("model must end with .keras or .h5/.hdf5")

def run(model_fp: str, test_fp: str, scaler_json: str,
        seq_len: int | None, label_col: str = "y", time_col: str | None = "timestamp"):

    model = _smart_load(model_fp)

    # infer L from model unless overridden
    if seq_len is None:
        in_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
        seq_len  = int(in_shape[1])

    # load test features/labels
    Xdf, y_raw = data_utils.load_features_labels(test_fp, label_col, drop_cols=time_col)

    # load scaler & align columns
    scaler = json.load(open(scaler_json))
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std  = np.asarray(scaler["std"],  dtype=np.float32)
    std[std == 0] = 1.0

    feat_names   = scaler.get("feature_names", None)
    label_shift  = int(scaler.get("label_shift", 0))
    label_order  = scaler.get("label_order", [-1,0,1])

    if feat_names:
        missing = [c for c in feat_names if c not in Xdf.columns]
        if missing:
            raise ValueError(f"Test set missing expected feature columns: {missing[:8]}...")
        Xdf = Xdf[feat_names]

    if Xdf.shape[1] != mean.shape[0]:
        raise ValueError(f"Feature dimension mismatch: X has {Xdf.shape[1]}, scaler has {mean.shape[0]}.")

    # scale
    X = (Xdf.to_numpy(dtype=np.float32, copy=False) - mean) / std

    # labels: APPLY THE TRAINING SHIFT
    y_raw = np.asarray(y_raw)
    y_int = (y_raw + label_shift).astype(np.int32, copy=False)

    # sequences
    Xseq, yseq = data_utils.create_sequences(X, y_int, seq_len)

    # predict
    y_pred_int = np.argmax(model.predict(Xseq, verbose=0), axis=1)

    # map back to original label space
    y_true = (yseq - label_shift).astype(int)
    y_pred = (y_pred_int - label_shift).astype(int)

    labels_fixed = [int(c) for c in label_order]

    # diagnostics
    uniq_true, cnt_true = np.unique(y_true, return_counts=True)
    uniq_pred, cnt_pred = np.unique(y_pred, return_counts=True)
    print("y_true distribution:", dict(zip(uniq_true.tolist(), cnt_true.tolist())))
    print("y_pred distribution:", dict(zip(uniq_pred.tolist(), cnt_pred.tolist())))
    if len(uniq_true) <= 1:
        print("⚠️  y_true has a single class. This usually means the label thresholding is too strict "
              "for this instrument/day/horizon. Rebuild features with a smaller --alpha (e.g., 0.10 or 0.05).")

    # metrics
    cm = confusion_matrix(y_true, y_pred, labels=labels_fixed)
    print("\nConfusion Matrix\n", cm)
    print("\nClassification Report\n",
          classification_report(y_true, y_pred, labels=labels_fixed, digits=4, zero_division=0))
    bal_acc = recall_score(y_true, y_pred, labels=labels_fixed, average="macro", zero_division=0)
    print("Balanced Accuracy:", round(float(bal_acc), 4))

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("model");  pa.add_argument("test");  pa.add_argument("scaler")
    pa.add_argument("--seq", type=int, help="Override sequence length (default: read from model)")
    pa.add_argument("--label", default="y"); pa.add_argument("--time_col", default="timestamp")
    args = pa.parse_args()
    run(args.model, args.test, args.scaler, args.seq, args.label, args.time_col)