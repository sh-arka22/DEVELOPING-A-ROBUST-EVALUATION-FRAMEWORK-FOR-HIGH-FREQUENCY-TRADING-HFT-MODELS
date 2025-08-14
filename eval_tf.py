# filepath: /Users/arkajyotisaha/Desktop/My-Thesis/code/eval_tf.py
#!/usr/bin/env python3
"""
eval_tf.py – Keras 3–safe loader + column alignment + label shift
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score, f1_score
from src.utils import data_utils
from src.model.transformer_tf import PositionalEncoding1D, AttentionPool1D, SqueezeExcite1D  # ensure registered

_CUSTOMS = {
    "PositionalEncoding1D": PositionalEncoding1D,
    "AttentionPool1D":      AttentionPool1D,
    "SqueezeExcite1D":      SqueezeExcite1D,
}

def _smart_load(path: str) -> tf.keras.Model:
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".keras",".h5",".hdf5"}:
        raise ValueError("model must end with .keras or .h5/.hdf5")
    # First try standard safe load (works with our new model)
    try:
        return tf.keras.models.load_model(path, compile=False, custom_objects=_CUSTOMS)
    except Exception as e1:
        # Fallback to unsafe (older model MAY contain Lambda layers)
        try:
            return tf.keras.models.load_model(path, compile=False, custom_objects=_CUSTOMS, safe_mode=False)
        except Exception as e2:
            raise RuntimeError(
                f"[eval] Failed to load model '{path}'. This usually happens when the checkpoint "
                f"contains legacy Lambda layers without output_shape. Please re-train once with the "
                f"updated model code (no Lambda) and try again.\n\nSafe error: {e1}\nUnsafe error: {e2}"
            )

def _infer_seq_len(model: tf.keras.Model) -> int:
    shp = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    if shp is None or len(shp) < 3 or shp[1] is None:
        raise ValueError("Cannot infer sequence length from model; pass --seq")
    return int(shp[1])

def run(model_fp: str, test_fp: str, scaler_json: str,
        seq_len: int | None, label_col: str = "y", time_col: str | None = "timestamp"):

    model = _smart_load(model_fp)
    if seq_len is None:
        seq_len = _infer_seq_len(model)

    # load test features/labels
    Xdf, y_raw = data_utils.load_features_labels(test_fp, label_col, drop_cols=time_col)

    # load scaler & align columns
    scaler = json.load(open(scaler_json))
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std  = np.asarray(scaler["std"],  dtype=np.float32)
    std[std == 0] = 1.0

    feat_names  = scaler.get("feature_names", list(Xdf.columns))
    label_shift = int(scaler.get("label_shift", 0))
    label_order = scaler.get("label_order", None)

    missing = [c for c in feat_names if c not in Xdf.columns]
    if missing:
        raise ValueError(f"Test set missing expected feature columns: {missing[:8]}...")
    Xdf = Xdf[feat_names]

    if Xdf.shape[1] != mean.shape[0]:
        raise ValueError(f"Feature dimension mismatch: X has {Xdf.shape[1]}, scaler has {mean.shape[0]}.")

    # scale
    X = (Xdf.to_numpy(dtype=np.float32, copy=False) - mean) / std

    # labels: APPLY THE TRAINING SHIFT
    y_int = (np.asarray(y_raw) + label_shift).astype(np.int32, copy=False)

    # sequences
    Xseq, yseq = data_utils.create_sequences(X, y_int, seq_len)

    # predict (probs)
    proba = model.predict(Xseq, verbose=0)
    y_pred_int = np.argmax(proba, axis=1)

    # map back to original label space
    y_true = yseq - label_shift
    y_pred = y_pred_int - label_shift

    # print distributions
    from collections import Counter
    print("y_true distribution:", dict(Counter(y_true.tolist())))
    print("y_pred distribution:", dict(Counter(y_pred.tolist())))

    # choose reporting order (fixed order helps when some classes absent)
    if label_order is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    else:
        labels = [int(c) for c in label_order if (c in set(np.concatenate([y_true, y_pred])) or True)]

    print("\nConfusion Matrix\n", confusion_matrix(y_true, y_pred, labels=labels))
    print("\nClassification Report\n",
          classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0))
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Balanced Accuracy:", round(balanced_accuracy_score(y_true, y_pred), 4))
    # You can also compute macro-F1 if desired:
    print("Macro F1:", round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4))

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("model");  pa.add_argument("test");  pa.add_argument("scaler")
    pa.add_argument("--seq", type=int, help="Override sequence length (default: read from model)")
    pa.add_argument("--label", default="y"); pa.add_argument("--time_col", default="timestamp")
    args = pa.parse_args()
    run(args.model, args.test, args.scaler, args.seq, args.label, args.time_col)