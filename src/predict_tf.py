#!/usr/bin/env python3
# src/predict_tf.py — generate per-bar probabilities for backtest
from __future__ import annotations
import argparse, json, os
import numpy as np, pandas as pd, tensorflow as tf
from src.utils import data_utils

def _smart_load(path: str) -> tf.keras.Model:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".keras", ".h5", ".hdf5"}:
        return tf.keras.models.load_model(path, compile=False)
    raise ValueError("model must end with .keras or .h5/.hdf5")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--scaler_json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq_len", type=int, help="override; default inferred from model")
    ap.add_argument("--label_col", default="y")
    ap.add_argument("--time_col", default="timestamp")
    args = ap.parse_args()

    # load model + scaler
    model = _smart_load(args.model)
    if args.seq_len is None:
        in_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
        L = int(in_shape[1])
    else:
        L = int(args.seq_len)

    scaler = json.load(open(args.scaler_json))
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std  = np.asarray(scaler["std"],  dtype=np.float32); std[std == 0] = 1.0
    feat_names  = scaler.get("feature_names", None)
    label_shift = int(scaler.get("label_shift", 0))        # model’s label mapping
    label_order = scaler.get("label_order", [-1, 0, 1])    # how to name outputs

    # load raw df (need timestamp + mid_px alongside features)
    df = pd.read_parquet(args.data_file)
    if args.time_col in df.columns:
        df[args.time_col] = pd.to_datetime(df[args.time_col], utc=True, errors="coerce")
        df = df.sort_values(args.time_col).reset_index(drop=True)

    # feature frame aligned to training feature order
    if feat_names is None:
        # fall back to TRAIN-like selection: drop time + label, keep numeric/bool
        drop_cols = [c for c in [args.time_col, args.label_col] if c in df.columns]
        Xdf = df.drop(columns=drop_cols).select_dtypes(include=["number","bool"])
        feat_names = Xdf.columns.tolist()
    else:
        missing = [c for c in feat_names if c not in df.columns]
        if missing:
            raise ValueError(f"data missing required features: {missing[:8]}...")
        Xdf = df[feat_names].copy()

    X = (Xdf.to_numpy(dtype=np.float32, copy=False) - mean) / std

    # build sequences (use a dummy y just for shape)
    dummy_y = np.zeros(len(X), dtype=np.int32)
    Xseq, _ = data_utils.create_sequences(X, dummy_y, L)

    # predict class probabilities
    probs = model.predict(Xseq, verbose=0)
    if probs.ndim == 1:
        probs = np.stack([1.0 - probs, probs], axis=1)
    n_out = probs.shape[1]

    # map model output indices → original label space via label_order + label_shift
    # model was trained on labels shifted to [0..K-1]; scaler[label_order] is in original space.
    if len(label_order) != n_out:
        # if model was trained with fewer classes present, pad/truncate conservatively
        # keep existing order and label names as 0..n_out-1 in original space = (idx - label_shift)
        label_order = [(i - label_shift) for i in range(n_out)]

    # argmax in original label space
    pred_int = np.argmax(probs, axis=1)
    pred_lbl = np.array([label_order[i] for i in pred_int], dtype=np.int32)

    # align timestamps & mid_px with window ends
    ts = df[args.time_col].iloc[L-1:].reset_index(drop=True) if args.time_col in df else pd.Series(np.arange(len(probs)))
    if "mid_px" in df.columns:
        mid = df["mid_px"].iloc[L-1:].reset_index(drop=True)
    else:
        # fallback: try to reconstruct from bid/ask if present
        if {"bid_px","ask_px"}.issubset(df.columns):
            mid = ((df["bid_px"] + df["ask_px"]) * 0.5).iloc[L-1:].reset_index(drop=True)
        else:
            raise ValueError("mid_px (or bid_px/ask_px) required to compute forward returns for backtest.")

    out = pd.DataFrame({"timestamp": ts, "mid_px": mid, "pred": pred_lbl})
    # probability columns named by label_order
    for j, lab in enumerate(label_order):
        out[f"p_{lab}"] = probs[:, j].astype(np.float32)

    # write parquet
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"✔ wrote {args.out}  rows={len(out)}  seq_len={L}")

if __name__ == "__main__":
    main()