# filepath: src/predict_tf.py
#!/usr/bin/env python3
"""
src/predict_tf.py — V2 (sanitation-aware inference)
...
"""
from __future__ import annotations
import argparse, json, os, sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from src.utils import data_utils  # uses create_sequences()

# ▶ Import custom layers to ensure they are registered with Keras’ serializer
from src.model.transformer_tf import (
    PositionalEncoding1D, AttentionPool1D, SqueezeExcite1D
)

# Also provide explicit custom_objects mappings (helps when registry names are used)
_CUSTOMS = {
    "PositionalEncoding1D": PositionalEncoding1D,
    "AttentionPool1D": AttentionPool1D,
    "SqueezeExcite1D": SqueezeExcite1D,
    # Registered names (package>ClassName) for extra safety:
    "hft>PositionalEncoding1D": PositionalEncoding1D,
    "hft>AttentionPool1D": AttentionPool1D,
    "hft>SqueezeExcite1D": SqueezeExcite1D,
}

# ─────────────────────────── helpers ───────────────────────────
def _smart_load(path: str) -> tf.keras.Model:
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".keras", ".h5", ".hdf5"}:
        raise ValueError("model must end with .keras or .h5/.hdf5")
    # First try standard safe load with custom_objects
    try:
        return tf.keras.models.load_model(path, compile=False, custom_objects=_CUSTOMS)
    except Exception as e1:
        # Fallback: allow unsafe deserialization (for legacy Lambda models, if any)
        try:
            return tf.keras.models.load_model(path, compile=False, custom_objects=_CUSTOMS, safe_mode=False)
        except Exception as e2:
            raise RuntimeError(
                f"[predict] Failed to load model '{path}'. "
                f"Safe load error: {e1}\nUnsafe load error: {e2}"
            )

def _coerce_time(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if time_col in df.columns:
        ts = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.loc[ts.notna()].copy()
        df[time_col] = ts.loc[ts.notna()]
        df = df.sort_values(time_col).reset_index(drop=True)
    return df

def _ensure_features(df: pd.DataFrame, feat_names: List[str] | None,
                     label_col: str, time_col: str | None) -> Tuple[pd.DataFrame, List[str]]:
    # Drop label/time, keep numeric/bool; then reorder to training feature list.
    drop_cols = [c for c in [label_col, time_col] if (c and c in df.columns)]
    Xdf = df.drop(columns=drop_cols).select_dtypes(include=["number", "bool"]).copy()
    if feat_names is None:
        feat_names = Xdf.columns.tolist()
    else:
        missing = [c for c in feat_names if c not in Xdf.columns]
        if missing:
            raise ValueError(f"[predict] data missing required features: {missing[:8]}...")
        Xdf = Xdf[feat_names]
    return Xdf, feat_names

def _sanitize_apply(Xdf: pd.DataFrame, params: Dict[str, dict]) -> pd.DataFrame:
    """
    Apply training-time sanitation to inference:
    • replace infs → NaN → LOCF → BFill
    • fill remaining NaN with **training medians**
    • winsorize to training quantile bounds (q_low/q_high)
    """
    X = Xdf.copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.ffill(inplace=True)
    X.bfill(inplace=True)

    med = params.get("medians")
    ql  = params.get("q_low")
    qh  = params.get("q_high")
    if med is None or ql is None or qh is None:
        print("[predict][warn] scaler JSON missing medians/q_low/q_high; "
              "falling back to per-column median without winsorization.", file=sys.stderr)
        # best-effort fallback
        med_local = X.median(numeric_only=True).to_dict()
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].fillna(med_local.get(c, 0.0))
        return X

    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            if c in med:
                X[c] = X[c].fillna(med[c])
            lo, hi = ql.get(c, None), qh.get(c, None)
            if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                X[c] = X[c].clip(lo, hi)
    return X

def _zapply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = std.copy()
    std[std == 0] = 1.0
    return ((X - mean) / std).astype(np.float32)

def _infer_seq_len(model: tf.keras.Model) -> int:
    shp = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    if shp is None or len(shp) < 3 or shp[1] is None:
        raise ValueError("Cannot infer sequence length from model; pass --seq_len")
    return int(shp[1])

def _index_to_labels(n_out: int, scaler: dict) -> List[int]:
    # Primary: scaler['label_order'] (e.g., [-1,0,1]) → index j names
    order = scaler.get("label_order", None)
    if order is not None and len(order) == n_out:
        return [int(x) for x in order]
    # Secondary: use label_shift (index j → j - shift)
    if "label_shift" in scaler:
        shift = int(scaler["label_shift"])
        return [j - shift for j in range(n_out)]
    # Fallback: common defaults
    if n_out == 3:
        return [-1, 0, 1]
    return list(range(n_out))

# ─────────────────────────── main ───────────────────────────

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

    # Load model + scaler
    model = _smart_load(args.model)
    L = int(args.seq_len) if args.seq_len else _infer_seq_len(model)

    scaler = json.load(open(args.scaler_json))
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std  = np.asarray(scaler["std"],  dtype=np.float32)
    feat_names = scaler.get("feature_names", None)

    # Load data
    df = pd.read_parquet(args.data_file)
    df = _coerce_time(df, args.time_col)

    # Feature frame → sanitation → scaling
    Xdf, feat_names = _ensure_features(df, feat_names, args.label_col, args.time_col)
    Xdf = _sanitize_apply(Xdf, scaler)        # LOCF/BFill/medians/winsor
    if Xdf.shape[1] != mean.shape[0]:
        raise ValueError(f"[predict] Feature dimension mismatch: X has {Xdf.shape[1]}, scaler has {mean.shape[0]}.")
    X = _zapply(Xdf.to_numpy(dtype=np.float32, copy=False), mean, std)

    # Build sequences (dummy y for shape)
    dummy_y = np.zeros(len(X), dtype=np.int32)
    Xseq, _ = data_utils.create_sequences(X, dummy_y, L)

    # Predict probabilities
    probs = model.predict(Xseq, verbose=0)
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim == 1:  # binary logistic → [p0] → make 2-col probs
        probs = np.stack([1.0 - probs, probs], axis=1)
    n_out = probs.shape[1]

    # Index → original label names
    label_names = _index_to_labels(n_out, scaler)

    # Argmax in original label space
    pred_int = np.argmax(probs, axis=1)
    pred_lbl = np.array([label_names[i] for i in pred_int], dtype=np.int32)

    # Align timestamps & mid price to window ends
    ts = df[args.time_col].iloc[L-1:].reset_index(drop=True) if args.time_col in df.columns else pd.Series(np.arange(len(probs)))
    if "mid_px" in df.columns:
        mid = df["mid_px"].iloc[L-1:].reset_index(drop=True)
    elif {"bid_px","ask_px"}.issubset(df.columns):
        mid = ((df["bid_px"] + df["ask_px"]) * 0.5).iloc[L-1:].reset_index(drop=True)
    else:
        raise ValueError("Need 'mid_px' (or bid_px+ask_px) in data_file for backtesting alignment.")

    # Assemble output
    out = pd.DataFrame({"timestamp": ts, "mid_px": mid, "pred": pred_lbl})
    for j, lab in enumerate(label_names):
        out[f"p_{lab}"] = probs[:, j].astype(np.float32)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"✔ wrote {args.out}  rows={len(out)}  seq_len={L}")

if __name__ == "__main__":
    main()