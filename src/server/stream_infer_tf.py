#!/usr/bin/env python3
"""
Micro-batch streaming inference:
• Tails a growing Parquet/CSV (or reads from stdin)
• Maintains rolling L-length window
• Emits class + probabilities with timestamps
"""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np, pandas as pd, tensorflow as tf
from src.model.transformer_tf import PositionalEncoding1D, AttentionPool1D, SqueezeExcite1D
from src.utils import data_utils

def _load_scaler(fp):
    s = json.load(open(fp)); mu=np.array(s["mean"],np.float32); sd=np.array(s["std"],np.float32); sd[sd==0]=1.0
    return s, mu, sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scaler_json", required=True)
    ap.add_argument("--source", required=True, help="csv/parquet to tail (must contain features + timestamp)")
    ap.add_argument("--time_col", default="timestamp"); ap.add_argument("--label_col", default="y")
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--interval", type=float, default=0.25, help="poll seconds")
    ap.add_argument("--out", default="results/stream_preds.csv")
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model,
        compile=False,
        custom_objects={
            "PositionalEncoding1D": PositionalEncoding1D,
            "AttentionPool1D": AttentionPool1D,
            "SqueezeExcite1D": SqueezeExcite1D,
            "hft>PositionalEncoding1D": PositionalEncoding1D,
            "hft>AttentionPool1D": AttentionPool1D,
            "hft>SqueezeExcite1D": SqueezeExcite1D,
        })
    L = args.seq_len or (model.input_shape[1] if isinstance(model.input_shape,tuple) else model.input_shape[0][1])
    scaler, mu, sd = _load_scaler(args.scaler_json)
    feats = scaler["feature_names"]; shift = scaler.get("label_shift",0)

    seen = 0
    out_cols = ["timestamp","pred","p_-1","p_0","p_1"]
    pd.DataFrame(columns=out_cols).to_csv(args.out, index=False)
    while True:
        df = (pd.read_parquet if args.source.endswith(".parquet") else pd.read_csv)(args.source)
        df = df.sort_values(args.time_col)
        Xdf = df.drop(columns=[c for c in [args.label_col, args.time_col] if c in df.columns]).select_dtypes(include=["number","bool"])
        Xdf = Xdf[feats]
        X = ((Xdf.to_numpy(np.float32) - mu) / sd)
        if len(X) >= L and len(df) > seen:
            Xseq = np.stack([X[i:i+L] for i in range(len(X)-L+1)], 0)
            proba = model.predict(Xseq[-1:,:], verbose=0)[0]
            pred  = int(np.argmax(proba) - shift)
            row = [df[args.time_col].iloc[-1], pred, float(proba[0]), float(proba[1] if len(proba)>1 else np.nan), float(proba[2] if len(proba)>2 else np.nan)]
            pd.DataFrame([row], columns=out_cols).to_csv(args.out, mode="a", header=False, index=False)
            seen = len(df)
        time.sleep(args.interval)

if __name__ == "__main__":
    main()