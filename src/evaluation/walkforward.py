#!/usr/bin/env python3
"""
Rolling walk-forward over time for parquet datasets.
Creates temp train/val/test slices per window, calls train_tf, evaluates,
and aggregates classification + backtest risk metrics.
"""
from __future__ import annotations
import argparse, json, os, shutil, tempfile, subprocess
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # repo root (…/code)
TRAIN = HERE / "train_tf.py"
EVAL  = HERE / "eval_tf.py"
PRED  = HERE / "src/predict_tf.py"
BT    = HERE / "src/evaluation/backtest.py"

def _slice(df, start, end, time_col):
    m = (df[time_col] >= start) & (df[time_col] < end)
    return df.loc[m].copy()

def _write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_cfg", required=True)     # e.g., configs/nasdaq_transformer.yaml
    ap.add_argument("--data",     required=True)     # one parquet with timestamp + features + y
    ap.add_argument("--time_col", default="timestamp")
    ap.add_argument("--train_days", type=int, default=5)
    ap.add_argument("--val_days",   type=int, default=1)
    ap.add_argument("--test_days",  type=int, default=1)
    ap.add_argument("--step_days",  type=int, default=1)
    ap.add_argument("--out_dir", default="results/walk_forward")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.data).sort_values(args.time_col)
    df[args.time_col] = pd.to_datetime(df[args.time_col], utc=True, errors="coerce")

    tmin, tmax = df[args.time_col].min(), df[args.time_col].max()
    # daily grid in UTC
    days = pd.date_range(tmin.floor("D"), tmax.ceil("D"), freq="1D")
    runs = []
    for i in range(0, len(days)-(args.train_days+args.val_days+args.test_days), args.step_days):
        t_tr0, t_tr1 = days[i],   days[i+args.train_days]
        t_va0, t_va1 = t_tr1,     days[i+args.train_days+args.val_days]
        t_te0, t_te1 = t_va1,     days[i+args.train_days+args.val_days+args.test_days]
        runs.append((t_tr0, t_tr1, t_va0, t_va1, t_te0, t_te1))

    agg = []
    for k,(t0,t1,u0,u1,v0,v1) in enumerate(runs, 1):
        with tempfile.TemporaryDirectory() as tmpd:
            tmp = Path(tmpd)
            tr = _slice(df, t0, t1, args.time_col); va = _slice(df, u0, u1, args.time_col); te = _slice(df, v0, v1, args.time_col)
            _write(tr, tmp/"train.parquet"); _write(va, tmp/"val.parquet"); _write(te, tmp/"test.parquet")

            # make per-fold YAML based on base config but swapping file paths
            cfg = Path(args.base_cfg).read_text()
            cfg = cfg.replace("nasdaq_AAPL_10ms_train.parquet", str(tmp/"train.parquet"))
            cfg = cfg.replace("nasdaq_AAPL_10ms_val.parquet",   str(tmp/"val.parquet"))
            cfg = cfg.replace("nasdaq_AAPL_10ms_test.parquet",  str(tmp/"test.parquet"))
            fold_cfg = tmp/"cfg.yaml"; fold_cfg.write_text(cfg)

            # train
            subprocess.check_call([sys.executable, str(TRAIN), "--config", str(fold_cfg)])
            # model/scaler paths are taken from the cfg
            model = "results/models/nasdaq_tf.keras"
            scaler= "results/scalers/nasdaq.json"

            # evaluate classification
            out_eval = subprocess.check_output([sys.executable, str(EVAL), model, str(tmp/"test.parquet"), scaler]).decode()
            # predict for backtest
            pred_out = tmp/"preds.parquet"
            subprocess.check_call([sys.executable, str(PRED),
                                   "--model", model, "--data_file", str(tmp/"test.parquet"),
                                   "--scaler_json", scaler, "--out", str(pred_out),
                                   "--seq_len", "100", "--label_col", "y", "--time_col", args.time_col])
            # backtest
            subprocess.check_call([sys.executable, str(BT), "--preds", str(pred_out),
                                   "--bar_ms", "10", "--cost_bps", "0.5",
                                   "--out_dir", str(Path(args.out_dir)/f"fold_{k:02d}")])
            agg.append({"fold": k, "train": f"{t0}→{t1}", "val": f"{u0}→{u1}", "test": f"{v0}→{v1}"} )

    pd.DataFrame(agg).to_csv(Path(args.out_dir)/"walk_forward_summary.csv", index=False)
    print(f"✓ Walk‑forward finished: {len(agg)} folds → {args.out_dir}")

if __name__ == "__main__":
    import sys; main()