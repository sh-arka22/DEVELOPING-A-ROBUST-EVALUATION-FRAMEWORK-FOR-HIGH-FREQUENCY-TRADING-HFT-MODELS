#!/usr/bin/env python3
"""
build_features.py — Phase 2 feature & label builder (robust to missing timestamps)

Adds:
  • Advanced moving averages: DEMA/TEMA/HMA/KAMA/TRIX (configurable windows)
  • 1-second GARCH(1,1) conditional volatility (resampled to original grid)
  • Centralized label generation with tunable alpha:
      - abs:      y = -1 if r <= -alpha; +1 if r >= +alpha; else 0
      - quantile: alpha from TRAIN abs-return tail so only top |r| fraction is ±1

CLI stays identical to prior version.
"""

from __future__ import annotations
import argparse, os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for robustness (timestamps + mid detection)
# ─────────────────────────────────────────────────────────────────────────────

# def ensure_timestamp(df: pd.DataFrame, bar_ms: int, ts_col: str = "timestamp") -> pd.DataFrame:
#     """
#     Guarantee a UTC 'timestamp' column for sorting/resample. If missing/unparsable,
#     synthesize a monotonic timeline from row order (bar_ms). Keeps prior behavior if present.
#     """
#     df = df.copy()
#     ts = None
#     if ts_col in df.columns:
#         ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
#         if ts.isna().all():  # fully unparsable → synthesize
#             ts = None
#     elif isinstance(df.index, pd.DatetimeIndex):
#         ts = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")

#     if ts is None:
#         base = pd.Timestamp("1970-01-01", tz="UTC")
#         ts = base + pd.to_timedelta(np.arange(len(df)) * int(bar_ms), unit="ms")

#     df[ts_col] = pd.DatetimeIndex(ts)
#     return df.sort_values(ts_col).reset_index(drop=True)
def ensure_timestamp(df: pd.DataFrame, bar_ms: int, ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Guarantee a UTC 'timestamp' column for sorting/resample. If missing/unparsable,
    synthesize a monotonic timeline from row order (bar_ms).
    Robust to cases where 'timestamp' is both an index level and a column.
    """
    df = df.copy()

    # 1) Pull/create a proper UTC time vector *before* touching the index
    ts = None
    if ts_col in df.columns:
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        if ts.isna().all():
            ts = None
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")

    if ts is None:
        base = pd.Timestamp("1970-01-01", tz="UTC")
        step_ms = max(int(bar_ms), 1)
        ts = base + pd.to_timedelta(np.arange(len(df)) * step_ms, unit="ms")

    # 2) Nuke any possibility of name collision: drop index to a clean RangeIndex
    df.index = pd.RangeIndex(len(df))  # removes names/levels entirely

    # 3) Sort by a temporary unique column, then commit to 'timestamp'
    tmp = "__ts__"
    df[tmp] = pd.DatetimeIndex(ts)
    df.sort_values(tmp, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df[ts_col] = df[tmp]
    df.drop(columns=[tmp], inplace=True)

    return df


def auto_mid_col(df: pd.DataFrame, preferred: str = "mid_px") -> str:
    """Pick a sensible mid-price column if the preferred one is missing."""
    candidates = [preferred, "mid_px", "mid", "mid_price", "midprice", "midpoint"]
    for c in dict.fromkeys(candidates):
        if c in df.columns:
            return c
    raise SystemExit(f"Mid-price column not found. Tried: {candidates}")


# ─────────────────────────────────────────────────────────────────────────────
# Indicators
# ─────────────────────────────────────────────────────────────────────────────

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def dema(s: pd.Series, span: int) -> pd.Series:
    e1 = ema(s, span)
    e2 = ema(e1, span)
    return 2.0 * e1 - e2

def tema(s: pd.Series, span: int) -> pd.Series:
    e1 = ema(s, span)
    e2 = ema(e1, span)
    e3 = ema(e2, span)
    return 3.0 * (e1 - e2) + e3

def wma(s: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return s.copy()
    w = np.arange(1, window + 1, dtype=float)
    w /= w.sum()
    return s.rolling(window, min_periods=window).apply(lambda x: np.dot(w, x), raw=True)

def hma(s: pd.Series, window: int) -> pd.Series:
    # HMA(n) = WMA( 2*WMA(P, n/2) - WMA(P, n), sqrt(n) )
    n = max(int(window), 1)
    n2 = max(int(round(n / 2)), 1)
    n_sqrt = max(int(np.sqrt(n)), 1)
    wma_n  = wma(s, n)
    wma_n2 = wma(s, n2)
    tmp = 2.0 * wma_n2 - wma_n
    return wma(tmp, n_sqrt)

def kama(s: pd.Series, er_window: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    # Kaufman Adaptive Moving Average (ER-based)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    out = np.full(len(s), np.nan, dtype=np.float64)
    s_np = s.to_numpy(dtype=np.float64)
    if len(s_np) < er_window + 2:
        return pd.Series(out, index=s.index)
    out[er_window] = np.nanmean(s_np[:er_window])  # seed
    for t in range(er_window + 1, len(s_np)):
        change = abs(s_np[t] - s_np[t - er_window])
        vol = np.sum(np.abs(np.diff(s_np[t - er_window:t + 1])))
        er = (change / vol) if vol > 0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        out[t] = out[t - 1] + sc * (s_np[t] - out[t - 1])
    return pd.Series(out, index=s.index)

def trix(s: pd.Series, span: int = 15) -> pd.Series:
    # TRIX: 1‑period diff of triple‑smoothed EMA of log price
    logp = np.log(s.replace(0, np.nan))
    e1 = ema(logp, span)
    e2 = ema(e1, span)
    e3 = ema(e2, span)
    return e3.diff()

def garch_vol_1s(mid: pd.Series,
                 omega: float = 1e-6, alpha: float = 0.05, beta: float = 0.90,
                 min_points: int = 30) -> pd.Series:
    """Compute GARCH(1,1) conditional vol on 1-second log returns; upsample to the original grid."""
    if mid.isna().all():
        return pd.Series(np.nan, index=mid.index)
    mid_1s = mid.resample("1S").last().dropna()
    r = np.log(mid_1s).diff().dropna()
    if len(r) < min_points:
        return pd.Series(np.nan, index=mid.index)
    var = np.empty(len(r), dtype=np.float64)
    var[0] = r.var() if np.isfinite(r.var()) else 1e-6
    for t in range(1, len(r)):
        var[t] = omega + alpha * (r.iloc[t - 1] ** 2) + beta * var[t - 1]
    vol_1s = pd.Series(np.sqrt(var), index=r.index)
    return vol_1s.reindex(mid.index, method="ffill")


# ─────────────────────────────────────────────────────────────────────────────
# Labeling
# ─────────────────────────────────────────────────────────────────────────────

def forward_return(mid: pd.Series, horizon_steps: int) -> pd.Series:
    return (mid.shift(-horizon_steps) - mid) / mid

def label_by_alpha(ret_fwd: pd.Series, alpha: float) -> pd.Series:
    y = np.zeros(len(ret_fwd), dtype=np.int8)
    y[ret_fwd >= alpha]  = 1
    y[ret_fwd <= -alpha] = -1
    return pd.Series(y, index=ret_fwd.index)

def calibrate_alpha_from_quantile(abs_rets_train: pd.Series, target_tail_frac: float) -> float:
    # If target_tail_frac = 0.01, ~1% of samples have |r| >= alpha (≈ half up, half down).
    q = 1.0 - float(target_tail_frac)
    q = min(max(q, 0.50), 0.999999)
    return float(abs_rets_train.quantile(q))


# ─────────────────────────────────────────────────────────────────────────────
# Core feature pipeline
# ─────────────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame,
                   mid_col: str,
                   *,
                   bar_ms: int,
                   fast_ms: int,
                   slow_ms: int,
                   trix_ms: int,
                   kama_window: int = 10) -> pd.DataFrame:
    df = ensure_timestamp(df.copy(), bar_ms)
    df[mid_col] = df[mid_col].astype("float64")

    def ms_to_n(ms: int) -> int:
        return max(int(round(ms / max(bar_ms, 1))), 1)

    n_fast = ms_to_n(fast_ms)
    n_slow = ms_to_n(slow_ms)
    n_trix = ms_to_n(trix_ms)

    px = df[mid_col]
    df[f"dema_{fast_ms}ms"] = dema(px, n_fast)
    df[f"dema_{slow_ms}ms"] = dema(px, n_slow)
    df[f"tema_{fast_ms}ms"] = tema(px, n_fast)
    df[f"tema_{slow_ms}ms"] = tema(px, n_slow)
    df[f"hma_{fast_ms}ms"]  = hma(px, n_fast)
    df[f"hma_{slow_ms}ms"]  = hma(px, n_slow)
    df[f"kama_{kama_window}"] = kama(px, er_window=kama_window, fast=2, slow=30)
    df[f"trix_{trix_ms}ms"] = trix(px, n_trix)

    # 1s GARCH vol (needs DatetimeIndex)
    df = df.set_index("timestamp")
    df["garch_vol_1s"] = garch_vol_1s(df[mid_col])
    df = df.reset_index()

    return df


def build_one_split(df_in: pd.DataFrame,
                    *,
                    mid_col: str,
                    bar_ms: int,
                    fast_ms: int,
                    slow_ms: int,
                    trix_ms: int,
                    kama_window: int,
                    horizon_steps: int,
                    alpha: float) -> pd.DataFrame:
    df = ensure_timestamp(df_in.copy(), bar_ms)

    # indicators
    df = add_indicators(df, mid_col, bar_ms=bar_ms,
                        fast_ms=fast_ms, slow_ms=slow_ms, trix_ms=trix_ms,
                        kama_window=kama_window)

    # forward return + labels
    df = df.set_index("timestamp")
    r_fwd = forward_return(df[mid_col], horizon_steps=horizon_steps)
    y = label_by_alpha(r_fwd, alpha)
    df["y"] = y.astype(np.int8)

    # back to columns + light types
    df = df.reset_index()
    for c in df.columns:
        if c == "y": continue
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype(np.float32)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI / IO
# ─────────────────────────────────────────────────────────────────────────────

def _load(fp: str) -> pd.DataFrame:
    if fp is None: return None
    return pd.read_parquet(fp)

def _save(df: pd.DataFrame, fp_out: str):
    Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(fp_out, compression="snappy")
    print("✔ wrote", fp_out, f"rows={len(df):,}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train parquet path")
    ap.add_argument("--val",   required=True, help="val parquet path")
    ap.add_argument("--test",  required=True, help="test parquet path")
    ap.add_argument("--out_tag", default="", help="optional suffix for outputs (e.g., _v2). If empty, overwrite in place.")

    # grid & horizons
    ap.add_argument("--bar_ms", type=int, default=10, help="bar size in milliseconds (e.g., 10 for 10ms grid)")
    ap.add_argument("--horizon_steps", type=int, default=1, help="forward steps for labeling (on the bar grid)")

    # indicator params in milliseconds (converted to bar counts internally)
    ap.add_argument("--fast_ms", type=int, default=50,  help="fast MA window in ms")
    ap.add_argument("--slow_ms", type=int, default=200, help="slow MA window in ms")
    ap.add_argument("--trix_ms", type=int, default=150, help="TRIX EMA span in ms")
    ap.add_argument("--kama_n", type=int, default=10,   help="KAMA efficiency-ratio window (bars)")

    # labeling
    ap.add_argument("--alpha_mode", choices=["abs", "quantile"], default="abs",
                    help="'abs' uses a fixed alpha; 'quantile' learns alpha from TRAIN abs-return tail.")
    ap.add_argument("--alpha", type=float, default=0.0005, help="abs mode: threshold on fwd return (e.g., 5 bps = 0.0005)")
    ap.add_argument("--target_tail_frac", type=float, default=0.01,
                    help="quantile mode: fraction of TRAIN samples labeled ±1 in total (≈1% gives ~0.5% up/down)")

    ap.add_argument("--mid_col", default="mid_px", help="mid-price column name")
    args = ap.parse_args()

    tr, va, te = _load(args.train), _load(args.val), _load(args.test)
    if tr is None or va is None or te is None:
        raise SystemExit("Failed to load one or more input files.")

    # Ensure timestamps + mid col before calibrating alpha
    tr = ensure_timestamp(tr, args.bar_ms); va = ensure_timestamp(va, args.bar_ms); te = ensure_timestamp(te, args.bar_ms)
    mid_col = args.mid_col if args.mid_col in tr.columns else auto_mid_col(tr, args.mid_col)
    if mid_col != args.mid_col:
        print(f"[mid_col] '{args.mid_col}' not found. Using '{mid_col}'.")

    # Compute TRAIN forward return to calibrate quantile alpha if needed
    tr_ret = forward_return(tr.set_index("timestamp")[mid_col], args.horizon_steps)

    if args.alpha_mode == "quantile":
        abs_ret = tr_ret.abs().dropna()
        if len(abs_ret) == 0:
            raise SystemExit("Cannot calibrate alpha: TRAIN forward returns are empty.")
        alpha = calibrate_alpha_from_quantile(abs_ret, args.target_tail_frac)
        print(f"[labels] quantile mode: target_tail_frac={args.target_tail_frac} → alpha={alpha:.6g}")
    else:
        alpha = float(args.alpha)
        print(f"[labels] abs mode: alpha={alpha:.6g}")

    # Build each split
    def _outpath(in_path: str) -> str:
        if args.out_tag:
            b, ext = os.path.splitext(in_path)
            return f"{b}{args.out_tag}{ext}"
        return in_path

    print("[features] generating indicators and labels …")
    tr_out = build_one_split(tr, mid_col=mid_col, bar_ms=args.bar_ms,
                             fast_ms=args.fast_ms, slow_ms=args.slow_ms, trix_ms=args.trix_ms,
                             kama_window=args.kama_n, horizon_steps=args.horizon_steps, alpha=alpha)
    va_out = build_one_split(va, mid_col=mid_col, bar_ms=args.bar_ms,
                             fast_ms=args.fast_ms, slow_ms=args.slow_ms, trix_ms=args.trix_ms,
                             kama_window=args.kama_n, horizon_steps=args.horizon_steps, alpha=alpha)
    te_out = build_one_split(te, mid_col=mid_col, bar_ms=args.bar_ms,
                             fast_ms=args.fast_ms, slow_ms=args.slow_ms, trix_ms=args.trix_ms,
                             kama_window=args.kama_n, horizon_steps=args.horizon_steps, alpha=alpha)

    # quick label diagnostics
    def _dist(y: pd.Series) -> dict:
        u, c = np.unique(y.dropna().astype(int), return_counts=True)
        return {int(i): int(j) for i, j in zip(u, c)}
    print("label distribution (train):", _dist(tr_out["y"]))
    print("label distribution (val):  ", _dist(va_out["y"]))
    print("label distribution (test): ", _dist(te_out["y"]))

    # Save
    _save(tr_out, _outpath(args.train))
    _save(va_out, _outpath(args.val))
    _save(te_out, _outpath(args.test))

if __name__ == "__main__":
    main()
