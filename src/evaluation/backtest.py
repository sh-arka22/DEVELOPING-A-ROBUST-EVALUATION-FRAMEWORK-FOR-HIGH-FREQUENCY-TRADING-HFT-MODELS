# #!/usr/bin/env python3
# # src/evaluation/backtest.py — map predictions → positions → PnL → risk metrics
# from __future__ import annotations
# import argparse, json, os
# from pathlib import Path
# import numpy as np
# import pandas as pd

# from .metrics import sharpe, sortino, max_drawdown, var_es

# def _annualization_factor(bar_ms: int, hours_per_day: float = 6.5, trading_days: int = 252) -> float:
#     steps_per_day = int((hours_per_day * 3600 * 1000) // max(1, bar_ms))
#     return float(steps_per_day * trading_days)

# def _to_positions(df: pd.DataFrame, mode: str, neutral_p: float, long_p: float, short_p: float) -> pd.Series:
#     """
#     mode='argmax' (default) → sign(argmax probs); if p_0 >= neutral_p → 0
#     mode='threshold' → 1 if p_1 >= long_p; -1 if p_-1 >= short_p; else 0
#     """
#     p_neg = df.get("p_-1")
#     p_neu = df.get("p_0")
#     p_pos = df.get("p_1")
#     if p_neg is None or p_neu is None or p_pos is None:
#         # fall back to predicted class if probs are missing
#         return df["pred"].astype(int).clip(-1, 1)

#     if mode == "threshold":
#         pos = np.where(p_pos >= long_p, 1, np.where(p_neg >= short_p, -1, 0))
#         return pd.Series(pos.astype(np.int8), index=df.index)
#     else:
#         # argmax with neutral band
#         arg = np.argmax(np.stack([p_neg, p_neu, p_pos], axis=1), axis=1) - 1  # [-1,0,1]
#         arg = arg.astype(np.int8)
#         if neutral_p > 0.0:
#             arg = np.where(p_neu >= neutral_p, 0, arg)
#         return pd.Series(arg, index=df.index)

# def run(
#     preds_file: str,
#     *,
#     bar_ms: int = 10,
#     hours_per_day: float = 6.5,
#     cost_bps: float = 0.5,
#     slip_bps: float = 0.0,
#     mode: str = "argmax",
#     neutral_p: float = 0.45,
#     long_p: float = 0.40,
#     short_p: float = 0.40,
#     out_dir: str = "results/backtests",
# ):
#     Path(out_dir).mkdir(parents=True, exist_ok=True)
#     df = pd.read_parquet(preds_file)
#     if "timestamp" in df.columns:
#         df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
#         df = df.sort_values("timestamp").reset_index(drop=True)

#     # positions from probabilities
#     pos = _to_positions(df, mode=mode, neutral_p=neutral_p, long_p=long_p, short_p=short_p)

#     # forward return (mid_{t+1}/mid_t - 1), last row has no next
#     r_fwd = df["mid_px"].pct_change().shift(-1)
#     r_fwd = r_fwd.iloc[:-1].reset_index(drop=True)
#     pos   = pos.iloc[:-1].reset_index(drop=True)

#     # simple transaction costs on changes in position (enter/exit/flip)
#     delta = pos.diff().fillna(pos.iloc[0]).abs()
#     cost  = (cost_bps + slip_bps) / 1e4 * delta  # proportional to |Δposition|

#     pnl_step = (pos * r_fwd) - cost
#     equity = (1.0 + pnl_step).cumprod()

#     # risk metrics
#     ann = _annualization_factor(bar_ms, hours_per_day=hours_per_day)
#     sr  = sharpe(pnl_step, ann_factor=ann)
#     so  = sortino(pnl_step, ann_factor=ann)
#     mdd = max_drawdown(equity)  # negative
#     var95, es95 = var_es(pnl_step, alpha=0.95)
#     var99, es99 = var_es(pnl_step, alpha=0.99)

#     summary = {
#         "bars": int(len(pnl_step)),
#         "bar_ms": bar_ms,
#         "cost_bps": float(cost_bps),
#         "slip_bps": float(slip_bps),
#         "mode": mode,
#         "neutral_p": float(neutral_p),
#         "long_p": float(long_p),
#         "short_p": float(short_p),
#         "Sharpe": float(sr),
#         "Sortino": float(so),
#         "MaxDrawdown": float(mdd),
#         "VaR95": float(var95), "ES95": float(es95),
#         "VaR99": float(var99), "ES99": float(es99),
#     }

#     base = os.path.splitext(os.path.basename(preds_file))[0]
#     pd.DataFrame({"timestamp": df["timestamp"].iloc[:len(equity)], "equity": equity}).to_csv(
#         f"{out_dir}/{base}_equity.csv", index=False
#     )
#     with open(f"{out_dir}/{base}_summary.json", "w") as f:
#         json.dump(summary, f, indent=2)

#     print(json.dumps(summary, indent=2))
#     print(f"✔ wrote {out_dir}/{base}_equity.csv")
#     print(f"✔ wrote {out_dir}/{base}_summary.json")

# if __name__ == "__main__":
#     pa = argparse.ArgumentParser()
#     pa.add_argument("--preds", required=True)
#     pa.add_argument("--bar_ms", type=int, default=10)
#     pa.add_argument("--hours_per_day", type=float, default=6.5)
#     pa.add_argument("--cost_bps", type=float, default=0.5)
#     pa.add_argument("--slip_bps", type=float, default=0.0)
#     pa.add_argument("--mode", choices=["argmax","threshold"], default="argmax")
#     pa.add_argument("--neutral_p", type=float, default=0.45)
#     pa.add_argument("--long_p", type=float, default=0.40)
#     pa.add_argument("--short_p", type=float, default=0.40)
#     pa.add_argument("--out_dir", default="results/backtests")
#     args = pa.parse_args()

#     run(
#         preds_file=args.preds,
#         bar_ms=args.bar_ms,
#         hours_per_day=args.hours_per_day,
#         cost_bps=args.cost_bps,
#         slip_bps=args.slip_bps,
#         mode=args.mode,
#         neutral_p=args.neutral_p,
#         long_p=args.long_p,
#         short_p=args.short_p,
#         out_dir=args.out_dir,
#     )


#!/usr/bin/env python3
# src/evaluation/backtest.py — predictions → positions → PnL → risk metrics
# v2: risk metrics on 1-second PnL (default) + diagnostics for “flat + cost”
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd

from .metrics import sharpe, sortino, max_drawdown, var_es


def _annualization_factor_for(freq: str, bar_ms: int, hours_per_day: float, trading_days: int = 252) -> float:
    f = freq.strip().lower()
    if f == "bar":
        steps = (hours_per_day * 3600.0 * 1000.0) / max(1, bar_ms)
    elif f.endswith("s"):  # e.g. "1s", "2s"
        seconds = float(f[:-1]) if len(f) > 1 else 1.0
        steps = (hours_per_day * 3600.0) / max(1.0, seconds)
    else:
        steps = hours_per_day * 3600.0
    return float(steps * trading_days)


def _to_positions(df: pd.DataFrame, mode: str, neutral_p: float, long_p: float, short_p: float) -> pd.Series:
    """
    mode='argmax' (default) → sign(argmax probs); if p_0 >= neutral_p → 0
    mode='threshold' → 1 if p_1 >= long_p; -1 if p_-1 >= short_p; else 0
    Falls back to 'pred' column if probability columns missing.
    """
    p_neg = df.get("p_-1")
    p_neu = df.get("p_0")
    p_pos = df.get("p_1")

    if p_neg is None or p_neu is None or p_pos is None:
        return df["pred"].astype(int).clip(-1, 1)

    if mode == "threshold":
        pos = np.where(p_pos.to_numpy() >= long_p, 1,
                       np.where(p_neg.to_numpy() >= short_p, -1, 0))
        return pd.Series(pos.astype(np.int8), index=df.index)
    else:
        # argmax with optional neutral band
        stack = np.stack([p_neg.to_numpy(), p_neu.to_numpy(), p_pos.to_numpy()], axis=1)
        arg = (np.argmax(stack, axis=1) - 1).astype(np.int8)  # [-1,0,1]
        if neutral_p > 0.0:
            arg = np.where(p_neu.to_numpy() >= neutral_p, 0, arg)
        return pd.Series(arg, index=df.index)


def _aggregate_returns(pnl_bar: pd.Series, ts: pd.Series, freq: str) -> pd.Series:
    f = freq.strip().lower()
    if f == "bar":
        return pnl_bar
    if "datetime64" not in str(ts.dtype).lower():
        raise ValueError("timestamp column is required (datetime) to resample metrics.")
    buckets = ts.dt.floor(f)  # lowercase avoids FutureWarning
    return (1.0 + pnl_bar).groupby(buckets).prod() - 1.0


def run(
    preds_file: str,
    *,
    bar_ms: int = 10,
    hours_per_day: float = 6.5,
    cost_bps: float = 0.5,
    slip_bps: float = 0.0,
    mode: str = "argmax",
    neutral_p: float = 0.45,
    long_p: float = 0.40,
    short_p: float = 0.40,
    metrics_freq: str = "1s",
    out_dir: str = "results/backtests",
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(preds_file)

    # time order
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

    # positions from probabilities or predicted class
    pos_full = _to_positions(df, mode=mode, neutral_p=neutral_p, long_p=long_p, short_p=short_p)

    # forward return (mid_{t+1}/mid_t - 1), last row has no next
    r_fwd_full = df["mid_px"].pct_change().shift(-1)

    # align (drop the last bar with no next return)
    pos = pos_full.iloc[:-1].reset_index(drop=True)
    r_fwd = r_fwd_full.iloc[:-1].reset_index(drop=True)

    # transaction costs on |Δposition|
    delta = pos.diff().abs().fillna(pos.iloc[0]).astype(float)
    cost_per_step = (cost_bps + slip_bps) / 1e4
    cost = cost_per_step * delta

    # per-bar PnL
    pnl_bar = (pos * r_fwd) - cost
    equity_bar = (1.0 + pnl_bar.fillna(0.0)).cumprod()

    # ---------- Diagnostics (helps detect “flat + cost” regimes) ----------
    nz_mask = r_fwd.notna() & (r_fwd != 0.0)
    share_nonzero_returns = float(nz_mask.mean())
    share_neutral_pos = float((pos == 0).mean())
    flip_rate = float(delta.mean())
    flips = int(delta.sum())
    median_abs_r_nonzero = float(np.nanmedian(np.abs(r_fwd[nz_mask]))) if nz_mask.any() else 0.0

    diags = {
        "share_nonzero_future_returns": share_nonzero_returns,
        "share_neutral_positions": share_neutral_pos,
        "flip_rate_per_bar": flip_rate,
        "flips": flips,
        "median_abs_return_nonzero": median_abs_r_nonzero,
        "bars": int(len(pnl_bar)),
    }
    print("Diagnostics:", json.dumps(diags, indent=2))

    # ---------- Risk metrics ----------
    # metrics on bar-level (old behavior)
    ann_bar = _annualization_factor_for("BAR", bar_ms, hours_per_day)
    sharpe_bar = sharpe(pnl_bar, ann_factor=ann_bar)
    sortino_bar = sortino(pnl_bar, ann_factor=ann_bar)
    mdd_bar = max_drawdown(equity_bar)
    var95_bar, es95_bar = var_es(pnl_bar, alpha=0.95)
    var99_bar, es99_bar = var_es(pnl_bar, alpha=0.99)

    # metrics on resampled horizon (default: 1S)
    try:
        pnl_resampled = _aggregate_returns(pnl_bar.fillna(0.0), df["timestamp"].iloc[:len(pnl_bar)], metrics_freq)
        equity_res = (1.0 + pnl_resampled).cumprod()
        ann_res = _annualization_factor_for(metrics_freq, bar_ms, hours_per_day)
        sharpe_res = sharpe(pnl_resampled, ann_factor=ann_res)
        sortino_res = sortino(pnl_resampled, ann_factor=ann_res)
        mdd_res = max_drawdown(equity_res)
        var95_res, es95_res = var_es(pnl_resampled, alpha=0.95)
        var99_res, es99_res = var_es(pnl_resampled, alpha=0.99)
    except Exception as e:
        # If timestamp is missing or resample fails, fall back to bar metrics
        print(f"⚠️ metrics resample ({metrics_freq}) failed: {e} — falling back to per-bar metrics")
        metrics_freq = "BAR"
        pnl_resampled = pnl_bar
        equity_res = equity_bar
        ann_res = ann_bar
        sharpe_res, sortino_res, mdd_res = sharpe_bar, sortino_bar, mdd_bar
        var95_res, es95_res = var95_bar, es95_bar
        var99_res, es99_res = var99_bar, es99_bar

    # ---------- Output ----------
    base = os.path.splitext(os.path.basename(preds_file))[0]
    # Keep the original equity output (per-bar)
    pd.DataFrame({"timestamp": df["timestamp"].iloc[:len(equity_bar)], "equity": equity_bar}).to_csv(
        f"{out_dir}/{base}_equity.csv", index=False
    )
    # Also write resampled equity if different
    if metrics_freq.upper() != "BAR":
        equity_resampled_fp = f"{out_dir}/{base}_equity_{metrics_freq.lower()}.csv"
        pd.DataFrame({"bucket": pnl_resampled.index, "equity": equity_res}).to_csv(equity_resampled_fp, index=False)
        print(f"✔ wrote {equity_resampled_fp}")

    summary = {
        "bars": int(len(pnl_bar)),
        "bar_ms": bar_ms,
        "cost_bps": float(cost_bps),
        "slip_bps": float(slip_bps),
        "mode": mode,
        "neutral_p": float(neutral_p),
        "long_p": float(long_p),
        "short_p": float(short_p),
        "diagnostics": diags,

        # Top-level metrics = resampled horizon (default 1S)
        "metrics_freq": metrics_freq,
        "Sharpe": float(sharpe_res),
        "Sortino": float(sortino_res),
        "MaxDrawdown": float(mdd_res),
        "VaR95": float(var95_res), "ES95": float(es95_res),
        "VaR99": float(var99_res), "ES99": float(es99_res),

        # For reference: per-bar metrics
        "bar_metrics": {
            "Sharpe": float(sharpe_bar),
            "Sortino": float(sortino_bar),
            "MaxDrawdown": float(mdd_bar),
            "VaR95": float(var95_bar), "ES95": float(es95_bar),
            "VaR99": float(var99_bar), "ES99": float(es99_bar),
        },
    }

    with open(f"{out_dir}/{base}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"✔ wrote {out_dir}/{base}_equity.csv")
    print(f"✔ wrote {out_dir}/{base}_summary.json")


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--preds", required=True)
    pa.add_argument("--bar_ms", type=int, default=10)
    pa.add_argument("--hours_per_day", type=float, default=6.5)
    pa.add_argument("--cost_bps", type=float, default=0.5)
    pa.add_argument("--slip_bps", type=float, default=0.0)
    pa.add_argument("--mode", choices=["argmax","threshold"], default="argmax")
    pa.add_argument("--neutral_p", type=float, default=0.45)
    pa.add_argument("--long_p", type=float, default=0.40)
    pa.add_argument("--short_p", type=float, default=0.40)
    pa.add_argument("--metrics_freq", default="1S", help="Horizon for risk metrics (e.g., '1S' or 'BAR').")
    pa.add_argument("--out_dir", default="results/backtests")
    args = pa.parse_args()

    run(
        preds_file=args.preds,
        bar_ms=args.bar_ms,
        hours_per_day=args.hours_per_day,
        cost_bps=args.cost_bps,
        slip_bps=args.slip_bps,
        mode=args.mode,
        neutral_p=args.neutral_p,
        long_p=args.long_p,
        short_p=args.short_p,
        metrics_freq=args.metrics_freq,
        out_dir=args.out_dir,
    )