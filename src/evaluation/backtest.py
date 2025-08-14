# filepath: /Users/arkajyotisaha/Desktop/My-Thesis/code/src/evaluation/backtest.py
#!/usr/bin/env python3
# src/evaluation/backtest.py — predictions → positions → PnL → risk metrics (v2)
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import sharpe, sortino, max_drawdown, var_es

def _annualization_factor(bar_ms: int, hours_per_day: float = 6.5, trading_days: int = 252) -> float:
    steps_per_day = int((hours_per_day * 3600 * 1000) // max(1, bar_ms))
    return float(steps_per_day * trading_days)

def _to_positions(df: pd.DataFrame, mode: str, neutral_p: float, long_p: float, short_p: float) -> pd.Series:
    p_neg = df.get("p_-1"); p_neu = df.get("p_0"); p_pos = df.get("p_1")
    if p_neg is None or p_neu is None or p_pos is None:
        return df["pred"].astype(int).clip(-1, 1)
    if mode == "threshold":
        pos = np.where(p_pos >= long_p, 1, np.where(p_neg >= short_p, -1, 0))
        return pd.Series(pos.astype(np.int8), index=df.index)
    # argmax with neutral band
    arg = np.argmax(np.stack([p_neg, p_neu, p_pos], axis=1), axis=1) - 1  # [-1,0,1]
    arg = arg.astype(np.int8)
    if neutral_p > 0.0:
        arg = np.where(p_neu >= neutral_p, 0, arg)
    return pd.Series(arg, index=df.index)

def _resample_1s(pnl: pd.Series, ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    return (1.0 + pnl).groupby(ts.dt.floor("1s")).prod() - 1.0

def run(
    preds_file: str,
    *,
    bar_ms: int = 10,
    hours_per_day: float = 6.5,
    cost_bps: float = 0.5,
    slip_bps: float = 0.0,
    mode: str = "argmax",
    neutral_p: float = 0.45,
    long_p: float = 0.50,
    short_p: float = 0.50,
    out_dir: str = "results/backtests",
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(preds_file).sort_values("timestamp")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # positions from probabilities
    pos = _to_positions(df, mode=mode, neutral_p=neutral_p, long_p=long_p, short_p=short_p)

    # forward return (mid_{t+1}/mid_t - 1), last row has no next
    r_fwd = df["mid_px"].pct_change().shift(-1)
    r_fwd = r_fwd.iloc[:-1].reset_index(drop=True)
    pos   = pos.iloc[:-1].reset_index(drop=True)
    ts    = df["timestamp"].iloc[:-1].reset_index(drop=True)

    # simple transaction costs on changes in position (enter/exit/flip)
    delta = pos.diff().fillna(pos.iloc[0]).abs()
    cost  = ((cost_bps + slip_bps) / 1e4) * delta

    pnl_bar = (pos * r_fwd) - cost
    equity  = (1.0 + pnl_bar.fillna(0.0)).cumprod()

    # 1-second aggregation for risk metrics (more stable w.r.t bar size)
    pnl_1s = _resample_1s(pnl_bar.fillna(0.0), ts)
    eq_1s  = (1.0 + pnl_1s).cumprod()

    # risk metrics
    ann_bar = _annualization_factor(bar_ms, hours_per_day=hours_per_day)
    sr  = sharpe(pnl_1s)           # 1s metrics (relative compare is enough)
    so  = sortino(pnl_1s)
    mdd = max_drawdown(eq_1s)      # negative
    var95, es95 = var_es(pnl_1s, alpha=0.95)
    var99, es99 = var_es(pnl_1s, alpha=0.99)

    # diagnostics for “flat + cost” (if the strategy stayed mostly flat but paid costs)
    flat_frac = float((pos == 0).mean())
    turn = float(delta.mean())
    diagnostics = {
        "flat_fraction": flat_frac,
        "avg_position_change": turn,
        "bars": int(len(pnl_bar)),
    }

    summary = {
        "bars": int(len(pnl_bar)),
        "bar_ms": bar_ms,
        "cost_bps": float(cost_bps),
        "slip_bps": float(slip_bps),
        "mode": mode,
        "neutral_p": float(neutral_p),
        "long_p": float(long_p),
        "short_p": float(short_p),
        "metrics_freq": "1s",
        "Sharpe": float(sr),
        "Sortino": float(so),
        "MaxDrawdown": float(mdd),
        "VaR95": float(var95), "ES95": float(es95),
        "VaR99": float(var99), "ES99": float(es99),
        "diagnostics": diagnostics,
    }

    base = os.path.splitext(os.path.basename(preds_file))[0]
    # save per‑bar equity and 1s equity for plotting/report
    pd.DataFrame({"timestamp": ts, "equity": equity}).to_csv(f"{out_dir}/{base}_equity.csv", index=False)
    pd.DataFrame({"timestamp": pnl_1s.index.tz_convert(None), "equity": eq_1s.values}).to_csv(
        f"{out_dir}/{base}_equity_1s.csv", index=False
    )
    with open(f"{out_dir}/{base}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"✔ wrote {out_dir}/{base}_equity.csv")
    print(f"✔ wrote {out_dir}/{base}_equity_1s.csv")
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
    pa.add_argument("--long_p", type=float, default=0.50)
    pa.add_argument("--short_p", type=float, default=0.50)
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
        out_dir=args.out_dir,
    )
