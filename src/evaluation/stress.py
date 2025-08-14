# filepath: /Users/arkajyotisaha/Desktop/My-Thesis/code/src/evaluation/stress.py
#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, numpy as np, pandas as pd
from .metrics import sharpe, sortino, max_drawdown, var_es

def stress_pnl(pnl_bar: pd.Series, ts: pd.Series, *,
               shock_scale: float = 2.0, drop_frac: float = 0.02, seed: int = 42):
    np.random.seed(seed)
    # scale returns up/down
    pnl_scaled = pnl_bar * shock_scale
    # random drop-outs (liquidity holes)
    mask = np.random.rand(len(pnl_bar)) < drop_frac
    pnl_dropout = pnl_bar.copy()
    pnl_dropout[mask] = pnl_dropout[mask] * -1.0  # flip sign as adverse move

    def _score(pnl):
        pnl = pnl.fillna(0.0)
        pnl_1s = (1.0 + pnl).groupby(pd.to_datetime(ts, utc=True, errors="coerce").dt.floor("1s")).prod() - 1.0
        eq = (1.0 + pnl_1s).cumprod()
        return {
            "Sharpe": float(sharpe(pnl_1s)),
            "Sortino": float(sortino(pnl_1s)),
            "MDD": float(max_drawdown(eq)),
            "VaR95": float(var_es(pnl_1s, 0.95)[0]),
            "ES95":  float(var_es(pnl_1s, 0.95)[1]),
        }

    return {"scale": _score(pnl_scaled), "dropout": _score(pnl_dropout)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)  # parquet with timestamp, mid_px, p_-1/p_0/p_1 OR pred
    ap.add_argument("--neutral_p", type=float, default=0.45)
    ap.add_argument("--long_p", type=float, default=0.50)
    ap.add_argument("--short_p", type=float, default=0.50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.preds).sort_values("timestamp")
    ts = df["timestamp"]
    # naive positions from argmax with neutral band
    if {"p_-1","p_0","p_1"}.issubset(df.columns):
        pneg, pneu, ppos = df["p_-1"].values, df["p_0"].values, df["p_1"].values
        arg = np.argmax(np.stack([pneg,pneu,ppos],1),1) - 1
        arg = np.where(pneu >= args.neutral_p, 0, arg).astype(np.int8)
    else:
        arg = df["pred"].astype(int).clip(-1,1).values
    r_fwd = df["mid_px"].pct_change().shift(-1).iloc[:-1].reset_index(drop=True)
    pos   = pd.Series(arg).iloc[:-1].reset_index(drop=True)
    ts    = ts.iloc[:-1].reset_index(drop=True)
    pnl   = (pos * r_fwd).astype(float)
    res = stress_pnl(pnl, ts)
    with open(args.out,"w") as f: json.dump(res,f,indent=2)
    print(json.dumps(res, indent=2))
if __name__ == "__main__":
    main()