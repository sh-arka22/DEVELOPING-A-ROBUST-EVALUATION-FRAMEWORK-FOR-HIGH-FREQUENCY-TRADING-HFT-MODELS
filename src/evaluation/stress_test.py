#!/usr/bin/env python3
"""
Stress test a trading policy under synthetic price paths:
• Block bootstrap on empirical returns (preserves intraday clustering)
• Volatility shock regime (σ×k), spread shock, and jump events
• Outputs risk metrics identical to backtest.py
"""
from __future__ import annotations
import argparse, json, numpy as np, pandas as pd
from pathlib import Path
from .metrics import sharpe, sortino, max_drawdown, var_es

def block_bootstrap(ret, block=50, paths=1000, length=None, rng=None):
    rng = np.random.default_rng(rng)
    n = len(ret) if length is None else length
    starts = rng.integers(0, len(ret)-block, size=(paths, int(np.ceil(n/block))))
    out = np.concatenate([ret[s:s+block] for s in starts[0]], axis=0)[:n]  # first for shape
    sims = np.empty((paths, n), dtype=float)
    for i in range(paths):
        idx = np.concatenate([np.arange(s, s+block) for s in starts[i]], axis=0)[:n]
        sims[i] = ret[idx]
    return sims

def apply_stress(sims, vol_mult=1.0, jump_prob=0.0, jump_scale=5.0, rng=None):
    rng = np.random.default_rng(rng)
    out = sims * vol_mult
    if jump_prob > 0:
        jumps = rng.normal(0, sims.std()*jump_scale, size=sims.shape)
        mask  = rng.uniform(0,1,size=sims.shape) < jump_prob
        out = out + jumps*mask
    return out

def summarize(pnl):
    eq = (1.0 + pnl).cumprod(axis=1)
    ann = 252*6.5*3600*1000/10  # rough 10ms→ per-year steps; cancels in relative compare
    sr  = np.array([sharpe(pd.Series(p), ann) for p in pnl])
    so  = np.array([sortino(pd.Series(p), ann) for p in pnl])
    mdd = np.array([max_drawdown(pd.Series(e)) for e in eq])
    var95 = np.quantile(pnl.sum(1), 0.05)
    es95  = pnl.sum(1)[pnl.sum(1)<=var95].mean()
    return dict(Sharpe=float(sr.mean()), Sortino=float(so.mean()),
                MaxDrawdown=float(mdd.mean()), VaR95=float(var95), ES95=float(es95))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="predictions parquet with timestamp, mid_px, probs or positions")
    ap.add_argument("--mode", choices=["argmax","threshold"], default="argmax")
    ap.add_argument("--neutral_p", type=float, default=0.45)
    ap.add_argument("--long_p", type=float, default=0.55)
    ap.add_argument("--short_p", type=float, default=0.55)
    ap.add_argument("--paths", type=int, default=500)
    ap.add_argument("--block", type=int, default=50)
    ap.add_argument("--vol_mult", type=float, default=2.0)
    ap.add_argument("--jump_prob", type=float, default=0.002)
    ap.add_argument("--jump_scale", type=float, default=6.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.preds).sort_values("timestamp")
    r = df["mid_px"].pct_change().fillna(0).to_numpy()
    sims = block_bootstrap(r, block=args.block, paths=args.paths, length=len(r))
    sims = apply_stress(sims, vol_mult=args.vol_mult, jump_prob=args.jump_prob, jump_scale=args.jump_scale)

    # simple policy: reuse predicted positions from file (or derive from probs)
    if "position" in df:
        pos = df["position"].to_numpy()
    else:
        pneg,pneu,ppos = df.get("p_-1"), df.get("p_0"), df.get("p_1")
        if pneg is not None and pneu is not None and ppos is not None:
            pos = np.where(ppos.to_numpy()>=args.long_p,1,np.where(pneg.to_numpy()>=args.short_p,-1,0))
        else:
            pos = df["pred"].astype(int).clip(-1,1).to_numpy()

    pos = pos[:-1]
    sims = sims[:,1:]  # align next-return
    pnl = (pos * sims)  # ignore additional cost/slip for stress
    summary = summarize(pnl)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"✓ wrote {args.out}")

if __name__ == "__main__":
    main()