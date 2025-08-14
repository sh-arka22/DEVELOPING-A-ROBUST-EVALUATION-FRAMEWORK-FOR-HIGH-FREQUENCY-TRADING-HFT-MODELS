# filepath: /Users/arkajyotisaha/Desktop/My-Thesis/code/src/evaluation/threshold_search.py -->
#!/usr/bin/env python3
# Grid-search thresholds on validation predictions (Sharpe or minority F1 objective)
from __future__ import annotations
import argparse, json, itertools, numpy as np, pandas as pd
from typing import Tuple
from sklearn.metrics import f1_score
from .metrics import sharpe, sortino, max_drawdown, var_es

def _to_positions(df: pd.DataFrame, mode: str, neutral_p: float, long_p: float, short_p: float) -> np.ndarray:
    p_neg, p_neu, p_pos = df.get("p_-1"), df.get("p_0"), df.get("p_1")
    if p_neg is None or p_neu is None or p_pos is None:
        return df["pred"].astype(int).clip(-1, 1).to_numpy()
    if mode == "threshold":
        return np.where(p_pos.to_numpy() >= long_p, 1,
                        np.where(p_neg.to_numpy() >= short_p, -1, 0)).astype(np.int8)
    # argmax with neutral band
    stack = np.stack([p_neg.to_numpy(), p_neu.to_numpy(), p_pos.to_numpy()], axis=1)
    arg = (np.argmax(stack, axis=1) - 1).astype(np.int8)
    if neutral_p > 0.0:
        arg = np.where(p_neu.to_numpy() >= neutral_p, 0, arg)
    return arg

def _resample_1s(pnl: pd.Series, ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    return (1.0 + pnl).groupby(ts.dt.floor("1s")).prod() - 1.0

def evaluate(df: pd.DataFrame, mode: str, neutral_p: float, long_p: float, short_p: float,
             cost_bps: float, objective: str) -> Tuple[float, dict]:
    ts  = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    mid = df["mid_px"].astype(float)
    pos_full = _to_positions(df, mode, neutral_p, long_p, short_p)

    # forward return (drop last)
    r_fwd = mid.pct_change().shift(-1).iloc[:-1].reset_index(drop=True)
    pos   = pd.Series(pos_full).iloc[:-1].reset_index(drop=True)
    ts    = ts.iloc[:-1].reset_index(drop=True)

    delta = pos.diff().abs().fillna(pos.iloc[0]).astype(float)
    cost  = (cost_bps / 1e4) * delta
    pnl_bar = (pos * r_fwd) - cost

    pnl_1s = _resample_1s(pnl_bar.fillna(0.0), ts)
    sr = sharpe(pnl_1s)  # relative compare sufficient
    so = sortino(pnl_1s)
    mdd = max_drawdown((1.0 + pnl_1s).cumprod())
    var95, es95 = var_es(pnl_1s, alpha=0.95)

    score = float(sr)
    if objective == "f1_rare" and "y" in df.columns:
        # macro F1 on {-1, +1}; ignore 0
        y_true = df["y"].iloc[len(df) - len(pos_full):] if len(df["y"]) == len(pos_full) else df["y"]
        y_true = y_true.iloc[:-1].astype(int).to_numpy()
        y_pred = pos.to_numpy()
        mask = np.isin(y_true, [-1, 1]) | np.isin(y_pred, [-1, 1])
        score = f1_score(y_true[mask], y_pred[mask], labels=[-1, 1], average="macro", zero_division=0.0) if mask.any() else 0.0

    return score, {"Sharpe_1s": float(sr), "Sortino_1s": float(so), "MDD_1s": float(mdd),
                   "VaR95_1s": float(var95), "ES95_1s": float(es95)}

def parse_grid(spec: str) -> dict:
    # "neutral=0.35:0.50:0.01,long=0.50:0.70:0.01,short=0.50:0.70:0.01"
    out = {}
    for part in spec.split(","):
        k,v = part.split("=")
        a,b,s = map(float, v.split(":"))
        out[k] = np.round(np.arange(a, b+1e-12, s), 6)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--mode", choices=["argmax","threshold"], default="threshold")
    ap.add_argument("--grid", default="neutral=0.35:0.50:0.01,long=0.50:0.70:0.01,short=0.50:0.70:0.01")
    ap.add_argument("--cost_bps", type=float, default=0.5)
    ap.add_argument("--objective", choices=["sharpe","f1_rare"], default="sharpe")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.preds).sort_values("timestamp")
    grid = parse_grid(args.grid)

    best, tried = None, []
    for neutral_p in grid.get("neutral",[0.0]):
        for long_p in grid.get("long",[0.5]):
            for short_p in grid.get("short",[0.5]):
                score, detail = evaluate(df, args.mode, float(neutral_p), float(long_p), float(short_p),
                                         args.cost_bps, args.objective)
                rec = {"neutral_p": float(neutral_p), "long_p": float(long_p), "short_p": float(short_p),
                       "score": float(score)} | detail
                tried.append(rec)
                if (best is None) or (rec["score"] > best["score"]):
                    best = rec

    tried = sorted(tried, key=lambda r: r["score"], reverse=True)
    out = {"objective": args.objective, "mode": args.mode, "best": best, "top5": tried[:5], "searched": len(tried)}
    with open(args.out, "w") as f: json.dump(out, f, indent=2)
    print(json.dumps(out["best"], indent=2))
    print(f"✔ wrote {args.out}")

if __name__ == "__main__":
    main()