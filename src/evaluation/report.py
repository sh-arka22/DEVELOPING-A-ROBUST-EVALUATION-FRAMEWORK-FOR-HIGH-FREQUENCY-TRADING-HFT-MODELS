#!/usr/bin/env python3
# src/evaluation/report.py – v3 (always-visible equity line)
from __future__ import annotations
import argparse, base64, io, json, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def _png_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _ensure_time_index(df: pd.DataFrame, *, candidates=("timestamp","bucket"), label="") -> pd.DataFrame:
    # find a time column, coerce to UTC, drop NaT, set tz-naive index (Matplotlib-friendly)
    for c in candidates:
        if c in df.columns:
            ts = pd.to_datetime(df[c], utc=True, errors="coerce")
            good = ts.notna()
            dropped = int((~good).sum())
            if dropped:
                print(f"[report] dropped {dropped} rows with NaT in '{c}' for {label or 'frame'}")
            ts = ts.loc[good].dt.tz_convert(None)
            df = df.loc[good].copy()
            df.index = pd.Index(ts, name="time")
            return df
    raise ValueError(f"No time column found in {label or 'frame'} (expected one of {candidates}).")

def _drawdown(equity: pd.Series) -> pd.Series:
    e = equity.to_numpy(dtype=float)
    peak = np.maximum.accumulate(np.where(np.isfinite(e), e, -np.inf))
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak <= 0.0, 0.0, e / peak - 1.0)
    return pd.Series(dd, index=equity.index)

def _make_fig():
    return plt.subplots(figsize=(10, 3), dpi=160)

def _visible_band(y: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    y_min = np.nanmin(y) if y.size else 0.0
    y_max = np.nanmax(y) if y.size else 0.0
    ptp   = float(y_max - y_min)
    yabs  = float(np.nanmax(np.abs(y))) if y.size else 0.0
    # at least ±1e-6 band; otherwise 5% of amplitude or magnitude
    pad = max(1e-6, ptp * 0.05, yabs * 0.05)
    if not np.isfinite(pad) or pad <= 0:
        pad = 1e-6
    if ptp < 1e-12:  # flat series
        return -pad, +pad
    return y_min - pad, y_max + pad

def _plot_equity(ax, idx, y, title: str):
    y = np.asarray(y, dtype=float)
    # draw baseline FIRST so the equity line sits on top (otherwise it hides a flat equity)
    ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.35)
    ax.plot(idx, y, drawstyle="steps-post", linewidth=1.8)  # equity on top

    # optional markers so a near-flat line is still visible
    if len(y) <= 2000:
        step = max(1, len(y) // 200)
        ax.plot(idx[::step], y[::step], linestyle="None", marker=".", markersize=2.5, alpha=0.8)

    ylo, yhi = _visible_band(y)
    ax.set_ylim(ylo, yhi)
    ax.set_title(title); ax.set_xlabel("time"); ax.set_ylabel("equity")
    ax.grid(True, alpha=0.25)

def _plot_drawdown(ax, idx, dd, title: str):
    dd = np.asarray(dd, dtype=float)
    ax.plot(idx, dd, linewidth=1.6)
    ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.35)
    dmin = np.nanmin(dd) if dd.size else 0.0
    if not np.isfinite(dmin) or dmin > -1e-6:
        ax.set_ylim(-1e-6, 0.0)
    ax.set_title(title); ax.set_xlabel("time"); ax.set_ylabel("drawdown")
    ax.grid(True, alpha=0.25)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--equity_csv", required=True)
    ap.add_argument("--equity_resampled_csv", help="optional (e.g., *_equity_1s.csv)")
    ap.add_argument("--out_html", required=True)
    args = ap.parse_args()

    summary = json.load(open(args.summary_json))

    # per-bar equity
    eq_bar = pd.read_csv(args.equity_csv)
    eq_bar = _ensure_time_index(eq_bar, candidates=("timestamp","bucket"), label="equity_csv")
    if "equity" not in eq_bar.columns:
        raise KeyError("Column 'equity' not found in --equity_csv")
    eq_bar["equity"] = pd.to_numeric(eq_bar["equity"], errors="coerce")
    eq_bar = eq_bar.dropna(subset=["equity"])

    dd_bar = _drawdown(eq_bar["equity"])

    fig1, ax1 = _make_fig()
    _plot_equity(ax1, eq_bar.index, eq_bar["equity"].to_numpy(), "Equity Curve (per‑bar)")
    img_equity_bar = _png_base64(fig1)

    fig2, ax2 = _make_fig()
    _plot_drawdown(ax2, dd_bar.index, dd_bar.to_numpy(), "Drawdown (per‑bar)")
    img_dd_bar = _png_base64(fig2)

    # resampled equity (optional)
    img_equity_res = None
    if args.equity_resampled_csv and os.path.exists(args.equity_resampled_csv):
        eq_res = pd.read_csv(args.equity_resampled_csv)
        eq_res = _ensure_time_index(eq_res, candidates=("bucket","timestamp"), label="equity_resampled_csv")
        if "equity" in eq_res.columns:
            eq_res["equity"] = pd.to_numeric(eq_res["equity"], errors="coerce")
            eq_res = eq_res.dropna(subset=["equity"])
            fig3, ax3 = _make_fig()
            _plot_equity(ax3, eq_res.index, eq_res["equity"].to_numpy(), "Equity Curve (resampled)")
            img_equity_res = _png_base64(fig3)

    metrics = {k: v for k, v in summary.items()
               if k in ["metrics_freq","Sharpe","Sortino","MaxDrawdown","VaR95","ES95","VaR99","ES99"]}
    diags = summary.get("diagnostics", {})

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Backtest Report</title></head>
<body>
<h1>Backtest Report</h1>
<p><b>Bars:</b> {summary['bars']} &nbsp; <b>bar_ms:</b> {summary['bar_ms']} &nbsp;
<b>cost_bps:</b> {summary['cost_bps']} &nbsp; <b>slip_bps:</b> {summary['slip_bps']} &nbsp;
<b>mode:</b> {summary['mode']} &nbsp; <b>metrics_freq:</b> {summary.get('metrics_freq','BAR')}</p>

<h2>Diagnostics</h2>
<pre>{json.dumps(diags, indent=2)}</pre>

<h2>Risk Metrics</h2>
<pre>{json.dumps(metrics, indent=2)}</pre>

<h2>Equity (per-bar)</h2>
<img src="data:image/png;base64,{img_equity_bar}" />

<h2>Drawdown (per-bar)</h2>
<img src="data:image/png;base64,{img_dd_bar}" />
"""

    if img_equity_res:
        html += f"""
<h2>Equity (resampled)</h2>
<img src="data:image/png;base64,{img_equity_res}" />
"""

    html += "</body></html>"
    os.makedirs(os.path.dirname(args.out_html), exist_ok=True)
    with open(args.out_html, "w") as f:
        f.write(html)
    print(f"✔ wrote {args.out_html}")

if __name__ == "__main__":
    main()