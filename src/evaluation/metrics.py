#!/usr/bin/env python3
# src/evaluation/metrics.py — risk metrics used in the HFT evaluation framework
from __future__ import annotations
import numpy as np
import pandas as pd

def _safe_std(x: np.ndarray) -> float:
    s = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return s if np.isfinite(s) and s > 0 else 0.0

def sharpe(returns: pd.Series, ann_factor: float = 1.0) -> float:
    r = returns.dropna().to_numpy()
    if len(r) == 0: return 0.0
    m, s = float(np.mean(r)), _safe_std(r)
    return float((m / s) * np.sqrt(ann_factor)) if s > 0 else 0.0

def sortino(returns: pd.Series, ann_factor: float = 1.0) -> float:
    r = returns.dropna().to_numpy()
    if len(r) == 0: return 0.0
    downside = r[r < 0.0]
    sdn = float(np.sqrt(np.mean(downside ** 2))) if len(downside) > 0 else 0.0
    m = float(np.mean(r))
    return float((m / sdn) * np.sqrt(ann_factor)) if sdn > 0 else 0.0

def max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna().to_numpy()
    if len(eq) == 0: return 0.0
    peaks = np.maximum.accumulate(eq)
    drawdowns = (eq - peaks) / peaks
    return float(np.min(drawdowns))  # negative number

def var_es(returns: pd.Series, alpha: float = 0.99) -> tuple[float, float]:
    """
    Historical VaR/ES on simple returns. Returns (VaR, ES) as negative numbers for losses.
    """
    r = returns.dropna().to_numpy()
    if len(r) == 0: return (0.0, 0.0)
    q = float(np.quantile(r, 1.0 - alpha))
    tail = r[r <= q]
    es = float(np.mean(tail)) if len(tail) > 0 else q
    return (q, es)