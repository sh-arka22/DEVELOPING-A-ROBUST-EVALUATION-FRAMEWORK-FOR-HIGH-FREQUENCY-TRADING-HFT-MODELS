#!/usr/bin/env python
# ── src/data/clean_crypto_tick.py ───────────────────────────────────────────
"""
Clean heterogeneous Crypto‑Tick files
→ data/processed/CRYPTO_TICK/<symbol>_<interval>.parquet

Guarantees
──────────
• tz‑aware UTC DatetimeIndex with no duplicates
• Four Level‑1 columns **bid_px, ask_px, bid_sz, ask_sz** always present
• Derives bid/ask from midpoint & spread, sizes from buys / sells if needed
• Drops obviously corrupt (≤ 0) prices
• Forward‑fills gaps so rolling windows remain well‑behaved
"""

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ─────────────── paths ──────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]          # …/src/data
RAW_DIR   = ROOT / "data" / "raw"       / "CRYPTO_TICK"
PROC_DIR  = ROOT / "data" / "processed" / "CRYPTO_TICK"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────── alias tables ───────────────────────────────────────────────
_ALIAS_PRICE = {
    "bid_px": {"bid_px", "bidprice", "bid_price", "best_bid_price"},
    "ask_px": {"ask_px", "askprice", "ask_price", "best_ask_price"},
}
_ALIAS_SIZE = {
    "bid_sz": {"bid_sz", "bidsize", "bid_size", "best_bid_size",
               "bid_qty", "bidvolume", "buys"},
    "ask_sz": {"ask_sz", "asksize", "ask_size", "best_ask_size",
               "ask_qty", "askvolume", "sells"},
}

# ─────────────── helper: standardise to L1 columns ──────────────────────────
def _standardise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame contains bid_px, ask_px, bid_sz, ask_sz.
    Rules (run in‑place, in order):
        1. Rename known aliases.
        2. If bid/ask *prices* missing but midpoint & spread present:
              bid_px = midpoint - spread/2
              ask_px = midpoint + spread/2
        3. If bid/ask *sizes* missing but buys / sells present:
              bid_sz = buys ; ask_sz = sells
        4. Fallback: sizes → 0.
    Raises KeyError if we still cannot build bid_px & ask_px.
    """
    out = df.copy()

    # 1. rename obvious aliases ------------------------------------------------
    for tgt, alts in {**_ALIAS_PRICE, **_ALIAS_SIZE}.items():
        for a in alts:
            if a in out.columns:
                out.rename(columns={a: tgt}, inplace=True)
                break

    # 2. derive bid/ask from midpoint & spread --------------------------------
    if {"bid_px", "ask_px"} - set(out.columns):
        if {"midpoint", "spread"}.issubset(out.columns):
            mid = out["midpoint"]
            spr = out["spread"].clip(lower=0)          # avoid negative spread
            if "bid_px" not in out.columns:
                out["bid_px"] = mid - spr / 2
            if "ask_px" not in out.columns:
                out["ask_px"] = mid + spr / 2

    # 3. derive sizes from buys / sells ---------------------------------------
    if "bid_sz" not in out.columns and "buys" in out.columns:
        out["bid_sz"] = out["buys"]
    if "ask_sz" not in out.columns and "sells" in out.columns:
        out["ask_sz"] = out["sells"]

    # 4. fallback size = 0 -----------------------------------------------------
    if "bid_sz" not in out.columns:
        out["bid_sz"] = 0.0
    if "ask_sz" not in out.columns:
        out["ask_sz"] = 0.0

    # final sanity
    req = {"bid_px", "ask_px", "bid_sz", "ask_sz"}
    missing = req - set(out.columns)
    if missing:
        raise KeyError(f"cannot derive core L1 columns {missing}")

    # keep the four essentials first, retain everything else afterwards
    front = out[list(req)]
    rest  = out.drop(columns=front.columns, errors="ignore")
    return pd.concat([front, rest], axis=1)

# ─────────────── helper: infer pandas frequency from filename ───────────────
def _infer_freq(interval: str) -> str | None:
    """Return a pandas‑compatible offset string or None if unknown."""
    if interval.endswith("sec"):
        return f"{interval.rstrip('sec')}S"
    if interval.endswith("min"):
        return f"{interval.rstrip('min')}T"
    if interval.endswith("ms"):
        return f"{interval}ms"
    return None

# ─────────────── single‑file cleaner ────────────────────────────────────────
def _clean_one(path: Path) -> Path:
    tag        = path.stem                      # e.g. BTC_1sec
    sym, interval = tag.split("_", 1)

    # read CSV or Parquet ------------------------------------------------------
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    # timestamp → UTC index ----------------------------------------------------
    if not isinstance(df.index, pd.DatetimeIndex):
        ts_col = next((c for c in df.columns
                       if "time" in c.lower() or "ts" in c.lower()), None)
        if ts_col is None:
            raise ValueError(f"{path.name}: no timestamp column found")
        df = df.set_index(pd.to_datetime(df[ts_col], utc=True)).drop(columns=[ts_col])

    df = df[~df.index.duplicated(keep="first")].sort_index()

    # drop non‑positive prices -------------------------------------------------
    price_cols = [c for c in df.columns if "price" in c.lower() or c.lower() == "midpoint"]
    df = df[(df[price_cols] > 0).all(axis=1)]

    # enforce continuity -------------------------------------------------------
    freq = _infer_freq(interval)
    if freq:
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
        df = df.reindex(full_idx).ffill()

    # attach the four L1 columns ----------------------------------------------
    df = _standardise(df)

    # write Parquet ------------------------------------------------------------
    out = PROC_DIR / f"{tag}.parquet"
    df.to_parquet(out, compression="snappy")
    return out

# ─────────────── CLI entry point ────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--pattern", default="*.parquet",
                    help="glob pattern inside data/raw/CRYPTO_TICK (default *.parquet)")
    args = pa.parse_args()

    files = sorted(RAW_DIR.glob(args.pattern))
    if not files:
        print("No files found in", RAW_DIR)
        return

    for f in tqdm(files, desc="clean crypto", unit="file"):
        try:
            out = _clean_one(f)
            tqdm.write(f"✔ {f.name}  →  {out.name}")
        except Exception as e:
            tqdm.write(f"⚠️  {f.name} skipped – {e}")

if __name__ == "__main__":
    main()
