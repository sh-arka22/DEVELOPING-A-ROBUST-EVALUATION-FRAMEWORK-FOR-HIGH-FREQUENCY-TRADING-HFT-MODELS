# filepath: /Users/arkajyotisaha/Desktop/My-Thesis/code/src/data/crypto_tick_features.py -->
#!/usr/bin/env python
# ── src/data/crypto_tick_features.py ────────────────────────────────────────
"""
Week‑2: build bar‑level features + causal labels for *cleaned* Crypto‑Tick
Parquets (written by clean_crypto_tick.py).

Output  →  data/processed/CRYPTO_TICK/<symbol>_<bar_ms>ms_feat.parquet
Registry → data/features/crypto_feature_registry.json
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────── paths ──────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]                   # …/src/data
PROC_DIR  = ROOT / "data" / "processed" / "CRYPTO_TICK"           # cleaned
FEAT_DIR  = ROOT / "data" / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────── feature helpers ────────────────────────────────────────────
def _zscore(s: pd.Series, w: int = 20) -> pd.Series:
    mu  = s.rolling(w, min_periods=1).mean()
    sd  = s.rolling(w, min_periods=1).std().replace(0, 1e-9)
    return (s - mu) / sd

def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal micro‑structure feature block (Plan §2)."""
    f          = pd.DataFrame(index=df.index)
    f["mid_px"]    = (df["bid_px"] + df["ask_px"]) / 2
    f["ret"]       = f["mid_px"].pct_change()
    f["log_ret"]   = np.log(f["mid_px"]).diff()
    f["spread"]    = df["ask_px"] - df["bid_px"]
    f["imbalance"] = (df["bid_sz"] - df["ask_sz"]) / (df["bid_sz"] + df["ask_sz"] + 1e-9)
    f["bid_sz_z"]  = _zscore(df["bid_sz"])
    f["ask_sz_z"]  = _zscore(df["ask_sz"])
    f["volatility"] = _zscore(f["ret"].abs())
    f["hour_of_day"] = f.index.hour.astype("int8")
    f["day_of_week"] = f.index.dayofweek.astype("int8")
    return f.dropna()

def _add_label(feat: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """3‑class direction label (+1 / 0 / ‑1) horizon bars ahead (Plan §3)."""
    tgt  = feat["mid_px"].shift(-horizon) - feat["mid_px"]
    tick = feat["mid_px"].diff().abs().median()
    feat["y"] = np.select([tgt > tick, tgt < -tick], [1, -1], default=0).astype("int8")
    return feat.iloc[:-horizon]                      # drop rows w/out future price

# ─────────────── batch builder ──────────────────────────────────────────────
def build_crypto_features(bar_ms: int, horizon: int) -> Dict[str, Path]:
    """
    Iterate over every *cleaned* Parquet in data/processed/CRYPTO_TICK and
    emit feature/label Parquets.  Returns {symbol_interval: path}.
    """
    out_map: Dict[str, Path] = {}
    files = sorted(PROC_DIR.glob("*.parquet"))
    if not files:
        print("⚠️  No cleaned Crypto‑Tick Parquets in", PROC_DIR)
        return out_map

    pbar = tqdm(total=len(files), desc="crypto‑features", unit="file", dynamic_ncols=True)
    for p in files:
        tag = p.stem                    # e.g. BTC_1sec
        pbar.set_postfix_str(tag)       # show current file name in bar

        try:
            df = pd.read_parquet(p)

            # sanity check – cleaner guarantees these four but re‑check anyway
            req = {"bid_px", "ask_px", "bid_sz", "ask_sz"}
            if missing := req - set(df.columns):
                pbar.write(f"⚠️  {p.name} skipped – missing {missing}")
                pbar.update(1)
                continue

            # resample to uniform bar grid
            df_bar = df.resample(f"{bar_ms}ms").last().ffill()

            feat = _add_label(_make_features(df_bar), horizon)

            out = PROC_DIR / f"{tag}_{bar_ms}ms_feat.parquet"
            feat.to_parquet(out, compression="snappy")
            out_map[tag] = out

            pbar.write(f"[{tag}] {len(df_bar):,} bars → {len(feat):,} rows")

        except Exception as e:
            pbar.write(f"⚠️  {p.name} skipped – {e}")

        pbar.update(1)                  # progress tick as *soon* as file processed
    pbar.close()

    print(f"✔ Crypto‑Tick   → {len(out_map)} feature files")
    return out_map

# ─────────────── CLI wrapper (optional standalone use) ──────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar_ms",  type=int, default=100,
                    help="bar width in ms (default 100)")
    ap.add_argument("--horizon", type=int, default=10,
                    help="label horizon in bars (default 10)")
    args = ap.parse_args()

    registry = build_crypto_features(args.bar_ms, args.horizon)

    reg_path = FEAT_DIR / "crypto_feature_registry.json"
    reg_path.write_text(json.dumps({k: str(v) for k, v in registry.items()},
                                   indent=2))
    print("✓ Registry written →", reg_path.relative_to(ROOT))

if __name__ == "__main__":
    main()
