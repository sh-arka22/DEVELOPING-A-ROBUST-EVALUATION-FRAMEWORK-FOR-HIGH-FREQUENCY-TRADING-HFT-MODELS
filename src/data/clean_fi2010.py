# filepath: /Users/arkajyotisaha/Desktop/My-Thesis/code/src/data/clean_fi2010.py -->
# ── src/data/clean_fi2010.py ────────────────────────────────────────────────
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

HERE      = Path(__file__).resolve().parent
RAW_BASE  = HERE / "raw" / "FI2010"
PROC_DIR  = HERE / "processed" / "FI2010"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ---------- robust raw loader ----------------------------------------------
def _load_raw(fp: Path) -> np.ndarray:
    """Return ndarray shaped (n_snapshots, 150). Accepts 149‑column files."""
    try:
        df = pd.read_csv(fp, header=None, sep=r"\s+", engine="c")
    except Exception:
        df = pd.read_csv(fp, header=None)         # comma‑delimited fallback

    # handle orientation + padding
    if df.shape[1] in (149, 150):                 # row‑major
        arr = df.values.astype(np.float32, copy=False)
        if arr.shape[1] == 149:                   # pad missing y100
            arr = np.hstack([arr, np.zeros((arr.shape[0], 1), np.float32)])
    elif df.shape[0] in (149, 150):               # column‑major
        arr = df.T.values.astype(np.float32, copy=False)
        if arr.shape[1] == 149:
            arr = np.hstack([arr, np.zeros((arr.shape[0], 1), np.float32)])
    else:
        raise ValueError(
            f"{fp.name}: cannot interpret shape {df.shape} as FI‑2010 "
            "(needs 149/150 columns after possible transpose)."
        )
    return arr

def read_one(fp: Path) -> pd.DataFrame:
    arr = _load_raw(fp)
    feat, labels = arr[:, :145], arr[:, 145:150].astype(np.int8)

    df = pd.DataFrame(feat)
    df = pd.concat(
        [df,
         pd.DataFrame(labels, columns=["y10", "y20", "y30", "y50", "y100"])],
        axis=1
    )

    ask_p, bid_p = df[0], df[2]
    df["mid_price"] = (ask_p + bid_p) / 2
    df["spread"]    =  ask_p - bid_p

    ask_vol = df[[1 + 2*k for k in range(10)]].sum(axis=1)
    bid_vol = df[[1 + 2*k + 40 for k in range(10)]].sum(axis=1)
    df["imbalance"] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6)

    df.insert(0, "event_id", np.arange(len(df), dtype=np.int64))
    return df

# ---------- helpers ---------------------------------------------------------
def zscore(df: pd.DataFrame, exclude: list[str]) -> None:
    cols = df.columns.difference(exclude)
    mu, sd = df[cols].mean(), df[cols].std().replace(0, 1)
    df[cols] = (df[cols] - mu) / sd

def write_split(df: pd.DataFrame, split: str) -> None:
    out = PROC_DIR / f"FI2010_{split}.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   out, compression="snappy")
    print(f"   ✔ {out.name:18}  ({len(df):,} rows)")

# ---------- main ------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    txt_files = list(RAW_BASE.rglob("*NoAuction_DecPre_CF_*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No raw FI‑2010 files under {RAW_BASE}")

    by_day: dict[int, Path] = {}
    for fp in sorted(txt_files):
        day = int(fp.stem.split("_")[-1])
        if day not in range(1, 10):
            continue
        if (day not in by_day) or ("Training" in fp.parts and "Testing" in by_day[day].parts):
            by_day[day] = fp

    print("Found CF days:", sorted(by_day))

    train_days = [d for d in range(1, 7) if d in by_day]
    val_days   = [7] if 7 in by_day else []
    test_days  = [d for d in (8, 9) if d in by_day]

    if not val_days and train_days:
        val_days = [train_days.pop()]
    if not test_days and val_days:
        test_days = [val_days.pop()]
    if not train_days:
        raise RuntimeError("No data available for training split")

    splits = {"train": train_days, "val": val_days, "test": test_days}

    for split, days in splits.items():
        out_path = PROC_DIR / f"FI2010_{split}.parquet"
        if out_path.exists() and not args.overwrite:
            print(f"↻ {out_path.name} exists – skip")
            continue

        dfs = [read_one(by_day[d]) for d in days]
        df  = pd.concat(dfs, ignore_index=True)

        zscore(df, exclude=["event_id", "y10", "y20", "y30", "y50", "y100"])
        write_split(df, split)

    print(f"\n✅  All splits written to {PROC_DIR}")

if __name__ == "__main__":
    main()
