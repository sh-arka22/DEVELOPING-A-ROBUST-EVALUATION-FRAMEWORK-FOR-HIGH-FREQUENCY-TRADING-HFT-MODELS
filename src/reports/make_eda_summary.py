# filepath: /Users/arkajyotisaha/Desktop/My-Thesis/code/src/reports/make_eda_summary.py -->
#!/usr/bin/env python
# ── src/reports/make_eda_summary.py ─────────────────────────────────────────
from pathlib import Path
import json, pandas as pd, numpy as np, matplotlib.pyplot as plt
plt.rcParams["figure.autolayout"] = True

ROOT   = Path(__file__).resolve().parents[1]        # …/src
DATA   = ROOT / "data"
REG    = json.loads((DATA/"features/feature_registry.json").read_text())
OUTDIR = ROOT / "reports" / "eds"
OUTDIR.mkdir(parents=True, exist_ok=True)

def _hist(series: pd.Series, title: str, outfile: Path):
    series.hist(bins=100, figsize=(4,3))
    plt.title(title); plt.tight_layout(); plt.savefig(outfile, dpi=120); plt.close()

def _label_bar(series: pd.Series, title: str, outfile: Path):
    series.value_counts().reindex([-1,0,1]).plot(kind="bar")
    plt.title(title); plt.tight_layout(); plt.savefig(outfile, dpi=120); plt.close()

# ─── ★ helper to obtain a *return* series safely ★ ──────────────────────────
def _get_return(df: pd.DataFrame) -> pd.Series:
    if "ret" in df.columns:
        return df["ret"]
    if "ret_1" in df.columns:
        return df["ret_1"]
    # fall‑back: recompute from mid_px if present
    if {"mid_px"}.issubset(df.columns):
        return df["mid_px"].pct_change().fillna(0)
    raise KeyError("No return column (ret / ret_1) and cannot recompute")

# ─── main -------------------------------------------------------------------
def main():
    rows = []
    # (collect all <tag,path> pairs exactly as before) ------------------------
    pairs = []
    for p in REG["nasdaq_itch"]:
        tag = Path(p).stem.split("_")[-1]
        pairs.append((f"ITCH_{tag}", p))
    pairs.append(("FI2010", REG["fi2010"]))
    pairs.extend(REG["crypto"].items())

    for tag, path in pairs:
        df = pd.read_parquet(path)
        ret = _get_return(df)

        rows.append({
            "dataset":  tag,
            "rows":     len(df),
            "from":     str(df.index[0])[:19],
            "to":       str(df.index[-1])[:19],
            "+1 (%)":   round((df["y"]==1).mean()*100, 2),
            " 0 (%)":   round((df["y"]==0).mean()*100, 2),
            "-1 (%)":   round((df["y"]==-1).mean()*100, 2),
            "ret σ":    round(ret.std(), 6),
            "spread µ": round(df["spread"].mean(), 6) if "spread" in df else np.nan
        })

        _hist(ret,            f"{tag} – return",        OUTDIR/f"{tag}_ret_hist.png")
        if "imbalance" in df:
            _hist(df["imbalance"], f"{tag} – imbalance",    OUTDIR/f"{tag}_imbalance_hist.png")
        _label_bar(df["y"],   f"{tag} – label balance", OUTDIR/f"{tag}_label_counts.png")

    pd.DataFrame(rows).to_csv(OUTDIR/"summary.csv", index=False)
    print("✓ EDA artefacts written →", OUTDIR.relative_to(ROOT))

if __name__ == "__main__":
    main()
