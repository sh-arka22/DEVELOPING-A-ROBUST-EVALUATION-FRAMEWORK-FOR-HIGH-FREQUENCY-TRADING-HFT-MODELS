# filepath: /Users/arkajyotisaha/Desktop/My-Thesis/code/src/data/build_features.py -->
# #!/usr/bin/env python3
# # Build bar‑level features & labels for one NASDAQ instrument.
# from __future__ import annotations
# import argparse, json, math, warnings, re
# from pathlib import Path
# from typing import Dict, Callable, Optional
# import numpy as np
# import pandas as pd
# import numba as nb
# from tqdm import tqdm
# from arch import arch_model

# warnings.filterwarnings("ignore", category=FutureWarning)

# ROOT     = Path(__file__).resolve().parents[1]
# ITCH_DIR = ROOT / "data" / "processed" / "NASDAQ_ITCH"
# ITCH_DIR.mkdir(parents=True, exist_ok=True)

# # ── helpers ───────────────────────────────────────────────────────────────
# @nb.njit(cache=True)
# def _wma(arr: np.ndarray, n: int) -> np.ndarray:
#     out = np.empty_like(arr)
#     w   = np.arange(1, n+1, dtype=arr.dtype); ws = w.sum()
#     for i in range(arr.shape[0]):
#         out[i] = np.nan if i < n-1 else (arr[i-n+1:i+1]*w).sum()/ws
#     return out

# def ema(s: pd.Series, n:int) -> pd.Series:  return s.ewm(span=n, adjust=False).mean()
# def dema(s: pd.Series, n:int) -> pd.Series: e1=ema(s,n); return 2*e1-ema(e1,n)
# def tema(s: pd.Series, n:int) -> pd.Series: e1=ema(s,n); e2=ema(e1,n); e3=ema(e2,n); return 3*e1-3*e2+e3
# def hma (s: pd.Series, n:int) -> pd.Series:
#     half, root = int(n/2), int(math.sqrt(n))
#     return pd.Series(_wma(2*_wma(s.values, half) - _wma(s.values, n), root), index=s.index)

# def kama(price: pd.Series, er_period=10, fast=2, slow=30) -> pd.Series:
#     change = price.diff(er_period).abs()
#     vol    = price.diff().abs().rolling(er_period).sum()
#     er     = (change / vol).fillna(0)
#     sc     = (er*(2/(fast+1)-2/(slow+1)) + 2/(slow+1))**2
#     out = pd.Series(index=price.index, dtype=float); out.iloc[0] = price.iloc[0]
#     for i in range(1, len(price)):
#         out.iloc[i] = out.iloc[i-1] + sc.iloc[i]*(price.iloc[i]-out.iloc[i-1])
#     return out

# def zscore(s: pd.Series, win=500) -> pd.Series:
#     mu = s.rolling(win).mean() if len(s)>=win else s.expanding().mean()
#     sd = s.rolling(win).std()  if len(s)>=win else s.expanding().std()
#     sd = sd.replace(0, 1e-9)
#     return (s - mu) / sd

# def _ffill_resample(df: pd.DataFrame, bar_ms:int, snap_ms:int) -> pd.DataFrame:
#     """Bring snapshots to exact bar grid with controlled ffill gap."""
#     limit = max(1, math.ceil(snap_ms / bar_ms))
#     def is_px(c:str) -> bool: return c.endswith("_px") or "_px_" in c or c in ("bid_px","ask_px")
#     def is_sz(c:str) -> bool: return c.endswith("_sz") or "_sz_" in c or c.endswith("sz1")
#     agg_map = {}
#     for c in df.columns:
#         if is_px(c):   agg_map[c] = "last"
#         elif is_sz(c): agg_map[c] = "mean"
#         else:          agg_map[c] = "last"
#     out = df.resample(f"{bar_ms}ms").agg(agg_map).ffill(limit=limit)
#     return out

# # ── features ───────────────────────────────────────────────────────────────
# def make_features(df: pd.DataFrame, *, bar_ms:int) -> pd.DataFrame:
#     need = {"mid_px","microprice","spread","imbalance","last_trade_px","cum_vol",
#             "bid_sz1","ask_sz1","ask_px","bid_px"}
#     miss = need - set(df.columns)
#     if miss:
#         raise KeyError(f"Cleaner must provide columns: {miss}")

#     # Drop crossed books just in case (ask<=bid)
#     df = df[df["ask_px"] > df["bid_px"]].copy()

#     f = pd.DataFrame(index=df.index)
#     f["mid_px"]    = df["mid_px"]
#     f["micropx"]   = df["microprice"]
#     f["spread"]    = df["spread"]
#     f["imbalance"] = df["imbalance"]
#     f["ret_mid"]   = f["mid_px"].pct_change(fill_method=None).fillna(0)
#     f["ret_micro"] = f["micropx"].pct_change(fill_method=None).fillna(0)
#     f["log_ret"]   = np.log(f["mid_px"]).diff().fillna(0)

#     # depth (L1..L5 sums & shapes)
#     bid_depth_cols = [c for c in df.columns if re.match(r"^bid_sz(_\d+)?$", c)]
#     ask_depth_cols = [c for c in df.columns if re.match(r"^ask_sz(_\d+)?$", c)]
#     bid_depth = df[bid_depth_cols].fillna(0).to_numpy()
#     ask_depth = df[ask_depth_cols].fillna(0).to_numpy()
#     f["depth_bid_L5"] = bid_depth.sum(1)
#     f["depth_ask_L5"] = ask_depth.sum(1)
#     f["depth_imb5"]   = (f["depth_bid_L5"] - f["depth_ask_L5"]) / \
#                         (f["depth_bid_L5"] + f["depth_ask_L5"] + 1e-9)
#     f["q_iratio"] = df["bid_sz1"] / (df["bid_sz1"] + df["ask_sz1"] + 1e-9)
#     f["depth_slope_bid"] = (df["bid_sz1"] - df.get("bid_sz_5", df["bid_sz1"])) / 4
#     f["depth_slope_ask"] = (df["ask_sz1"] - df.get("ask_sz_5", df["ask_sz1"])) / 4

#     # order‑flow
#     trade_px = df["last_trade_px"].replace(0, np.nan).ffill()
#     f["trade_dir"] = np.sign(trade_px - f["mid_px"]).fillna(0)
#     vol_deltas = df["cum_vol"].diff().fillna(0)
#     intensity_n = max(5, int(1000/bar_ms))
#     f["of_intensity"] = vol_deltas.rolling(intensity_n).mean().fillna(0)
#     f["trade_gap"]    = (trade_px - f["micropx"]).fillna(0)

#     # volatility proxies
#     atr_n = max(3, int(1000/bar_ms))
#     rng   = f["mid_px"].rolling(atr_n).max() - f["mid_px"].rolling(atr_n).min()
#     f["atr_fast"] = (rng / f["mid_px"]).fillna(method="bfill")

#     n_fast = max(4, int( 800 / bar_ms))
#     n_slow = max(8, int(2400 / bar_ms))
#     f["dema_f"] = dema(f["micropx"], n_fast)
#     f["dema_s"] = dema(f["micropx"], n_slow)
#     f["tema_f"] = tema(f["micropx"], n_fast)
#     f["hma16"]  = hma (f["micropx"], 16)
#     f["kama"]   = kama(f["micropx"])

#     n_trix = max(5, int(1500/bar_ms))
#     ema1 = ema(f["micropx"], n_trix); ema2 = ema(ema1, n_trix); ema3 = ema(ema2, n_trix)
#     f["trix"] = (ema3 / ema3.shift(1) - 1).fillna(0) * 100

#     # GARCH on 1‑second returns (optional, ffill to bar grid)
#     f["garch_vol"] = np.nan
#     r1s = f["micropx"].resample("1s").last().pct_change().dropna() * 100
#     if len(r1s) > 300:
#         vol = arch_model(r1s, p=1, q=1).fit(disp=False).conditional_volatility
#         f["garch_vol"] = vol.reindex(f.index, method="ffill")

#     # z‑scores
#     for col in ["spread","imbalance","depth_imb5","of_intensity"]:
#         f[col+"_z"] = zscore(f[col])

#     return f.dropna()

# # ── labels (horizon in **bars**) ──────────────────────────────────────────
# def add_label(f: pd.DataFrame, *, bar_ms:int, horizon_bars:int,
#               alpha: float = 0.20) -> pd.DataFrame:
#     """
#     3‑class label at t for ΔP over next `horizon_bars`:
#        +1 if ΔP >  max(alpha * dyn_tick, min_tick)
#         0 if |ΔP| ≤ threshold
#        -1 if ΔP < -max(alpha * dyn_tick, min_tick)
#     where dyn_tick = rolling median |ΔP(1 bar)|.
#     """
#     steps = int(horizon_bars)
#     ref   = f["micropx"]
#     tgt   = ref.shift(-steps) - ref

#     # robust local movement scale
#     dyn_tick = ref.diff().abs().rolling(100).median().bfill()

#     # min tick by price regime (tweak as needed)
#     min_tick = np.where(ref < 2.0, 0.0005, 0.001)

#     thr = np.maximum(alpha * dyn_tick, min_tick)
#     f = f.copy()
#     f["y"] = np.select([tgt > thr, tgt < -thr], [1, -1], 0).astype(np.int8)
#     return f.dropna()

# # ── NASDAQ pipeline ───────────────────────────────────────────────────────
# def process_nasdaq(bar_ms:int, horizon:int, snap_ms:int,
#                    stock: Optional[str], locate: Optional[int]) -> tuple[Path,Path,Path]:
#     if locate is not None:
#         candidate = ITCH_DIR / f"mid_{locate}_{snap_ms}ms.parquet"
#         tag = str(locate)
#     elif stock:
#         candidate = ITCH_DIR / f"mid_{stock}_{snap_ms}ms.parquet"
#         tag = stock
#     else:
#         raise SystemExit("Select instrument with --stock or --locate (same as cleaning).")

#     if not candidate.exists():
#         raise FileNotFoundError(f"Snapshot file not found: {candidate}\n"
#                                 f"Run clean_nasdaq_itch.py with the same --snap_ms and instrument.")

#     df = pd.read_parquet(candidate)
#     if "timestamp" in df and not isinstance(df.index, pd.DatetimeIndex):
#         df = df.set_index(pd.to_datetime(df["timestamp"], utc=True)).drop(columns="timestamp")

#     bars = _ffill_resample(df, bar_ms, snap_ms)
#     bars = bars[bars["ask_px"] > bars["bid_px"]]  # paranoia

#     feat = make_features(bars, bar_ms=bar_ms)
#     feat = add_label(feat, bar_ms=bar_ms, horizon_bars=horizon)  # ✅ horizon in BARS

#     n = len(feat)
#     tr, va, te = feat[:int(.7*n)], feat[int(.7*n):int(.85*n)], feat[int(.85*n):]
#     paths = [ITCH_DIR / f"nasdaq_{tag}_{bar_ms}ms_{p}.parquet" for p in ("train","val","test")]
#     for subset, p in zip((tr,va,te), paths):
#         p.parent.mkdir(parents=True, exist_ok=True)
#         subset.to_parquet(p, compression="snappy")
#     print(f"✔ NASDAQ_ITCH {tag} @ {bar_ms} ms | horizon {horizon} bars | rows {n:,}")
#     return tuple(paths)

# _PIPE: Dict[str, Callable[...,object]] = {"NASDAQ_ITCH": process_nasdaq}

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--dataset", required=True, choices=_PIPE.keys())
#     ap.add_argument("--bar_ms",  type=int, default=20)
#     ap.add_argument("--horizon", type=int, default=50, help="Lookahead in bars (e.g., 50 bars @20ms ≈1s)")
#     ap.add_argument("--snap_ms", type=int, default=20, help="Must match cleaning grid")
#     ap.add_argument("--stock")
#     ap.add_argument("--locate", type=int)
#     args = ap.parse_args()

#     print(f"▶ Build {args.dataset} | instrument {args.stock or args.locate} "
#           f"@ {args.bar_ms} ms | horizon {args.horizon} bars | snap {args.snap_ms} ms")

#     result = _PIPE[args.dataset](args.bar_ms, args.horizon, args.snap_ms, args.stock, args.locate)

#     reg_path = ITCH_DIR / "feature_registry.json"
#     registry = json.loads(reg_path.read_text()) if reg_path.exists() else {}
#     key = f"nasdaq_itch_{args.stock or args.locate}"
#     registry[key] = [str(p) for p in result]
#     reg_path.write_text(json.dumps(registry, indent=2))
#     print("✓ Registry updated:", reg_path)

# if __name__ == "__main__":
#     main()



# #!/usr/bin/env python3
# # Build bar‑level features & labels for one NASDAQ instrument.
# from __future__ import annotations
# import argparse, json, math, warnings, re
# from pathlib import Path
# from typing import Dict, Callable, Optional
# import numpy as np
# import pandas as pd
# import numba as nb
# from tqdm import tqdm
# from arch import arch_model

# warnings.filterwarnings("ignore", category=FutureWarning)

# ROOT     = Path(__file__).resolve().parents[1]
# ITCH_DIR = ROOT / "data" / "processed" / "NASDAQ_ITCH"
# ITCH_DIR.mkdir(parents=True, exist_ok=True)

# # ── helpers ───────────────────────────────────────────────────────────────
# @nb.njit(cache=True)
# def _wma(arr: np.ndarray, n: int) -> np.ndarray:
#     out = np.empty_like(arr)
#     w   = np.arange(1, n+1, dtype=arr.dtype); ws = w.sum()
#     for i in range(arr.shape[0]):
#         out[i] = np.nan if i < n-1 else (arr[i-n+1:i+1]*w).sum()/ws
#     return out

# def ema(s: pd.Series, n:int) -> pd.Series:  return s.ewm(span=n, adjust=False).mean()
# def dema(s: pd.Series, n:int) -> pd.Series: e1=ema(s,n); return 2*e1-ema(e1,n)
# def tema(s: pd.Series, n:int) -> pd.Series: e1=ema(s,n); e2=ema(e1,n); e3=ema(e2,n); return 3*e1-3*e2+e3
# def hma (s: pd.Series, n:int) -> pd.Series:
#     half, root = int(n/2), int(math.sqrt(n))
#     return pd.Series(_wma(2*_wma(s.values, half) - _wma(s.values, n), root), index=s.index)

# def kama(price: pd.Series, er_period=10, fast=2, slow=30) -> pd.Series:
#     change = price.diff(er_period).abs()
#     vol    = price.diff().abs().rolling(er_period).sum()
#     er     = (change / vol).fillna(0)
#     sc     = (er*(2/(fast+1)-2/(slow+1)) + 2/(slow+1))**2
#     out = pd.Series(index=price.index, dtype=float); out.iloc[0] = price.iloc[0]
#     for i in range(1, len(price)):
#         out.iloc[i] = out.iloc[i-1] + sc.iloc[i]*(price.iloc[i]-out.iloc[i-1])
#     return out

# def zscore(s: pd.Series, win=500) -> pd.Series:
#     mu = s.rolling(win).mean() if len(s)>=win else s.expanding().mean()
#     sd = s.rolling(win).std()  if len(s)>=win else s.expanding().std()
#     sd = sd.replace(0, 1e-9)
#     return (s - mu) / sd

# def _ffill_resample(df: pd.DataFrame, bar_ms:int, snap_ms:int) -> pd.DataFrame:
#     """Bring snapshots to exact bar grid with controlled ffill gap."""
#     limit = max(1, math.ceil(snap_ms / bar_ms))
#     def is_px(c:str) -> bool: return c.endswith("_px") or "_px_" in c or c in ("bid_px","ask_px")
#     def is_sz(c:str) -> bool: return c.endswith("_sz") or "_sz_" in c or c.endswith("sz1")
#     agg_map = {}
#     for c in df.columns:
#         if is_px(c):   agg_map[c] = "last"
#         elif is_sz(c): agg_map[c] = "mean"
#         else:          agg_map[c] = "last"
#     out = df.resample(f"{bar_ms}ms").agg(agg_map).ffill(limit=limit)
#     return out

# # ── features ───────────────────────────────────────────────────────────────
# def make_features(df: pd.DataFrame, *, bar_ms:int) -> pd.DataFrame:
#     need = {"mid_px","microprice","spread","imbalance","last_trade_px","cum_vol",
#             "bid_sz1","ask_sz1","ask_px","bid_px"}
#     miss = need - set(df.columns)
#     if miss:
#         raise KeyError(f"Cleaner must provide columns: {miss}")

#     # Drop crossed books just in case (ask<=bid)
#     df = df[df["ask_px"] > df["bid_px"]].copy()

#     f = pd.DataFrame(index=df.index)
#     f["mid_px"]    = df["mid_px"]
#     f["micropx"]   = df["microprice"]
#     f["spread"]    = df["spread"]
#     f["imbalance"] = df["imbalance"]
#     f["ret_mid"]   = f["mid_px"].pct_change(fill_method=None).fillna(0)
#     f["ret_micro"] = f["micropx"].pct_change(fill_method=None).fillna(0)
#     f["log_ret"]   = np.log(f["mid_px"]).diff().fillna(0)

#     # depth (L1..L5 sums & shapes)
#     bid_depth_cols = [c for c in df.columns if re.match(r"^bid_sz(_\d+)?$", c)]
#     ask_depth_cols = [c for c in df.columns if re.match(r"^ask_sz(_\d+)?$", c)]
#     bid_depth = df[bid_depth_cols].fillna(0).to_numpy()
#     ask_depth = df[ask_depth_cols].fillna(0).to_numpy()
#     f["depth_bid_L5"] = bid_depth.sum(1)
#     f["depth_ask_L5"] = ask_depth.sum(1)
#     f["depth_imb5"]   = (f["depth_bid_L5"] - f["depth_ask_L5"]) / \
#                         (f["depth_bid_L5"] + f["depth_ask_L5"] + 1e-9)
#     f["q_iratio"] = df["bid_sz1"] / (df["bid_sz1"] + df["ask_sz1"] + 1e-9)
#     f["depth_slope_bid"] = (df["bid_sz1"] - df.get("bid_sz_5", df["bid_sz1"])) / 4
#     f["depth_slope_ask"] = (df["ask_sz1"] - df.get("ask_sz_5", df["ask_sz1"])) / 4

#     # order‑flow
#     trade_px = df["last_trade_px"].replace(0, np.nan).ffill()
#     f["trade_dir"] = np.sign(trade_px - f["mid_px"]).fillna(0)
#     vol_deltas = df["cum_vol"].diff().fillna(0)
#     intensity_n = max(5, int(1000/bar_ms))
#     f["of_intensity"] = vol_deltas.rolling(intensity_n).mean().fillna(0)
#     f["trade_gap"]    = (trade_px - f["micropx"]).fillna(0)

#     # volatility proxies
#     atr_n = max(3, int(1000/bar_ms))
#     rng   = f["mid_px"].rolling(atr_n).max() - f["mid_px"].rolling(atr_n).min()
#     f["atr_fast"] = (rng / f["mid_px"]).fillna(method="bfill")

#     n_fast = max(4, int( 800 / bar_ms))
#     n_slow = max(8, int(2400 / bar_ms))
#     f["dema_f"] = dema(f["micropx"], n_fast)
#     f["dema_s"] = dema(f["micropx"], n_slow)
#     f["tema_f"] = tema(f["micropx"], n_fast)
#     f["hma16"]  = hma (f["micropx"], 16)
#     f["kama"]   = kama(f["micropx"])

#     n_trix = max(5, int(1500/bar_ms))
#     ema1 = ema(f["micropx"], n_trix); ema2 = ema(ema1, n_trix); ema3 = ema(ema2, n_trix)
#     f["trix"] = (ema3 / ema3.shift(1) - 1).fillna(0) * 100

#     # GARCH on 1‑second returns (optional, ffill to bar grid)
#     f["garch_vol"] = np.nan
#     r1s = f["micropx"].resample("1s").last().pct_change().dropna() * 100
#     if len(r1s) > 300:
#         vol = arch_model(r1s, p=1, q=1).fit(disp=False).conditional_volatility
#         f["garch_vol"] = vol.reindex(f.index, method="ffill")

#     # z‑scores
#     for col in ["spread","imbalance","depth_imb5","of_intensity"]:
#         f[col+"_z"] = zscore(f[col])

#     return f.dropna()

# # ── labels (horizon in **bars**) ──────────────────────────────────────────
# def add_label(f: pd.DataFrame, *, bar_ms:int, horizon_bars:int,
#               alpha: float = 0.20) -> pd.DataFrame:
#     """
#     3‑class label at t for ΔP over next `horizon_bars`:
#        +1 if ΔP >  max(alpha * dyn_tick, min_tick)
#         0 if |ΔP| ≤ threshold
#        -1 if ΔP < -max(alpha * dyn_tick, min_tick)
#     where dyn_tick = rolling median |ΔP(1 bar)|.
#     """
#     steps = int(horizon_bars)
#     ref   = f["micropx"]
#     tgt   = ref.shift(-steps) - ref

#     # robust local movement scale
#     dyn_tick = ref.diff().abs().rolling(100).median().bfill()

#     # min tick by price regime (tweak as needed)
#     min_tick = np.where(ref < 2.0, 0.0005, 0.001)

#     thr = np.maximum(alpha * dyn_tick, min_tick)
#     f = f.copy()
#     f["y"] = np.select([tgt > thr, tgt < -thr], [1, -1], 0).astype(np.int8)
#     return f.dropna()

# # ── NASDAQ pipeline ───────────────────────────────────────────────────────
# def process_nasdaq(bar_ms:int, horizon:int, snap_ms:int,
#                    stock: Optional[str], locate: Optional[int]) -> tuple[Path,Path,Path]:
#     if locate is not None:
#         candidate = ITCH_DIR / f"mid_{locate}_{snap_ms}ms.parquet"
#         tag = str(locate)
#     elif stock:
#         candidate = ITCH_DIR / f"mid_{stock}_{snap_ms}ms.parquet"
#         tag = stock
#     else:
#         raise SystemExit("Select instrument with --stock or --locate (same as cleaning).")

#     if not candidate.exists():
#         raise FileNotFoundError(f"Snapshot file not found: {candidate}\n"
#                                 f"Run clean_nasdaq_itch.py with the same --snap_ms and instrument.")

#     df = pd.read_parquet(candidate)
#     if "timestamp" in df and not isinstance(df.index, pd.DatetimeIndex):
#         df = df.set_index(pd.to_datetime(df["timestamp"], utc=True)).drop(columns="timestamp")

#     bars = _ffill_resample(df, bar_ms, snap_ms)
#     bars = bars[bars["ask_px"] > bars["bid_px"]]  # paranoia

#     feat = make_features(bars, bar_ms=bar_ms)
#     feat = add_label(feat, bar_ms=bar_ms, horizon_bars=horizon)  # ✅ horizon in BARS

#     n = len(feat)
#     tr, va, te = feat[:int(.7*n)], feat[int(.7*n):int(.85*n)], feat[int(.85*n):]
#     paths = [ITCH_DIR / f"nasdaq_{tag}_{bar_ms}ms_{p}.parquet" for p in ("train","val","test")]
#     for subset, p in zip((tr,va,te), paths):
#         p.parent.mkdir(parents=True, exist_ok=True)
#         subset.to_parquet(p, compression="snappy")
#     print(f"✔ NASDAQ_ITCH {tag} @ {bar_ms} ms | horizon {horizon} bars | rows {n:,}")
#     return tuple(paths)

# _PIPE: Dict[str, Callable[...,object]] = {"NASDAQ_ITCH": process_nasdaq}

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--dataset", required=True, choices=_PIPE.keys())
#     ap.add_argument("--bar_ms",  type=int, default=20)
#     ap.add_argument("--horizon", type=int, default=50, help="Lookahead in bars (e.g., 50 bars @20ms ≈1s)")
#     ap.add_argument("--snap_ms", type=int, default=20, help="Must match cleaning grid")
#     ap.add_argument("--stock")
#     ap.add_argument("--locate", type=int)
#     args = ap.parse_args()

#     print(f"▶ Build {args.dataset} | instrument {args.stock or args.locate} "
#           f"@ {args.bar_ms} ms | horizon {args.horizon} bars | snap {args.snap_ms} ms")

#     result = _PIPE[args.dataset](args.bar_ms, args.horizon, args.snap_ms, args.stock, args.locate)

#     reg_path = ITCH_DIR / "feature_registry.json"
#     registry = json.loads(reg_path.read_text()) if reg_path.exists() else {}
#     key = f"nasdaq_itch_{args.stock or args.locate}"
#     registry[key] = [str(p) for p in result]
#     reg_path.write_text(json.dumps(registry, indent=2))
#     print("✓ Registry updated:", reg_path)

# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# # Build bar‑level features & labels for one NASDAQ instrument.
# from __future__ import annotations
# import argparse, json, math, warnings, re
# from pathlib import Path
# from typing import Dict, Callable, Optional
# import numpy as np
# import pandas as pd
# import numba as nb
# from tqdm import tqdm
# from arch import arch_model

# warnings.filterwarnings("ignore", category=FutureWarning)

# ROOT     = Path(__file__).resolve().parents[1]
# ITCH_DIR = ROOT / "data" / "processed" / "NASDAQ_ITCH"
# ITCH_DIR.mkdir(parents=True, exist_ok=True)

# # ── helpers ───────────────────────────────────────────────────────────────
# @nb.njit(cache=True)
# def _wma(arr: np.ndarray, n: int) -> np.ndarray:
#     out = np.empty_like(arr)
#     w   = np.arange(1, n+1, dtype=arr.dtype); ws = w.sum()
#     for i in range(arr.shape[0]):
#         out[i] = np.nan if i < n-1 else (arr[i-n+1:i+1]*w).sum()/ws
#     return out

# def ema(s: pd.Series, n:int) -> pd.Series:  return s.ewm(span=n, adjust=False).mean()
# def dema(s: pd.Series, n:int) -> pd.Series: e1=ema(s,n); return 2*e1-ema(e1,n)
# def tema(s: pd.Series, n:int) -> pd.Series: e1=ema(s,n); e2=ema(e1,n); e3=ema(e2,n); return 3*e1-3*e2+e3
# def hma (s: pd.Series, n:int) -> pd.Series:
#     half, root = int(n/2), int(math.sqrt(n))
#     return pd.Series(_wma(2*_wma(s.values, half) - _wma(s.values, n), root), index=s.index)

# def kama(price: pd.Series, er_period=10, fast=2, slow=30) -> pd.Series:
#     change = price.diff(er_period).abs()
#     vol    = price.diff().abs().rolling(er_period).sum()
#     er     = (change / vol).fillna(0)
#     sc     = (er*(2/(fast+1)-2/(slow+1)) + 2/(slow+1))**2
#     out = pd.Series(index=price.index, dtype=float); out.iloc[0] = price.iloc[0]
#     for i in range(1, len(price)):
#         out.iloc[i] = out.iloc[i-1] + sc.iloc[i]*(price.iloc[i]-out.iloc[i-1])
#     return out

# def zscore(s: pd.Series, win=500) -> pd.Series:
#     mu = s.rolling(win).mean() if len(s)>=win else s.expanding().mean()
#     sd = s.rolling(win).std()  if len(s)>=win else s.expanding().std()
#     sd = sd.replace(0, 1e-9)
#     return (s - mu) / sd

# def _ffill_resample(df: pd.DataFrame, bar_ms:int, snap_ms:int) -> pd.DataFrame:
#     """Bring snapshots to exact bar grid with controlled ffill gap."""
#     limit = max(1, math.ceil(snap_ms / bar_ms))
#     def is_px(c:str) -> bool: return c.endswith("_px") or "_px_" in c or c in ("bid_px","ask_px")
#     def is_sz(c:str) -> bool: return c.endswith("_sz") or "_sz_" in c or c.endswith("sz1")
#     agg_map = {}
#     for c in df.columns:
#         if is_px(c):   agg_map[c] = "last"
#         elif is_sz(c): agg_map[c] = "mean"
#         else:          agg_map[c] = "last"
#     out = df.resample(f"{bar_ms}ms").agg(agg_map).ffill(limit=limit)
#     return out

# # ── features ───────────────────────────────────────────────────────────────
# def make_features(df: pd.DataFrame, *, bar_ms:int) -> pd.DataFrame:
#     need = {"mid_px","microprice","spread","imbalance","last_trade_px","cum_vol",
#             "bid_sz1","ask_sz1","ask_px","bid_px"}
#     miss = need - set(df.columns)
#     if miss:
#         raise KeyError(f"Cleaner must provide columns: {miss}")

#     # Drop crossed books just in case (ask<=bid)
#     df = df[df["ask_px"] > df["bid_px"]].copy()

#     f = pd.DataFrame(index=df.index)
#     f["mid_px"]    = df["mid_px"]
#     f["micropx"]   = df["microprice"]
#     f["spread"]    = df["spread"]
#     f["imbalance"] = df["imbalance"]
#     f["ret_mid"]   = f["mid_px"].pct_change(fill_method=None).fillna(0)
#     f["ret_micro"] = f["micropx"].pct_change(fill_method=None).fillna(0)
#     f["log_ret"]   = np.log(f["mid_px"]).diff().fillna(0)

#     # depth (L1..L5 sums & shapes)
#     bid_depth_cols = [c for c in df.columns if re.match(r"^bid_sz(_\d+)?$", c)]
#     ask_depth_cols = [c for c in df.columns if re.match(r"^ask_sz(_\d+)?$", c)]
#     bid_depth = df[bid_depth_cols].fillna(0).to_numpy()
#     ask_depth = df[ask_depth_cols].fillna(0).to_numpy()
#     f["depth_bid_L5"] = bid_depth.sum(1)
#     f["depth_ask_L5"] = ask_depth.sum(1)
#     f["depth_imb5"]   = (f["depth_bid_L5"] - f["depth_ask_L5"]) / \
#                         (f["depth_bid_L5"] + f["depth_ask_L5"] + 1e-9)
#     f["q_iratio"] = df["bid_sz1"] / (df["bid_sz1"] + df["ask_sz1"] + 1e-9)
#     f["depth_slope_bid"] = (df["bid_sz1"] - df.get("bid_sz_5", df["bid_sz1"])) / 4
#     f["depth_slope_ask"] = (df["ask_sz1"] - df.get("ask_sz_5", df["ask_sz1"])) / 4

#     # order‑flow
#     trade_px = df["last_trade_px"].replace(0, np.nan).ffill()
#     f["trade_dir"] = np.sign(trade_px - f["mid_px"]).fillna(0)
#     vol_deltas = df["cum_vol"].diff().fillna(0)
#     intensity_n = max(5, int(1000/bar_ms))
#     f["of_intensity"] = vol_deltas.rolling(intensity_n).mean().fillna(0)
#     f["trade_gap"]    = (trade_px - f["micropx"]).fillna(0)

#     # volatility proxies
#     atr_n = max(3, int(1000/bar_ms))
#     rng   = f["mid_px"].rolling(atr_n).max() - f["mid_px"].rolling(atr_n).min()
#     f["atr_fast"] = (rng / f["mid_px"]).fillna(method="bfill")

#     n_fast = max(4, int( 800 / bar_ms))
#     n_slow = max(8, int(2400 / bar_ms))
#     f["dema_f"] = dema(f["micropx"], n_fast)
#     f["dema_s"] = dema(f["micropx"], n_slow)
#     f["tema_f"] = tema(f["micropx"], n_fast)
#     f["hma16"]  = hma (f["micropx"], 16)
#     f["kama"]   = kama(f["micropx"])

#     n_trix = max(5, int(1500/bar_ms))
#     ema1 = ema(f["micropx"], n_trix); ema2 = ema(ema1, n_trix); ema3 = ema(ema2, n_trix)
#     f["trix"] = (ema3 / ema3.shift(1) - 1).fillna(0) * 100

#     # GARCH on 1‑second returns (optional, ffill to bar grid)
#     f["garch_vol"] = np.nan
#     r1s = f["micropx"].resample("1s").last().pct_change().dropna() * 100
#     if len(r1s) > 300:
#         vol = arch_model(r1s, p=1, q=1).fit(disp=False).conditional_volatility
#         f["garch_vol"] = vol.reindex(f.index, method="ffill")

#     # z‑scores
#     for col in ["spread","imbalance","depth_imb5","of_intensity"]:
#         f[col+"_z"] = zscore(f[col])

#     return f.dropna()

# # ── labels (horizon in **bars**) ──────────────────────────────────────────
# def add_label(f: pd.DataFrame, *, bar_ms:int, horizon_bars:int,
#               alpha: float = 0.20) -> pd.DataFrame:
#     """
#     3‑class label at t for ΔP over next `horizon_bars`:
#        +1 if ΔP >  max(alpha * dyn_tick, min_tick)
#         0 if |ΔP| ≤ threshold
#        -1 if ΔP < -max(alpha * dyn_tick, min_tick)

#     If you see only a single class after building, LOWER alpha (e.g. 0.10 or 0.05)
#     and rebuild features for this instrument/day/horizon.
#     """
#     steps = int(horizon_bars)
#     ref   = f["micropx"]
#     tgt   = ref.shift(-steps) - ref

#     # robust local movement scale
#     dyn_tick = ref.diff().abs().rolling(100).median().bfill()

#     # min tick by price regime (tweak as needed)
#     min_tick = np.where(ref < 2.0, 0.0005, 0.001)

#     thr = np.maximum(alpha * dyn_tick, min_tick)
#     f = f.copy()
#     f["y"] = np.select([tgt > thr, tgt < -thr], [1, -1], 0).astype(np.int8)
#     return f.dropna()

# # ── NASDAQ pipeline ───────────────────────────────────────────────────────
# def process_nasdaq(bar_ms:int, horizon:int, snap_ms:int,
#                    stock: Optional[str], locate: Optional[int],
#                    alpha: float) -> tuple[Path,Path,Path]:
#     if locate is not None:
#         candidate = ITCH_DIR / f"mid_{locate}_{snap_ms}ms.parquet"
#         tag = str(locate)
#     elif stock:
#         candidate = ITCH_DIR / f"mid_{stock}_{snap_ms}ms.parquet"
#         tag = stock
#     else:
#         raise SystemExit("Select instrument with --stock or --locate (same as cleaning).")

#     if not candidate.exists():
#         raise FileNotFoundError(f"Snapshot file not found: {candidate}\n"
#                                 f"Run clean_nasdaq_itch.py with the same --snap_ms and instrument.")

#     df = pd.read_parquet(candidate)
#     if "timestamp" in df and not isinstance(df.index, pd.DatetimeIndex):
#         df = df.set_index(pd.to_datetime(df["timestamp"], utc=True)).drop(columns="timestamp")

#     bars = _ffill_resample(df, bar_ms, snap_ms)
#     bars = bars[bars["ask_px"] > bars["bid_px"]]  # paranoia

#     feat = make_features(bars, bar_ms=bar_ms)
#     feat = add_label(feat, bar_ms=bar_ms, horizon_bars=horizon, alpha=alpha)  # ✅ alpha exposed

#     # quick sanity: label distribution
#     vc = feat["y"].value_counts(dropna=False).sort_index()
#     print("Label distribution (y):", vc.to_dict())
#     if vc.shape[0] <= 1:
#         print("⚠️  Only a single class present. Rebuild with a smaller --alpha "
#               "(e.g., 0.10 or 0.05) for this instrument/day/horizon.")

#     n = len(feat)
#     tr, va, te = feat[:int(.7*n)], feat[int(.7*n):int(.85*n)], feat[int(.85*n):]
#     paths = [ITCH_DIR / f"nasdaq_{tag}_{bar_ms}ms_{p}.parquet" for p in ("train","val","test")]
#     for subset, p in zip((tr,va,te), paths):
#         p.parent.mkdir(parents=True, exist_ok=True)
#         subset.to_parquet(p, compression="snappy")
#     print(f"✔ NASDAQ_ITCH {tag} @ {bar_ms} ms | horizon {horizon} bars | alpha {alpha} | rows {n:,}")
#     return tuple(paths)

# _PIPE: Dict[str, Callable[...,object]] = {"NASDAQ_ITCH": process_nasdaq}

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--dataset", required=True, choices=_PIPE.keys())
#     ap.add_argument("--bar_ms",  type=int, default=20)
#     ap.add_argument("--horizon", type=int, default=50, help="Lookahead in bars (e.g., 50 bars @20ms ≈1s)")
#     ap.add_argument("--snap_ms", type=int, default=20, help="Must match cleaning grid")
#     ap.add_argument("--alpha",   type=float, default=0.20, help="Label threshold scale (lower → more ±1s)")
#     ap.add_argument("--stock")
#     ap.add_argument("--locate", type=int)
#     args = ap.parse_args()

#     print(f"▶ Build {args.dataset} | instrument {args.stock or args.locate} "
#           f"@ {args.bar_ms} ms | horizon {args.horizon} bars | snap {args.snap_ms} ms | alpha {args.alpha}")

#     result = _PIPE[args.dataset](args.bar_ms, args.horizon, args.snap_ms, args.stock, args.locate, args.alpha)

#     reg_path = ITCH_DIR / "feature_registry.json"
#     registry = json.loads(reg_path.read_text()) if reg_path.exists() else {}
#     key = f"nasdaq_itch_{args.stock or args.locate}"
#     registry[key] = [str(p) for p in result]
#     reg_path.write_text(json.dumps(registry, indent=2))
#     print("✓ Registry updated:", reg_path)

# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# # Build bar‑level features & labels for one NASDAQ instrument.
# from __future__ import annotations
# import argparse, json, math, warnings, re
# from pathlib import Path
# from typing import Dict, Callable, Optional
# import numpy as np
# import pandas as pd
# import numba as nb
# from arch import arch_model

# warnings.filterwarnings("ignore", category=FutureWarning)

# ROOT     = Path(__file__).resolve().parents[1]
# ITCH_DIR = ROOT / "data" / "processed" / "NASDAQ_ITCH"
# ITCH_DIR.mkdir(parents=True, exist_ok=True)

# # ── helpers ───────────────────────────────────────────────────────────────
# @nb.njit(cache=True)
# def _wma(arr: np.ndarray, n: int) -> np.ndarray:
#     out = np.empty_like(arr)
#     w   = np.arange(1, n+1, dtype=arr.dtype); ws = w.sum()
#     for i in range(arr.shape[0]):
#         out[i] = np.nan if i < n-1 else (arr[i-n+1:i+1]*w).sum()/ws
#     return out

# def ema(s: pd.Series, n:int) -> pd.Series:  return s.ewm(span=n, adjust=False).mean()
# def dema(s: pd.Series, n:int) -> pd.Series: e1=ema(s,n); return 2*e1-ema(e1,n)
# def tema(s: pd.Series, n:int) -> pd.Series: e1=ema(s,n); e2=ema(e1,n); e3=ema(e2,n); return 3*e1-3*e2+e3
# def hma (s: pd.Series, n:int) -> pd.Series:
#     half, root = int(n/2), int(math.sqrt(n))
#     return pd.Series(_wma(2*_wma(s.values, half) - _wma(s.values, n), root), index=s.index)

# def kama(price: pd.Series, er_period=10, fast=2, slow=30) -> pd.Series:
#     change = price.diff(er_period).abs()
#     vol    = price.diff().abs().rolling(er_period).sum()
#     er     = (change / vol).fillna(0)
#     sc     = (er*(2/(fast+1)-2/(slow+1)) + 2/(slow+1))**2
#     out = pd.Series(index=price.index, dtype=float); out.iloc[0] = price.iloc[0]
#     for i in range(1, len(price)):
#         out.iloc[i] = out.iloc[i-1] + sc.iloc[i]*(price.iloc[i]-out.iloc[i-1])
#     return out

# def zscore(s: pd.Series, win=500) -> pd.Series:
#     mu = s.rolling(win).mean() if len(s)>=win else s.expanding().mean()
#     sd = s.rolling(win).std()  if len(s)>=win else s.expanding().std()
#     sd = sd.replace(0, 1e-9)
#     return (s - mu) / sd

# def _ffill_resample(df: pd.DataFrame, bar_ms:int, snap_ms:int) -> pd.DataFrame:
#     """Bring snapshots to exact bar grid with controlled ffill gap."""
#     import math as _m
#     limit = max(1, _m.ceil(snap_ms / bar_ms))
#     def is_px(c:str) -> bool: return c.endswith("_px") or "_px_" in c or c in ("bid_px","ask_px")
#     def is_sz(c:str) -> bool: return c.endswith("_sz") or "_sz_" in c or c.endswith("sz1")
#     agg_map = {}
#     for c in df.columns:
#         if is_px(c):   agg_map[c] = "last"
#         elif is_sz(c): agg_map[c] = "mean"
#         else:          agg_map[c] = "last"
#     out = df.resample(f"{bar_ms}ms").agg(agg_map).ffill(limit=limit)
#     return out

# # ── features ───────────────────────────────────────────────────────────────
# def make_features(df: pd.DataFrame, *, bar_ms:int) -> pd.DataFrame:
#     need = {"mid_px","microprice","spread","imbalance","last_trade_px","cum_vol",
#             "bid_sz1","ask_sz1","ask_px","bid_px"}
#     miss = need - set(df.columns)
#     if miss:
#         raise KeyError(f"Cleaner must provide columns: {miss}")

#     # Drop crossed books just in case (ask<=bid)
#     df = df[df["ask_px"] > df["bid_px"]].copy()

#     f = pd.DataFrame(index=df.index)
#     f["mid_px"]    = df["mid_px"]
#     f["micropx"]   = df["microprice"]
#     f["spread"]    = df["spread"]
#     f["imbalance"] = df["imbalance"]
#     f["ret_mid"]   = f["mid_px"].pct_change(fill_method=None).fillna(0)
#     f["ret_micro"] = f["micropx"].pct_change(fill_method=None).fillna(0)
#     f["log_ret"]   = np.log(f["mid_px"]).diff().fillna(0)

#     # depth (L1..L5 sums & shapes)
#     bid_depth_cols = [c for c in df.columns if re.match(r"^bid_sz(_\d+)?$", c)]
#     ask_depth_cols = [c for c in df.columns if re.match(r"^ask_sz(_\d+)?$", c)]
#     bid_depth = df[bid_depth_cols].fillna(0).to_numpy()
#     ask_depth = df[ask_depth_cols].fillna(0).to_numpy()
#     f["depth_bid_L5"] = bid_depth.sum(1)
#     f["depth_ask_L5"] = ask_depth.sum(1)
#     f["depth_imb5"]   = (f["depth_bid_L5"] - f["depth_ask_L5"]) / \
#                         (f["depth_bid_L5"] + f["depth_ask_L5"] + 1e-9)
#     f["q_iratio"] = df["bid_sz1"] / (df["bid_sz1"] + df["ask_sz1"] + 1e-9)
#     f["depth_slope_bid"] = (df["bid_sz1"] - df.get("bid_sz_5", df["bid_sz1"])) / 4
#     f["depth_slope_ask"] = (df["ask_sz1"] - df.get("ask_sz_5", df["ask_sz1"])) / 4

#     # order‑flow
#     trade_px = df["last_trade_px"].replace(0, np.nan).ffill()
#     f["trade_dir"] = np.sign(trade_px - f["mid_px"]).fillna(0)
#     vol_deltas = df["cum_vol"].diff().fillna(0)
#     intensity_n = max(5, int(1000/bar_ms))
#     f["of_intensity"] = vol_deltas.rolling(intensity_n).mean().fillna(0)
#     f["trade_gap"]    = (trade_px - f["micropx"]).fillna(0)

#     # volatility proxies
#     atr_n = max(3, int(1000/bar_ms))
#     rng   = f["mid_px"].rolling(atr_n).max() - f["mid_px"].rolling(atr_n).min()
#     f["atr_fast"] = (rng / f["mid_px"]).fillna(method="bfill")

#     n_fast = max(4, int( 800 / bar_ms))
#     n_slow = max(8, int(2400 / bar_ms))
#     f["dema_f"] = dema(f["micropx"], n_fast)
#     f["dema_s"] = dema(f["micropx"], n_slow)
#     f["tema_f"] = tema(f["micropx"], n_fast)
#     f["hma16"]  = hma (f["micropx"], 16)
#     f["kama"]   = kama(f["micropx"])

#     n_trix = max(5, int(1500/bar_ms))
#     ema1 = ema(f["micropx"], n_trix); ema2 = ema(ema1, n_trix); ema3 = ema(ema2, n_trix)
#     f["trix"] = (ema3 / ema3.shift(1) - 1).fillna(0) * 100

#     # GARCH on 1‑second returns (optional, ffill to bar grid)
#     f["garch_vol"] = np.nan
#     r1s = f["micropx"].resample("1s").last().pct_change().dropna() * 100
#     if len(r1s) > 300:
#         vol = arch_model(r1s, p=1, q=1).fit(disp=False).conditional_volatility
#         f["garch_vol"] = vol.reindex(f.index, method="ffill")

#     # z‑scores
#     for col in ["spread","imbalance","depth_imb5","of_intensity"]:
#         f[col+"_z"] = zscore(f[col])

#     return f.dropna()

# # ── labels (horizon in **bars**) ──────────────────────────────────────────
# def add_label(f: pd.DataFrame, *, bar_ms:int, horizon_bars:int,
#               alpha: float = 0.20) -> pd.DataFrame:
#     """
#     3‑class label at t for ΔP over next `horizon_bars`:
#        +1 if ΔP >  max(alpha * dyn_tick, min_tick)
#         0 if |ΔP| ≤ threshold
#        -1 if ΔP < -max(alpha * dyn_tick, min_tick)

#     If you see only a single class after building, LOWER alpha (e.g. 0.10, 0.05, 0.005)
#     and/or filter to RTH. Then rebuild.
#     """
#     steps = int(horizon_bars)
#     ref   = f["micropx"]
#     tgt   = ref.shift(-steps) - ref

#     # robust local movement scale
#     dyn_tick = ref.diff().abs().rolling(100).median().bfill()

#     # min tick by price regime
#     min_tick = np.where(ref < 2.0, 0.0005, 0.01)  # 1 cent for typical US stocks

#     thr = np.maximum(alpha * dyn_tick, min_tick)
#     f = f.copy()
#     f["y"] = np.select([tgt > thr, tgt < -thr], [1, -1], 0).astype(np.int8)
#     return f.dropna()

# # ── NASDAQ pipeline ───────────────────────────────────────────────────────
# def process_nasdaq(bar_ms:int, horizon:int, snap_ms:int,
#                    stock: Optional[str], locate: Optional[int],
#                    alpha: float, rth_only: bool) -> tuple[Path,Path,Path]:
#     # resolve snapshot file
#     if locate is not None:
#         candidate = ITCH_DIR / f"mid_{locate}_{snap_ms}ms.parquet"
#         tag = str(locate)
#     elif stock:
#         candidate = ITCH_DIR / f"mid_{stock}_{snap_ms}ms.parquet"
#         tag = stock
#     else:
#         raise SystemExit("Select instrument with --stock or --locate (same as cleaning).")

#     if not candidate.exists():
#         raise FileNotFoundError(
#             f"Snapshot file not found: {candidate}\n"
#             f"Run clean_nasdaq_itch.py with the same --snap_ms and instrument."
#         )

#     df = pd.read_parquet(candidate)
#     if "timestamp" in df and not isinstance(df.index, pd.DatetimeIndex):
#         df = df.set_index(pd.to_datetime(df["timestamp"], utc=True)).drop(columns="timestamp")

#     # Optional: filter to regular trading hours in America/New_York
#     if rth_only:
#         df_ny = df.tz_convert("America/New_York")
#         df_ny = df_ny.between_time("09:30", "16:00", include_end=False)
#         df = df_ny.tz_convert("UTC")

#     # resample to bar grid
#     bars = _ffill_resample(df, bar_ms, snap_ms)
#     bars = bars[bars["ask_px"] > bars["bid_px"]]  # paranoia

#     # features + labels
#     feat = make_features(bars, bar_ms=bar_ms)
#     feat = add_label(feat, bar_ms=bar_ms, horizon_bars=horizon, alpha=alpha)

#     # split and write
#     n = len(feat)
#     tr, va, te = feat[:int(.7*n)], feat[int(.7*n):int(.85*n)], feat[int(.85*n):]
#     paths = [ITCH_DIR / f"nasdaq_{tag}_{bar_ms}ms_{p}.parquet" for p in ("train","val","test")]
#     for subset, p in zip((tr,va,te), paths):
#         p.parent.mkdir(parents=True, exist_ok=True)
#         subset.to_parquet(p, compression="snappy")

#     # Diagnostics
#     print(f"✔ NASDAQ_ITCH {tag} @ {bar_ms} ms | horizon {horizon} bars | alpha {alpha} | rows {n:,}")
#     for name, part in zip(("train","val","test"), (tr,va,te)):
#         vc = part["y"].value_counts().to_dict()
#         print(f"  {name:>5} label counts → {vc}")

#     return tuple(paths)

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--dataset", required=True, choices=["NASDAQ_ITCH"])
#     ap.add_argument("--bar_ms",  type=int, default=20)
#     ap.add_argument("--horizon", type=int, default=50, help="Lookahead in bars (e.g., 50 bars @20ms ≈1s)")
#     ap.add_argument("--snap_ms", type=int, default=20, help="Must match cleaning grid")
#     ap.add_argument("--stock")
#     ap.add_argument("--locate", type=int)
#     ap.add_argument("--alpha", type=float, default=0.20,
#                     help="Label threshold multiplier for dyn_tick (try 0.20, 0.10, 0.05, 0.005)")
#     ap.add_argument("--rth", action="store_true",
#                     help="Use only 09:30–16:00 America/New_York to avoid dead time")
#     args = ap.parse_args()

#     print(f"▶ Build {args.dataset} | instrument {args.stock or args.locate} "
#           f"@ {args.bar_ms} ms | horizon {args.horizon} bars | snap {args.snap_ms} ms | "
#           f"alpha={args.alpha} | rth={args.rth}")

#     train, val, test = process_nasdaq(
#         args.bar_ms, args.horizon, args.snap_ms, args.stock, args.locate, args.alpha, args.rth
#     )

#     reg_path = ITCH_DIR / "feature_registry.json"
#     registry = json.loads(reg_path.read_text()) if reg_path.exists() else {}
#     key = f"nasdaq_itch_{args.stock or args.locate}"
#     registry[key] = [str(train), str(val), str(test)]
#     reg_path.write_text(json.dumps(registry, indent=2))
#     print("✓ Registry updated:", reg_path)

# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# # Build bar‑level features & labels for one NASDAQ instrument.
# from __future__ import annotations
# import argparse, json, math, warnings, re
# from pathlib import Path
# from typing import Dict, Callable, Optional
# import numpy as np
# import pandas as pd
# import numba as nb
# from arch import arch_model

# warnings.filterwarnings("ignore", category=FutureWarning)

# ROOT     = Path(__file__).resolve().parents[1]
# ITCH_DIR = ROOT / "data" / "processed" / "NASDAQ_ITCH"
# ITCH_DIR.mkdir(parents=True, exist_ok=True)

# # ── helpers ───────────────────────────────────────────────────────────────
# @nb.njit(cache=True)
# def _wma(arr: np.ndarray, n: int) -> np.ndarray:
#     out = np.empty_like(arr)
#     w   = np.arange(1, n+1, dtype=arr.dtype); ws = w.sum()
#     for i in range(arr.shape[0]):
#         out[i] = np.nan if i < n-1 else (arr[i-n+1:i+1]*w).sum()/ws
#     return out

# def ema(s: pd.Series, n:int) -> pd.Series:  return s.ewm(span=n, adjust=False).mean()
# def dema(s: pd.Series, n:int) -> pd.Series: e1=ema(s,n); return 2*e1-ema(e1,n)
# def tema(s: pd.Series, n:int) -> pd.Series: e1=ema(s,n); e2=ema(e1,n); e3=ema(e2,n); return 3*e1-3*e2+e3
# def hma (s: pd.Series, n:int) -> pd.Series:
#     half, root = int(n/2), int(math.sqrt(n))
#     return pd.Series(_wma(2*_wma(s.values, half) - _wma(s.values, n), root), index=s.index)

# def kama(price: pd.Series, er_period=10, fast=2, slow=30) -> pd.Series:
#     change = price.diff(er_period).abs()
#     vol    = price.diff().abs().rolling(er_period).sum()
#     er     = (change / vol).fillna(0)
#     sc     = (er*(2/(fast+1)-2/(slow+1)) + 2/(slow+1))**2
#     out = pd.Series(index=price.index, dtype=float); out.iloc[0] = price.iloc[0]
#     for i in range(1, len(price)):
#         out.iloc[i] = out.iloc[i-1] + sc.iloc[i]*(price.iloc[i]-out.iloc[i-1])
#     return out

# def zscore(s: pd.Series, win=500) -> pd.Series:
#     mu = s.rolling(win).mean() if len(s)>=win else s.expanding().mean()
#     sd = s.rolling(win).std()  if len(s)>=win else s.expanding().std()
#     sd = sd.replace(0, 1e-9)
#     return (s - mu) / sd

# def _ffill_resample(df: pd.DataFrame, bar_ms:int, snap_ms:int) -> pd.DataFrame:
#     """Bring snapshots to exact bar grid with controlled ffill gap."""
#     limit = max(1, math.ceil(snap_ms / bar_ms))
#     def is_px(c:str) -> bool: return c.endswith("_px") or "_px_" in c or c in ("bid_px","ask_px")
#     def is_sz(c:str) -> bool: return c.endswith("_sz") or "_sz_" in c or c.endswith("sz1")
#     agg_map = {}
#     for c in df.columns:
#         if is_px(c):   agg_map[c] = "last"
#         elif is_sz(c): agg_map[c] = "mean"
#         else:          agg_map[c] = "last"
#     out = df.resample(f"{bar_ms}ms").agg(agg_map).ffill(limit=limit)
#     return out

# # ── new: time‑of‑day filter ───────────────────────────────────────────────
# def _filter_time_of_day(bars: pd.DataFrame,
#                         start: str = "09:00:00",
#                         end:   str = "17:00:00",
#                         tz:    str = "America/New_York") -> pd.DataFrame:
#     """
#     Keep only rows whose *local* clock (tz) falls inside [start, end).
#     bars index is UTC; we convert to tz and slice by local time of day.
#     """
#     if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
#         raise ValueError("bars index must be tz‑aware DateTimeIndex (UTC).")
#     local_idx = bars.index.tz_convert(tz)
#     t = local_idx.time
#     start_t = pd.to_datetime(start).time()
#     end_t   = pd.to_datetime(end).time()

#     if start_t <= end_t:
#         mask = (t >= start_t) & (t < end_t)
#     else:
#         # window crossing midnight (not your case, but safe)
#         mask = (t >= start_t) | (t < end_t)
#     out = bars[mask]
#     return out

# # ── features ───────────────────────────────────────────────────────────────
# def make_features(df: pd.DataFrame, *, bar_ms:int) -> pd.DataFrame:
#     need = {"mid_px","microprice","spread","imbalance","last_trade_px","cum_vol",
#             "bid_sz1","ask_sz1","ask_px","bid_px"}
#     miss = need - set(df.columns)
#     if miss:
#         raise KeyError(f"Cleaner must provide columns: {miss}")

#     # Drop crossed books just in case (ask<=bid)
#     df = df[df["ask_px"] > df["bid_px"]].copy()

#     f = pd.DataFrame(index=df.index)
#     f["mid_px"]    = df["mid_px"]
#     f["micropx"]   = df["microprice"]
#     f["spread"]    = df["spread"]
#     f["imbalance"] = df["imbalance"]
#     f["ret_mid"]   = f["mid_px"].pct_change(fill_method=None).fillna(0)
#     f["ret_micro"] = f["micropx"].pct_change(fill_method=None).fillna(0)
#     f["log_ret"]   = np.log(f["mid_px"]).diff().fillna(0)

#     # depth (L1..L5 sums & shapes)
#     bid_depth_cols = [c for c in df.columns if re.match(r"^bid_sz(_\d+)?$", c)]
#     ask_depth_cols = [c for c in df.columns if re.match(r"^ask_sz(_\d+)?$", c)]
#     bid_depth = df[bid_depth_cols].fillna(0).to_numpy()
#     ask_depth = df[ask_depth_cols].fillna(0).to_numpy()
#     f["depth_bid_L5"] = bid_depth.sum(1)
#     f["depth_ask_L5"] = ask_depth.sum(1)
#     f["depth_imb5"]   = (f["depth_bid_L5"] - f["depth_ask_L5"]) / \
#                         (f["depth_bid_L5"] + f["depth_ask_L5"] + 1e-9)
#     f["q_iratio"] = df["bid_sz1"] / (df["bid_sz1"] + df["ask_sz1"] + 1e-9)
#     f["depth_slope_bid"] = (df["bid_sz1"] - df.get("bid_sz_5", df["bid_sz1"])) / 4
#     f["depth_slope_ask"] = (df["ask_sz1"] - df.get("ask_sz_5", df["ask_sz1"])) / 4

#     # order‑flow
#     trade_px = df["last_trade_px"].replace(0, np.nan).ffill()
#     f["trade_dir"] = np.sign(trade_px - f["mid_px"]).fillna(0)
#     vol_deltas = df["cum_vol"].diff().fillna(0)
#     intensity_n = max(5, int(1000/bar_ms))
#     f["of_intensity"] = vol_deltas.rolling(intensity_n).mean().fillna(0)
#     f["trade_gap"]    = (trade_px - f["micropx"]).fillna(0)

#     # volatility proxies
#     atr_n = max(3, int(1000/bar_ms))
#     rng   = f["mid_px"].rolling(atr_n).max() - f["mid_px"].rolling(atr_n).min()
#     f["atr_fast"] = (rng / f["mid_px"]).fillna(method="bfill")

#     n_fast = max(4, int( 800 / bar_ms))
#     n_slow = max(8, int(2400 / bar_ms))
#     f["dema_f"] = dema(f["micropx"], n_fast)
#     f["dema_s"] = dema(f["micropx"], n_slow)
#     f["tema_f"] = tema(f["micropx"], n_fast)
#     f["hma16"]  = hma (f["micropx"], 16)
#     f["kama"]   = kama(f["micropx"])

#     n_trix = max(5, int(1500/bar_ms))
#     ema1 = ema(f["micropx"], n_trix); ema2 = ema(ema1, n_trix); ema3 = ema(ema2, n_trix)
#     f["trix"] = (ema3 / ema3.shift(1) - 1).fillna(0) * 100

#     # GARCH on 1‑second returns (optional, ffill to bar grid)
#     f["garch_vol"] = np.nan
#     r1s = f["micropx"].resample("1s").last().pct_change().dropna() * 100
#     if len(r1s) > 300:
#         vol = arch_model(r1s, p=1, q=1).fit(disp=False).conditional_volatility
#         f["garch_vol"] = vol.reindex(f.index, method="ffill")

#     # z‑scores
#     for col in ["spread","imbalance","depth_imb5","of_intensity"]:
#         f[col+"_z"] = zscore(f[col])

#     return f.dropna()

# # ── labels (horizon in **bars**) ──────────────────────────────────────────
# def add_label(f: pd.DataFrame, *, bar_ms:int, horizon_bars:int,
#               alpha: float = 0.20) -> pd.DataFrame:
#     """
#     3‑class label at t for ΔP over next `horizon_bars`:
#        +1 if ΔP >  max(alpha * dyn_tick, min_tick)
#         0 if |ΔP| ≤ threshold
#        -1 if ΔP < -max(alpha * dyn_tick, min_tick)

#     If you see only a single class after building, LOWER alpha (e.g. 0.10 or 0.05)
#     and rebuild features for this instrument/day/horizon.
#     """
#     steps = int(horizon_bars)
#     ref   = f["micropx"]
#     tgt   = ref.shift(-steps) - ref

#     # robust local movement scale
#     dyn_tick = ref.diff().abs().rolling(100).median().bfill()

#     # min tick by price regime
#     min_tick = np.where(ref < 2.0, 0.0005, 0.001)

#     thr = np.maximum(alpha * dyn_tick, min_tick)
#     f = f.copy()
#     f["y"] = np.select([tgt > thr, tgt < -thr], [1, -1], 0).astype(np.int8)
#     return f.dropna()

# # ── NASDAQ pipeline ───────────────────────────────────────────────────────
# def process_nasdaq(bar_ms:int, horizon:int, snap_ms:int,
#                    stock: Optional[str], locate: Optional[int],
#                    alpha: float,
#                    tod_start: str, tod_end: str, tz: str) -> tuple[Path,Path,Path]:

#     if locate is not None:
#         candidate = ITCH_DIR / f"mid_{locate}_{snap_ms}ms.parquet"
#         tag = str(locate)
#     elif stock:
#         candidate = ITCH_DIR / f"mid_{stock}_{snap_ms}ms.parquet"
#         tag = stock
#     else:
#         raise SystemExit("Select instrument with --stock or --locate (same as cleaning).")

#     if not candidate.exists():
#         raise FileNotFoundError(f"Snapshot file not found: {candidate}\n"
#                                 f"Run clean_nasdaq_itch.py with the same --snap_ms and instrument.")

#     df = pd.read_parquet(candidate)
#     if "timestamp" in df and not isinstance(df.index, pd.DatetimeIndex):
#         df = df.set_index(pd.to_datetime(df["timestamp"], utc=True)).drop(columns="timestamp")

#     # resample to bar grid first
#     bars = _ffill_resample(df, bar_ms, snap_ms)
#     bars = bars[bars["ask_px"] > bars["bid_px"]]  # paranoia

#     # NEW: time-of-day window in local market time
#     before = len(bars)
#     bars = _filter_time_of_day(bars, start=tod_start, end=tod_end, tz=tz)
#     after = len(bars)
#     print(f"⏱  kept {after:,}/{before:,} bars in [{tod_start},{tod_end}) {tz}")

#     feat = make_features(bars, bar_ms=bar_ms)
#     feat = add_label(feat, bar_ms=bar_ms, horizon_bars=horizon, alpha=alpha)

#     n = len(feat)
#     tr, va, te = feat[:int(.7*n)], feat[int(.7*n):int(.85*n)], feat[int(.85*n):]
#     paths = [ITCH_DIR / f"nasdaq_{tag}_{bar_ms}ms_{p}.parquet" for p in ("train","val","test")]
#     for subset, p in zip((tr,va,te), paths):
#         p.parent.mkdir(parents=True, exist_ok=True)
#         subset.to_parquet(p, compression="snappy")

#     # diagnostics
#     def _counts(y): 
#         u,c = np.unique(y, return_counts=True); return dict(zip(u.tolist(), c.tolist()))
#     print(f"✔ NASDAQ_ITCH {tag} @ {bar_ms} ms | horizon {horizon} bars | alpha {alpha} | rows {n:,}")
#     print(f"  train label counts → {_counts(tr['y'].to_numpy())}")
#     print(f"    val label counts → {_counts(va['y'].to_numpy())}")
#     print(f"   test label counts → {_counts(te['y'].to_numpy())}")

#     return tuple(paths)

# _PIPE: Dict[str, Callable[...,object]] = {"NASDAQ_ITCH": process_nasdaq}

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--dataset", required=True, choices=_PIPE.keys())
#     ap.add_argument("--bar_ms",  type=int, default=20)
#     ap.add_argument("--horizon", type=int, default=50, help="Lookahead in bars (e.g., 50 bars @20ms ≈1s)")
#     ap.add_argument("--snap_ms", type=int, default=20, help="Must match cleaning grid")
#     ap.add_argument("--stock")
#     ap.add_argument("--locate", type=int)
#     ap.add_argument("--alpha", type=float, default=0.20,
#                     help="Label threshold multiplier (lower to get fewer zeros)")
#     # NEW: time-of-day window (local market time)
#     ap.add_argument("--tod_start", default="09:00:00", help="Local start HH:MM[:SS]")
#     ap.add_argument("--tod_end",   default="17:00:00", help="Local end HH:MM[:SS] (exclusive)")
#     ap.add_argument("--tz",        default="America/New_York", help="Market timezone")

#     args = ap.parse_args()

#     print(f"▶ Build {args.dataset} | instrument {args.stock or args.locate} "
#           f"@ {args.bar_ms} ms | horizon {args.horizon} bars | snap {args.snap_ms} ms | "
#           f"alpha={args.alpha} | {args.tod_start}-{args.tod_end} {args.tz}")

#     result = _PIPE[args.dataset](args.bar_ms, args.horizon, args.snap_ms,
#                                  args.stock, args.locate,
#                                  args.alpha, args.tod_start, args.tod_end, args.tz)

#     reg_path = ITCH_DIR / "feature_registry.json"
#     registry = json.loads(reg_path.read_text()) if reg_path.exists() else {}
#     key = f"nasdaq_itch_{args.stock or args.locate}"
#     registry[key] = [str(p) for p in result]
#     reg_path.write_text(json.dumps(registry, indent=2))
#     print("✓ Registry updated:", reg_path)

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# Build bar‑level features & labels for one NASDAQ instrument.
from __future__ import annotations
import argparse, json, math, warnings, re
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple
import numpy as np
import pandas as pd
import numba as nb
from arch import arch_model

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT     = Path(__file__).resolve().parents[1]
ITCH_DIR = ROOT / "data" / "processed" / "NASDAQ_ITCH"
ITCH_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────
@nb.njit(cache=True)
def _wma(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(arr)
    w   = np.arange(1, n+1, dtype=arr.dtype); ws = w.sum()
    for i in range(arr.shape[0]):
        out[i] = np.nan if i < n-1 else (arr[i-n+1:i+1]*w).sum()/ws
    return out

def ema(s: pd.Series, n:int) -> pd.Series:  return s.ewm(span=n, adjust=False).mean()
def dema(s: pd.Series, n:int) -> pd.Series: e1=ema(s,n); return 2*e1-ema(e1,n)
def tema(s: pd.Series, n:int) -> pd.Series: e1=ema(s,n); e2=ema(e1,n); e3=ema(e2,n); return 3*e1-3*e2+e3
def hma (s: pd.Series, n:int) -> pd.Series:
    half, root = int(n/2), int(math.sqrt(n))
    return pd.Series(_wma(2*_wma(s.values, half) - _wma(s.values, n), root), index=s.index)

def kama(price: pd.Series, er_period=10, fast=2, slow=30) -> pd.Series:
    if len(price) == 0:
        return price.copy()
    change = price.diff(er_period).abs()
    vol    = price.diff().abs().rolling(er_period).sum()
    er     = (change / vol).fillna(0)
    sc     = (er*(2/(fast+1)-2/(slow+1)) + 2/(slow+1))**2
    out = pd.Series(index=price.index, dtype=float)
    out.iloc[0] = price.iloc[0]
    for i in range(1, len(price)):
        out.iloc[i] = out.iloc[i-1] + sc.iloc[i]*(price.iloc[i]-out.iloc[i-1])
    return out

def zscore(s: pd.Series, win=500) -> pd.Series:
    if len(s) == 0:
        return s.copy()
    mu = s.rolling(win).mean() if len(s)>=win else s.expanding().mean()
    sd = s.rolling(win).std()  if len(s)>=win else s.expanding().std()
    sd = sd.replace(0, 1e-9)
    return (s - mu) / sd

def _ffill_resample(df: pd.DataFrame, bar_ms:int, snap_ms:int) -> pd.DataFrame:
    """Bring snapshots to exact bar grid with controlled ffill gap."""
    limit = max(1, math.ceil(snap_ms / bar_ms))
    def is_px(c:str) -> bool: return c.endswith("_px") or "_px_" in c or c in ("bid_px","ask_px")
    def is_sz(c:str) -> bool: return c.endswith("_sz") or "_sz_" in c or c.endswith("sz1")
    agg_map = {}
    for c in df.columns:
        if is_px(c):   agg_map[c] = "last"
        elif is_sz(c): agg_map[c] = "mean"
        else:          agg_map[c] = "last"
    out = df.resample(f"{bar_ms}ms").agg(agg_map).ffill(limit=limit)
    return out

# ── features ───────────────────────────────────────────────────────────────
def make_features(df: pd.DataFrame, *, bar_ms:int) -> pd.DataFrame:
    need = {"mid_px","microprice","spread","imbalance","last_trade_px","cum_vol",
            "bid_sz1","ask_sz1","ask_px","bid_px"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"Cleaner must provide columns: {miss}")

    # Drop crossed books just in case (ask<=bid)
    df = df[df["ask_px"] > df["bid_px"]].copy()

    f = pd.DataFrame(index=df.index)
    f["mid_px"]    = df["mid_px"]
    f["micropx"]   = df["microprice"]
    f["spread"]    = df["spread"]
    f["imbalance"] = df["imbalance"]
    f["ret_mid"]   = f["mid_px"].pct_change(fill_method=None).fillna(0)
    f["ret_micro"] = f["micropx"].pct_change(fill_method=None).fillna(0)
    f["log_ret"]   = np.log(f["mid_px"]).diff().fillna(0)

    # depth (L1..L5 sums & shapes)
    bid_depth_cols = [c for c in df.columns if re.match(r"^bid_sz(_\d+)?$", c)]
    ask_depth_cols = [c for c in df.columns if re.match(r"^ask_sz(_\d+)?$", c)]
    bid_depth = df[bid_depth_cols].fillna(0).to_numpy()
    ask_depth = df[ask_depth_cols].fillna(0).to_numpy()
    f["depth_bid_L5"] = bid_depth.sum(1)
    f["depth_ask_L5"] = ask_depth.sum(1)
    f["depth_imb5"]   = (f["depth_bid_L5"] - f["depth_ask_L5"]) / \
                        (f["depth_bid_L5"] + f["depth_ask_L5"] + 1e-9)
    f["q_iratio"] = df["bid_sz1"] / (df["bid_sz1"] + df["ask_sz1"] + 1e-9)
    f["depth_slope_bid"] = (df["bid_sz1"] - df.get("bid_sz_5", df["bid_sz1"])) / 4
    f["depth_slope_ask"] = (df["ask_sz1"] - df.get("ask_sz_5", df["ask_sz1"])) / 4

    # order‑flow
    trade_px = df["last_trade_px"].replace(0, np.nan).ffill()
    f["trade_dir"] = np.sign(trade_px - f["mid_px"]).fillna(0)
    vol_deltas = df["cum_vol"].diff().fillna(0)
    intensity_n = max(5, int(1000/bar_ms))
    f["of_intensity"] = vol_deltas.rolling(intensity_n).mean().fillna(0)
    f["trade_gap"]    = (trade_px - f["micropx"]).fillna(0)

    # volatility proxies
    atr_n = max(3, int(1000/bar_ms))
    rng   = f["mid_px"].rolling(atr_n).max() - f["mid_px"].rolling(atr_n).min()
    f["atr_fast"] = (rng / f["mid_px"]).fillna(method="bfill")

    n_fast = max(4, int( 800 / bar_ms))
    n_slow = max(8, int(2400 / bar_ms))
    f["dema_f"] = dema(f["micropx"], n_fast)
    f["dema_s"] = dema(f["micropx"], n_slow)
    f["tema_f"] = tema(f["micropx"], n_fast)
    f["hma16"]  = hma (f["micropx"], 16)
    f["kama"]   = kama(f["micropx"])

    n_trix = max(5, int(1500/bar_ms))
    ema1 = ema(f["micropx"], n_trix); ema2 = ema(ema1, n_trix); ema3 = ema(ema2, n_trix)
    f["trix"] = (ema3 / ema3.shift(1) - 1).fillna(0) * 100

    # GARCH on 1‑second returns (optional, ffill to bar grid)
    f["garch_vol"] = np.nan
    r1s = f["micropx"].resample("1s").last().pct_change().dropna() * 100
    if len(r1s) > 300:
        vol = arch_model(r1s, p=1, q=1).fit(disp=False).conditional_volatility
        f["garch_vol"] = vol.reindex(f.index, method="ffill")

    # z‑scores
    for col in ["spread","imbalance","depth_imb5","of_intensity"]:
        f[col+"_z"] = zscore(f[col])

    return f.dropna()

# ── labels (horizon in **bars**) ──────────────────────────────────────────
def add_label(f: pd.DataFrame, *, bar_ms:int, horizon_bars:int,
              alpha: float = 0.20) -> pd.DataFrame:
    """
    3‑class label at t for ΔP over next `horizon_bars`:
       +1 if ΔP >  max(alpha * dyn_tick, min_tick)
        0 if |ΔP| ≤ threshold
       -1 if ΔP < -max(alpha * dyn_tick, min_tick)
    dyn_tick = rolling median |ΔP(1 bar)|.
    """
    steps = int(horizon_bars)
    ref   = f["micropx"]
    tgt   = ref.shift(-steps) - ref

    # robust local movement scale
    dyn_tick = ref.diff().abs().rolling(100).median().bfill()

    # min tick by price regime
    min_tick = np.where(ref < 2.0, 0.0005, 0.001)

    thr = np.maximum(alpha * dyn_tick, min_tick)
    f = f.copy()
    f["y"] = np.select([tgt > thr, tgt < -thr], [1, -1], 0).astype(np.int8)
    return f.dropna()

# ── NASDAQ pipeline ───────────────────────────────────────────────────────
def _load_mid(snap_ms:int, stock: Optional[str], locate: Optional[int]) -> Tuple[pd.DataFrame, str]:
    if locate is not None:
        candidate = ITCH_DIR / f"mid_{locate}_{snap_ms}ms.parquet"
        tag = str(locate)
    elif stock:
        candidate = ITCH_DIR / f"mid_{stock}_{snap_ms}ms.parquet"
        tag = stock
    else:
        raise SystemExit("Select instrument with --stock or --locate (same as cleaning).")

    if not candidate.exists():
        raise FileNotFoundError(f"Snapshot file not found: {candidate}\n"
                                f"Run clean_nasdaq_itch.py with the same --snap_ms and instrument.")
    df = pd.read_parquet(candidate)
    if "timestamp" in df and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df["timestamp"], utc=True)).drop(columns="timestamp")
    return df, tag

def process_nasdaq(bar_ms:int, horizon:int, snap_ms:int,
                   stock: Optional[str], locate: Optional[int],
                   alpha: float, tail_frac: float) -> tuple[Path,Path,Path]:
    df, tag = _load_mid(snap_ms, stock, locate)

    bars = _ffill_resample(df, bar_ms, snap_ms)
    bars = bars[bars["ask_px"] > bars["bid_px"]]  # paranoia

    feat = make_features(bars, bar_ms=bar_ms)
    feat = add_label(feat, bar_ms=bar_ms, horizon_bars=horizon, alpha=alpha)  # horizon in BARS

    # drop the last tail_frac (e.g., 0.15 = last 15% of the day)
    n_all = len(feat)
    keep_n = int((1.0 - max(0.0, min(0.999, tail_frac))) * n_all)
    feat = feat.iloc[:keep_n].copy()
    print(f"⏱  kept {len(feat):,}/{n_all:,} bars in the first {int((1-tail_frac)*100)}% of the day")

    # chronological split inside the kept window: 70/15/15
    n = len(feat)
    i1, i2 = int(.70*n), int(.85*n)
    tr, va, te = feat.iloc[:i1], feat.iloc[i1:i2], feat.iloc[i2:]

    def _counts(name, y):
        u, c = np.unique(y, return_counts=True)
        print(f"  {name:5s} label counts → " + str({int(k): int(v) for k, v in zip(u, c)}))

    _counts("train", tr["y"].to_numpy())
    _counts("val",   va["y"].to_numpy())
    _counts("test",  te["y"].to_numpy())

    paths = [ITCH_DIR / f"nasdaq_{tag}_{bar_ms}ms_{p}.parquet" for p in ("train","val","test")]
    for subset, p in zip((tr,va,te), paths):
        p.parent.mkdir(parents=True, exist_ok=True)
        subset.to_parquet(p, compression="snappy")
    print(f"✔ NASDAQ_ITCH {tag} @ {bar_ms} ms | horizon {horizon} bars | alpha {alpha} | rows {n:,}")
    return tuple(paths)

_PIPE: Dict[str, Callable[...,object]] = {"NASDAQ_ITCH": process_nasdaq}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=_PIPE.keys())
    ap.add_argument("--bar_ms",  type=int, default=20)
    ap.add_argument("--horizon", type=int, default=50, help="Lookahead in bars (e.g., 50 bars @20ms ≈1s)")
    ap.add_argument("--snap_ms", type=int, default=20, help="Must match cleaning grid")
    ap.add_argument("--stock")
    ap.add_argument("--locate", type=int)
    ap.add_argument("--alpha", type=float, default=0.20, help="Move threshold scale; lower ⇒ fewer 0s")
    ap.add_argument("--tail_frac", type=float, default=0.15, help="Drop last fraction of the day")
    args = ap.parse_args()

    print(f"▶ Build {args.dataset} | instrument {args.stock or args.locate} "
          f"@ {args.bar_ms} ms | horizon {args.horizon} bars | snap {args.snap_ms} ms | "
          f"alpha={args.alpha} | drop_last={int(args.tail_frac*100)}%")

    result = _PIPE[args.dataset](args.bar_ms, args.horizon, args.snap_ms,
                                 args.stock, args.locate, args.alpha, args.tail_frac)

    reg_path = ITCH_DIR / "feature_registry.json"
    registry = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    key = f"nasdaq_itch_{args.stock or args.locate}"
    registry[key] = [str(p) for p in result]
    reg_path.write_text(json.dumps(registry, indent=2))
    print("✓ Registry updated:", reg_path)

if __name__ == "__main__":
    main()