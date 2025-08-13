#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# src/data/clean_nasdaq_itch.py – HFT‑grade ITCH replay → instrument snapshots
#
# Writes:  src/data/processed/NASDAQ_ITCH/mid_{SYMBOL|locate}_{snap_ms}ms.parquet
# Sidecar: src/data/processed/NASDAQ_ITCH/mid_{...}_{snap_ms}ms_meta.json
#
# Snapshots include L1–L5 px/sz, midpoint, micro‑price, spread, imbalance,
# last trade px/sz, cumulative executed volume, and (optionally) shape features:
#   tot_bid5, tot_ask5, qimb5, b_slope, a_slope, bq_share, aq_share
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import heapq
import itertools
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sortedcontainers import SortedDict
from tqdm import tqdm

# ─── paths ───────────────────────────────────────────────────────────────────
HERE   = os.path.dirname(__file__)
RAW_H5 = os.path.join(HERE, "raw",      "NASDAQ_ITCH", "itch.h5")
PROC   = os.path.join(HERE, "processed", "NASDAQ_ITCH")
os.makedirs(PROC, exist_ok=True)

# ITCH table codes
CORE = {"A", "C", "D", "E", "F", "U", "X"}   # core order book messages
MAPS = {"R"}                                 # symbol directory / stock_locate map

PRICE_COLS = {
    "price", "execution_price",
    "auction_ref_price", "upper_auction_collar_price",
    "lower_auction_collar_price", "cross_price"
}

NEEDED_COLS = {
    "timestamp", "stock_locate", "order_reference_number", "price", "shares",
    "buy_sell_indicator", "executed_shares", "cancelled_shares",
    "original_order_reference_number", "new_order_reference_number"
}

SNAP_INTERVAL_NS: int  # set in main()


# ═════════════════════════════ utilities ════════════════════════════════════
def _ns_to_utc(ns: np.ndarray, day: str) -> pd.DatetimeIndex:
    base = pd.Timestamp(day).tz_localize("America/New_York")
    return pd.DatetimeIndex(base + pd.to_timedelta(ns, unit="ns")).tz_convert("UTC")


def _detect_ts(cols) -> str:
    for c in cols:
        if "time" in c.lower() or "stamp" in c.lower():
            return c
    raise ValueError("timestamp column not found")


# ═════════════════════════ HDF5 → core Parquet ══════════════════════════════
def _clean_one(store: pd.HDFStore, code: str, day: str, overwrite: bool):
    out = f"{PROC}/{code}.parquet"
    if os.path.exists(out) and not overwrite:
        print(f"↻  {code}.parquet exists – skip")
        return

    df = store[f"/{code}"]
    ts = _detect_ts(df.columns)

    # timestamps → UTC
    if pd.api.types.is_timedelta64_dtype(df[ts]):
        df[ts] = _ns_to_utc(df[ts].astype("int64"), day)
    elif np.issubdtype(df[ts].dtype, np.integer):
        df[ts] = _ns_to_utc(df[ts], day)
    else:
        df[ts] = pd.to_datetime(df[ts], utc=True, errors="coerce")

    df = df.dropna(subset=[ts]).sort_values(ts).reset_index(drop=True)

    # scale prices (ITCH integer 1/10000 USD → float)
    for c in PRICE_COLS & set(df.columns):
        if np.issubdtype(df[c].dtype, np.integer):
            df[c] = df[c].astype("float64") / 10_000.0

    # normalize side flag
    if "buy_sell_indicator" in df.columns:
        df["buy_sell_indicator"] = df["buy_sell_indicator"].map(
            {1: "B", -1: "S", b"B": "B", b"S": "S", "B": "B", "S": "S"}
        )

    df.to_parquet(out, compression="snappy")
    print(f"✔  {code}.parquet ({len(df):,} rows)")


def initial_clean(day: str, overwrite: bool):
    if not os.path.exists(RAW_H5):
        raise FileNotFoundError(RAW_H5)
    with pd.HDFStore(RAW_H5) as store:
        for key in store.keys():
            code = key.lstrip("/")
            if code in CORE | MAPS:
                _clean_one(store, code, day, overwrite)


def read_symbol_map() -> dict[int, str]:
    path = f"{PROC}/R.parquet"
    if not os.path.exists(path):
        return {}
    df = pd.read_parquet(path)
    if "stock_locate" in df and "stock" in df:
        df = df.drop_duplicates("stock_locate")[["stock_locate", "stock"]]
        return dict(zip(df["stock_locate"].astype(int), df["stock"].astype(str)))
    return {}


# ═════════════════════ iterator over parquet batches ════════════════════════
def event_stream(code: str, batch: int = 1_000_000) -> Iterable[Tuple[pd.Timestamp, str, dict]]:
    path = f"{PROC}/{code}.parquet"
    if not os.path.exists(path):
        return
    cols = list(NEEDED_COLS & set(pq.read_schema(path).names))
    scan = ds.dataset(path).scanner(columns=cols, batch_size=batch, use_threads=True)
    for rb in scan.to_batches():
        d  = rb.to_pydict()
        ts = d["timestamp"]
        for i in range(len(ts)):
            row = {k: d[k][i] for k in d if k != "timestamp"}
            yield ts[i], code, row


# ═════════════════════ main replay → snapshot logic ═════════════════════════
def build_midpoint(
    day: str,
    snap_ms: int,
    target_locate: Optional[int],
    *,
    drop_crossed: bool = True,
    shape_feats: bool = False,
) -> str:
    total = sum(
        pq.ParquetFile(f"{PROC}/{c}.parquet").metadata.num_rows
        for c in CORE
        if os.path.exists(f"{PROC}/{c}.parquet")
    )
    if total == 0:
        raise RuntimeError("No core messages; run initial_clean first.")

    # streams
    ev_heap, streams, COUNTER = [], {}, itertools.count()
    for c in CORE:
        g = event_stream(c)
        streams[c] = g
        try:
            t, _, row = next(g)
            heapq.heappush(ev_heap, (t, next(COUNTER), c, row))
        except StopIteration:
            pass

    # per‑instrument state
    bids, asks = {}, {}
    depth_bid  = SortedDict(lambda x: -x)  # price → qty (desc)
    depth_ask  = SortedDict()              # price → qty (asc)
    side_cache: Dict[int, str] = {}

    last_trade_px = 0.0
    last_trade_sz = 0
    cum_vol       = 0

    # buffer + column schema
    FLUSH, buf = 100_000, []
    cols = [
        "timestamp",
        "bid_px", "bid_sz1", "ask_px", "ask_sz1",
        "bid_px_2", "bid_sz_2", "bid_px_3", "bid_sz_3", "bid_px_4", "bid_sz_4", "bid_px_5", "bid_sz_5",
        "ask_px_2", "ask_sz_2", "ask_px_3", "ask_sz_3", "ask_px_4", "ask_sz_4", "ask_px_5", "ask_sz_5",
        "mid_px", "microprice", "spread", "imbalance",
        "last_trade_px", "last_trade_sz", "cum_vol",
    ]
    if shape_feats:
        cols += ["tot_bid5", "tot_ask5", "qimb5", "b_slope", "a_slope", "bq_share", "aq_share"]

    # progress + QA counters
    pbar = tqdm(total=total, unit="ev", desc="Replay")
    writer = None
    crossed_kept = 0
    crossed_dropped = 0

    # first snapshot boundary
    next_snap_ns = None
    if ev_heap:
        first_ns = int(pd.Timestamp(ev_heap[0][0]).value)
        next_snap_ns = ((first_ns // SNAP_INTERVAL_NS) + 1) * SNAP_INTERVAL_NS

    # symbol map for naming
    locate_to_sym = read_symbol_map()
    sym = locate_to_sym.get(target_locate, None)
    out_name = f"mid_{sym or target_locate}_{snap_ms}ms.parquet"
    out_fp   = os.path.join(PROC, out_name)

    # slope helper
    xs = np.arange(1, 6, dtype=float)
    xm = xs.mean()
    denom = float(((xs - xm) ** 2).sum() + 1e-12)

    def _slope(level_sizes: list[float]) -> float:
        y = np.asarray(level_sizes, dtype=float)
        ym = y.mean()
        return float(((xs - xm) * (y - ym)).sum() / denom)

    # ─── event loop ─────────────────────────────────────────────────────────
    while ev_heap:
        t, _, code, r = heapq.heappop(ev_heap)
        ts_ns = int(pd.Timestamp(t).value)

        loc = r.get("stock_locate")
        if loc is None:
            pass
        elif target_locate is not None and int(loc) != int(target_locate):
            pass
        else:
            # unpack
            oid  = r.get("order_reference_number")
            side = r.get("buy_sell_indicator")
            px   = r.get("price") if r.get("price") is not None else r.get("execution_price")
            qty  = r.get("shares")
            exe  = r.get("executed_shares") or 0
            canc = r.get("cancelled_shares") or 0
            old  = r.get("original_order_reference_number")
            new  = r.get("new_order_reference_number")

            if side in ("B", "S") and oid:
                side_cache[int(oid)] = side

            # --- book updates ---------------------------------------------
            if code in ("A", "F") and oid and px is not None and qty:
                s = "B" if side == "B" else "S"
                book, depth = (bids, depth_bid) if s == "B" else (asks, depth_ask)
                book[oid]   = (px, int(qty))
                depth[px]   = depth.get(px, 0) + int(qty)

            elif code in ("E", "C") and oid:
                # execution reduces outstanding qty; capture last trade
                book = bids if oid in bids else asks
                if oid in book:
                    p, rem  = book[oid]
                    delta   = int(exe or canc)
                    new_rem = max(rem - delta, 0)
                    if new_rem == 0:
                        book.pop(oid)
                    else:
                        book[oid] = (p, new_rem)
                    depth = depth_bid if book is bids else depth_ask
                    depth[p] = max(depth.get(p, 0) - delta, 0)
                    if depth[p] == 0:
                        depth.pop(p, None)
                    if exe:
                        last_trade_px, last_trade_sz = p, delta
                        cum_vol += delta

            elif code == "X" and oid:
                book = bids if oid in bids else asks
                if oid in book:
                    p, rem  = book[oid]
                    delta   = int(canc)
                    new_rem = max(rem - delta, 0)
                    if new_rem == 0:
                        book.pop(oid)
                    else:
                        book[oid] = (p, new_rem)
                    depth = depth_bid if book is bids else depth_ask
                    depth[p] = max(depth.get(p, 0) - delta, 0)
                    if depth[p] == 0:
                        depth.pop(p, None)

            elif code == "D" and oid:
                if oid in bids:
                    p, rem = bids.pop(oid)
                    depth_bid[p] = max(depth_bid.get(p, 0) - rem, 0)
                    if depth_bid[p] == 0:
                        depth_bid.pop(p, None)
                elif oid in asks:
                    p, rem = asks.pop(oid)
                    depth_ask[p] = max(depth_ask.get(p, 0) - rem, 0)
                    if depth_ask[p] == 0:
                        depth_ask.pop(p, None)

            elif code == "U" and old and new and px is not None:
                s = side_cache.get(int(old))
                if s is not None:
                    old_book, old_depth = (bids, depth_bid) if s == "B" else (asks, depth_ask)
                    if old in old_book:
                        p_old, rem_old = old_book.pop(old)
                        old_depth[p_old] = max(old_depth.get(p_old, 0) - rem_old, 0)
                        if old_depth[p_old] == 0:
                            old_depth.pop(p_old, None)
                    side_cache[int(new)] = s
                    new_book, new_depth = (bids, depth_bid) if s == "B" else (asks, depth_ask)
                    if qty:
                        new_book[new] = (px, int(qty))
                        new_depth[px] = new_depth.get(px, 0) + int(qty)

            # --- emit snapshots ------------------------------------------
            while next_snap_ns is not None and ts_ns >= next_snap_ns:
                if depth_bid and depth_ask:
                    bid_px, bid_sz = next(iter(depth_bid.items()))
                    ask_px, ask_sz = next(iter(depth_ask.items()))
                    is_crossed = not (ask_px > bid_px)
                    if is_crossed:
                        if drop_crossed:
                            crossed_dropped += 1
                            next_snap_ns += SNAP_INTERVAL_NS
                            continue
                        else:
                            crossed_kept += 1

                    # collect L2–L5 and interleave (px2, sz2, px3, sz3, …)
                    b_levels = list(itertools.islice(depth_bid.items(), 1, 5))
                    a_levels = list(itertools.islice(depth_ask.items(), 1, 5))
                    while len(b_levels) < 4:
                        b_levels.append((np.nan, 0))
                    while len(a_levels) < 4:
                        a_levels.append((np.nan, 0))
                    b_flat = list(itertools.chain.from_iterable((p, q) for p, q in b_levels))
                    a_flat = list(itertools.chain.from_iterable((p, q) for p, q in a_levels))

                    mid = (bid_px + ask_px) / 2.0
                    mpr = (bid_px * ask_sz + ask_px * bid_sz) / (bid_sz + ask_sz + 1e-9)
                    spr = (ask_px - bid_px)
                    imb = (bid_sz - ask_sz) / (bid_sz + ask_sz + 1e-9)

                    extra = ()
                    if shape_feats:
                        b_sizes = [bid_sz] + [q for _, q in b_levels]
                        a_sizes = [ask_sz] + [q for _, q in a_levels]
                        tot_bid5 = int(np.nansum(b_sizes))
                        tot_ask5 = int(np.nansum(a_sizes))
                        qimb5    = (tot_bid5 - tot_ask5) / (tot_bid5 + tot_ask5 + 1e-9)
                        b_slope  = _slope(b_sizes)
                        a_slope  = _slope(a_sizes)
                        bq_share = float(bid_sz) / (tot_bid5 + 1e-9)
                        aq_share = float(ask_sz) / (tot_ask5 + 1e-9)
                        extra = (tot_bid5, tot_ask5, qimb5, b_slope, a_slope, bq_share, aq_share)

                    buf.append(
                        (
                            pd.Timestamp(next_snap_ns, unit="ns", tz="UTC"),
                            bid_px, bid_sz, ask_px, ask_sz,
                            *b_flat, *a_flat,
                            mid, mpr, spr, imb,
                            last_trade_px, last_trade_sz, cum_vol,
                            *extra,
                        )
                    )

                next_snap_ns += SNAP_INTERVAL_NS
                if len(buf) >= FLUSH:
                    writer = _flush(buf, cols, out_fp, snap_ms, drop_crossed, shape_feats, writer)

        # pull next from same stream
        try:
            nxt = next(streams[code])
            heapq.heappush(ev_heap, (nxt[0], next(COUNTER), nxt[1], nxt[2]))
        except StopIteration:
            pass

        pbar.update(1)

    pbar.close()
    if buf:
        writer = _flush(buf, cols, out_fp, snap_ms, drop_crossed, shape_feats, writer)
    if writer:
        writer.close()

    # Sidecar meta JSON for auditability (counts + params)
    pf = pq.ParquetFile(out_fp)
    meta_json = {
        "symbol": sym,
        "locate": int(target_locate) if target_locate is not None else None,
        "trade_day": day,
        "snap_ms": int(snap_ms),
        "drop_crossed": bool(drop_crossed),
        "shape_feats": bool(shape_feats),
        "crossed_kept": int(crossed_kept),
        "crossed_dropped": int(crossed_dropped),
        "rows_written": int(pf.metadata.num_rows if pf.metadata else 0),
        "clean_commit": subprocess.getoutput("git rev-parse HEAD")[:8],
    }
    sidecar = out_fp.replace(".parquet", "_meta.json")
    with open(sidecar, "w") as f:
        json.dump(meta_json, f, indent=2)
    print(f"✔ {os.path.basename(out_fp)} written → {out_fp}")
    print(f"↳ crossed kept={crossed_kept:,}, dropped={crossed_dropped:,} → {sidecar}")

    return out_fp


# ─── helper to flush ---------------------------------------------------------
def _flush(
    buf: list,
    cols: list,
    path: str,
    snap_ms: int,
    drop_crossed: bool,
    shape_feats: bool,
    writer: Optional[pq.ParquetWriter],
):
    table = pa.Table.from_pandas(pd.DataFrame(buf, columns=cols), preserve_index=False)
    buf.clear()
    if writer is None:
        meta_bytes = {
            b"snap_ms":       str(snap_ms).encode(),
            b"drop_crossed":  b"1" if drop_crossed else b"0",
            b"shape_feats":   b"1" if shape_feats else b"0",
            b"clean_commit":  subprocess.getoutput("git rev-parse HEAD")[:8].encode(),
        }
        schema_with_meta = table.schema.with_metadata(meta_bytes)
        writer = pq.ParquetWriter(path, schema_with_meta, compression="snappy")
    writer.write_table(table)
    return writer


# ═════════════════════════ CLI ═══════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap_ms", type=int, default=20, help="Snapshot grid (ms)")
    ap.add_argument("--stock",   type=str, help="Ticker symbol to extract (uses R table)")
    ap.add_argument("--locate",  type=int, help="Stock‑locate (overrides --stock)")
    ap.add_argument("--overwrite", action="store_true", help="Re‑run core clean")
    ap.add_argument("--keep_crossed", action="store_true",
                    help="Keep snapshots with ask<=bid (default: drop)")
    ap.add_argument("--shape_feats", action="store_true",
                    help="Compute extra L2–L5 shape features (totals, slopes, queue shares)")
    args = ap.parse_args()

    global SNAP_INTERVAL_NS
    SNAP_INTERVAL_NS = args.snap_ms * 1_000_000

    # infer trade day from RAW_H5 path if present
    m = re.search(r"\d{4}-\d{2}-\d{2}", RAW_H5)
    trade_day = m.group(0) if m else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"Trade day {trade_day} | snap grid {args.snap_ms} ms")

    # clean / convert core tables once
    initial_clean(trade_day, args.overwrite)

    # resolve locate
    target_loc = args.locate
    if target_loc is None and args.stock:
        sym_map = read_symbol_map()
        inverse = {v: k for k, v in sym_map.items()}
        if args.stock not in inverse:
            raise SystemExit(
                f"Ticker {args.stock!r} not found in R.parquet."
                f" Available: {sorted(set(sym_map.values()))[:10]} …"
            )
        target_loc = inverse[args.stock]

    if target_loc is None:
        raise SystemExit("Pick an instrument: use --stock AAPL or --locate 496")

    build_midpoint(
        trade_day,
        args.snap_ms,
        int(target_loc),
        drop_crossed=(not args.keep_crossed),
        shape_feats=bool(args.shape_feats),
    )


if __name__ == "__main__":
    main()