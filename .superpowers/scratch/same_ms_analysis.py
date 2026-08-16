"""Same-millisecond two-sided activity characterization, BTC vs ETH, 2023-06.

One symbol at a time; frames deleted after use. Prints JSON.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import polars as pl

from microstructure.estimators.acf import sign_acf
from microstructure.signals.load import load_events

ROOT = Path("/Users/varshithkotagiri/Projects/QR project/data")
PERIOD = ["2023-06"]
MAX_LAG = 6


def analyze(symbol: str) -> dict:
    ev = load_events(ROOT, symbol, PERIOD)
    n = ev.height

    # Per-ms grouping
    grp = ev.group_by("ts").agg(
        pl.len().alias("n_ev"),
        (pl.col("sign") == 1).sum().alias("n_buy"),
        (pl.col("sign") == -1).sum().alias("n_sell"),
    )
    n_ms = grp.height
    n_events_in_shared_ms = int(grp.filter(pl.col("n_ev") > 1)["n_ev"].sum())
    frac_events_sharing_ts = n_events_in_shared_ms / n
    n_both = int(grp.filter((pl.col("n_buy") > 0) & (pl.col("n_sell") > 0)).height)
    frac_ms_both_signs = n_both / n_ms
    mean_events_per_ms = n / n_ms
    del grp
    gc.collect()

    # Consecutive-pair flip probabilities, conditioned on same-ts vs different-ts
    ts = ev["ts"].to_numpy()
    signs = ev["sign"].to_numpy().astype(np.int8)
    del ev
    gc.collect()

    same_ts = ts[1:] == ts[:-1]
    flip = signs[1:] != signs[:-1]
    n_same = int(same_ts.sum())
    n_diff = int((~same_ts).sum())
    p_flip_same_ts = float(flip[same_ts].mean())
    p_flip_diff_ts = float(flip[~same_ts].mean())
    del ts, same_ts, flip
    gc.collect()

    acf = sign_acf(signs, MAX_LAG)
    del signs
    gc.collect()

    return {
        "n_events": n,
        "n_occupied_ms": n_ms,
        "frac_events_sharing_ts": round(frac_events_sharing_ts, 6),
        "frac_ms_with_both_signs": round(frac_ms_both_signs, 6),
        "mean_events_per_occupied_ms": round(mean_events_per_ms, 6),
        "n_pairs_same_ts": n_same,
        "n_pairs_diff_ts": n_diff,
        "p_flip_given_same_ts": round(p_flip_same_ts, 6),
        "p_flip_given_diff_ts": round(p_flip_diff_ts, 6),
        "acf_1_to_6": [round(float(a), 6) for a in acf[1:]],
    }


def main() -> None:
    out = {}
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        out[symbol] = analyze(symbol)
        gc.collect()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
