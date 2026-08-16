"""Raw prints vs aggregated events: lag-1 ACF and power-law gamma contrast.

One symbol-month per run (argv: SYMBOL PERIOD); prints a JSON line.
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from microstructure.estimators.acf import fit_power_law, sign_acf
from microstructure.signals.load import load_events

ROOT = Path("/Users/varshithkotagiri/Projects/QR project/data")
MAX_LAG = 1000
FIT_LO, FIT_HI = 10, 500


def stats_from_signs(signs: np.ndarray) -> dict:
    acf = sign_acf(signs, max_lag=MAX_LAG)
    fit = fit_power_law(acf, lo=FIT_LO, hi=FIT_HI)
    return {
        "n": int(signs.size),
        "acf1": float(acf[1]),
        "gamma": float(fit.exponent),
        "gamma_stderr": float(fit.stderr),
    }


def main() -> None:
    symbol, period = sys.argv[1], sys.argv[2]

    # RAW: read parquet in file order (sorted by agg_trade_id)
    raw_path = ROOT / "parquet" / "aggTrades" / symbol / f"{period}.parquet"
    raw_df = pl.read_parquet(raw_path, columns=["is_buyer_maker"])
    raw_signs = np.where(raw_df["is_buyer_maker"].to_numpy(), -1.0, 1.0)
    n_raw = raw_signs.size
    del raw_df
    gc.collect()
    raw_stats = stats_from_signs(raw_signs)
    del raw_signs
    gc.collect()

    # AGG: load_events pipeline
    events = load_events(ROOT, symbol, [period])
    agg_signs = events["sign"].to_numpy().astype(np.float64)
    del events
    gc.collect()
    agg_stats = stats_from_signs(agg_signs)
    del agg_signs
    gc.collect()

    out = {
        "symbol": symbol,
        "period": period,
        "raw": raw_stats,
        "agg": agg_stats,
        "prints_per_event": n_raw / agg_stats["n"],
    }
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
