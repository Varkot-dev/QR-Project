"""ETHUSDT gamma robustness: month-level and ISO-week sub-period fits.

Reads events via load_events, computes sign_acf(max_lag=1000) and
fit_power_law(lo=10, hi=500) per sub-period. One symbol at a time.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import polars as pl

from microstructure.estimators.acf import fit_power_law, sign_acf
from microstructure.signals.load import load_events

ROOT = Path("/Users/varshithkotagiri/Projects/QR project/data")
MAX_LAG = 1000
LO, HI = 10, 500
MIN_WEEK_EVENTS = 1_000_000

results: list[dict] = []


def fit(label: str, signs: np.ndarray) -> None:
    n = signs.size
    g = sign_acf(signs, MAX_LAG)
    pf = fit_power_law(g, LO, HI)
    results.append(
        {"label": label, "n_events": int(n), "gamma": round(pf.exponent, 4),
         "stderr": round(pf.stderr, 4)}
    )
    print(f"{label:28s} n={n:>10,d}  gamma={pf.exponent:.4f}  se={pf.stderr:.4f}",
          flush=True)


# ---- ETHUSDT: both months loaded once (needed for cross-month ISO weeks) ----
ev = load_events(ROOT, "ETHUSDT", ["2023-06", "2023-07"])
ev = ev.select(
    "sign",
    pl.col("ts").dt.month().alias("month"),
    pl.col("ts").dt.week().alias("iso_week"),
)

for month, tag in [(6, "ETHUSDT 2023-06"), (7, "ETHUSDT 2023-07")]:
    signs = ev.filter(pl.col("month") == month)["sign"].to_numpy()
    fit(tag, signs)
    del signs
    gc.collect()

weeks = ev["iso_week"].unique().sort().to_list()
for w in weeks:
    signs = ev.filter(pl.col("iso_week") == w)["sign"].to_numpy()
    if signs.size < MIN_WEEK_EVENTS:
        print(f"ETHUSDT week {w}: skipped, only {signs.size:,d} events", flush=True)
        results.append({"label": f"ETHUSDT week {w} (skipped)",
                        "n_events": int(signs.size), "gamma": None, "stderr": None})
        del signs
        continue
    fit(f"ETHUSDT 2023 ISO week {w}", signs)
    del signs
    gc.collect()

del ev
gc.collect()

# ---- BTCUSDT: month at a time ----
for period in ["2023-06", "2023-07"]:
    ev = load_events(ROOT, "BTCUSDT", [period])
    signs = ev["sign"].to_numpy()
    del ev
    gc.collect()
    fit(f"BTCUSDT {period}", signs)
    del signs
    gc.collect()

out = Path(__file__).parent / "gamma_subperiods.json"
out.write_text(json.dumps(results, indent=2))
print(f"\nwrote {out}")
