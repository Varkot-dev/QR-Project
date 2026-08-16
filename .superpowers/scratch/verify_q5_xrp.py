"""Independent reproduction of the Q5 XRPUSDT record.

Loads June aggTrades filtered to 2023-06-01..07 plus daily bookTicker,
joins strictly-prior mids, computes gamma_week and blocked beta, and
compares against results/q5_kernel_panel.json.
"""
from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

from microstructure.estimators.acf import fit_power_law, sign_acf
from microstructure.estimators.propagator import kernel_exponent_blocked
from microstructure.signals.load import events_with_prior_mid, load_book_ticker, load_events

ROOT = Path("data")
SYMBOL = "XRPUSDT"
MAX_LAG = 300

daily = [(date(2023, 6, 1) + timedelta(days=i)).isoformat() for i in range(7)]

events = load_events(ROOT, SYMBOL, ["2023-06"])
ts_min = datetime(2023, 6, 1, tzinfo=UTC)
ts_max = datetime(2023, 6, 8, tzinfo=UTC)
events = events.filter((events["ts"] >= ts_min) & (events["ts"] < ts_max))

bt = load_book_ticker(ROOT, SYMBOL, daily)
joined, n_dropped = events_with_prior_mid(events, bt)

signs = joined["sign"].to_numpy()
mids = joined["mid"].to_numpy()
n_events = signs.size

acf_full = sign_acf(signs, 1000)
gamma_fit = fit_power_law(acf_full, lo=10, hi=500)

dm = np.diff(mids)
blocked = kernel_exponent_blocked(signs[:-1], dm, max_lag=MAX_LAG)

ref = next(
    r
    for r in json.load(open("results/q5_kernel_panel.json"))["records"]
    if r["symbol"] == SYMBOL
)

print(f"n_events   repro={n_events}  json={ref['n_events']}")
print(f"gamma_week repro={gamma_fit.exponent:.6f}  json={ref['gamma_week']:.6f}  "
      f"diff={abs(gamma_fit.exponent - ref['gamma_week']):.2e}")
print(f"beta       repro={blocked.exponent:.6f}  json={ref['beta']:.6f}  "
      f"diff={abs(blocked.exponent - ref['beta']):.2e}")
print(f"block_sd   repro={blocked.block_sd:.6f}  json={ref['beta_block_sd']:.6f}")
print("within tol 0.02:",
      abs(gamma_fit.exponent - ref["gamma_week"]) <= 0.02
      and abs(blocked.exponent - ref["beta"]) <= 0.02)
