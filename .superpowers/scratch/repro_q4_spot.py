"""Spot-check gamma and p_flip from raw parquet for 3 symbols spanning activity.

Uses signals.load.load_events for event collapse (as instructed), then a fully
naive O(n*lag) ACF and my own closed-form log-log OLS at lags 10..500.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from microstructure.signals.load import load_events  # noqa: E402

ROOT = Path("data")
PERIOD = "2023-06"
SYMBOLS = ["LPTUSDT", "SXPUSDT", "BTCUSDT"]  # low / median / high activity

table = pl.read_parquet("results/q4_cross_section.parquet")


def naive_acf(signs: np.ndarray, max_lag: int) -> np.ndarray:
    x = signs.astype(np.float64) - signs.mean()
    n = x.size
    acf = np.empty(max_lag + 1)
    var = (x @ x) / n  # unbiased-at-lag-0 normalizer, matches acov[0]/(n-0)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = ((x[:-lag] @ x[lag:]) / (n - lag)) / var
    return acf


def loglog_gamma(acf: np.ndarray, lo: int = 10, hi: int = 500) -> float:
    lags = np.arange(len(acf))
    m = (lags >= lo) & (lags <= hi) & (acf > 0)
    lx, ly = np.log(lags[m]), np.log(acf[m])
    dx, dy = lx - lx.mean(), ly - ly.mean()
    slope = (dx @ dy) / (dx @ dx)
    return -float(slope)


for sym in SYMBOLS:
    ev = load_events(ROOT, sym, [PERIOD])
    signs = ev["sign"].to_numpy()
    n = signs.size
    p_flip = float(np.mean(signs[1:] != signs[:-1]))
    acf = naive_acf(signs, 500)
    gamma = loglog_gamma(acf, 10, 500)
    row = table.filter(pl.col("symbol") == sym)
    tg = row["gamma"][0]
    tp = row["p_flip"][0]
    tn = row["n_events"][0]
    print(f"{sym}: n={n} (table {tn}, match={n==tn})")
    print(f"  gamma  mine={gamma:.6f} table={tg:.6f} diff={abs(gamma-tg):.6f} pass={abs(gamma-tg)<=0.02}")
    print(f"  p_flip mine={p_flip:.6f} table={tp:.6f} diff={abs(p_flip-tp):.6f} pass={abs(p_flip-tp)<=0.005}")
