"""Independent recompute: BTCUSDT 2023-07 raw vs aggregated lag-1 ACF and gamma.

ACF computed naively (direct dot products per lag, unbiased normalization),
NOT via the project's FFT sign_acf. fit_power_law is reused per task instructions,
but an independent OLS cross-check is also computed inline.
"""
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("/Users/varshithkotagiri/Projects/QR project/data/parquet")
SYMBOL = "BTCUSDT"
PERIOD = "2023-07"
MAX_LAG = 500


def naive_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Direct per-lag dot products; unbiased normalization (divide by n - lag)."""
    xc = x.astype(np.float64) - x.mean()
    n = xc.size
    acov = np.empty(max_lag + 1)
    acov[0] = np.dot(xc, xc) / n
    for k in range(1, max_lag + 1):
        acov[k] = np.dot(xc[:-k], xc[k:]) / (n - k)
    return acov / acov[0]


def my_power_law(acf: np.ndarray, lo: int, hi: int) -> tuple[float, float]:
    """Independent OLS of log acf vs log lag over [lo, hi], skipping acf <= 0."""
    lags = np.arange(len(acf))
    mask = (lags >= lo) & (lags <= hi) & (acf > 0)
    lx, ly = np.log(lags[mask]), np.log(acf[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    return -slope, float(mask.sum())


def report(label: str, signs: np.ndarray) -> None:
    from microstructure.estimators.acf import fit_power_law

    acf = naive_acf(signs, MAX_LAG)
    gamma_own, npts = my_power_law(acf, 10, 500)
    fit = fit_power_law(acf, lo=10, hi=500)
    print(
        f"{label}: n={signs.size} acf1={acf[1]:.6f} "
        f"gamma_fit_power_law={fit.exponent:.6f} gamma_own_ols={gamma_own:.6f} "
        f"fit_points={int(npts)}"
    )


def main() -> None:
    import sys

    if "--agg-only" not in sys.argv:
        # RAW: file order, sign from is_buyer_maker
        path = ROOT / "aggTrades" / SYMBOL / f"{PERIOD}.parquet"
        raw = pl.read_parquet(path, columns=["is_buyer_maker"])
        raw_signs = np.where(raw["is_buyer_maker"].to_numpy(), -1.0, 1.0)
        del raw
        report("RAW", raw_signs)
        del raw_signs

    # AGG: project loader (collapses same-(ts, side) prints, sorts by (ts, sign))
    from microstructure.signals.load import load_events

    events = load_events(ROOT.parent, SYMBOL, [PERIOD])
    agg_signs = events["sign"].to_numpy().astype(np.float64)
    del events
    report("AGG", agg_signs)


if __name__ == "__main__":
    main()
