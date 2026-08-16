"""Autocorrelation and power-law decay estimation for sign series.

sign_acf uses the FFT (Wiener-Khinchin): O(n log n) vs O(n*max_lag) naive.
fit_power_law is OLS on log(y) vs log(lag) — standard in the order-flow
memory literature; its stderr is the OLS slope standard error, which
understates true uncertainty for autocorrelated data (documented caveat,
addressed with block bootstrap at the analysis level if needed).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sign_acf(signs: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized autocorrelation of a ±1 (or real) series, lags 0..max_lag."""
    x = signs.astype(np.float64) - signs.mean()
    n = x.size
    if max_lag >= n:
        raise ValueError(f"max_lag {max_lag} must be < series length {n}")
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, nfft)
    acov = np.fft.irfft(f * np.conj(f), nfft)[: max_lag + 1]
    # Use unbiased ACF: divide by (n - lag) for each lag
    lags = np.arange(len(acov))
    acov = acov / (n - lags)
    return acov / acov[0]


@dataclass(frozen=True)
class PowerLawFit:
    exponent: float  # gamma in y ~ lag^(-gamma)
    intercept: float
    stderr: float


def fit_power_law(y: np.ndarray, lo: int, hi: int) -> PowerLawFit:
    """OLS fit of log y vs log lag over [lo, hi], skipping y <= 0 points."""
    lags = np.arange(len(y))
    mask = (lags >= lo) & (lags <= hi) & (y > 0)
    if mask.sum() < 3:
        raise ValueError("fewer than 3 positive points in fit window")
    lx, ly = np.log(lags[mask]), np.log(y[mask])
    (slope, intercept), cov = np.polyfit(lx, ly, 1, cov=True)
    return PowerLawFit(exponent=-slope, intercept=intercept, stderr=float(np.sqrt(cov[0, 0])))
