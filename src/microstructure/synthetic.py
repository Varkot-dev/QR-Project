"""Series with KNOWN statistical properties, for validating estimators.

Rule (spec section 5): no estimator touches real data until it recovers the
known answers generated here within stated error bars.
"""
from __future__ import annotations

import numpy as np


def iid_signs(n: int, seed: int) -> np.ndarray:
    """±1 coin flips: ACF is exactly 0 at every positive lag."""
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=n)


def markov_signs(n: int, p_repeat: float, seed: int) -> np.ndarray:
    """±1 chain repeating previous sign w.p. p_repeat.

    Theoretical ACF(k) = (2*p_repeat - 1)**k  (geometric, short memory).
    """
    if not 0.0 < p_repeat < 1.0:
        raise ValueError("p_repeat must be in (0, 1)")
    rng = np.random.default_rng(seed)
    flips = rng.random(n) >= p_repeat  # True -> switch sign
    signs = np.empty(n, dtype=np.int8)
    signs[0] = 1
    switches = np.where(flips[1:], -1, 1)
    signs[1:] = np.cumprod(switches)
    return signs


def fractional_signs(n: int, d: float, seed: int) -> np.ndarray:
    """Signs of FARIMA(0, d, 0) noise: power-law (long-memory) sign ACF.

    Built by MA(inf) truncation with coefficients
    psi_k = Gamma(k + d) / (Gamma(k + 1) Gamma(d)), computed recursively:
    psi_0 = 1, psi_k = psi_{k-1} * (k - 1 + d) / k.
    """
    if not 0.0 < d < 0.5:
        raise ValueError("d must be in (0, 0.5)")
    rng = np.random.default_rng(seed)
    n_lags = 2000
    psi = np.empty(n_lags)
    psi[0] = 1.0
    for k in range(1, n_lags):
        psi[k] = psi[k - 1] * (k - 1 + d) / k
    eps = rng.standard_normal(n + n_lags)
    x = np.convolve(eps, psi, mode="full")[n_lags : n_lags + n]
    return np.where(x >= 0, 1, -1).astype(np.int8)
