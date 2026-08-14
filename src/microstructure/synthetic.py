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


def _fft_convolve_full(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Linear ("full") convolution of a and b via zero-padded FFT.

    Equivalent to np.convolve(a, b, mode="full") but O((n+m) log(n+m))
    instead of O(n*m); needed because the MA truncation below uses
    n_lags large enough that np.convolve is too slow.
    """
    out_len = len(a) + len(b) - 1
    size = 1 << (out_len - 1).bit_length()  # next power of two >= out_len
    fa = np.fft.rfft(a, n=size)
    fb = np.fft.rfft(b, n=size)
    return np.fft.irfft(fa * fb, n=size)[:out_len]


def fractional_signs(n: int, d: float, seed: int) -> np.ndarray:
    """Signs of FARIMA(0, d, 0) noise: power-law (long-memory) sign ACF.

    Built by MA(inf) truncation with coefficients
    psi_k = Gamma(k + d) / (Gamma(k + 1) Gamma(d)), computed recursively:
    psi_0 = 1, psi_k = psi_{k-1} * (k - 1 + d) / k.

    n_lags = 50_000 terms: the truncation must be large enough that the
    discarded tail (~k^(d-1) decay) does not bias the sign-ACF power-law
    exponent gamma = 1 - 2*d. At 2000 terms gamma is measurably biased
    high (~0.30 vs theoretical 0.20 at d=0.4); at 50_000 terms it recovers
    to ~0.202. FFT-based convolution (O((n+n_lags) log(...))) is used
    instead of np.convolve (O(n*n_lags)) because n_lags=50_000 makes the
    direct convolution too slow to be practical.
    """
    if not 0.0 < d < 0.5:
        raise ValueError("d must be in (0, 0.5)")
    rng = np.random.default_rng(seed)
    n_lags = 50_000
    psi = np.empty(n_lags)
    psi[0] = 1.0
    for k in range(1, n_lags):
        psi[k] = psi[k - 1] * (k - 1 + d) / k
    eps = rng.standard_normal(n + n_lags)
    x = _fft_convolve_full(eps, psi)[n_lags : n_lags + n]
    return np.where(x >= 0, 1, -1).astype(np.int8)
