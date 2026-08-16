"""Propagator kernel deconvolution estimator.

Model assumption: a discrete linear propagator / MA representation for the
mid-price increment,

    dm[t] = m[t+1] - m[t] = sum_n kappa[n] * signs[t-n] + noise[t],

i.e. each signed event at time t-n contributes kappa[n] to the price change
realized at step t, and the contributions of all past events superpose
linearly. Cross-correlating both sides with signs[t-j] gives

    b[j] = E[dm[t] * signs[t-j]] = sum_n kappa[n] * C[|j-n|]

where C is the (normalized) sign autocorrelation function. This is a
Toeplitz linear system in kappa given b and C, and inverting it is exactly
what separates the bare impact kernel from the confound of order-flow
memory: naively reading kappa off of b (or off of the response function
R(l), which is a partial sum of b) implicitly assumes C = delta (iid signs).
When signs are long-memory, that assumption is wrong and the naive read
mixes the kernel's decay with the flow's persistence — this module performs
the correction.

Conditioning caveat: the Toeplitz matrix built from an estimated ACF is only
as well-conditioned as that ACF estimate. Near-unit-root/long-memory C can
make the system nearly singular (small singular values), so the solve uses
least-squares (np.linalg.lstsq) rather than a direct solve, and an explicit
rank check raises before returning a meaningless answer. Callers relying on
this estimator on real (as opposed to synthetic ground-truth) data should
treat the recovered kernel as sensitive to the lag window L and to acf
estimation noise, and should sanity-check L against the effective sample
size.
"""
from __future__ import annotations

import numpy as np


def sign_price_cross_cov(signs: np.ndarray, dm: np.ndarray, max_lag: int) -> np.ndarray:
    """b[j] = E[dm_t * signs_{t-j}] for j = 0..max_lag-1.

    `signs` and `dm` must be the same length and already aligned so that
    dm[t] is the price change caused at/after event t (dm[t] = m[t+1] -
    m[t], with signs[t] the event at time t). To respect causality (kappa
    cannot depend on future signs), b[j] only uses signs at or before dm's
    time index, i.e. signs[t-j] for t >= j.
    """
    if signs.shape != dm.shape:
        raise ValueError(f"signs {signs.shape} and dm {dm.shape} must match")
    n = signs.size
    if max_lag > n:
        raise ValueError(f"max_lag {max_lag} must be <= series length {n}")
    s = signs.astype(np.float64)
    x = dm.astype(np.float64)
    b = np.empty(max_lag)
    for j in range(max_lag):
        if j == 0:
            b[j] = np.mean(x * s)
        else:
            b[j] = np.mean(x[j:] * s[:-j])
    return b


def deconvolve_kernel(b: np.ndarray, acf: np.ndarray) -> np.ndarray:
    """Solve sum_n kappa[n] * C[|j-n|] = b[j] for kappa via Toeplitz lstsq.

    `acf` must have the same length as `b` (acf[0..L-1], normalized so
    acf[0] == 1), giving a (L, L) Toeplitz matrix T[j, n] = acf[|j-n|].
    Solved with np.linalg.lstsq (rcond=None) rather than a direct solve
    because the estimated ACF can make T ill-conditioned; an explicit rank
    check raises ValueError before returning an unreliable answer.
    """
    if b.ndim != 1 or acf.ndim != 1:
        raise ValueError("b and acf must be 1-D arrays")
    if b.shape != acf.shape:
        raise ValueError(f"b {b.shape} and acf {acf.shape} must have the same length")
    length = b.shape[0]
    idx = np.arange(length)
    lag_matrix = np.abs(idx[:, None] - idx[None, :])
    toeplitz = acf[lag_matrix]

    rank = np.linalg.matrix_rank(toeplitz)
    if rank < length:
        raise ValueError(
            f"Toeplitz ACF matrix is singular (rank {rank} < {length}); "
            "cannot deconvolve kernel reliably"
        )

    kappa, *_ = np.linalg.lstsq(toeplitz, b, rcond=None)
    return kappa


def cumulative_kernel(kappa: np.ndarray) -> np.ndarray:
    """G[l] = sum_{n<l} kappa[n], G[0] = 0.

    In the discrete MA representation dm_t = sum_n kappa[n] * signs_{t-n} +
    noise, this partial sum is exactly the propagator's response to a
    single unit event after l steps.
    """
    if kappa.ndim != 1:
        raise ValueError("kappa must be a 1-D array")
    G = np.empty_like(kappa, dtype=np.float64)
    G[0] = 0.0
    np.cumsum(kappa[:-1], out=G[1:])
    return G
