"""Propagator kernel deconvolution estimator.

Model assumption: a discrete linear propagator / MA representation for the
mid-price increment,

    dm[t] = m[t+1] - m[t] = sum_n kappa[n] * signs[t-n] + noise[t],

i.e. each signed event at time t-n contributes kappa[n] to the price change
realized at step t, and the contributions of all past events superpose
linearly. Cross-correlating both sides with signs[t-j] gives

    b[j] = E[dm[t] * signs[t-j]] = sum_n kappa[n] * C[|j-n|]

where C is the (normalized) sign autocorrelation function. This identity
relies on E[signs[t]^2] = 1, i.e. +-1 signs; if using signed volume instead
of raw +-1 signs, the cross-covariance must be rescaled by the volume
variance before this identity applies. This is a Toeplitz linear system in
kappa given b and C, and inverting it is exactly what separates the bare
impact kernel from the confound of order-flow memory: naively reading kappa
off of b (or off of the response function R(l), which is a partial sum of
b) implicitly assumes C = delta (iid signs). When signs are long-memory,
that assumption is wrong and the naive read mixes the kernel's decay with
the flow's persistence — this module performs the correction.

Conditioning caveat: the Toeplitz matrix built from an estimated ACF is only
as well-conditioned as that ACF estimate. Near-unit-root/long-memory C can
make the system nearly singular (small singular values), so the solve uses
least-squares (np.linalg.lstsq) rather than a direct solve, and an explicit
rank check raises before returning a meaningless answer. HOWEVER: the rank
and condition number of the Toeplitz matrix are properties of the ACF
*values*, not of how well those values were estimated from finite data.
A well-conditioned Toeplitz matrix built from a noisy, short-sample ACF
estimate can still produce a garbage kappa, because lstsq happily fits the
noise. Empirically (see task-1-report.md for the exact reproduction), at
n_samples=1,000 and L=300 (ratio 3.3), Toeplitz condition numbers of
~40-500 (well within the range that "looks fine") were observed across
seeds while the recovered power-law exponent ranged roughly 0.15..0.65 in
a 20-seed sweep -- pure estimation noise, undetectable by rank/condition
checks alone. This is why deconvolve_kernel requires n_samples and
enforces a minimum samples-per-degree-of-freedom ratio (see its docstring),
and why callers must use kernel_exponent_blocked's block_sd rather than
fit_power_law's OLS stderr to characterize uncertainty on any recovered
exponent (see that docstring for the measured magnitude of the OLS
stderr's under-statement).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from microstructure.estimators.acf import fit_power_law, sign_acf

MIN_SAMPLES_PER_LAG = 100


def sign_price_cross_cov(signs: np.ndarray, dm: np.ndarray, max_lag: int) -> np.ndarray:
    """b[j] = E[(dm_t - mean(dm)) * (signs_{t-j} - mean(signs))] for j = 0..max_lag-1.

    `signs` and `dm` must be the same length and already aligned so that
    dm[t] is the price change caused at/after event t (dm[t] = m[t+1] -
    m[t], with signs[t] the event at time t). To respect causality (kappa
    cannot depend on future signs), b[j] only uses signs at or before dm's
    time index, i.e. signs[t-j] for t >= j.

    Both series are mean-centered before cross-multiplication, consistent
    with sign_acf's centering convention. For near-zero-mean +-1 signs this
    changes nothing; for directional order flow (mean substantially away
    from zero) omitting the centering biases b by mean(dm)*mean(signs).
    """
    if signs.shape != dm.shape:
        raise ValueError(f"signs {signs.shape} and dm {dm.shape} must match")
    n = signs.size
    if max_lag > n:
        raise ValueError(f"max_lag {max_lag} must be <= series length {n}")
    s = signs.astype(np.float64) - signs.astype(np.float64).mean()
    x = dm.astype(np.float64) - dm.astype(np.float64).mean()
    b = np.empty(max_lag)
    for j in range(max_lag):
        if j == 0:
            b[j] = np.mean(x * s)
        else:
            b[j] = np.mean(x[j:] * s[:-j])
    return b


def deconvolve_kernel(
    b: np.ndarray,
    acf: np.ndarray,
    n_samples: int,
    allow_low_sample: bool = False,
) -> np.ndarray:
    """Solve sum_n kappa[n] * C[|j-n|] = b[j] for kappa via Toeplitz lstsq.

    `acf` must have the same length as `b` (acf[0..L-1], normalized so
    acf[0] == 1 within 1e-9 -- pass the normalized sign ACF, not a raw
    autocovariance), giving a (L, L) Toeplitz matrix T[j, n] = acf[|j-n|].
    Solved with np.linalg.lstsq (rcond=None) rather than a direct solve
    because the estimated ACF can make T ill-conditioned; an explicit rank
    check raises ValueError before returning an unreliable answer.

    `n_samples` is the number of (signs, dm) observations `b` and `acf`
    were estimated from. This is REQUIRED, separately from the rank/
    condition checks on the Toeplitz matrix, because those checks are blind
    to estimation noise: a short sample can produce a well-conditioned
    Toeplitz matrix (built from noisy but not-degenerate ACF values) whose
    lstsq solution is still dominated by that noise rather than the true
    kernel. Measured example (see task-1-report.md): n_samples=1_000,
    L=300 (ratio 3.3) gives Toeplitz condition numbers in the range
    ~40-500 across seeds (well within what "looks fine"), while the
    recovered power-law exponent on synthetic long-memory data ranged
    roughly 0.15..0.65 across a 20-seed sweep against a planted 0.35 --
    i.e. even the rough magnitude is not reliably recovered. A floor of
    n_samples / len(b) >= 100 is enforced by default; pass
    allow_low_sample=True to bypass it (e.g. for deliberate small-sample
    diagnostics) -- doing so does not make the result trustworthy, it only
    silences this guard.
    """
    if b.ndim != 1 or acf.ndim != 1:
        raise ValueError("b and acf must be 1-D arrays")
    if b.shape != acf.shape:
        raise ValueError(f"b {b.shape} and acf {acf.shape} must have the same length")
    length = b.shape[0]

    if abs(acf[0] - 1.0) > 1e-9:
        raise ValueError(
            f"acf[0] = {acf[0]!r}, expected 1.0 (within 1e-9); pass the "
            "normalized sign ACF (e.g. from sign_acf), not a raw autocovariance"
        )

    ratio = n_samples / length
    if ratio < MIN_SAMPLES_PER_LAG and not allow_low_sample:
        raise ValueError(
            f"n_samples/len(b) = {n_samples}/{length} = {ratio:.2f} is below the "
            f"minimum ratio of {MIN_SAMPLES_PER_LAG}; the Toeplitz rank/condition "
            "checks cannot detect this kind of estimation-noise failure (a "
            "well-conditioned matrix built from a noisy short-sample ACF can "
            "still yield a garbage kappa). Increase n_samples, decrease L "
            "(len(b)), or pass allow_low_sample=True to proceed anyway (the "
            "result is not thereby made trustworthy)."
        )

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


@dataclass(frozen=True)
class BlockedExponent:
    exponent: float  # full-sample beta_hat
    block_exponents: list[float]
    block_sd: float
    n_samples: int


def _kernel_power_law_exponent(
    signs: np.ndarray, dm: np.ndarray, max_lag: int, fit_lo: int, fit_hi: int
) -> float:
    """Run the full pipeline: cross-cov -> deconvolve -> cumulative -> fit."""
    b = sign_price_cross_cov(signs, dm, max_lag=max_lag)
    acf = sign_acf(signs, max_lag=max_lag - 1)
    kappa = deconvolve_kernel(b, acf, n_samples=signs.size)
    G = cumulative_kernel(kappa)
    return fit_power_law(G, lo=fit_lo, hi=fit_hi).exponent


def kernel_exponent_blocked(
    signs: np.ndarray,
    dm: np.ndarray,
    max_lag: int,
    n_blocks: int = 5,
    fit_lo: int = 5,
    fit_hi: int | None = None,
) -> BlockedExponent:
    """Full-sample beta_hat plus a block-bootstrap uncertainty estimate.

    fit_power_law's OLS stderr is NOT a usable uncertainty on the
    deconvolved power-law exponent beta_hat -- do not report it as such.
    It is the standard error of a single log-log regression fit to one
    already-noisy point estimate of the cumulative kernel; it does not
    account for the sampling noise in b and the ACF that feeds into
    deconvolve_kernel, nor for autocorrelation among the fitted points.
    Measured on synthetic long-memory data (fractional_signs d=0.35,
    n=300_000, L=300, 20 seeds): OLS stderr ~= 0.00136 while the actual
    Monte-Carlo standard deviation of beta_hat across seeds was ~= 0.0092
    (about 6.8x larger), and the mean recovered beta_hat was ~= 0.386
    against a planted 0.35 -- a systematic finite-L bias of roughly +0.03
    to +0.04 at L=300. Any downstream comparison of a recovered beta_hat
    against a reference value (a "balance residual") must be judged against
    block_sd from this function AND against that ~0.03-0.04 bias scale,
    never against fit_power_law's stderr alone.

    Computes the full (cross-cov, deconvolve, cumulative, fit) pipeline on
    the full sample (-> exponent) and on n_blocks contiguous, non-
    overlapping blocks of (signs, dm) (-> block_exponents), then reports
    block_sd = np.std(block_exponents, ddof=1) as the uncertainty estimate.
    """
    if signs.shape != dm.shape:
        raise ValueError(f"signs {signs.shape} and dm {dm.shape} must match")
    if n_blocks < 2:
        raise ValueError(f"n_blocks must be >= 2 to compute a standard deviation, got {n_blocks}")
    n = signs.size
    if fit_hi is None:
        fit_hi = max_lag // 2

    full_exponent = _kernel_power_law_exponent(signs, dm, max_lag, fit_lo, fit_hi)

    block_size = n // n_blocks
    if block_size <= max_lag:
        raise ValueError(
            f"block size {block_size} (n={n} / n_blocks={n_blocks}) must exceed "
            f"max_lag {max_lag}; use fewer blocks or a longer series"
        )
    block_exponents: list[float] = []
    for i in range(n_blocks):
        start = i * block_size
        end = n if i == n_blocks - 1 else start + block_size
        block_exponents.append(
            _kernel_power_law_exponent(signs[start:end], dm[start:end], max_lag, fit_lo, fit_hi)
        )

    block_sd = float(np.std(np.array(block_exponents), ddof=1))
    return BlockedExponent(
        exponent=full_exponent,
        block_exponents=block_exponents,
        block_sd=block_sd,
        n_samples=n,
    )
