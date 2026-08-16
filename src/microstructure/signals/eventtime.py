"""Business-time rescaling: seasonality-robust event-time deseasonalization.

Motivation (Filimonov & Sornette 2015; see the regime-switching trap test
in tests/estimators/test_hawkes.py,
`test_regime_switching_poisson_produces_spurious_endogeneity_trap`): fitting
a Hawkes model directly on clock-time event data cannot distinguish genuine
self-excitation from a time-varying but non-self-exciting baseline rate
mu(t) — e.g. the intraday U-shape / funding-hour clustering universally
present in crypto/equity microstructure data. Both a likelihood fit and the
model-free count-variance estimator report spurious positive endogeneity
when the true process is just a non-stationary-rate Poisson process.

This module implements the standard fix: a deterministic time change to
"business time" (also called "theta time" / operational time in the market
microstructure literature) that flattens the known intraday seasonality
BEFORE any Hawkes fitting is attempted. Given a piecewise-constant estimate
of the intraday rate profile rate(s) (period 24h, repeating across days,
mean exactly 1 by construction), business time is

    tau(t) = integral_0^t rate(s) ds

Under tau, a Poisson process with rate mu(t) = mu_bar * rate(time_of_day(t))
becomes a HOMOGENEOUS Poisson process with rate mu_bar in tau-time (standard
time-change / random time-change theorem for point processes), so a Hawkes
fit performed on tau(events) is no longer confounded by the daily cycle.

Usage: estimate the profile from (ideally) a longer/representative sample
via `intraday_rate_profile`, then rescale the event times you intend to fit
via `rescale_to_business_time` before calling `fit_hawkes_exp` /
`branching_count_variance` from microstructure.estimators.hawkes.
"""
from __future__ import annotations

import numpy as np

MS_PER_DAY = 86_400_000
_EMPTY_BIN_FLOOR = 0.01


def intraday_rate_profile(ts: np.ndarray, n_bins: int = 48) -> np.ndarray:
    """Estimate the normalized intraday event-rate profile from event times.

    `ts` is a 1-D array of int64 epoch-milliseconds (UTC). If timestamps
    originate from a polars `Datetime` series, the caller is responsible for
    converting first, e.g. `df["ts"].dt.epoch("ms").to_numpy()`.

    Each event is assigned to one of `n_bins` equal-width bins covering the
    24h time-of-day-in-UTC cycle (bin width = 86_400_000 / n_bins ms), by
    `(ts % MS_PER_DAY) // bin_width_ms`. The returned profile is the per-bin
    event count divided by the mean per-bin count (total_events / n_bins),
    so the profile has mean exactly 1 by construction — a value of 2.0 in a
    bin means events are twice as frequent in that time-of-day window as the
    all-day average.

    Any bin with zero observed events is floored at `_EMPTY_BIN_FLOOR`
    (0.01) rather than left at 0, to avoid division-by-zero / a permanently
    frozen business clock in `rescale_to_business_time`'s integral. This
    floor is a deliberate approximation: it assumes sparse bins are due to
    sampling noise (short/data-starved windows), not a truly dead trading
    period, and callers estimating a profile from very short or highly
    intermittent samples should be aware the floor can distort empty hours.

    Raises ValueError if `ts` is empty or `n_bins` is not a positive
    integer.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    ts = np.asarray(ts, dtype=np.int64)
    if ts.size == 0:
        raise ValueError("ts must not be empty")

    bin_width_ms = MS_PER_DAY / n_bins
    time_of_day_ms = ts % MS_PER_DAY
    bin_idx = np.minimum((time_of_day_ms / bin_width_ms).astype(np.int64), n_bins - 1)

    counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
    mean_count = ts.size / n_bins
    profile = counts / mean_count
    profile = np.maximum(profile, _EMPTY_BIN_FLOOR)
    return profile


def rescale_to_business_time(ts: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Deterministic time-change from clock time to business (operational) time.

    `ts` is a 1-D array of int64 epoch-milliseconds (UTC), and `profile` is
    an `n_bins`-length normalized rate profile as returned by
    `intraday_rate_profile` (piecewise-constant over time-of-day, period
    24h, repeating across days). Returns a float64 array (same length as
    `ts`) of business-time coordinates in seconds:

        tau(t) = integral_0^t rate(s) ds,  anchored so tau(ts[0]) == 0.

    The integral is computed exactly for a piecewise-constant rate: for a
    query time t, decompose t = n_full_days * 86400s + within_day_seconds.
    Each full day contributes exactly `sum(profile) * bin_width_seconds`
    (== 86400 seconds, since profile has mean 1 by construction) to tau.
    The partial day contributes the cumulative integral of `profile` over
    whole bins strictly before the query's bin, plus a linear partial-bin
    term `profile[bin_idx] * (leftover time within the bin)`. This is
    vectorized (no per-event Python loop): every event's decomposition is
    computed with numpy array ops.

    Determinism: identical input arrays always produce identical output
    (no RNG, no data-dependent iteration order).

    Raises ValueError if `ts` is empty, if `ts` has fewer than 2 events (a
    single event has no informative inter-event structure so rescaling is
    meaningless for downstream Hawkes fitting), or if `profile` has an
    invalid shape (empty, non-1-D, or containing non-positive values).
    """
    ts = np.asarray(ts, dtype=np.int64)
    profile = np.asarray(profile, dtype=np.float64)

    if ts.size == 0:
        raise ValueError("ts must not be empty")
    if ts.size < 2:
        raise ValueError("need at least 2 events to rescale to business time")
    if profile.ndim != 1 or profile.size == 0:
        raise ValueError("profile must be a non-empty 1-D array")
    if np.any(profile <= 0.0):
        raise ValueError("profile must be strictly positive everywhere")

    n_bins = profile.size
    bin_width_ms = MS_PER_DAY / n_bins
    bin_width_s = bin_width_ms / 1000.0

    # Cumulative integral of the profile over whole bins, in seconds:
    # cum_profile[k] = integral of rate(s) ds over time-of-day bins [0, k).
    cum_profile_s = np.concatenate(([0.0], np.cumsum(profile))) * bin_width_s
    day_integral_s = cum_profile_s[-1]  # integral over one full 24h cycle

    day_idx = ts // MS_PER_DAY
    time_of_day_ms = ts - day_idx * MS_PER_DAY
    bin_idx = np.minimum((time_of_day_ms / bin_width_ms).astype(np.int64), n_bins - 1)
    within_bin_ms = time_of_day_ms - bin_idx * bin_width_ms

    tau = (
        day_idx.astype(np.float64) * day_integral_s
        + cum_profile_s[bin_idx]
        + profile[bin_idx] * (within_bin_ms / 1000.0)
    )

    return tau - tau[0]
