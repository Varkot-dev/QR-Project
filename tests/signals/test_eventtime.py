"""Tests for business-time rescaling (seasonality-robust Hawkes fitting).

These tests ARE the deseasonalization justification for Phase-3 Task 2:
Group 3's two `_amplitude_*` tests directly parallel the regime-switching
trap documented in
tests/estimators/test_hawkes.py::test_regime_switching_poisson_produces_spurious_endogeneity_trap
— a non-stationary intraday rate profile, left in clock time, inflates a
Hawkes fit's alpha even though (in Group 3's construction, via
`simulate_seasonal_hawkes_exp`) there IS genuine self-excitation, just with
its true alpha obscured by seasonality. Rescaling to business time before
fitting recovers the true alpha. A secondary, explicitly-labeled
construction-bias-documentation test (post-hoc thinning of an unseasonal
Hawkes simulation) is kept separately to document a measured, systematic
downward bias in that alternative (rejected) construction — see that test's
docstring.
"""
from __future__ import annotations

import numpy as np
import pytest

from microstructure.estimators.hawkes import (
    fit_hawkes_exp,
    simulate_hawkes_exp,
    simulate_seasonal_hawkes_exp,
)
from microstructure.signals.eventtime import (
    MS_PER_DAY,
    intraday_rate_profile,
    rescale_to_business_time,
)

# ---------------------------------------------------------------------------
# Shared synthetic data helpers.
# ---------------------------------------------------------------------------

_DAY_MS = MS_PER_DAY


def _planted_daily_shape(n_bins: int, amplitude: float = 1.05) -> np.ndarray:
    """A 2-cycle-per-day sinusoidal shape, normalized to mean 1, strictly positive.

    2 full cosine cycles across the 24h period (period = 12h), matching the
    brief's "planted 2-cycle sinusoidal daily rate." `amplitude` controls
    trough depth (raw = amplitude + cos(...), so smaller amplitude -> deeper,
    narrower trough relative to the mean); the default (1.05) is used for
    the profile-recovery and CV-flattening tests (groups 1-2), where a deep
    trough gives a cleaner, more distinctive shape to recover/flatten. Group
    3 (the justification test) uses a shallower amplitude=1.4 instead — see
    that test's docstring for why.
    """
    bin_centers = (np.arange(n_bins) + 0.5) / n_bins  # in [0, 1), fraction of day
    raw = amplitude + np.cos(2 * 2 * np.pi * bin_centers)  # 2 cycles/day
    return raw / raw.mean()


def _simulate_inhomogeneous_poisson(
    shape: np.ndarray, mean_rate_per_ms: float, t_end_ms: float, seed: int
) -> np.ndarray:
    """Thinning-based simulation of a Poisson process with rate mean_rate_per_ms * shape(tod).

    `shape` is piecewise-constant over `len(shape)` time-of-day bins,
    period 24h, mean 1. Uses simple rejection (thinning) against the
    max of `shape` as the bounding constant.
    """
    rng = np.random.default_rng(seed)
    n_bins = shape.size
    bin_width_ms = _DAY_MS / n_bins
    shape_max = shape.max()
    lambda_bar = mean_rate_per_ms * shape_max

    events: list[float] = []
    t = 0.0
    while t < t_end_ms:
        t += rng.exponential(1.0 / lambda_bar)
        if t >= t_end_ms:
            break
        tod_ms = t % _DAY_MS
        bin_idx = min(int(tod_ms / bin_width_ms), n_bins - 1)
        accept_prob = (mean_rate_per_ms * shape[bin_idx]) / lambda_bar
        if rng.random() <= accept_prob:
            events.append(t)
    return np.asarray(events, dtype=np.int64)


# ---------------------------------------------------------------------------
# Group 1: profile recovery.
# ---------------------------------------------------------------------------


def test_intraday_rate_profile_recovers_planted_2cycle_shape():
    n_bins = 48
    true_shape = _planted_daily_shape(n_bins)
    # 20 days, mean rate high enough for a clean per-bin recovery.
    t_end_ms = 20 * _DAY_MS
    mean_rate_per_ms = 2000.0 / _DAY_MS  # ~2000 events/day
    ts = _simulate_inhomogeneous_poisson(true_shape, mean_rate_per_ms, t_end_ms, seed=1)

    profile = intraday_rate_profile(ts, n_bins=n_bins)

    assert profile.mean() == pytest.approx(1.0, abs=1e-9)
    corr = np.corrcoef(profile, true_shape)[0, 1]
    assert corr > 0.95, f"expected profile-shape correlation > 0.95, got {corr}"


def test_intraday_rate_profile_floors_empty_bins():
    # All events land in bin 0 only (time-of-day 0..bin_width); other bins
    # must be floored, not zero, to keep rescale_to_business_time invertible.
    # The floor is applied then the whole profile is renormalized to mean 1
    # (fix for HIGH-severity review finding), so the floored bins land very
    # close to, but not exactly at, the raw 0.01 floor constant.
    n_bins = 48
    bin_width_ms = _DAY_MS / n_bins
    ts = np.arange(0, 100) * _DAY_MS + 10  # every event at ~10ms into the day
    assert (ts % _DAY_MS < bin_width_ms).all()

    profile = intraday_rate_profile(ts, n_bins=n_bins)

    assert np.all(profile > 0.0)
    assert profile[1:].min() == pytest.approx(0.01, rel=0.02)
    assert profile.mean() == pytest.approx(1.0, abs=1e-9)


def test_intraday_rate_profile_floor_preserves_exact_mean_one():
    """HIGH-severity fix: floor THEN renormalize, so mean stays exactly 1
    (not approximately 1) even with multiple forced-empty bins, and a full
    24h cycle's business-time integral (see
    test_rescale_to_business_time_full_day_integrates_to_exact_86400s below)
    stays exactly 86400 seconds rather than merely close to it.
    """
    n_bins = 48
    bin_width_ms = _DAY_MS / n_bins
    # Force events into only the first 36 of 48 bins, leaving 12 bins
    # (indices 36-47) with zero observed events.
    active_bins = np.arange(36)
    rng = np.random.default_rng(3)
    bin_choices = rng.choice(active_bins, size=2000)
    offsets_ms = rng.integers(0, int(bin_width_ms), size=2000)
    ts = (bin_choices * bin_width_ms + offsets_ms).astype(np.int64)

    profile = intraday_rate_profile(ts, n_bins=n_bins)

    assert profile.mean() == pytest.approx(1.0, abs=1e-12)
    assert np.all(profile[36:] > 0.0)
    assert np.all(profile[36:] < profile[:36].min())  # floored bins are still the smallest


def test_intraday_rate_profile_rejects_empty_input():
    with pytest.raises(ValueError):
        intraday_rate_profile(np.array([], dtype=np.int64))


def test_intraday_rate_profile_rejects_non_positive_n_bins():
    with pytest.raises(ValueError):
        intraday_rate_profile(np.array([0, 1000], dtype=np.int64), n_bins=0)


# ---------------------------------------------------------------------------
# Group 2: rescaling flattens the inter-event-time distribution.
# ---------------------------------------------------------------------------


def _coefficient_of_variation(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1) / np.mean(x))


def _max_ks_deviation_from_exponential(gaps: np.ndarray) -> float:
    """Max |empirical CDF - exponential CDF| for inter-event gaps (KS-style).

    Compares the empirical CDF of `gaps` against 1 - exp(-x/mean(gaps)), the
    CDF of an Exp(1/mean) distribution -- the theoretical inter-event-gap
    distribution for a homogeneous Poisson process, which is what
    business-time-rescaled events from a deseasonalized Poisson-like process
    should look like. Simple sorted-empirical-CDF comparison, per the
    brief's suggested check (item 6 of the review's fix list).
    """
    sorted_gaps = np.sort(gaps)
    n = sorted_gaps.size
    empirical_cdf = (np.arange(1, n + 1)) / n
    mean_gap = sorted_gaps.mean()
    theoretical_cdf = 1.0 - np.exp(-sorted_gaps / mean_gap)
    return float(np.max(np.abs(empirical_cdf - theoretical_cdf)))


def test_rescale_to_business_time_flattens_cv_to_near_one():
    n_bins = 48
    true_shape = _planted_daily_shape(n_bins)
    t_end_ms = 20 * _DAY_MS
    mean_rate_per_ms = 2000.0 / _DAY_MS
    ts = _simulate_inhomogeneous_poisson(true_shape, mean_rate_per_ms, t_end_ms, seed=2)

    raw_gaps = np.diff(ts.astype(np.float64))
    raw_cv = _coefficient_of_variation(raw_gaps)
    # Inhomogeneous-rate Poisson clumps into high/low periods, inflating CV
    # of the raw (clock-time) inter-event gaps well above the CV=1 that a
    # homogeneous Poisson process would show.
    assert raw_cv > 1.15, f"expected raw clock-time CV > 1.15, got {raw_cv}"

    profile = intraday_rate_profile(ts, n_bins=n_bins)
    tau = rescale_to_business_time(ts, profile)
    tau_gaps = np.diff(tau)
    tau_cv = _coefficient_of_variation(tau_gaps)

    assert abs(tau_cv - 1.0) < 0.05, f"expected business-time CV within 5% of 1.0, got {tau_cv}"

    # KS-style distribution check (brief's originally-specified check, added
    # per review item 6): business-time gaps from a deseasonalized
    # inhomogeneous Poisson process should look exponentially distributed,
    # not just have CV~1 (CV~1 alone doesn't rule out non-exponential shapes
    # that happen to share that second moment).
    max_dev = _max_ks_deviation_from_exponential(tau_gaps)
    assert max_dev < 0.02, f"expected max KS deviation from Exp(1/mean) < 0.02, got {max_dev}"


def test_rescale_to_business_time_full_day_integrates_to_exact_86400s():
    """HIGH-severity fix follow-through: since intraday_rate_profile now
    floors THEN renormalizes (exact mean 1 even with floored bins), one full
    24h cycle must integrate to EXACTLY 86400.0 seconds in business time,
    not merely approximately -- including when some bins were floored.
    """
    n_bins = 48
    bin_width_ms = _DAY_MS / n_bins
    # Force 12 empty bins (as in the floor-preserves-mean-1 test) so the
    # floor is actually exercised here too.
    active_bins = np.arange(36)
    rng = np.random.default_rng(5)
    bin_choices = rng.choice(active_bins, size=2000)
    offsets_ms = rng.integers(0, int(bin_width_ms), size=2000)
    ts_for_profile = (bin_choices * bin_width_ms + offsets_ms).astype(np.int64)
    profile = intraday_rate_profile(ts_for_profile, n_bins=n_bins)

    # Two events exactly one full day apart, at the same time-of-day.
    ts = np.array([0, _DAY_MS], dtype=np.int64)
    tau = rescale_to_business_time(ts, profile)
    assert tau[1] - tau[0] == pytest.approx(86_400.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Group 3: THE JUSTIFICATION TEST — seasonality inflates a real Hawkes fit's
# alpha in clock time; business-time rescaling recovers the true alpha.
#
# This is the direct analogue, for Task 2, of
# tests/estimators/test_hawkes.py::test_regime_switching_poisson_produces_spurious_endogeneity_trap
# (Filimonov & Sornette 2015): there, a non-self-exciting but non-stationary
# process fools both the MLE and count-variance estimator into reporting
# spurious positive endogeneity. Here we go one step further: a GENUINE
# Hawkes process (real self-excitation, known alpha) has a genuinely
# time-varying (seasonal) baseline rate, simulated directly via
# `simulate_seasonal_hawkes_exp` (not approximated by post-hoc thinning — see
# the "construction-bias documentation" test below for why that distinction
# matters). A raw clock-time fit conflates the seasonality-driven clustering
# with the genuine self-excitation and reports an inflated alpha; rescaling
# to business time strips the seasonality out first and recovers alpha close
# to its true, planted value. This test is the reason the whole eventtime
# module exists.
#
# Tested at BOTH a shallow-trough amplitude (1.4) and a deep-trough
# amplitude (1.05): at amplitude=1.05 the raw fit lands near a high-alpha
# (~0.90) region of parameter space (a real, large seasonality-driven
# inflation this time -- not the Nelder-Mead degenerate-optimum artifact
# seen with thinning-based construction in Task 2's first attempt, since the
# seasonal simulator's baseline never actually goes as low as thinning could
# push effective density; see docstring below). Both amplitudes recover
# alpha within +/-0.06 of true after rescaling.
# ---------------------------------------------------------------------------


def test_business_time_rescaling_corrects_seasonality_inflated_alpha_amplitude_1_4():
    _run_seasonal_justification_case(amplitude=1.4, seed=7)


def test_business_time_rescaling_corrects_seasonality_inflated_alpha_amplitude_1_05_deep_trough():
    """Deep-trough case (amplitude=1.05): included specifically because a
    shallow daily swing could, in principle, understate how badly clock-time
    Hawkes fitting is fooled by realistic (deep, sharp) intraday seasonality.
    With the genuine seasonal-baseline simulator, this case does NOT trigger
    the degenerate near-alpha=1/near-beta=0 Nelder-Mead local optimum that
    the (now-removed) thinning-based construction hit at this same amplitude
    in Task 2's first attempt -- the raw fit here lands at a large but
    non-degenerate alpha (~0.90, beta staying near the true kernel
    timescale), which is itself evidence that construction (thinning vs. a
    genuine seasonal baseline) mattered, not just amplitude. Business-time
    rescaling still recovers true alpha within +/-0.06 regardless.
    """
    _run_seasonal_justification_case(amplitude=1.05, seed=7)


def _run_seasonal_justification_case(amplitude: float, seed: int) -> None:
    true_mu, true_alpha, true_beta = 0.4, 0.35, 1.5
    t_end_s = 10 * 24 * 3600.0  # ~10 days, in seconds (Hawkes sim uses float seconds)
    n_bins = 48

    shape = _planted_daily_shape(n_bins, amplitude=amplitude)
    hawkes_times_s = simulate_seasonal_hawkes_exp(
        true_mu, true_alpha, true_beta, t_end_s, shape, seed=seed
    )
    assert hawkes_times_s.size > 100, "need enough events for a stable fit"

    # --- Raw clock-time fit: seasonality inflates alpha. ---
    raw_t_end_s = float(hawkes_times_s[-1]) + 1.0
    raw_fit = fit_hawkes_exp(hawkes_times_s, raw_t_end_s)

    assert raw_fit.alpha > true_alpha + 0.08, (
        f"amplitude={amplitude}: expected seasonality to inflate raw-fit alpha above "
        f"true+0.08={true_alpha + 0.08}, got {raw_fit.alpha}"
    )

    # --- Business-time fit: rescaling removes the seasonality confound. ---
    # Convert to int64 epoch-ms so eventtime's ms-based API applies, anchored
    # at an arbitrary epoch far from 0 so day boundaries are non-trivial.
    epoch_anchor_ms = 1_700_000_000_000
    ts_ms = (epoch_anchor_ms + hawkes_times_s * 1000.0).astype(np.int64)
    profile = intraday_rate_profile(ts_ms, n_bins=n_bins)
    tau_s = rescale_to_business_time(ts_ms, profile)
    tau_t_end_s = float(tau_s[-1]) + 1.0
    rescaled_fit = fit_hawkes_exp(tau_s, tau_t_end_s)

    assert abs(rescaled_fit.alpha - true_alpha) < 0.06, (
        f"amplitude={amplitude}: expected business-time fit alpha within +/-0.06 of "
        f"true alpha={true_alpha}, got {rescaled_fit.alpha}"
    )


# ---------------------------------------------------------------------------
# Secondary control test: post-hoc thinning by a daily profile, kept as
# CONSTRUCTION-BIAS DOCUMENTATION, not as the justification test.
#
# Task 2's first attempt approximated "a Hawkes process with seasonal
# baseline" by simulating an unseasonal (constant-mu) Hawkes process and then
# thinning its realized events by an independent time-of-day acceptance
# probability. That approximation is systematically biased: thinning
# discards genuinely-excited child events along with baseline events, which
# depresses a branching-ratio estimate on its own, independent of any
# seasonality confound. This is preserved here (not as the justification
# test, since it validates against a construction that has its own known
# bias baked in) to document that finding with the actual measured numbers,
# for anyone tempted to reintroduce thinning as a shortcut later.
# ---------------------------------------------------------------------------


def _thin_by_daily_profile(ts: np.ndarray, shape: np.ndarray, seed: int) -> np.ndarray:
    """Thin a Hawkes event-time array by the (normalized) daily shape.

    Thinning a Hawkes process by an independent, deterministic time-of-day
    acceptance probability is an APPROXIMATION of a true "Hawkes process
    with seasonal baseline mu(t)": it modulates realized event density by
    time-of-day but does not change the underlying self-excitation kernel's
    dependence on the (unthinned) parent event times, and thinning removes
    some real excited children along with baseline events -- see
    `test_thinning_construction_has_documented_downward_bias` below for the
    measured magnitude of that bias.
    """
    rng = np.random.default_rng(seed)
    n_bins = shape.size
    bin_width_ms = _DAY_MS / n_bins
    shape_norm = shape / shape.max()  # in (0, 1], so it's a valid acceptance prob

    tod_ms = ts % _DAY_MS
    bin_idx = np.minimum((tod_ms / bin_width_ms).astype(np.int64), n_bins - 1)
    accept_prob = shape_norm[bin_idx]
    accept = rng.random(ts.size) <= accept_prob
    return ts[accept]


def test_thinning_construction_has_documented_downward_bias():
    """CONSTRUCTION-BIAS DOCUMENTATION (not the justification test).

    Measured numbers from Task 2's first attempt, reproduced here: thinning
    an unseasonal Hawkes simulation by a daily profile (amplitude=1.4,
    thinning seed=8) gives a business-time-rescaled fit of alpha ~ 0.28-0.29
    against true_alpha=0.35 -- a reproducible ~0.06-0.07 downward bias (std
    ~0.001 across 5 independent thinning seeds), confirmed by a control
    experiment: UNIFORM (non-seasonal) random thinning of the same simulated
    series, at the same overall keep-fraction, drops fitted alpha to ~0.25
    entirely on its own, with zero seasonality confound present. This
    isolates the bias to thinning's removal of genuinely-excited child
    events, not to any flaw in `rescale_to_business_time`'s math. This is
    why the justification test above uses `simulate_seasonal_hawkes_exp`
    (a genuine time-varying-baseline generative model) instead.
    """
    true_mu, true_alpha, true_beta = 0.4, 0.35, 1.5
    t_end_s = 10 * 24 * 3600.0
    n_bins = 48

    hawkes_times_s = simulate_hawkes_exp(true_mu, true_alpha, true_beta, t_end_s, seed=7)
    epoch_anchor_ms = 1_700_000_000_000
    ts_ms = (epoch_anchor_ms + hawkes_times_s * 1000.0).astype(np.int64)

    shape = _planted_daily_shape(n_bins, amplitude=1.4)
    thinned_ts_ms = _thin_by_daily_profile(ts_ms, shape, seed=8)
    assert thinned_ts_ms.size > 100, "need enough events left after thinning for a stable fit"

    profile = intraday_rate_profile(thinned_ts_ms, n_bins=n_bins)
    tau_s = rescale_to_business_time(thinned_ts_ms, profile)
    tau_t_end_s = float(tau_s[-1]) + 1.0
    rescaled_fit = fit_hawkes_exp(tau_s, tau_t_end_s)

    # Document the reproducible downward bias: rescaled alpha lands well
    # below true_alpha specifically because of thinning's information loss,
    # not because rescaling failed to remove the seasonality confound.
    assert 0.20 < rescaled_fit.alpha < 0.32, (
        f"expected the thinning construction's documented downward bias "
        f"(alpha in (0.20, 0.32) vs true_alpha={true_alpha}), got {rescaled_fit.alpha}"
    )


# ---------------------------------------------------------------------------
# Group 4: determinism + edge cases.
# ---------------------------------------------------------------------------


def test_intraday_rate_profile_is_deterministic():
    ts = np.array([0, 1000, 90_000, _DAY_MS + 500, 2 * _DAY_MS + 12_345], dtype=np.int64)
    p1 = intraday_rate_profile(ts, n_bins=48)
    p2 = intraday_rate_profile(ts, n_bins=48)
    np.testing.assert_array_equal(p1, p2)


def test_rescale_to_business_time_is_deterministic():
    ts = np.array([0, 1000, 90_000, _DAY_MS + 500, 2 * _DAY_MS + 12_345], dtype=np.int64)
    profile = intraday_rate_profile(ts, n_bins=48)
    tau1 = rescale_to_business_time(ts, profile)
    tau2 = rescale_to_business_time(ts, profile)
    np.testing.assert_array_equal(tau1, tau2)


def test_rescale_to_business_time_anchors_first_event_at_zero():
    ts = np.array([12_345, 90_000, _DAY_MS + 500], dtype=np.int64)
    profile = intraday_rate_profile(ts, n_bins=48)
    tau = rescale_to_business_time(ts, profile)
    assert tau[0] == 0.0


def test_intraday_rate_profile_rejects_empty_array():
    with pytest.raises(ValueError):
        intraday_rate_profile(np.array([], dtype=np.int64))


def test_rescale_to_business_time_rejects_empty_array():
    with pytest.raises(ValueError):
        rescale_to_business_time(np.array([], dtype=np.int64), np.ones(48))


def test_rescale_to_business_time_rejects_single_event():
    with pytest.raises(ValueError):
        rescale_to_business_time(np.array([1000], dtype=np.int64), np.ones(48))


def test_rescale_to_business_time_rejects_non_positive_profile():
    ts = np.array([0, 1000], dtype=np.int64)
    bad_profile = np.ones(48)
    bad_profile[3] = 0.0
    with pytest.raises(ValueError):
        rescale_to_business_time(ts, bad_profile)


def test_rescale_to_business_time_rejects_empty_profile():
    ts = np.array([0, 1000], dtype=np.int64)
    with pytest.raises(ValueError):
        rescale_to_business_time(ts, np.array([]))


def test_rescale_to_business_time_rejects_nan_in_profile():
    """HIGH-severity fix: non-finite profile values must be rejected
    explicitly rather than silently propagating NaN through the integral."""
    ts = np.array([0, 1000], dtype=np.int64)
    bad_profile = np.ones(48)
    bad_profile[5] = np.nan
    with pytest.raises(ValueError):
        rescale_to_business_time(ts, bad_profile)


def test_rescale_to_business_time_rejects_inf_in_profile():
    ts = np.array([0, 1000], dtype=np.int64)
    bad_profile = np.ones(48)
    bad_profile[5] = np.inf
    with pytest.raises(ValueError):
        rescale_to_business_time(ts, bad_profile)


def test_rescale_to_business_time_rejects_unsorted_ts():
    """MEDIUM-severity fix: unsorted ts must be rejected explicitly, since
    the day/bin decomposition and the tau[0]-anchor both assume ts[0] is the
    earliest event -- an unsorted input would otherwise silently produce a
    nonsensical business-time axis instead of erroring."""
    ts = np.array([2000, 1000, 3000], dtype=np.int64)
    profile = np.ones(48)
    with pytest.raises(ValueError):
        rescale_to_business_time(ts, profile)
