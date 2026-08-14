import numpy as np
import pytest

from microstructure.synthetic import fractional_signs, iid_signs, markov_signs


def _acf(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    return float((x[:-lag] * x[lag:]).mean() / (x * x).mean())


def test_iid_signs_values_and_no_memory():
    s = iid_signs(200_000, seed=1)
    assert set(np.unique(s)) == {-1, 1}
    assert abs(_acf(s, 1)) < 0.01


def test_iid_signs_reproducible():
    assert np.array_equal(iid_signs(1000, seed=7), iid_signs(1000, seed=7))


def test_markov_signs_match_theoretical_acf():
    p = 0.75  # theoretical ACF(k) = (2p-1)^k = 0.5^k
    s = markov_signs(400_000, p_repeat=p, seed=2)
    for k, expected in [(1, 0.5), (2, 0.25), (3, 0.125)]:
        assert abs(_acf(s, k) - expected) < 0.02


@pytest.mark.parametrize("p_repeat", [0.0, 1.0, -0.1, 1.1])
def test_markov_signs_rejects_p_repeat_outside_open_unit_interval(p_repeat):
    with pytest.raises(ValueError):
        markov_signs(10, p_repeat=p_repeat, seed=1)


@pytest.mark.parametrize("d", [0.0, 0.5, -0.1, 0.6])
def test_fractional_signs_rejects_d_outside_valid_range(d):
    with pytest.raises(ValueError):
        fractional_signs(10, d=d, seed=1)


def test_fractional_signs_smoke_long_memory_slower_than_markov():
    """Fast smoke check: long memory persists much further than a short-memory chain."""
    s = fractional_signs(400_000, d=0.4, seed=3)
    # long memory: ACF at lag 50 must remain clearly positive,
    # whereas any short-memory chain with same lag-1 ACF would be ~0 by lag 50
    assert _acf(s, 1) > 0.05
    assert _acf(s, 50) > 0.02


def test_fractional_signs_recovers_theoretical_power_law_exponent():
    """Real exponent-recovery test.

    Theory: sign ACF of FARIMA(0, d, 0) noise decays as lag^-gamma with
    gamma = 1 - 2*d. At d=0.4, gamma = 0.20. Fit gamma via log-log linear
    regression of ACF vs lag over lags 10..200 (restricted to lags where
    the empirical ACF is positive, since log is undefined otherwise).

    This test fails against the old 2000-term MA truncation (measured
    gamma_hat ~ 0.31, outside tolerance) and passes against the current
    50_000-term FFT-based implementation (measured gamma_hat ~ 0.25).
    """
    s = fractional_signs(400_000, d=0.4, seed=3)
    lags = np.arange(10, 201)
    acfs = np.array([_acf(s, int(k)) for k in lags])
    mask = acfs > 0
    log_lags = np.log(lags[mask])
    log_acfs = np.log(acfs[mask])
    slope, _intercept = np.polyfit(log_lags, log_acfs, 1)
    gamma_hat = -slope
    theoretical_gamma = 1 - 2 * 0.4  # 0.20
    assert abs(gamma_hat - theoretical_gamma) < 0.06, (
        f"gamma_hat={gamma_hat:.4f} outside tolerance of theory={theoretical_gamma:.2f}"
    )
