import numpy as np

from microstructure.estimators.acf import fit_power_law, sign_acf
from microstructure.synthetic import fractional_signs, iid_signs, markov_signs


def test_acf_lag0_is_one_and_iid_is_zero():
    a = sign_acf(iid_signs(200_000, seed=1), max_lag=100)
    assert a.shape == (101,)
    assert a[0] == 1.0
    assert np.all(np.abs(a[1:]) < 0.02)


def test_acf_matches_naive_computation():
    x = markov_signs(5_000, p_repeat=0.7, seed=4).astype(float)
    a = sign_acf(x, max_lag=5)
    xc = x - x.mean()
    for k in range(1, 6):
        naive = (xc[:-k] * xc[k:]).mean() / (xc * xc).mean()
        assert abs(a[k] - naive) < 1e-6  # FFT path must equal the definition


def test_acf_markov_known_answer():
    a = sign_acf(markov_signs(400_000, p_repeat=0.75, seed=2), max_lag=3)
    for k, expected in [(1, 0.5), (2, 0.25), (3, 0.125)]:
        assert abs(a[k] - expected) < 0.02


def test_power_law_fit_exact():
    lags = np.arange(201, dtype=float)
    y = np.zeros(201)
    y[1:] = 2.0 * lags[1:] ** -0.4
    fit = fit_power_law(y, lo=10, hi=200)
    assert abs(fit.exponent - 0.4) < 1e-9
    assert fit.stderr < 1e-9


def test_power_law_fit_recovers_fractional_gamma():
    a = sign_acf(fractional_signs(400_000, d=0.4, seed=3), max_lag=200)
    fit = fit_power_law(a, lo=10, hi=200)
    assert 0.14 < fit.exponent < 0.26  # theory: gamma = 1 - 2d = 0.2


def test_power_law_fit_skips_nonpositive_values():
    y = np.zeros(101)
    y[1:] = 1.5 * np.arange(1, 101, dtype=float) ** -0.3
    y[50] = -0.001  # one noisy negative point must not crash or poison the fit
    fit = fit_power_law(y, lo=10, hi=100)
    assert abs(fit.exponent - 0.3) < 0.02
