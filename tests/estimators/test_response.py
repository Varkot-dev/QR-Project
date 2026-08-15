import numpy as np
import pytest

from microstructure.estimators.response import response_function
from microstructure.synthetic import iid_signs


def _mids_from_kernel(signs: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """mid before event t = sum over k<t of signs[k] * kernel[t-k]."""
    n = signs.size
    full = np.convolve(signs.astype(float), kernel, mode="full")[:n]
    mids = np.zeros(n)
    mids[:] = full  # strictly-before convention: full[t] already excludes signs[t] due to kernel[0]=0
    return mids


def test_permanent_impact_gives_flat_response():
    signs = iid_signs(200_000, seed=5)
    c = 0.7  # each event permanently moves mid by c*sign, forever
    mids = c * np.concatenate(([0.0], np.cumsum(signs)[:-1]))  # mid strictly before event t
    r = response_function(signs, mids, max_lag=20)
    assert r[0] == 0.0
    assert np.all(np.abs(r[1:] - c) < 0.02)  # theory: R(l) = c for all l >= 1


def test_exponential_kernel_recovered():
    signs = iid_signs(400_000, seed=6)
    g0, phi = 0.5, 0.8
    taus = np.arange(1, 400)
    kernel = np.concatenate(([0.0], g0 * phi ** (taus - 1)))
    mids = _mids_from_kernel(signs, kernel)
    r = response_function(signs, mids, max_lag=10)
    for lag in range(1, 11):
        assert abs(r[lag] - g0 * phi ** (lag - 1)) < 0.02  # R(l) = g(l) for iid signs


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        response_function(np.ones(10), np.ones(9), max_lag=2)
