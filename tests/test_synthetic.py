import numpy as np

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


def test_fractional_signs_long_memory_slower_than_markov():
    s = fractional_signs(400_000, d=0.4, seed=3)
    # long memory: ACF at lag 50 must remain clearly positive,
    # whereas any short-memory chain with same lag-1 ACF would be ~0 by lag 50
    assert _acf(s, 1) > 0.05
    assert _acf(s, 50) > 0.02
