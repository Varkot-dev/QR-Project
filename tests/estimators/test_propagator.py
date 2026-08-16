"""Tests for the propagator kernel deconvolution estimator (Phase 2, Task 1).

Contract under test (see task-1-brief.md and the follow-up review fixes):
1. White-noise signs: C = delta, so kappa recovered == b exactly (up to noise
   used to build the synthetic mids).
2. Long-memory signs (fractional_signs d=0.35): plant a power-law kernel
   G0(l) = l^(-0.35), build mids by convolution, estimate b and the sign ACF
   from the DATA (not the theory), deconvolve, and recover beta close to
   0.35 via fit_power_law on the cumulative kernel, across multiple seeds.
   This is the key test: it also shows that the naive response function (no
   deconvolution) does NOT recover 0.35 as well as the deconvolved kernel
   does, on the same data.
3. Guards: mismatched lengths, singular ACF, unnormalized ACF (acf[0] != 1),
   and too-few-samples-per-lag all raise ValueError.
4. kernel_exponent_blocked reports a block-bootstrap uncertainty (block_sd)
   that must be used instead of fit_power_law's OLS stderr, which is known
   to badly understate beta_hat's true uncertainty.
"""
from __future__ import annotations

import numpy as np
import pytest

from microstructure.estimators.acf import fit_power_law, sign_acf
from microstructure.estimators.propagator import (
    cumulative_kernel,
    deconvolve_kernel,
    kernel_exponent_blocked,
    sign_price_cross_cov,
)
from microstructure.estimators.response import response_function
from microstructure.synthetic import fractional_signs, iid_signs


def _fft_convolve_full(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Linear ("full") convolution via zero-padded FFT (mirrors synthetic.py)."""
    out_len = len(a) + len(b) - 1
    size = 1 << (out_len - 1).bit_length()
    fa = np.fft.rfft(a, n=size)
    fb = np.fft.rfft(b, n=size)
    return np.fft.irfft(fa * fb, n=size)[:out_len]


def _mids_from_kernel(signs: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """mid strictly before event t = sum_{k<t} signs[k] * kernel[t-k]."""
    n = signs.size
    full = _fft_convolve_full(signs.astype(float), kernel)[:n]
    return full


class TestSignPriceCrossCov:
    def test_length_matches_max_lag(self):
        signs = iid_signs(1000, seed=1)
        dm = np.diff(signs.astype(float))
        b = sign_price_cross_cov(signs[:-1], dm, max_lag=20)
        assert b.shape == (20,)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            sign_price_cross_cov(np.ones(10), np.ones(9), max_lag=2)

    def test_mean_centering_matters_for_directional_flow(self):
        """MEDIUM 1: both series must be mean-centered like sign_acf.

        With directional (non-zero-mean) signs and dm both held constant,
        an uncentered cross-cov would equal mean(dm)*mean(signs); after
        centering it must be (near) zero since there is no genuine
        covariation once the means are removed.
        """
        n = 5000
        rng = np.random.default_rng(42)
        # Directional flow: signs biased towards +1, non-trivial mean.
        signs = np.where(rng.random(n) < 0.8, 1.0, -1.0)
        dm = 0.5 + 0.01 * rng.standard_normal(n)  # mean 0.5, weak noise, no real link to signs

        assert abs(signs.mean()) > 0.4  # confirm this is a directional-flow scenario
        assert abs(dm.mean()) > 0.3

        b = sign_price_cross_cov(signs, dm, max_lag=5)
        # Centered cross-covariance with no real signs-dm link should be ~0,
        # not mean(dm)*mean(signs) ~= 0.5*0.6 = 0.3 (mean^2 discrepancy).
        assert np.all(np.abs(b) < 0.05)


class TestDeconvolveWhiteNoise:
    def test_white_noise_signs_kappa_equals_b(self):
        """C = delta (identity Toeplitz) => kappa recovered == b.

        Per the brief: dm is built directly by convolving signs with an
        arbitrary decaying kappa0 (the MA coefficients themselves), i.e.
        dm[t] = sum_n kappa0[n] * signs[t-n] + small noise. For iid signs
        the sign ACF is a delta function, so the Toeplitz system is the
        identity and the recovered kappa must equal b, which in turn must
        equal kappa0 (up to the injected noise).
        """
        n = 100_000
        signs = iid_signs(n, seed=10)
        rng = np.random.default_rng(11)

        # Plant an arbitrary decaying kappa0 (the MA/propagator coefficients).
        L = 50
        ell = np.arange(1, L + 1, dtype=float)
        kappa0 = 0.3 * 0.9 ** (ell - 1)  # geometric decay, arbitrary but decaying

        dm = _fft_convolve_full(signs.astype(float), kappa0)[:n]
        dm = dm + 0.001 * rng.standard_normal(n)  # small noise
        signs_aligned = signs

        b = sign_price_cross_cov(signs_aligned, dm, max_lag=L)
        acf = sign_acf(signs_aligned, max_lag=L - 1)

        kappa = deconvolve_kernel(b, acf, n_samples=n)
        assert kappa.shape == b.shape
        # Since signs are iid, acf[k] ~ 0 for k>0, so kappa ~= b == kappa0.
        assert np.max(np.abs(kappa - kappa0)) < 0.02


class TestDeconvolveLongMemory:
    def _run_recovery(self, seed: int):
        """Shared machinery for the key long-memory recovery test, per seed."""
        n = 300_000
        d = 0.35
        signs = fractional_signs(n, d=d, seed=seed)

        # Plant G0(l) = l^(-0.35) as the convolution kernel itself (mid_t
        # construction from Phase 1), truncated at n_lags terms for speed.
        n_lags = 2000
        ell = np.arange(1, n_lags + 1, dtype=float)
        G0 = ell**-0.35

        mids = _mids_from_kernel(signs, np.concatenate(([0.0], G0)))
        dm = np.diff(mids)
        signs_aligned = signs[:-1]

        L = 300
        b = sign_price_cross_cov(signs_aligned, dm, max_lag=L)
        acf = sign_acf(signs_aligned, max_lag=L - 1)

        kappa = deconvolve_kernel(b, acf, n_samples=signs_aligned.size)
        G = cumulative_kernel(kappa)
        assert G.shape == (L,)

        deconv_fit = fit_power_law(G, lo=5, hi=150)
        beta_hat = deconv_fit.exponent

        # Contrast: naive response function on the SAME data, same lag window.
        r = response_function(signs_aligned, mids[:-1], max_lag=150)
        naive_fit = fit_power_law(np.abs(r), lo=5, hi=150)
        naive_exponent = naive_fit.exponent

        return beta_hat, naive_exponent

    def test_recovers_power_law_kernel_and_beats_naive_response(self):
        """KEY TEST: deconvolution recovers beta=0.35 kernel across seeds;
        naive response does not, and the recovery is stable across seeds
        (LOW 2: loop 3 seeds, check per-seed tolerance AND across-seed sd).
        """
        target = 0.35
        seeds = [20, 21, 22]
        beta_hats = []
        for seed in seeds:
            beta_hat, naive_exponent = self._run_recovery(seed)
            beta_hats.append(beta_hat)

            deconv_err = abs(beta_hat - target)
            naive_err = abs(naive_exponent - target)

            assert deconv_err < 0.07, (
                f"seed={seed}: beta_hat={beta_hat} not within 0.07 of {target}"
            )
            assert naive_err > deconv_err, (
                f"seed={seed}: naive response exponent {naive_exponent} should be "
                f"farther from {target} than deconvolved beta_hat={beta_hat}"
            )

        sd = float(np.std(np.array(beta_hats), ddof=1))
        assert sd < 0.03, f"across-seed sd of beta_hat too large: {sd} (values={beta_hats})"


class TestCumulativeKernel:
    def test_g0_is_zero_and_partial_sums(self):
        kappa = np.array([1.0, 2.0, 3.0, 4.0])
        G = cumulative_kernel(kappa)
        assert G[0] == 0.0
        assert G.shape == kappa.shape
        assert np.allclose(G, [0.0, 1.0, 3.0, 6.0])


class TestGuards:
    def test_mismatched_lengths_raises(self):
        b = np.ones(10)
        acf = np.ones(5)  # wrong length: must be len(b)
        with pytest.raises(ValueError):
            deconvolve_kernel(b, acf, n_samples=10_000)

    def test_singular_acf_raises(self):
        # Constant ACF of 1 at all lags => every row of the Toeplitz matrix
        # is all-ones => rank 1, singular for L > 1.
        L = 10
        b = np.ones(L)
        acf = np.ones(L)
        with pytest.raises(ValueError):
            deconvolve_kernel(b, acf, n_samples=10_000)

    def test_unnormalized_acf_raises(self):
        """MEDIUM 2: acf[0] must equal 1.0 (normalized ACF), else ValueError."""
        L = 10
        b = np.ones(L)
        acf = np.zeros(L)
        acf[0] = 2.0  # raw autocovariance, not normalized
        with pytest.raises(ValueError):
            deconvolve_kernel(b, acf, n_samples=10_000)

    def test_low_sample_to_lag_ratio_raises(self):
        """HIGH 1: n_samples/len(b) below the 100 floor must raise, even
        though the Toeplitz matrix here is well-conditioned (an identity-
        like near-diagonal ACF), demonstrating the rank/condition checks
        alone cannot catch this failure mode.
        """
        L = 300
        rng = np.random.default_rng(99)
        acf = np.zeros(L)
        acf[0] = 1.0
        acf[1:] = 0.01 * rng.standard_normal(L - 1)  # tiny, near-delta ACF
        b = rng.standard_normal(L)
        n_samples = 1_000  # ratio = 1000/300 = 3.33, far below 100

        with pytest.raises(ValueError, match="n_samples"):
            deconvolve_kernel(b, acf, n_samples=n_samples)

    def test_low_sample_ratio_bypassed_with_allow_low_sample(self):
        L = 10
        rng = np.random.default_rng(1)
        acf = np.zeros(L)
        acf[0] = 1.0
        acf[1:] = 0.01 * rng.standard_normal(L - 1)
        b = rng.standard_normal(L)
        # Should not raise when explicitly bypassed.
        kappa = deconvolve_kernel(b, acf, n_samples=50, allow_low_sample=True)
        assert kappa.shape == (L,)


class TestKernelExponentBlocked:
    def test_returns_n_blocks_exponents_with_positive_sd_and_accurate_full_sample(self):
        n = 300_000
        d = 0.35
        signs = fractional_signs(n, d=d, seed=20)

        n_lags = 2000
        ell = np.arange(1, n_lags + 1, dtype=float)
        G0 = ell**-0.35
        mids = _mids_from_kernel(signs, np.concatenate(([0.0], G0)))
        dm = np.diff(mids)
        signs_aligned = signs[:-1]

        result = kernel_exponent_blocked(signs_aligned, dm, max_lag=300, n_blocks=5, fit_lo=5)

        assert len(result.block_exponents) == 5
        assert result.block_sd > 0.0
        assert result.n_samples == signs_aligned.size
        assert abs(result.exponent - 0.35) < 0.07
