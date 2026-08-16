"""Tests for the propagator kernel deconvolution estimator (Phase 2, Task 1).

Contract under test (see task-1-brief.md):
1. White-noise signs: C = delta, so kappa recovered == b exactly (up to noise
   used to build the synthetic mids).
2. Long-memory signs (fractional_signs d=0.35): plant a power-law kernel
   G0(l) = l^(-0.35), build mids by convolution, estimate b and the sign ACF
   from the DATA, deconvolve, and recover beta close to 0.35 via
   fit_power_law on the cumulative kernel. This is the key test: it also
   shows that the naive response function (no deconvolution) does NOT
   recover 0.35 as well as the deconvolved kernel does, on the same data.
3. Shape guards: mismatched lengths and singular ACF both raise ValueError.
"""
from __future__ import annotations

import numpy as np
import pytest

from microstructure.estimators.acf import fit_power_law, sign_acf
from microstructure.estimators.propagator import (
    cumulative_kernel,
    deconvolve_kernel,
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

        kappa = deconvolve_kernel(b, acf)
        assert kappa.shape == b.shape
        # Since signs are iid, acf[k] ~ 0 for k>0, so kappa ~= b == kappa0.
        assert np.max(np.abs(kappa - kappa0)) < 0.02


class TestDeconvolveLongMemory:
    def test_recovers_power_law_kernel_and_beats_naive_response(self):
        """KEY TEST: deconvolution recovers beta=0.35 kernel; naive response does not.

        Per the brief and Phase-1's kernel construction: mids are built by
        convolving signs directly with the power-law kernel
        G0(l) = l^(-0.35), l>=1 (truncated to n_lags terms). kappa0, the
        thing sign_price_cross_cov/deconvolve_kernel actually recover, is
        the first difference of G0. cumulative_kernel then undoes that
        differencing, so fitting fit_power_law on the recovered cumulative
        kernel should recover exponent ~0.35.
        """
        n = 300_000
        d = 0.35
        signs = fractional_signs(n, d=d, seed=20)

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

        kappa = deconvolve_kernel(b, acf)
        G = cumulative_kernel(kappa)
        assert G.shape == (L,)

        deconv_fit = fit_power_law(G, lo=5, hi=150)
        beta_hat = deconv_fit.exponent

        # Contrast: naive response function on the SAME data, same lag window.
        r = response_function(signs_aligned, mids[:-1], max_lag=150)
        naive_fit = fit_power_law(np.abs(r), lo=5, hi=150)
        naive_exponent = naive_fit.exponent

        target = 0.35
        deconv_err = abs(beta_hat - target)
        naive_err = abs(naive_exponent - target)

        assert deconv_err < 0.07, f"beta_hat={beta_hat} not within 0.07 of {target}"
        assert naive_err > deconv_err, (
            f"naive response exponent {naive_exponent} should be farther from "
            f"{target} than deconvolved beta_hat={beta_hat}"
        )


class TestCumulativeKernel:
    def test_g0_is_zero_and_partial_sums(self):
        kappa = np.array([1.0, 2.0, 3.0, 4.0])
        G = cumulative_kernel(kappa)
        assert G[0] == 0.0
        assert G.shape == kappa.shape
        assert np.allclose(G, [0.0, 1.0, 3.0, 6.0])


class TestShapeGuards:
    def test_mismatched_lengths_raises(self):
        b = np.ones(10)
        acf = np.ones(5)  # wrong length: must be len(b)
        with pytest.raises(ValueError):
            deconvolve_kernel(b, acf)

    def test_singular_acf_raises(self):
        # Constant ACF of 1 at all lags => every row of the Toeplitz matrix
        # is all-ones => rank 1, singular for L > 1.
        L = 10
        b = np.ones(L)
        acf = np.ones(L)
        with pytest.raises(ValueError):
            deconvolve_kernel(b, acf)
