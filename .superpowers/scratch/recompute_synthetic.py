"""Recompute the synthetic deconvolved-vs-naive contrast cited as 0.3767 vs 0.0633."""
import sys
sys.path.insert(0, "/Users/varshithkotagiri/Projects/QR project/tests")

import numpy as np

sys.path.insert(0, "/Users/varshithkotagiri/Projects/QR project/src")
from microstructure.estimators.acf import fit_power_law, sign_acf
from microstructure.estimators.propagator import (
    cumulative_kernel,
    deconvolve_kernel,
)
from microstructure.estimators.response import response_function
from microstructure.synthetic import fractional_signs


def _fft_convolve_full(a, b):
    out_len = len(a) + len(b) - 1
    size = 1 << (out_len - 1).bit_length()
    fa = np.fft.rfft(a, n=size)
    fb = np.fft.rfft(b, n=size)
    return np.fft.irfft(fa * fb, n=size)[:out_len]


def run(seed):
    n = 300_000
    signs = fractional_signs(n, d=0.35, seed=seed)
    n_lags = 2000
    ell = np.arange(1, n_lags + 1, dtype=float)
    G0 = ell**-0.35
    mids = _fft_convolve_full(signs.astype(float), np.concatenate(([0.0], G0)))[:n]
    dm = np.diff(mids)
    signs_aligned = signs[:-1]
    L = 300
    b = sign_price_cross_cov(signs_aligned, dm, max_lag=L)
    acf = sign_acf(signs_aligned, max_lag=L - 1)
    kappa = deconvolve_kernel(b, acf, n_samples=signs_aligned.size)
    G = cumulative_kernel(kappa)
    beta_hat = fit_power_law(G, lo=5, hi=150).exponent
    r = response_function(signs_aligned, mids[:-1], max_lag=150)
    naive = fit_power_law(np.abs(r), lo=5, hi=150).exponent
    return beta_hat, naive


from microstructure.estimators.propagator import sign_price_cross_cov  # noqa: E402

for seed in (20, 21, 22):
    b, nv = run(seed)
    print(f"seed={seed}: deconv={b:.4f} (err {abs(b-0.35):.4f})  naive={nv:.4f} (err {abs(nv-0.35):.4f})")
