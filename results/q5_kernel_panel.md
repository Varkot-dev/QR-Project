# Q5: kernel panel — critical balance across the cross-section

## Methodology

For each symbol in the requested panel, one week of aggTrades (2023-06-01..2023-06-07, from the monthly period `2023-06`) is loaded and filtered to that timestamp range, then joined to the bookTicker mid strictly before each event (`events_with_prior_mid`). Symbols are processed one at a time; any per-symbol failure (missing parquet, no events in range, or `kernel_exponent_blocked`'s n/L≥100 sample-sufficiency guard raising for a too-thin symbol) is caught and logged into `failures` without aborting the run for the remaining symbols.

**gamma_week**: the sign-ACF power-law exponent over the same week's trades (`fit_power_law(sign_acf(signs, 1000), lo=10, hi=500)`), Q1's order-flow-memory statistic.
**beta (deconvolved kernel exponent)**: `kernel_exponent_blocked` (Phase-2 Task 1's propagator deconvolution) recovers the BARE impact kernel by solving the Toeplitz system `sign_price_cross_cov = sign_ACF ⊛ kappa`, separating the kernel from the confound of order-flow memory that a naive read of the response function R(ℓ) would mix in. `beta_block_sd` (block-bootstrap standard deviation across 5 contiguous blocks) is used for uncertainty, NOT the fitted power law's OLS stderr, which `propagator.py`'s docstring measures to understate true uncertainty by roughly 6.8x on synthetic long-memory data.
**Critical balance**: the Bouchaud et al. (2004) propagator-diffusivity relation predicts beta = (1 - gamma_week) / 2 for a LINEAR propagator model whose accumulated price response grows no faster than diffusively (dm[t] = sum_n kappa[n] * signs[t-n] + noise, i.e. impacts superpose additively across events — see `propagator.py`'s module docstring for the exact model assumption). balance_delta = beta - (1-gamma_week)/2 measures the signed departure from that prediction per symbol.

**Judgement rule**: |balance_delta| <= 2*max(beta_block_sd, 0.04) => "consistent", else "violated". The 0.04 floor is not an arbitrary safety margin — `propagator.py`'s docstring for `kernel_exponent_blocked` reports a measured systematic finite-L bias of roughly +0.03 to +0.04 in the recovered beta at L=300 (20-seed Monte Carlo on `fractional_signs(d=0.35)`), i.e. even a genuinely balanced symbol's beta_hat typically reads ~0.03-0.04 too high purely from finite-sample deconvolution bias. Without this floor, a low-noise symbol (small beta_block_sd) with a truly-zero balance_delta could be flagged "violated" by nothing more than the method's own known, already-quantified bias — a dishonest false positive. Flooring the band at the measured bias scale is the honest choice: it states plainly that departures smaller than the method's own bias cannot be distinguished from zero, rather than implying a precision the estimate does not have.

## Run summary

Requested: 16. Successful: 16. Failed: 0. Of the successful symbols: **12 consistent**, **4 violated**.

## Panel table (sorted by n_events)

| symbol | n_events | γ̂_week | β̂ | β̂ block_sd | Δ (balance) | verdict | R(1) | drop_rate |
|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 4,703,378 | 0.3603 | 0.2399 | 0.0601 | -0.0799 | consistent | 0.112142 | 0.0000% |
| ETHUSDT | 3,112,419 | 0.2244 | 0.2895 | 0.0614 | -0.0983 | consistent | 0.010752 | 0.0000% |
| 1000PEPEUSDT | 2,060,998 | 0.4539 | 0.1625 | 0.0416 | -0.1106 | violated | 0.000000 | 0.0000% |
| XRPUSDT | 1,608,574 | 0.3524 | 0.3633 | 0.0615 | +0.0395 | consistent | 0.000010 | 0.0000% |
| LTCUSDT | 1,351,376 | 0.4184 | 0.3363 | 0.0706 | +0.0455 | consistent | 0.002099 | 0.0000% |
| OPUSDT | 1,279,604 | 0.4215 | 0.1343 | 0.0655 | -0.1549 | violated | 0.000076 | 0.0000% |
| SOLUSDT | 1,117,502 | 0.3038 | 0.2608 | 0.0218 | -0.0873 | violated | 0.000646 | 0.0000% |
| INJUSDT | 1,084,554 | 0.4282 | 0.2509 | 0.0493 | -0.0350 | consistent | 0.000509 | 0.0000% |
| SUIUSDT | 1,045,883 | 0.4433 | 0.2100 | 0.0370 | -0.0684 | consistent | 0.000060 | 0.0000% |
| ARBUSDT | 911,294 | 0.3436 | 0.2315 | 0.0346 | -0.0967 | violated | 0.000052 | 0.0000% |
| DOGEUSDT | 874,041 | 0.2109 | 0.4802 | 0.2025 | +0.0856 | consistent | 0.000001 | 0.0000% |
| EDUUSDT | 754,422 | 0.4326 | 0.1697 | 0.0571 | -0.1140 | consistent | 0.000092 | 0.0000% |
| APTUSDT | 667,892 | 0.5123 | 0.1911 | 0.0502 | -0.0527 | consistent | 0.000488 | 0.0000% |
| LINKUSDT | 514,284 | 0.3999 | 0.4183 | 0.0840 | +0.1183 | consistent | 0.000166 | 0.0000% |
| IDUSDT | 509,325 | 0.4837 | 0.2333 | 0.0508 | -0.0249 | consistent | 0.000046 | 0.0000% |
| BCHUSDT | 401,825 | 0.3565 | 0.3827 | 0.0412 | +0.0609 | consistent | 0.003580 | 0.0000% |

## Findings

12 of 16 symbols (75%) land within the balance band and 4 do not — critical balance holds for SOME but not all of the panel. Whether the split correlates with activity (n_events) or other symbol characteristics is visible in the panel table and right-hand plot above; no such correlation is asserted here beyond what the table shows, since a resolution below 16 points is not enough to fit a reliable trend.

## Caveats

- **7-day window** (2023-06-01..2023-06-07): one specific week is one specific market regime; per Phase 1.5 diagnostics, order-flow memory statistics are regime-dependent, and these results may not generalize to other weeks or volatility regimes.
- **L1 mids only**: the mid price used throughout is the best-bid/best-ask midpoint from bookTicker (top-of-book only); no order-book depth is used, so impact through queue depletion or hidden liquidity is not captured.
- **Linear-propagator model assumption**: the deconvolved kernel beta relies entirely on `propagator.py`'s model, dm[t] = sum_n kappa[n]*signs[t-n] + noise — i.e. that each signed event's price impact superposes additively and linearly with all other events' impacts. If real impact is nonlinear (e.g. saturating, or state-dependent on spread/depth), the deconvolved beta is a linear-model artifact, not evidence for or against a nonlinear propagator. The critical-balance relation itself (beta = (1-gamma)/2) is ALSO a linear/diffusive-propagator prediction (Bouchaud et al. 2004); a "violated" verdict is equally consistent with (a) the panel's true impact process being nonlinear, and (b) the linear model holding but with a genuinely different beta-gamma relationship than the diffusivity constraint predicts. This analysis cannot distinguish those two explanations.
- **0.04 bias floor**: see Methodology — this is the measured finite-L deconvolution bias at L=300 from Task 1's synthetic validation, not a generic safety margin; the true bias at this panel's actual max_lag (300) may differ from the L=300 measurement it is based on.
- **beta_block_sd** comes from only 5 contiguous non-overlapping blocks per symbol; with few blocks, the block standard deviation is itself a noisy estimate of uncertainty, and is not a formal confidence interval.
