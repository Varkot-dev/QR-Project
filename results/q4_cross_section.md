# Q4: trades-side cross-section — results

## Methodology

For each symbol in the requested universe, one month (2023-06) of aggTrades is loaded and collapsed into aggressor-level events (`load_events`), producing a ±1 sign series per symbol. Symbols are processed one at a time and their frames released (`del`) before moving to the next symbol, bounding peak memory across the full universe. Symbols with fewer than `min_events` = 1,000,000 events are **skipped** (logged, reason "below min_events") rather than analyzed; any other per-symbol failure (missing parquet, malformed data, or any other exception) is caught and logged into `failures` with the symbol and the exception, and never aborts the run for the remaining symbols.

Per successful symbol, five statistics are computed on the sign series:

- **n_events**: activity, the number of aggressor events in the period.
- **γ̂ + OLS stderr**: sign-ACF power-law exponent, fit the same way as Q1 (`fit_power_law(sign_acf(signs, max_lag), lo=10, hi=max_lag//2)`).
- **lag-1 ACF**: `sign_acf(signs, max_lag)[1]`.
- **p_flip**: `P(sign_{t+1} != sign_t)`, the empirical fraction of consecutive sign flips — 0.5 is the no-persistence benchmark (independent coin flips).
- **zigzag amplitude**: Phase-1.5's definition (Q1b) — mean ACF at even lags 2,4,6,8,10 minus mean ACF at odd lags 1,3,5,7,9.
- **total_qty**: sum of aggressor-event quantity over the period.

**Cross-sectional regressions**: on the successful set, ordinary least squares (`np.polyfit`, degree 1, with intercept — not through-origin, since there is no reason to expect γ̂ or p_flip to vanish at zero activity) is used to regress (a) γ̂ on log10(n_events) and (b) p_flip on log10(n_events).

**Heteroskedasticity caveat**: each symbol's own γ̂ stderr comes from `fit_power_law`'s i.i.d.-residual OLS assumption applied to autocorrelated ACF values, which already understates that symbol's true uncertainty (documented in Q1). That understatement is *not* uniform across symbols — it scales with each symbol's own n_events and ACF shape — so per-symbol γ̂ noise is heteroskedastic across the cross-section. The cross-sectional regressions above therefore also violate the homoskedastic-residual assumption behind their own OLS stderr; the reported regression stderr/R² should be read as descriptive, not as a valid confidence interval on the true relationship.

## Run summary

Requested: 207. Successful: 121. Skipped (below min_events): 86. Failed: 0.

## Highest activity (10 by n_events)

| symbol | n_events | γ̂ | stderr | acf1 | p_flip | zigzag | total_qty |
|---|---|---|---|---|---|---|---|
| BTCUSDT | 21,816,890 | 0.4617 | 0.0014 | -0.1676 | 0.5838 | 0.1382 | 14,099,299.27 |
| ETHUSDT | 14,239,099 | 0.2858 | 0.0017 | 0.0284 | 0.4858 | 0.0518 | 91,463,493.06 |
| 1000PEPEUSDT | 12,675,042 | 0.4366 | 0.0056 | -0.0073 | 0.5036 | 0.0211 | 19,186,219,451,822.00 |
| BCHUSDT | 10,755,244 | 0.3047 | 0.0031 | -0.0300 | 0.5150 | 0.0287 | 136,317,272.99 |
| TOMOUSDT | 9,258,867 | 0.3195 | 0.0035 | 0.0652 | 0.4674 | 0.0018 | 7,609,674,178.00 |
| LINAUSDT | 8,757,826 | 0.2394 | 0.0019 | -0.2767 | 0.6383 | 0.1316 | 1,291,640,842,664.00 |
| MTLUSDT | 8,611,397 | 0.3633 | 0.0036 | 0.0108 | 0.4946 | 0.0157 | 8,681,192,161.00 |
| XRPUSDT | 7,617,768 | 0.3003 | 0.0035 | -0.2976 | 0.6486 | 0.1725 | 46,981,370,141.90 |
| SOLUSDT | 6,820,337 | 0.3367 | 0.0036 | 0.0789 | 0.4605 | 0.0054 | 1,062,588,660.00 |
| WAVESUSDT | 6,269,680 | 0.2579 | 0.0060 | 0.0753 | 0.4623 | 0.0069 | 3,843,915,541.50 |

## Lowest activity (10 by n_events)

| symbol | n_events | γ̂ | stderr | acf1 | p_flip | zigzag | total_qty |
|---|---|---|---|---|---|---|---|
| ANKRUSDT | 1,076,265 | 0.3320 | 0.0026 | 0.0717 | 0.4640 | 0.0083 | 40,417,703,153.00 |
| 1000FLOKIUSDT | 1,071,346 | 0.4728 | 0.0048 | 0.1135 | 0.4432 | -0.0014 | 28,148,926,120.00 |
| YFIUSDT | 1,071,320 | 0.3764 | 0.0037 | 0.2173 | 0.3913 | -0.0169 | 126,256.17 |
| DASHUSDT | 1,046,537 | 0.3382 | 0.0024 | 0.1402 | 0.4296 | -0.0021 | 19,932,201.40 |
| COTIUSDT | 1,033,785 | 0.3073 | 0.0018 | 0.2294 | 0.3852 | -0.0166 | 8,141,718,570.00 |
| STGUSDT | 1,024,583 | 0.3638 | 0.0035 | 0.1965 | 0.4017 | -0.0176 | 937,434,891.00 |
| CTSIUSDT | 1,022,623 | 0.2378 | 0.0013 | 0.1118 | 0.4436 | 0.0061 | 7,670,616,899.00 |
| GALABUSD | 1,022,598 | 1.4294 | 0.0684 | 0.3842 | 0.3079 | -0.0311 | 15,754,528,152.00 |
| XRPBUSD | 1,010,206 | 0.0825 | 0.0005 | 0.1909 | 0.3978 | 0.0413 | 3,388,413,236.60 |
| LPTUSDT | 1,009,205 | 0.2722 | 0.0017 | 0.2022 | 0.3984 | -0.0118 | 92,955,694.20 |

## Cross-sectional regressions

**γ̂ on log10(n_events)**: slope = **-0.0112** (stderr 0.0547), intercept = 0.4093, R² = 0.0003, n = 121

**p_flip on log10(n_events)**: slope = **0.1114** (stderr 0.0171), intercept = -0.2547, R² = 0.2632, n = 121

## Findings

The fitted order-flow memory exponent γ̂ **decreases** with log-activity across the 121-symbol successful set (slope -0.0112, R² 0.0003), i.e. more actively traded symbols in this sample tend to show weaker long-memory decay than less actively traded ones.

The sign-flip probability p_flip **increases** with log-activity (slope 0.1114, R² 0.2632); since p_flip = 0.5 corresponds to no persistence, this indicates that persistence weakens as activity increases (a slope above zero means p_flip rises toward more anti-persistent behavior at higher activity).

## Skipped (below min_events)

| symbol | n_events | reason |
|---|---|---|
| BNBBUSD | 993,713 | below min_events |
| ENSUSDT | 938,356 | below min_events |
| IDEXUSDT | 911,298 | below min_events |
| BLZUSDT | 875,652 | below min_events |
| PEOPLEUSDT | 997,549 | below min_events |
| 1INCHUSDT | 993,914 | below min_events |
| ZECUSDT | 950,492 | below min_events |
| UNFIUSDT | 910,925 | below min_events |
| ZILUSDT | 938,528 | below min_events |
| LEVERUSDT | 903,468 | below min_events |
| SSVUSDT | 967,784 | below min_events |
| RSRUSDT | 887,261 | below min_events |
| ARUSDT | 863,727 | below min_events |
| IOSTUSDT | 785,884 | below min_events |
| FOOTBALLUSDT | 791,524 | below min_events |
| SPELLUSDT | 780,629 | below min_events |
| ONTUSDT | 888,126 | below min_events |
| ENJUSDT | 887,048 | below min_events |
| ALGOUSDT | 887,166 | below min_events |
| HOOKUSDT | 869,362 | below min_events |
| ASTRUSDT | 754,996 | below min_events |
| SKLUSDT | 778,602 | below min_events |
| APTBUSD | 776,768 | below min_events |
| ICXUSDT | 822,511 | below min_events |
| GMXUSDT | 777,831 | below min_events |
| AUDIOUSDT | 809,732 | below min_events |
| EGLDUSDT | 819,561 | below min_events |
| FLOWUSDT | 868,949 | below min_events |
| RVNUSDT | 763,338 | below min_events |
| CELRUSDT | 759,714 | below min_events |
| BALUSDT | 736,383 | below min_events |
| IOTAUSDT | 746,848 | below min_events |
| MATICBUSD | 691,459 | below min_events |
| BTCUSDT_230630 | 539,505 | below min_events |
| C98USDT | 727,467 | below min_events |
| LRCUSDT | 713,080 | below min_events |
| QTUMUSDT | 705,712 | below min_events |
| ONEUSDT | 691,452 | below min_events |
| LTCBUSD | 637,136 | below min_events |
| HFTUSDT | 663,186 | below min_events |
| KSMUSDT | 728,895 | below min_events |
| XTZUSDT | 748,247 | below min_events |
| CKBUSDT | 614,292 | below min_events |
| CELOUSDT | 712,572 | below min_events |
| ZRXUSDT | 656,595 | below min_events |
| PERPUSDT | 577,837 | below min_events |
| XEMUSDT | 708,385 | below min_events |
| AGIXBUSD | 631,096 | below min_events |
| CHRUSDT | 639,559 | below min_events |
| BAKEUSDT | 670,939 | below min_events |
| LITUSDT | 655,339 | below min_events |
| STMXUSDT | 658,360 | below min_events |
| REEFUSDT | 575,870 | below min_events |
| TRXBUSD | 441,474 | below min_events |
| HOTUSDT | 593,653 | below min_events |
| IOTXUSDT | 578,728 | below min_events |
| CTKUSDT | 544,938 | below min_events |
| DENTUSDT | 574,735 | below min_events |
| RUNEUSDT | 631,519 | below min_events |
| XVSUSDT | 525,608 | below min_events |
| KLAYUSDT | 548,117 | below min_events |
| BATUSDT | 538,723 | below min_events |
| GTCUSDT | 552,519 | below min_events |
| ALICEUSDT | 550,540 | below min_events |
| DOGEBUSD | 467,332 | below min_events |
| DARUSDT | 537,583 | below min_events |
| ETHUSDT_230630 | 398,039 | below min_events |
| ADABUSD | 497,296 | below min_events |
| CVXUSDT | 521,257 | below min_events |
| TRBUSDT | 528,943 | below min_events |
| UMAUSDT | 524,242 | below min_events |
| USDCUSDT | 619,244 | below min_events |
| FTMBUSD | 469,637 | below min_events |
| DGBUSDT | 475,555 | below min_events |
| OGNUSDT | 474,091 | below min_events |
| API3USDT | 472,303 | below min_events |
| BTCDOMUSDT | 381,793 | below min_events |
| TLMUSDT | 437,865 | below min_events |
| ATAUSDT | 451,152 | below min_events |
| BLUEBIRDUSDT | 419,780 | below min_events |
| 1000LUNCBUSD | 350,887 | below min_events |
| MAVUSDT | 274,271 | below min_events |
| DEFIUSDT | 370,015 | below min_events |
| DODOBUSD | 310,592 | below min_events |
| ETHBTC | 165,151 | below min_events |
| NMRUSDT | 170,600 | below min_events |

## Caveats

- Single month (2023-06); this is one specific market regime, and per Phase 1.5 diagnostics, order-flow memory statistics are regime-dependent — these results may not generalize to other months or volatility regimes.
- Each symbol's γ̂ OLS stderr understates true uncertainty (autocorrelated ACF values violate the i.i.d.-residual assumption, same caveat as Q1), and this understatement is heteroskedastic across the cross-section (see Methodology); the cross-sectional regression stderr/R² inherit this problem and should be read as descriptive summaries, not as valid confidence intervals.
- `q4_gamma_vs_activity.png` deliberately omits per-symbol error bars on γ̂: plotting the OLS stderr would imply a precision the estimate does not have, for the same heteroskedasticity/understatement reason given above.
- Symbols are only included in the regressions if they clear `min_events`; the cross-section is therefore a survivorship-filtered subset of the requested universe, not the full universe.
