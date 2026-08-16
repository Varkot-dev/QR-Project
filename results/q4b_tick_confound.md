# Q4b: tick-size confound test — results

## Question

Q4 found `p_flip ~ log10(n_events)` with slope **+0.1114** (R² = 0.2632, n = 121): more actively traded symbols flip sign more often. LEARNING.md Sec.6.2 named, but did not test, an alternative: **relative tick size** (`tickSize / price`) is a mechanical driver of bid-ask bounce, and it plausibly correlates with activity. If relative tick size is the real driver, "activity" in Q4's regression is a proxy variable and the competitive-response interpretation is decoration on a bid-ask-bounce artifact. This analysis runs the discriminating regression directly.

## Methodology

1. **Tick size**: fetched from Binance futures `exchangeInfo` (`https://testnet.binancefuture.com/fapi/v1/exchangeInfo`), a public unauthenticated endpoint. Each symbol's `PRICE_FILTER.tickSize` is extracted. The raw response is cached to `exchangeinfo_snapshot.json` for provenance.

   **Source substitution for this run**: The specified mainnet endpoint (fapi.binance.com/fapi/v1/exchangeInfo) returned HTTP 451 (geo-restricted) from this execution environment, as did every other fapi.binance.com/api.binance.com/dapi.binance.com path tried. The futures TESTNET exchangeInfo endpoint (testnet.binancefuture.com) was reachable and returns the same PRICE_FILTER schema; its BTCUSDT tickSize (0.10) matches the known mainnet value, but testnet contract specs are not guaranteed identical to mainnet for every symbol and this snapshot is missing 10 of Q4's 121 symbols (all delisted/renamed BUSD or discontinued pairs) that mainnet's live exchangeInfo would likely still list historically. Treat tick sizes in this run as a best-effort proxy for mainnet, not a verified mainnet snapshot.
2. **Mean price**: for each of Q4's 121 successful symbols, the mean aggTrades trade price over 2023-06 is computed via a lazy Polars scan (`pl.scan_parquet(...).select(pl.col("price").mean())`) of the same parquet Q4 used. `rel_tick = tickSize / mean_price`.
3. **Regressions**: three OLS fits via `numpy.linalg.lstsq` on the usable symbols (intersection of Q4's successful set, symbols present in the exchangeInfo snapshot, and symbols with a readable mean price): (a) `p_flip ~ log10(n_events)` — reproduces Q4's law as a baseline on this potentially-reduced sample; (b) `p_flip ~ log10(rel_tick)` — does tick size alone predict it; (c) `p_flip ~ log10(n_events) + log10(rel_tick)` — the discriminating regression: which variable's coefficient survives once the other is controlled for. `corr(log10(n_events), log10(rel_tick))` is also reported — the collinearity that motivates this whole test.

**Honesty caveat on t-ratios**: coefficient significance is reported as a t-ish ratio (coefficient / classical-OLS stderr), assuming i.i.d. homoskedastic residuals. That assumption is not verified and is likely violated — this is a heterogeneous cross-section of 121 different assets with no correction for cross-sectional dependence or heteroskedasticity (same caveat Q4 makes about its own regressions). Read these ratios as descriptive orientation on coefficient size relative to noise, not as a formal hypothesis test with a valid p-value.

## Run summary

Q4 successful symbols: 121. Usable for this analysis (tick size found + mean price computed): 111. Skipped: 10.

## Regressions

**(a) p_flip ~ log10(n_events)** [Q4's law, reproduced on this sample]: intercept = **-0.2140** (stderr 0.1000, t≈-2.14), log10_n_events = **0.1060** (stderr 0.0157, t≈6.75), R² = 0.2945, n = 111

**(b) p_flip ~ log10(rel_tick)**: intercept = **0.4814** (stderr 0.0439, t≈10.97), log10_rel_tick = **0.0053** (stderr 0.0108, t≈0.50), R² = 0.0022, n = 111

**(c) p_flip ~ log10(n_events) + log10(rel_tick)** [discriminating regression]: intercept = **-0.1812** (stderr 0.0997, t≈-1.82), log10_n_events = **0.1130** (stderr 0.0158, t≈7.14), log10_rel_tick = **0.0192** (stderr 0.0092, t≈2.09), R² = 0.3220, n = 111

**corr(log10(n_events), log10(rel_tick))** = -0.2113 — the collinearity between activity and relative tick size that motivates this test.

## Verdict

**Both variables survive jointly.** In regression (c), log10(n_events) (coef 0.1130, t≈7.14) and log10(rel_tick) (coef 0.0192, t≈2.09) both remain distinguishable from zero despite their collinearity (corr = -0.2113). Neither single-variable story is sufficient on its own: activity and relative tick size appear to carry at least partially independent information about p_flip in this cross-section, so the confound is real but does not fully explain away the activity effect. For context: univariate R² is 0.2945 for activity alone and 0.0022 for relative tick size alone, versus 0.3220 jointly.

## Skipped symbols

| symbol | reason |
|---|---|
| RNDRUSDT | no tickSize in exchangeInfo snapshot |
| MATICUSDT | no tickSize in exchangeInfo snapshot |
| BTCBUSD | no tickSize in exchangeInfo snapshot |
| ETHBUSD | no tickSize in exchangeInfo snapshot |
| GALUSDT | no tickSize in exchangeInfo snapshot |
| SOLBUSD | no tickSize in exchangeInfo snapshot |
| LDOBUSD | no tickSize in exchangeInfo snapshot |
| GALABUSD | no tickSize in exchangeInfo snapshot |
| ICPUSDT | no tickSize in exchangeInfo snapshot |
| XRPBUSD | no tickSize in exchangeInfo snapshot |

## Caveats

- **exchangeInfo source substitution**: The specified mainnet endpoint (fapi.binance.com/fapi/v1/exchangeInfo) returned HTTP 451 (geo-restricted) from this execution environment, as did every other fapi.binance.com/api.binance.com/dapi.binance.com path tried. The futures TESTNET exchangeInfo endpoint (testnet.binancefuture.com) was reachable and returns the same PRICE_FILTER schema; its BTCUSDT tickSize (0.10) matches the known mainnet value, but testnet contract specs are not guaranteed identical to mainnet for every symbol and this snapshot is missing 10 of Q4's 121 symbols (all delisted/renamed BUSD or discontinued pairs) that mainnet's live exchangeInfo would likely still list historically. Treat tick sizes in this run as a best-effort proxy for mainnet, not a verified mainnet snapshot.
- **Tick size is current, not June-2023.** `exchangeInfo` returns Binance's tick size as of whenever this analysis is run, not as of the June 2023 period the trade data and Q4's p_flip come from. Binance does change `PRICE_FILTER.tickSize` occasionally (usually only after large price moves, e.g. after a symbol's price falls by an order of magnitude), so for a symbol whose price regime shifted materially between June 2023 and today, `rel_tick` computed here may not reflect the tick size actually in force during the data window. This is a real, if probably small for most symbols, source of error and is not corrected for.
- **121-symbol sample**, further reduced to the usable subset above (symbols missing from the exchangeInfo snapshot — e.g. delisted or renamed since June 2023 — are dropped, not imputed).
- **Single month** (2023-06), same as Q4: one specific market regime; not tested for generalization to other periods.
- **OLS assumptions unverified** (see Methodology): reported stderr/t-ratios/R² are descriptive, not formal inference, for the same reasons Q4 gives about its own cross-sectional regressions (heteroskedastic, non-i.i.d. residuals across a heterogeneous set of assets).
- **Correlation is not causation either way**: even a clean result in (c) establishes which variable better explains this cross-section statistically, not the causal mechanism generating p_flip.
