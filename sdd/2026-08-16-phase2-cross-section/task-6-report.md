# Task 6 report: Q4b tick-size confound test

## Status

**Done.** Real analysis run, LEARNING.md and README updated, suite (103 tests) and ruff green on
`src/` and `tests/`, committed.

## Commit

`4c23eb5` — `feat: Q4b tick-size confound test with real results`

Branch: `phase2-cross-section`.

## Files

- `src/microstructure/analyses/q4b_tick_confound.py` — CLI (`--root`, `--out`, `--q4-json`,
  `--period`, `--exchange-info-url`, `--source-note`), regression machinery (`np.linalg.lstsq`
  with an explicit design matrix, classical-OLS stderr/t-ratio/R²), exchangeInfo fetch + cache,
  lazy-scan mean price, and md/json/PNG output.
- `tests/analyses/test_q4b.py` — 13 tests, fully offline (network mocked via
  `httpx.MockTransport`, everything else synthetic parquet/JSON). Covers regression math on
  known-answer synthetic inputs (univariate, bivariate, collinear-regressor discrimination,
  under-identified-system error), exchangeInfo parsing, mean-price scanning, record assembly
  with skip reasons, and two end-to-end runs (happy path + missing-symbol path + too-few-symbols
  path).
- `results/q4b_tick_confound.{md,json}`, `results/q4b_flip_vs_rel_tick.png`,
  `results/exchangeinfo_snapshot.json` (1.4 MB raw cache) — real run output.
- `LEARNING.md` §6.2 and Q13 (interview drill) — updated from "named but untested" to the actual
  result, style-matched to surrounding prose.
- `README.md` — new Q4b subsection under Phase 2, Q4's confound sentence updated to point at it,
  and the Status/queued-work line updated (tick-size regression moved from "queued" to "done, but
  rerun against mainnet once reachable").

## The three regressions (real run, n=111 of Q4's 121 symbols)

| | coefficient | t-ratio | R² |
|---|---|---|---|
| (a) `p_flip ~ log10(n_events)` | intercept −0.2140, slope **+0.1060** | slope t≈6.75 | 0.2945 |
| (b) `p_flip ~ log10(rel_tick)` | intercept 0.4814, slope **+0.0053** | slope t≈0.50 | 0.0022 |
| (c) `p_flip ~ log10(n_events) + log10(rel_tick)` | activity **+0.1130**, tick **+0.0192** | activity t≈7.14, tick t≈2.09 | 0.3220 |

`corr(log10(n_events), log10(rel_tick))` = **−0.2113** — weaker collinearity than the confound
hypothesis assumed.

## Verdict sentence

**Activity is the dominant driver of Q4's p_flip law; relative tick size is a real but minor
second contributor, not the reverse.** In the joint regression both coefficients are
distinguishable from zero by this project's rough t-ratio convention, but activity's effect is
roughly 6x larger and carries essentially all of the univariate explanatory power (0.294 vs
0.002), and the two regressors are only weakly collinear (−0.21), so the confound does not
explain away the law — it adds a small, statistically noisier second effect on top of it.

## Concerns

1. **Network geo-block (the main caveat).** This sandbox's egress IP gets HTTP 451 ("Service
   unavailable from a restricted location") from `fapi.binance.com`, `api.binance.com`, and
   `dapi.binance.com` — every mainnet Binance API path I tried, confirmed via direct `curl` and
   via `WebFetch`. Only the static historical-data bucket (`data.binance.vision`, already used
   elsewhere in this repo) is reachable. The specified mainnet `exchangeInfo` endpoint could not
   be hit. I substituted the futures **testnet** exchangeInfo mirror
   (`testnet.binancefuture.com`), which returned HTTP 200 with the identical
   `PRICE_FILTER.tickSize` schema and whose BTCUSDT tick (0.10) matches the known mainnet value,
   but testnet contract specs are not guaranteed identical to mainnet symbol-by-symbol, and the
   snapshot is missing 10 of Q4's 121 symbols (all delisted/discontinued BUSD pairs or renamed
   symbols mainnet's historical listing would likely still carry). This is disclosed prominently
   in `q4b_tick_confound.md`'s Methodology and Caveats sections, in the JSON's
   `exchange_info_source_note` field, and in LEARNING.md/README. **The `--exchange-info-url` CLI
   flag defaults to the correct mainnet endpoint** — re-running this command from a network with
   mainnet access, with `--exchange-info-url`/`--source-note` omitted, reproduces the intended
   provenance exactly; I recommend doing that before treating this as the final word.
2. **Current tick size, not June-2023's** — a smaller, separately-documented caveat present
   regardless of the mainnet/testnet substitution (this was called out explicitly in the task
   spec). Binance does occasionally change `PRICE_FILTER.tickSize`, usually after a large price
   move.
3. **t-ratios are descriptive, not formal inference** — same caveat Q4 makes about its own
   cross-sectional regressions: this project doesn't verify or correct for
   heteroskedasticity/cross-sectional dependence across 111-121 heterogeneous assets. Flagged in
   the md's Methodology and Caveats.
4. Given the substitution, I'd treat the exact coefficient magnitudes as provisional and the
   qualitative verdict (activity dominates, tick size is a minor real effect) as the more
   robust takeaway — it would be surprising for a mainnet rerun to flip the sign or ordering of
   effect sizes, but I can't rule it out from here.
