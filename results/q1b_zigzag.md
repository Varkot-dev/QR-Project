# Q1b: short-lag ACF zigzag — tie-break robustness

## Methodology

Aggressor events for BTCUSDT 2023-06 are loaded via `load_events` (sorted by `(ts, sign)`; within a shared millisecond, sells precede buys). Three sign series are compared, all evaluated on the identical FFT sign ACF (`sign_acf`) at lags 1-10:

- **A (baseline):** the series as `load_events` produces it -- deterministic `(ts, sign)` tie-break.
- **B (randomized tie-break):** same-timestamp adjacent pairs (after aggregation, always exactly one sell + one buy, since `to_aggressor_events` never emits two same-(ts, side) rows) have their order swapped with p=0.5 using a fixed-seed RNG (`numpy.random.default_rng(0)`), which destroys the deterministic ordering convention while leaving every other event untouched.
- **C (netted):** each same-timestamp group of opposite-signed events is collapsed into a single event with sign = sign(sum(sign * qty)); groups whose signed notional exactly nets to zero are dropped (undefined sign).

The **zigzag amplitude** is `mean(ACF at lags 2,4,6,8,10) - mean(ACF at lags 1,3,5,7,9)` -- large and positive when even lags run noticeably higher than odd lags, which is the pattern visible in Q1's log-log ACF plot at short lags.

## Results

n_events = 21,816,890.

Fraction of consecutive event pairs sharing a millisecond timestamp: **1.1987%** (261,515 same-ts pairs). Among those same-ts adjacent pairs, **100.00%** are opposite-signed (by construction: `to_aggressor_events` produces at most one buy and one sell per millisecond, so max same-ts group size = 2). Variant B is therefore a random 50/50 swap of 131,032 of those pairs, not a general permutation.

| lag | A: baseline | B: randomized tie-break | C: netted |
|---|---|---|---|
| 1 | -0.167615 | -0.166886 | -0.153383 |
| 2 | 0.268342 | 0.267467 | 0.271401 |
| 3 | 0.019937 | 0.020082 | 0.027652 |
| 4 | 0.153220 | 0.153224 | 0.151183 |
| 5 | 0.045971 | 0.045934 | 0.054323 |
| 6 | 0.109750 | 0.109815 | 0.107009 |
| 7 | 0.051406 | 0.051379 | 0.058521 |
| 8 | 0.087315 | 0.087359 | 0.085164 |
| 9 | 0.051650 | 0.051615 | 0.057827 |
| 10 | 0.073590 | 0.073620 | 0.071857 |

| variant | zigzag amplitude | relative change vs. A |
|---|---|---|
| A: baseline | 0.138174 | -- |
| B: randomized tie-break | 0.137872 | 0.22% |
| C: netted (21,550,823 events, 4,552 zero-net groups dropped) | 0.128335 | 7.12% |

## Verdict

**The zigzag is real structure, not a tie-break artifact.** The amplitude barely moves under the randomized tie-break (0.138174 -> 0.137872, a 0.22% change) and remains large under netting (0.128335, a 7.12% change). A sanity check agrees: only 1.20% of consecutive event pairs share a timestamp, so the deterministic tie-break simply does not touch enough adjacent pairs to manufacture an alternation of this size. The most likely explanation is genuine market structure -- alternation consistent with bid-ask bounce or interleaved liquidity-taking reversals -- surviving both perturbations.

**This does not affect Q1's headline gamma fits.** Q1's power-law fit window starts at lag 10 (`fit_power_law(..., lo=10, ...)`); the zigzag reported here is measured over lags 1-10, i.e. entirely at or before the start of the fit window, not inside it. Whether the alternation itself persists PAST lag 10 (and could therefore influence the fit) is not established by this analysis, which only computes lags 1-10 -- that would require extending this same three-way comparison to lags 11+ before the 'fit window unaffected' claim could be made without qualification.

## Caveats

- Single symbol-month (BTCUSDT 2023-06); this analysis was not repeated on other symbols or periods in this run.
- Among same-ts adjacent pairs, 100% are opposite-signed by construction (post-aggregation, each timestamp holds at most one buy and one sell event), so variant B reduces to random pair swaps rather than a general permutation -- equivalent here since groups never exceed size 2.
- `sign_acf` uses the unbiased normalization (divide by n-lag); at lags <=10 with n in the tens of millions this makes no practical difference.
- Netting (C) measurably dampens the zigzag amplitude relative to baseline, so same-ts buy/sell pairs do contribute something to it, but the dominant odd/even pattern (negative ACF(1), large positive ACF(2)) persists through both perturbations.
- Timestamps have millisecond resolution; finer-than-millisecond ordering information is unrecoverable, so 'real structure' means real at the millisecond-aggregated event level, not at the level of true arrival order within a millisecond.
- This analysis does not extend past lag 10, so it cannot by itself confirm or rule out zigzag-driven distortion of Q1's [10, 500] fit window; see the note under Verdict above.
