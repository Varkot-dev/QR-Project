# Q3: OFI linearity — results

## Methodology

For ETHUSDT, daily bookTicker L1 snapshots for the given periods are loaded and sorted by timestamp. Per-update order-flow imbalance (OFI) is computed with the Cont, Kukanov & Stoikov (2014) formula (`ofi_events`): each consecutive pair of L1 updates contributes a signed quantity reflecting bid/ask price improvements and same-price size changes. Each OFI value is attached to the timestamp of the later update in its pair. Updates are then bucketed into fixed `10s` bars via `pl.group_by_dynamic` on ts. Within each bar: OFI values are summed, delta_mid is computed as the last mid minus the first mid observed in the bar, and mean depth is the bar-average of (bid_qty + ask_qty)/2. Bars with fewer than 2 updates are dropped (no meaningful delta_mid). The resulting (summed OFI, delta_mid) pairs are regressed through the origin (`ols_through_origin`): delta_mid = beta * OFI_sum.

**Depth-scaling check**: bars are split into 5 depth quintiles by mean depth; a through-origin slope is fit per quintile, then log|slope| is regressed (ordinary least squares, not through origin) on log(mean depth) across quintiles. Cont's theory (slope ~ 1/depth) predicts this log-log regression's slope to be approximately -1. Quintiles whose fitted slope is zero or negative are excluded from the log-log fit (undefined under the log) and noted below.

Periods analyzed: 2023-06-01, 2023-06-02, 2023-06-03, 2023-06-04, 2023-06-05, 2023-06-06, 2023-06-07, 2023-06-08, 2023-06-09, 2023-06-10, 2023-06-11, 2023-06-12, 2023-06-13, 2023-06-14.

## Results

| metric | value |
|---|---|
| slope (β̂) | 0.000169 |
| stderr | 0.000001 |
| R² | 0.4019 |
| n_windows | 120,960 |

### Depth-scaling check

| quintile | mean depth | slope | n_bars |
|---|---|---|---|
| 0 | 53.3304 | 0.000303 | 24,192 |
| 1 | 71.5005 | 0.000182 | 24,192 |
| 2 | 84.2402 | 0.000153 | 24,192 |
| 3 | 100.0455 | 0.000137 | 24,192 |
| 4 | 158.1181 | 0.000125 | 24,192 |

log|slope| vs log(mean depth) regression exponent: **-0.7741** (Cont theory predicts ≈ -1).

## Benchmark vs. literature

Cont, Kukanov & Stoikov (2014) report R² ≈ 65%–70% for OFI-vs-price-change linear regressions on equities. Silantyev (2019) studied BitMEX order flow and found trade-flow imbalance (net signed trade volume) a stronger price-change predictor than book-based OFI in that venue.

| our R² | Cont equities R² | benchmark comparison |
|---|---|---|
| 0.4019 | 0.65–0.70 | below the Cont equities range |

## Caveats

- OFI is computed from L1 (best bid/ask) bookTicker snapshots only; no order-book depth beyond the top level is observed, so OFI understates true order-flow pressure from deeper levels.
- `ols_through_origin`'s stderr assumes i.i.d. residuals; bar-level delta_mid and OFI sums are likely autocorrelated across adjacent bars, so this stderr understates true uncertainty.
- The depth-scaling check uses only 5 quintiles over a fixed 14-day sample, giving a log-log regression with very few points; its exponent estimate carries wide (unreported) uncertainty and should be treated as suggestive, not confirmatory.
- The sample covers a single symbol over 14 days of a specific market regime; results may not generalize to other symbols, venues, or volatility regimes.
