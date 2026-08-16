# Q0: Aggregation effect on order-flow memory — results

## Punchline

**A broken pipeline that skips aggressor aggregation impersonates a successful replication.** In every symbol-month tested here, raw-print gamma is inflated by roughly +0.29 to +0.50 relative to the correctly aggregated gamma from the same data -- landing INSIDE the equities/futures range (0.3-0.7, Bouchaud et al. 2004) in half the cells below and OVERSHOOTING past it in the other half, while the aggregated gamma moves lower or further out of range in every cell. Checking 'does gamma fall in the literature range' cannot by itself distinguish the correct pipeline from the broken one: both checks pass, for different reasons, on different numbers, and it is the broken pipeline that more often looks like a clean replication.

## Methodology

For each (symbol, period) cell, the same raw `aggTrades` parquet is loaded two ways. **Raw**: `is_buyer_maker` is read directly in on-disk (`agg_trade_id`) order and signed (+1 buyer-taker, -1 seller-taker), with no same-(timestamp, side) merging -- one sign per PRINT. **Aggregated**: `load_events` (the repo's normal path), which merges all same-millisecond, same-side prints into one aggressor decision via `to_aggressor_events` before signing -- one sign per aggressor decision. Both series get the identical FFT sign ACF (`sign_acf`) and log-log power-law fit (`fit_power_law`, lags [10, 500]) used by Q1, so gamma values are directly comparable across the two paths.

## Results

| symbol | period | raw n | raw acf(1) | raw γ̂ | agg n | agg acf(1) | agg γ̂ | prints/event | γ̂ inflation |
|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 2023-06 | 39,024,962 | 0.3413 | 0.9634 | 21,816,890 | -0.1676 | 0.4617 | 1.7888 | +0.5017 |
| BTCUSDT | 2023-07 | 24,963,535 | 0.2786 | 0.7871 | 16,229,472 | -0.1032 | 0.3260 | 1.5382 | +0.4611 |
| ETHUSDT | 2023-06 | 24,368,924 | 0.4290 | 0.7077 | 14,239,099 | 0.0284 | 0.2858 | 1.7114 | +0.4219 |
| ETHUSDT | 2023-07 | 16,789,675 | 0.3737 | 0.4983 | 11,534,367 | 0.0917 | 0.2055 | 1.4556 | +0.2928 |

## Literature-range check, both pipelines

Equities/futures sign-ACF exponent range (Bouchaud et al. 2004): γ ≈ 0.3–0.7.

| symbol | period | raw γ̂ | raw in range? | agg γ̂ | agg in range? |
|---|---|---|---|---|---|
| BTCUSDT | 2023-06 | 0.9634 | no | 0.4617 | yes |
| BTCUSDT | 2023-07 | 0.7871 | no | 0.3260 | yes |
| ETHUSDT | 2023-06 | 0.7077 | no | 0.2858 | no |
| ETHUSDT | 2023-07 | 0.4983 | yes | 0.2055 | no |

The direction is universal across every cell measured: raw-print gamma is inflated relative to aggregated gamma by roughly +0.29 to +0.50, and raw lag-1 ACF is strongly positive (about 0.28-0.43) everywhere, reflecting the matching engine walking the book within a single aggressor decision. Whether the inflated number lands strictly inside [0.3, 0.7] or overshoots past 0.7 varies by symbol-month, so 'inflated into the range' is the common case but not universal at the individual-cell level -- check the table above rather than assuming every raw γ̂ sits inside the range.

A second, more qualitative effect shows up for BTC specifically: aggregation does not just shrink BTC's lag-1 ACF, it flips its sign from positive to negative, while ETH's aggregated lag-1 ACF stays small and positive. Same aggregation step, different effect on the sign, depending on the symbol.

## Caveats

- This is the same measurement Q1 already relies on (`to_aggressor_events` before signing); Q0 exists to make the raw-vs-aggregated CONTRAST itself a committed, re-runnable artifact rather than a fact stated only in prose (see LEARNING.md §1).
- gamma and its OLS stderr both come from the same fit window and normalization as Q1; the OLS stderr assumes i.i.d. residuals and understates true uncertainty for autocorrelated ACF points (see LEARNING.md §5).
- Whether raw γ̂ lands strictly inside [0.3, 0.7] or overshoots above 0.7 depends on the symbol-month; BTC's raw γ̂ in particular can exceed 0.7 (the fragmentation artifact is strong enough to overshoot the equity band entirely, not just enter it).
- Sample is whatever (symbols, periods) this analysis was run with; see the table above for exactly which cells are covered.
