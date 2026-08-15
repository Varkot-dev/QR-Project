# Q1: Order-flow memory — results

## Methodology

For each symbol, Binance aggTrades prints for the given periods are collapsed into aggressor-level events (all same-millisecond, same-side prints merged into one taker decision — see `to_aggressor_events`), producing a ±1 sign series where +1 is a buyer-initiated (taker-buy) event and -1 is a seller-initiated event. The sample autocorrelation function (ACF) of this sign series is computed via FFT (Wiener-Khinchin) out to `max_lag` events. A power law of the form ACF(lag) ~ lag^(-γ) is then fit by ordinary least squares on log(ACF) vs log(lag) over the window [10, max_lag // 2], skipping any non-positive ACF values in that window. The fitted exponent γ̂ is the analysis's estimate of the order-flow long-memory decay rate.

Periods analyzed: 2023-06, 2023-07.

## Results

| symbol | n_events | γ̂ | OLS stderr |
|---|---|---|---|
| BTCUSDT | 38,046,362 | 0.3803 | 0.0010 |
| ETHUSDT | 25,773,466 | 0.2380 | 0.0015 |

## Benchmark vs. literature

Literature range (equities/futures sign-ACF exponent, Bouchaud et al. 2004): γ ≈ 0.3–0.7, with persistence horizons of thousands of trades.

| symbol | γ̂ | literature range | in range? |
|---|---|---|---|
| BTCUSDT | 0.3803 | 0.3–0.7 | yes |
| ETHUSDT | 0.2380 | 0.3–0.7 | no |

Falling inside or outside this range is a documented empirical finding either way, not a pass/fail criterion for the analysis.

## Caveats

- The OLS standard error on γ̂ is computed from the log-log regression residuals under an i.i.d.-error assumption; because the ACF values at nearby lags are themselves autocorrelated, this stderr **understates** the true uncertainty in γ̂.
- Millisecond-timestamp ties are merged by aggressor aggregation before computing signs (multiple same-ms, same-side prints from one sweep become a single event), so the event count is smaller than the raw aggTrades row count and lag-1 structure reflects aggressor decisions, not raw prints.
- The sample covers 2 months of a specific market regime for each symbol; the estimated γ̂ may not generalize to other periods, volatility regimes, or symbols.
