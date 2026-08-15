# Q2: Response function — results

## Methodology

Aggressor events for ETHUSDT are loaded from monthly aggTrades period `2023-06` and filtered to the timestamp span covered by the daily bookTicker periods 2023-06-01, 2023-06-02, 2023-06-03, 2023-06-04, 2023-06-05, 2023-06-06, 2023-06-07, 2023-06-08, 2023-06-09, 2023-06-10, 2023-06-11, 2023-06-12, 2023-06-13, 2023-06-14. Each event is joined to the best bid/ask mid price prevailing STRICTLY BEFORE its timestamp (`events_with_prior_mid`, ms resolution). The average price response R(ℓ) = E[sign_t * (m_{t+ℓ} - m_t)] is then computed via `response_function` out to `max_lag` events. Two candidate decay shapes are fit over the window ℓ ∈ [10, 200]: a power law R(ℓ) ~ ℓ^(-γ) (OLS on log R vs log ℓ) and an exponential R(ℓ) ~ A·exp(-λℓ) (OLS on log R vs ℓ, positive R values only in both cases). The fit with the lower residual sum of squares (RSS) on the log scale is judged the better-fitting shape.

Events after joining to prior mid: 6,396,387 (dropped 0, 0.0000%).

## Results

**Note:** both fitted parameters are negative, i.e. R(ℓ) is *growing* with lag over the fit window, not decaying -- a negative γ̂ means R(ℓ) ~ ℓ^{+0.083} (growth), and a negative λ̂ means R(ℓ) ~ exp(+0.0009·ℓ) (growth). Read the sign of γ̂/λ̂ before reading their magnitude as a 'decay rate'.

| quantity | value |
|---|---|
| R(1) | 0.010402 |
| power-law exponent γ̂ (R ~ ℓ^-γ̂) | -0.0831 |
| power-law OLS stderr | 0.0034 |
| exponential rate λ̂ (R ~ exp(-λ̂ℓ)) | -0.0009 |
| exponential OLS stderr | 0.0001 |

## Fit comparison (log-scale RSS, lags 10-200)

| shape | log-scale RSS |
|---|---|
| power law | 0.2161 |
| exponential | 0.4718 |

**Verdict:** the power-law form has lower residual sum of squares on the log scale over lags 10-200 and is judged the better-fitting shape for ETHUSDT's response function in this sample -- which is growing (not the classic decaying-impact case) over lags 1-200.

## Benchmark vs. literature

Bouchaud et al. (2004) found response functions on equity markets that decay slowly, roughly as a power law, over hundreds to thousands of trades -- evidence that price impact is not a single-event, exponentially-forgotten shock but reflects long-range order-flow correlation.

## Caveats

- The OLS standard errors on γ̂ and λ̂ assume i.i.d. residuals; because R(ℓ) at nearby lags is itself autocorrelated (both through the impact kernel and any order-flow memory), these stderrs understate the true uncertainty.
- Real order flow is NOT i.i.d. (Q1 finds long-memory signs), so R(ℓ) here mixes the bare impact kernel with sign autocorrelation; it is not a clean kernel estimate the way it would be under the iid-sign assumption used to validate the estimator.
- The sample is a single 14-day window (2023-06-01..2023-06-14) for one symbol (ETHUSDT); the fitted decay shape and rate may not generalize to other periods, volatility regimes, or symbols.
- RSS comparison is on the log scale over a fixed window; a different window or a linear-scale comparison could favor the other shape, especially since power laws and exponentials with matched short-lag behavior often diverge only at large lag.
- A growing (not decaying) R(ℓ) over lags 1-200 departs from the classic single-event impact-decay picture in Bouchaud (2004); it is consistent with the strong short-lag order-flow persistence found in Q1 -- a run of same-sign events keeps pushing the mid in the same direction for many subsequent events, so the *average* response measured this way keeps rising before any decay could show up. Whether R(ℓ) eventually turns over past lag 200 is not addressed by this fit window.
