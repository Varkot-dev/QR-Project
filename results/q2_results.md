# Q2: Response function — results

## Methodology

Aggressor events for ETHUSDT are loaded from monthly aggTrades period `2023-06` and filtered to the timestamp span covered by the daily bookTicker periods 2023-06-01, 2023-06-02, 2023-06-03, 2023-06-04, 2023-06-05, 2023-06-06, 2023-06-07, 2023-06-08, 2023-06-09, 2023-06-10, 2023-06-11, 2023-06-12, 2023-06-13, 2023-06-14. Each event is joined to the best bid/ask mid price prevailing STRICTLY BEFORE its timestamp (`events_with_prior_mid`, ms resolution). The average price response R(ℓ) = E[sign_t * (m_{t+ℓ} - m_t)] is then computed via `response_function` out to `max_lag` events. Two candidate decay shapes are fit over the window ℓ ∈ [10, 200]: a power law R(ℓ) ~ ℓ^(-γ) (OLS on log R vs log ℓ) and an exponential R(ℓ) ~ A·exp(-λℓ) (OLS on log R vs ℓ, positive R values only in both cases). The fit with the lower residual sum of squares (RSS) on the log scale is judged the better-fitting shape.

Events after joining to prior mid: 6,396,387 (dropped 0, 0.0000%).

## Results

**Note:** both fitted parameters are negative, i.e. R(ℓ) is *growing* with lag over the fit window, not decaying -- a negative γ̂ means R(ℓ) ~ ℓ^{+0.0831} (growth), and a negative λ̂ means R(ℓ) ~ exp(+0.0009·ℓ) (growth). Read the sign of γ̂/λ̂ before reading their magnitude as a 'decay rate'. This growth is the EXPECTED shape given long-memory order flow, not an anomaly -- see the Benchmark and Caveats sections below for the R ≈ G + Σ G·C decomposition that explains why.

| quantity | value |
|---|---|
| R(1) | 0.010402 |
| R(500) | 0.056107 |
| R(500)/R(1) | 5.3939 |
| power-law exponent γ̂ (R ~ ℓ^-γ̂) | -0.0831 |
| power-law OLS stderr | 0.0034 |
| exponential rate λ̂ (R ~ exp(-λ̂ℓ)) | -0.0009 |
| exponential OLS stderr | 0.0001 |

## Fit comparison (log-scale RSS, lags 10-200)

| shape | log-scale RSS |
|---|---|
| power law | 0.2161 |
| exponential | 0.4718 |

**Verdict:** the power-law form has lower residual sum of squares on the log scale over lags 10-200 and is judged the better-fitting shape for ETHUSDT's response function in this sample, which is growing over lags 1-200.

## Benchmark vs. literature

Bouchaud et al. (2004) report that the bare impact KERNEL G(l) decays slowly, roughly as a power law, over hundreds to thousands of trades -- evidence that price impact is not a single-event, exponentially-forgotten shock but reflects long-range order-flow correlation. The MEASURED response function R(l) is a different object: it mixes G with order-flow memory C, and Bouchaud's own equity data shows R(l) rising to a maximum around 10^2-10^3 trades before any slow decline -- the same rise this analysis measures, not a contradiction of it.

## Caveats

- The OLS standard errors on γ̂ and λ̂ assume i.i.d. residuals; because R(ℓ) at nearby lags is itself autocorrelated (both through the impact kernel and any order-flow memory), these stderrs understate the true uncertainty.
- Real order flow is NOT i.i.d. (Q1 finds long-memory signs), so R(ℓ) here mixes the bare impact kernel with sign autocorrelation; it is not a clean kernel estimate the way it would be under the iid-sign assumption used to validate the estimator.
- The sample is a single 14-day window (2023-06-01..2023-06-14) for one symbol (ETHUSDT); the fitted decay shape and rate may not generalize to other periods, volatility regimes, or symbols.
- RSS comparison is on the log scale over a fixed window; a different window or a linear-scale comparison could favor the other shape, especially since power laws and exponentials with matched short-lag behavior often diverge only at large lag.
- A growing R(ℓ) over lags 1-200 is the EXPECTED response shape given long-memory order flow, not a departure from Bouchaud (2004). The measured response mixes the (decaying) bare impact kernel G with the sign autocorrelation C: R(ℓ) ≈ G(ℓ) + Σ_{n<ℓ} G(ℓ-n)·C(n). With Q1's measured sign-ACF exponent γ≈0.24 for ETH, the accumulation term Σ G·C dominates G itself, so R keeps climbing well past where G alone would have decayed -- Bouchaud's own equity response functions show the same rise-then-slow-decline shape, peaking around 10^2-10^3 trades before turning over. R(500)/R(1) = 5.39x in this sample; a toy transient-impact calculation using γ≈0.24 and the diffusivity-consistent kernel exponent β=(1-γ)/2≈0.38 predicts R(500)/R(1) in roughly the 3.5-6.9x range (depending on lag-1 sign autocorrelation in 0.2-0.4), and the measured 5.39x falls inside it. What decays in the literature is the KERNEL G(ℓ) itself, not R(ℓ) -- this analysis measures R only; separating G from C requires propagator deconvolution, which is out of scope here.
- R(ℓ) plateaus around ℓ≈300-500 (see the response array in `q2_results.json`), outside the fitted window of [10, 200]; the power-law/exponential fits above describe only the rising portion and say nothing about behavior at or past the plateau.
