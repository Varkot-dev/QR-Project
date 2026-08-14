# Design Spec: Empirical Market Microstructure on Binance

Date: 2026-08-13
Status: draft for user review

## 1. What this is

An empirical research project measuring price impact and order-flow dynamics on Binance from raw tick data. Learning-first: every Phase-1 result has a published benchmark to check against. Phase 2 carries the same machinery to a genuinely unoccupied spot (verified adversarially — see `research/04-novelty-verification-verdicts.md`).

**Primary goal:** the user learns market microstructure, statistics on real data, and large-dataset engineering well enough to defend every step in a quant interview.
**Secondary goal:** a GitHub-presentable research repo with a defensible "some novelty" Phase 2.
**Explicit non-goals:** publication, trading profitably, live trading of any kind.

## 2. Research questions

### Phase 1 — Core (target: 4–6 weeks). Replications with answer keys.

**Q1. Order-flow memory.** Is the sequence of buy/sell aggressor signs long-memory (power-law autocorrelation, exponent γ)?
Benchmark: equities/futures show power-law sign-ACF, γ ≈ 0.3–0.7, persisting for days (Bouchaud et al. 2004). Prior BTC results exist (Schnaubelt et al. 2019 stylized-facts battery).

**Q2. Response function.** Measure R(ℓ) = E[ε_t · (m_{t+ℓ} − m_t)] — the average mid-price move ℓ events after a signed trade. Extract the shape of impact decay; test exponential (Obizhaeva–Wang) vs power-law (propagator) decay.
Benchmark: Bouchaud et al. 2004 methodology; single-trade response shown near-universal across assets (arXiv:1702.08029).

**Q3. OFI linearity.** Replicate Cont–Kukanov–Stoikov: price change over short windows is linear in order-flow imbalance with slope ∝ 1/depth.
Benchmark: R² ≈ 65–70% in US equities; Silantyev 2019 replicated on BitMEX XBTUSD (trade-flow imbalance dominant).

**Q4 (stretch, if time in Phase 1).** Critical-balance check on BTC + ETH only: does the response-decay exponent β satisfy β ≈ (1−γ)/2?

### Phase 2 — Cross-section (semester). The novelty sliver.
Run Q1/Q2/Q4 as one package across 50–100 Binance USDT-M perp pairs spanning ≥3 decades of daily volume. Questions: do γ, β, and response amplitude vary systematically with liquidity (volume, spread, tick-to-price ratio)? Does the critical-balance relation hold across the cross-section? Per adversarial verification: this joint package exists for equities and FX, not crypto.

### Phase 3 — Optional endgame.
Fit Hawkes to the same tape (branching ratio with kernel-sensitivity analysis per the known pitfalls); build an execution replay simulator; compare TWAP vs impact-aware schedules. First published-pipeline transfer (energy/equities → crypto).

## 3. Data

- **Source:** data.binance.vision public dumps (free, no account). Primary: **USDT-M futures** `aggTrades` (taker-side pre-signed — no trade-sign classification needed) + `bookTicker` (best bid/ask price & size tape → mid-prices and L1 depth for Q2/Q3). Spot `aggTrades` as robustness check for Q1.
- **VERIFIED 2026-08-13 against the S3 bucket directly** (not from secondary sources):
  - `aggTrades` monthly: available through 2026-07 (current).
  - `bookTicker`: **only exists 2023-05 → 2024-04** (dumps discontinued after that). This bounds where clean mid-price/L1 analysis is possible.
  - `bookDepth` (daily, sampled depth snapshots) and `metrics`: available through present.
- **Phase 1 sample period (consequence of the above):** **2023-06 → 2024-03** (10 months) for BTCUSDT + ETHUSDT perps — the window where aggTrades and bookTicker overlap. Q1 (trades-only) additionally run on recent 2026 data as a regime robustness check. Estimated tens of GB raw; stored as monthly Parquet after ingestion.
- **Known data quirks to handle (from the literature review):** millisecond timestamp resolution (no kernel/ACF claims below ~10ms lags); one matching-engine event prints multiple times — **aggregate fills sharing (timestamp, side) into single aggressor events** before any analysis; 8h funding-clock seasonality on perps; deseasonalize or use event-time where appropriate.

## 4. Architecture

```
src/microstructure/
├── data/          # download, verify checksums, parse, cache to Parquet (AGENT-BUILT)
│   ├── binance.py     # dump-file URLs, download, integrity checks
│   ├── ingest.py      # CSV→Parquet, schema normalization, aggressor aggregation
│   └── catalog.py     # local dataset registry: what's downloaded, date ranges
├── signals/       # signed-trade series, mid-price series, OFI construction (TOGETHER)
├── estimators/    # sign ACF, response function, log-log fits w/ error bars (TOGETHER)
└── plots/         # standard figure styles (AGENT-BUILT)
notebooks/         # one notebook per research question, narrative form (TOGETHER)
tests/             # pytest; estimators validated on synthetic data with KNOWN answers
research/          # literature library (exists) + findings write-ups
LEARNING.md        # every concept explained as we hit it; interview-prep artifact
```

- **Tooling:** Python 3.12, polars (not pandas) for tick-scale data, uv for env, pytest.
- **Division of labor:** agents build `data/`, `plots/`, test scaffolding, CI. All research code (`signals/`, `estimators/`, notebooks) is written with the user step-by-step — this is the learning content, never delegated.

## 5. Validation strategy

1. **Synthetic ground truth for every estimator:** e.g., generate an AR/ARFIMA sign series with known γ — the ACF estimator must recover it within error bars before touching real data. Same for response-function and regression code.
2. **Published-benchmark comparison table** maintained in each notebook: our number vs literature number vs match/mismatch verdict.
3. **Data-integrity tests:** checksum verification on downloads; row counts vs Binance-published counts; no-gap continuity checks per day.
4. Standard repo hygiene: pytest ≥80% coverage on `src/`, CI on push.

## 6. Error handling & honesty rules

- Every empirical claim in write-ups carries: sample period, N, standard errors (block bootstrap for autocorrelated data), and known caveats.
- Mismatches with published benchmarks are findings, not failures — investigated and documented, never quietly dropped.
- Confidence labeling convention (per user preference): verified / sourced / inferred / unknown.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Tick data volume overwhelms laptop | Start 2 symbols × 6 months; Parquet + polars lazy scans; downsample only with explicit justification |
| bookTicker discontinued after 2024-04 (VERIFIED) | Resolved by design: Phase-1 window fixed to 2023-06→2024-03; recent-data analyses use trade prices + sampled bookDepth with documented caveats |
| Estimator subtleties (log-log fitting traps) | Synthetic-data validation first, always; consult the pitfalls sections in research/02 |
| Scope creep into Phase 2/3 early | Phase gates: Phase 1 write-up complete before any Phase 2 code |
| User time (recruiting season) | Each research question is independently completable; stopping after Q2 still yields a coherent repo |

## 8. Timeline (Phase 1)

- **Week 1:** agents build data layer; together: order-book/microstructure concepts, first look at the tape, sanity plots. Q1 started.
- **Weeks 2–3:** Q1 complete (ACF + γ estimate + benchmark comparison). Q2 response function.
- **Week 4:** Q2 decay-shape test. Q3 OFI regression.
- **Weeks 5–6:** Q3 complete, Q4 stretch, Phase-1 write-up (README with figures), repo polish.
