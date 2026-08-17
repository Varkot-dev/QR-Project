# Phase 3: Hawkes Endogeneity Cross-Section + Execution Simulator

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Checkbox steps.

**Goal (remaining verified-open ground per research/02 §7 and research/04):** (A) the branching-ratio cross-section — is crypto order flow near-critical, and does endogeneity n̂ scale with activity? (only BTC ever measured; the estimation-pitfall fight between Filimonov-Sornette and Hardiman-Bouchaud is the methodological terrain); (B) the execution-cost simulator on replayed real data with OUR measured kernels — TWAP vs front-loaded vs flow-reactive schedules (the Alfonsi-Blanc-style transfer, unpublished for crypto).

**Honesty doctrine (binding, from research/02 §4.2):** the branching ratio is estimator- and window-dependent; the SENSITIVITY IS THE FINDING. Every n̂ ships with: kernel-family sensitivity (exp vs 2-exp), window sensitivity, and the count-variance model-free cross-check. Never a single unqualified n̂.

**Data:** already on disk (207-symbol June aggTrades; 16-symbol 7-day bookTicker panel). No new downloads required.

**Tech:** existing stack; estimators pure numpy; TDD with synthetic ground truth (Ogata thinning simulator with KNOWN parameters is the ground-truth generator); suite+ruff green every commit.

---

### Task 1: Hawkes simulator + exponential-kernel MLE (`estimators/hawkes.py`)

**Interfaces:**
- `simulate_hawkes_exp(mu: float, alpha: float, beta: float, t_end: float, seed: int) -> np.ndarray` — event times via Ogata thinning for intensity λ(t) = mu + Σ alpha·beta·exp(−beta(t−t_i)) (branching ratio n = alpha; document the parameterization explicitly — alpha IS the branching ratio, kernel integral = alpha).
- `fit_hawkes_exp(times: np.ndarray, t_end: float) -> HawkesFit` frozen dataclass (mu, alpha, beta, loglik, converged: bool) — MLE via the O(N) recursion (research/02 §4.1: R_i = exp(−beta·Δt_i)(1+R_{i−1})); optimize with a small hand-rolled multi-start Nelder-Mead over (log mu, logit alpha, log beta) — numpy only, no scipy; 5 starts, document convergence criteria; alpha constrained to (0,1) via logistic transform.
- `branching_count_variance(times: np.ndarray, window: float, t_end: float) -> float` — Hardiman-Bouchaud model-free n̂ = 1 − sqrt(mean(counts)/var(counts))… IMPLEMENT THE CORRECT FORMULA: for a stationary Hawkes, var(N_W)/mean(N_W) → 1/(1−n)² for large windows, so n̂ = 1 − sqrt(mean/var). Document the large-window assumption and that windows must be ≫ kernel timescale.

**Tests (synthetic ground truth, the point of this task):**
1. Simulate (mu=0.5, alpha=0.4, beta=2.0, t_end=200_000, seed=7): fit recovers alpha within ±0.03, mu within ±0.05, beta within ±20% — loop 3 seeds, across-seed sd(alpha) < 0.02.
2. Count-variance estimator on the same sims with window=500: n̂ within ±0.05 of 0.4.
3. Near-critical sim (alpha=0.85): both estimators recover within stated (wider) tolerances — document the widening.
4. POISSON REFUTATION TEST (the Filimonov-Sornette trap): fit on a pure Poisson stream (alpha=0) — fitted alpha must be < 0.05 and count-variance n̂ < 0.05. THEN the trap itself: a REGIME-SWITCHING Poisson (rate 0.5 for 5000s, rate 2.0 for 5000s, alternating) — record what both estimators report (they WILL report spurious n̂ > 0; assert n̂ > 0.2 to document the trap is real) and assert the test's docstring explains this is the known spurious-endogeneity failure mode that motivates deseasonalization.
- [ ] TDD; commit `feat: Hawkes simulator, exp-kernel MLE, count-variance branching estimator`.

### Task 2: Event-time deseasonalization (`signals/eventtime.py`)

- `intraday_rate_profile(ts: np.ndarray_datetime_ms, n_bins: int = 48) -> np.ndarray` — mean event rate per time-of-day bin (UTC), normalized to mean 1.
- `rescale_to_business_time(ts, profile) -> np.ndarray` — deterministic time-change removing the daily+funding-clock seasonality (integrate the binned rate); returns float seconds in rescaled time. Test: a synthetic inhomogeneous Poisson with a planted 2-cycle daily profile becomes ACF-flat/CV≈1 after rescaling (KS-style check on inter-event CV), and a planted Hawkes ON TOP of seasonality: raw fit overestimates alpha, rescaled fit recovers within tolerance — this test IS the deseasonalization justification.
- [ ] TDD; commit `feat: business-time rescaling for seasonality-robust Hawkes fitting`.

### Task 3: Branching-ratio panel (`analyses/q6_endogeneity.py`)

Per symbol over the 16-symbol panel + the 40 most-active universe symbols (56 total, June 2023 aggTrades, aggressor events): rescale to business time; fit on K=6 disjoint 2-day sub-windows: report per symbol median alpha, IQR, count-variance n̂ (window ≫ fitted 1/beta, documented), raw-vs-rescaled alpha delta (seasonality bias), convergence failures → failures list. Cross-section: alpha vs log10 activity regression. Outputs: results/q6_endogeneity.{md,json,parquet,png} (alpha vs activity, errorbars = sub-window IQR; second panel: MLE vs count-variance scatter with y=x). md findings: median endogeneity level vs the near-critical claims in the literature (research/02: EJF 2022 found BTC ≈ FX levels); whether n̂ scales with activity; the estimator-disagreement honesty table. CLI as before. Synthetic test: planted Hawkes symbols through parquet fixtures recover alphas; regime-switching fixture lands with documented inflated n̂ flagged by the raw-vs-rescaled delta.
- [ ] TDD → REAL RUN → commit with headline numbers.

### Task 4: Execution simulator (`execution/simulator.py` + `analyses/q7_execution.py`)

- Simulator: replay a symbol-day's aggressor events + prior-mids (existing signals); execute a parent order of Q units over horizon H as child orders per schedule; cost model: each child of size q pays half-spread (from bookTicker at that ts) + temporary impact from the SYMBOL'S OWN measured kernel (Q5's G, scaled by q/typical-event-size — document the linear-scaling assumption and its violation risk per the square-root literature) + adverse drift = realized mid move over the child's interval. Implementation shortfall vs arrival mid. NO trading claims — this is a cost-model comparison, md must say so.
- Schedules: TWAP (uniform), front-loaded exponential (AC-flavored, decay parameter from the kernel timescale), flow-reactive (pause children when trailing signed-flow imbalance opposes the parent side beyond a threshold; parameters fixed a priori, no tuning on evaluation data — split days: calibrate thresholds on days 1–3, evaluate on days 4–7, state this).
- `q7_execution.py`: for 6 panel symbols spanning liquidity × both sides × Q ∈ {2, 10} × typical-event-size: mean shortfall per schedule with across-day dispersion; outputs results/q7_execution.{md,json,png}. Findings phrased from data; caveats: model-based own-impact, no queue/latency, 4 evaluation days.
- Synthetic test: constructed replay where the reactive schedule's advantage is planted (adverse flow clusters) → reactive beats TWAP; and a no-signal replay → all schedules statistically tie (dispersion overlap).
- [ ] TDD → REAL RUN → commit with the shortfall table.

### Task 5: Synthesis
LEARNING.md Phase-3 section (Hawkes intuition popcorn→criticality, the estimation-pitfall fight as OUR OWN measured raw-vs-rescaled deltas, endogeneity verdict vs literature, execution results and what they do NOT claim); README galleries + reproduction; drill += 4 (branching ratio meaning; why single-n̂ is dishonest; what the regime-switching trap does; why our execution numbers aren't a trading claim). All numbers artifact-verified programmatically.

**Then:** final verification workflow (independent reproduction incl. one full-pipeline symbol; whole-branch opus review; one fix wave) → PR → merge.
