# Phase 2: The Propagator Package Across the Binance Cross-Section

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Checkbox steps.

**Goal (the novel contribution, adversarially verified as unoccupied in research/04):** execute the joint propagator package — sign-ACF exponent γ, response R(ℓ), *deconvolved impact kernel* G(ℓ) with decay exponent β̂, and the critical-balance test β̂ ≈ (1−γ)/2 — as ONE package across a crypto cross-section spanning 3+ decades of liquidity. Secondary novel thread from Phase 1.5: the cross-sectional map of aggressor-flow persistence (BTC anti-persistent p_flip=0.579 vs ETH persistent 0.482 — does persistence flip with liquidity?).

**Data (scouted 2026-08-16, .superpowers/scratch/universe_ranked.txt):** 207 USDT-M perps with ≥1MB June-2023 aggTrades (7.7GB zipped total, syncing). Kernel panel: 16 symbols spanning ~509MB→~10MB monthly activity, all with daily bookTicker confirmed: BTCUSDT ETHUSDT 1000PEPEUSDT BCHUSDT XRPUSDT SOLUSDT LTCUSDT DOGEUSDT ARBUSDT OPUSDT SUIUSDT LINKUSDT APTUSDT INJUSDT EDUUSDT IDUSDT — 7-day bookTicker slices (2023-06-01..2023-06-07) each.

**Tech:** existing stack. Estimators pure numpy; analyses follow the established CLI/md/json/png conventions; every estimator synthetic-validated first; suite + ruff green at every commit; sample-limitation caveats in every results md.

---

### Task 1: Propagator deconvolution estimator (`estimators/propagator.py`) — THE new method

**Interfaces produced:**
- `sign_price_cross_cov(signs: np.ndarray, dm: np.ndarray, max_lag: int) -> np.ndarray` — b[j] = E[dm_t · signs_{t−j}] for j=0..max_lag−1 (dm[t] = m[t+1]−m[t] aligned so dm[t] is the price change caused at/after event t; arrays same length; last dm may be dropped).
- `deconvolve_kernel(b: np.ndarray, acf: np.ndarray) -> np.ndarray` — solve the Toeplitz system Σ_n κ[n]·C[|j−n|] = b[j] for κ (length = len(b)) by numpy lstsq on the explicitly-built Toeplitz matrix of the sign ACF C; document conditioning (report rcond via matrix rank check; raise if the system is singular).
- `cumulative_kernel(kappa: np.ndarray) -> np.ndarray` — G[ℓ] = Σ_{n<ℓ} κ[n], G[0]=0. (In the discrete MA representation Δm_t = Σ_n κ[n]ε_{t−n} + noise, the propagator response to one unit event after ℓ steps is exactly this partial sum.)

**Mathematical contract the tests enforce:**
1. White-noise signs (C = δ): κ must equal b exactly (system is identity) — plant an arbitrary decaying κ0, build dm by convolution + small noise, recover κ0 within noise tolerance.
2. Long-memory signs (fractional_signs d=0.35): plant power-law G0(ℓ)=ℓ^(−0.35) (κ0 = its first difference), build mids by convolution, compute b and acf from the DATA (not the theory), deconvolve, fit `fit_power_law` on recovered G over lags 5..max_lag//2 — recovered β within ±0.07 of 0.35. THIS IS THE KEY TEST: naive interpretation of R (without deconvolution) must FAIL this recovery (assert the response-based exponent differs from 0.35 by more — demonstrating the method matters).
3. Shape/度 guards: mismatched lengths ValueError; singular C ValueError.

- [ ] Steps: failing tests → red → implement → green → ruff → commit `feat: propagator kernel deconvolution estimator, synthetic-validated`.

### Task 2: Trades-side cross-section (`analyses/q4_cross_section.py`)

Per symbol over the 207-universe (aggTrades 2023-06 parquet): n_events, γ̂+stderr (lags 10..500 as Phase 1), lag-1 ACF, p_flip (P(sign_{t+1}≠sign_t)), zigzag amplitude (Phase-1.5 definition), total qty. Robustness: per-symbol try/except — failures logged to the results json with reason, never abort the run; memory: one symbol at a time. Outputs: `results/q4_cross_section.{md,json,parquet}` + two figures: γ̂ vs log(activity) and p_flip vs log(activity) (activity = n_events). md: methodology, top-line table (top/bottom 10 by activity), the cross-sectional regressions (OLS of γ̂ and p_flip on log n_events, with the honest caveat that stderr is heteroskedastic across symbols), findings statements, caveats (single month, June-2023 regime per Phase 1.5). CLI: `--root --out --symbols-file --min-events 1000000`.
- [ ] Synthetic test: 3 fake symbols written via parquet_path with markov_signs of different p_repeat — cross-section recovers their distinct lag-1 ACFs and skips a deliberately-missing symbol with a logged failure. TDD → run REAL (after sync completes) → commit with real headline numbers in the body.

### Task 3: Kernel panel (`analyses/q5_kernel_panel.py`)

For the 16 panel symbols: sync_days bookTicker 2023-06-01..07 (inside the task; ~16×7 daily files); events (June aggTrades filtered to the week) + strictly-prior mids → per symbol: R(ℓ) to 300, b, C, deconvolved G, β̂ = fit_power_law(G, 5, 150), γ̂_week (same week's trades), balance residual Δ = β̂ − (1−γ̂_week)/2. Outputs `results/q5_kernel_panel.{md,json,png}` (png: G(ℓ) log-log all symbols + Δ vs log activity). md must state the headline: does the critical-balance relation hold across the crypto cross-section (|Δ| small and unstructured) or fail systematically? Either answer is the novel result. Caveats: 7-day window, L1-mid only, deconvolution assumes the linear propagator model (state it).
- [ ] Synthetic test reusing Task-1 machinery through the real pipeline (planted kernel in parquet fixtures). TDD → run REAL → commit with the Δ table in the body.

### Task 4: CI (`.github/workflows/ci.yml`)

ubuntu-latest, checkout, install uv (astral-sh/setup-uv@v4), `uv sync`, `uv run ruff check src/ tests/`, `uv run pytest -m "not network" -q`. Python 3.12. Commit `ci: offline test + lint workflow`.

### Task 5: Synthesis

Update LEARNING.md (new section: the propagator model, deconvolution intuition — "unscrambling the echo from the crowd's reaction" — the critical-balance meaning, OUR cross-sectional verdicts with numbers) and README (Phase-2 results gallery + reproduction). Interview drill += 3 questions on kernel-vs-response and the balance test. All numbers from committed artifacts.

**Then:** verification workflow (independent reproduction of headline numbers + adversarial review), fix rounds, PR, merge — same machinery as Phase 1.5.
