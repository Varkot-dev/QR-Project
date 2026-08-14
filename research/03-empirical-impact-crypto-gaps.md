# Empirical Price Impact and the Crypto Gap: Literature Review and Feasible Research Corridor

## Part 1 — The empirical price impact canon

### 1.1 The square-root law: origin and establishment

The central empirical fact of modern microstructure: the average price impact of a **metaorder** (large parent order split into many child trades) of volume Q scales as

**I(Q) ≈ Y · σ · (Q/V)^δ, with δ ≈ 0.4–0.6**

(σ daily volatility, V daily volume). Impact is *concave*: trading twice as much moves the price far less than twice as far.

**Genealogy:**
- Torre & Ferrari (BARRA, ~1997), Grinold & Kahn — earliest practitioner statements.
- **Almgren, Thum, Hauptmann & Li (2005)** — canonical broker-side calibration (Citigroup internal executions); temporary impact concave, exponent ≈ 0.6.
- **Tóth et al. (2011), "Anomalous Price Impact and the Critical Nature of Liquidity," PRX** ([arXiv:1105.1694](https://arxiv.org/abs/1105.1694)) — THE landmark: ~500k CFM metaorders on futures; δ ≈ 0.5 over decades of Q/V; proposed **latent liquidity / locally-linear order book (LLOB)** theory — revealed liquidity vanishes linearly near the price, mechanically generating √ impact. Framed the law as critical/diffusive, not informational.
- Bershova & Rakhlin (2013), Brokmann et al. (2015) — confirmations + post-execution decay.
- **Donier & Bonart (2015), "A Million Metaorder Analysis of Market Impact on the Bitcoin"** ([arXiv:1412.4503](https://arxiv.org/pdf/1412.4503)) — **first crypto confirmation**, Mt. Gox data *with user IDs* (a historical accident enabling true metaorder reconstruction — never repeated): δ ≈ 0.5 over four decades of size, in a market with no HFT. Strongest evidence the law is *mechanical*, not equity-ecology-specific.
- Options too: [arXiv:1602.03043](https://arxiv.org/pdf/1602.03043).
- **Sato & Kanazawa (2024), complete TSE survey** ([arXiv:2411.13965](https://arxiv.org/abs/2411.13965)) — exchange trader-level data, ~2,000 stocks: strict universality δ = 1/2; statistically rejects non-universal alternatives. State of the art.
- Reviews: [Bouchaud Substack](https://bouchaud.substack.com/p/the-square-root-law-of-market-impact); [arXiv:2205.07385](https://arxiv.org/pdf/2205.07385).

### 1.2 Debates
1. **Functional form at edges:** Zarinelli et al. (2015, [arXiv:1412.2152](https://arxiv.org/pdf/1412.2152)) argued logarithmic; **Bucci et al. (2019, PRL** [arXiv:1811.05230](https://arxiv.org/pdf/1811.05230)**)** resolved much: linear (Kyle) regime at small Q crossing over to square-root, as LLOB predicts.
2. **Impact decay after execution:** power-law reversion to a nonzero plateau ([arXiv:1901.05332](https://ar5iv.labs.arxiv.org/html/1901.05332)); quantitatively unsettled.
3. **Mechanical vs informational origin:** Bitcoin result, options result, and **"Generating realistic metaorders from public data" (2025, [arXiv:2503.18199](https://arxiv.org/pdf/2503.18199))** — synthetic metaorders from public tape reproduce the law — all push mechanical. Active fault line.
4. Alternative theory routes: Kyle–Obizhaeva invariance; Gabaix power-law funds; LLOB. Tokyo survey discriminates in favor of strict universality.

### 1.3 The metaorder data bottleneck
Every clean study needed: (a) a fund's own records (CFM, AB, Citi), (b) consultancy client data (ANcerno), (c) regulator/exchange trader IDs (Spanish exchanges; TSE), or (d) a leak-like accident (Mt. Gox user IDs). **The tape alone doesn't say which trades share a parent — this is why academics without industry partnerships are locked out, and why tape-only methods are the entry point.**

Public-data workarounds: metaorder synthesis ([arXiv:2503.18199](https://arxiv.org/pdf/2503.18199)); conditioning on signed order flow instead ([arXiv:2004.08290](https://arxiv.org/pdf/2004.08290)).

### 1.4 What IS measurable from the public tape
- **Bouchaud et al. (2004)** propagator/response-function methodology ([arXiv:cond-mat/0307332](https://arxiv.org/abs/cond-mat/0307332)): R(ℓ) = ⟨ε_t · (p_{t+ℓ} − p_t)⟩ from trades+quotes alone; fit decaying kernel G(ℓ) ~ ℓ^(−β); the balance β ≈ (1−γ)/2 with flow-memory exponent γ. Canonical: *Trades, Quotes and Prices* Chs. 13–14.
- Single-trade response is universal across stocks ([arXiv:1702.08029](https://arxiv.org/pdf/1702.08029)).
- **Cont, Kukanov & Stoikov (2014), "The Price Impact of Order Book Events"** ([arXiv:1011.6402](https://arxiv.org/pdf/1011.6402)) — **order flow imbalance (OFI)**: price change linear in net best-quote imbalance, slope ~ 1/depth, R² ≈ 65–70%. Extensions: multi-level OFI; cross-asset OFI ([arXiv:2112.02947](https://arxiv.org/pdf/2112.02947)); nonparametric cross-impact ([arXiv:2510.06879](https://arxiv.org/pdf/2510.06879)).

## Part 2 — Crypto-specific scan (2019–2026)

### Impact / square-root on crypto
- Donier & Bonart (2015): still the *only* clean crypto metaorder study — and it describes 2011–2013 Bitcoin, a market that no longer exists (no HFT, no perps then).
- **Silantyev (2019)** (*Digital Finance*, [Springer](https://link.springer.com/article/10.1007/s42521-019-00007-w)) — Cont-style OFI/TFI on BitMEX XBTUSD; trade-flow imbalance beats quote-level OFI; linear regime confirmed.
- **Albers et al. (2021)** ([arXiv:2108.09750](https://ideas.repec.org/p/arx/papers/2108.09750.html)) — sub-second cross-exchange lead-lag + cross-impact predicting 500ms returns (R² 10–37%), validated with $1.5M live trading.
- **Talos (2025)** industry model ([link](https://www.talos.com/insights/an-empirical-model-of-market-impact-in-cryptocurrency-trading)) — 50k+ parent metaorders, top-60 pairs. **Proprietary OMS data — the classic bottleneck reappearing in crypto.**
- Community replication ([crypto-market-impact GitHub](https://github.com/SLMolenaar/crypto-market-impact)): naive pseudo-metaorder reconstruction on Binance gives exponent ≈ 0.1 — far off 0.5 — showing **how badly naive reconstruction biases the exponent**. "Does the √ law hold on modern Binance?" is *not settled publicly*.
- DEXs: AMM slippage is mechanical (bonding curve) — [Angeris et al.](https://arxiv.org/pdf/1911.03380), [SoK](https://arxiv.org/pdf/2103.12732); the open question is *realized* aggregate impact including arb reversion.

### Hawkes on crypto
Endogeneity of BTC near FX-like criticality ([EJF 2022](https://www.tandfonline.com/doi/abs/10.1080/1351847X.2020.1791925)); Hawkes LOB forecasting ([arXiv:2312.16190](https://arxiv.org/pdf/2312.16190), [Springer 2026](https://link.springer.com/article/10.1007/s10203-026-00570-z)); state-dependent Hawkes portable from equities ([QF](https://www.tandfonline.com/doi/full/10.1080/14697688.2021.1983199)).

### Liquidation cascades on perps
- Soska et al. (WWW '21, [ACM](https://dl.acm.org/doi/fullHtml/10.1145/3442381.3450059)) — BitMEX leverage ecology.
- **Early-warning signals across seven BTC cascades 2022–2025** ([arXiv:2607.27070](https://arxiv.org/abs/2607.27070)) incl. the ~$19B Oct 10–11, 2025 event — only taker-flow variance compression is a consistent precursor; cascades look like **shock-driven first-order transitions, not critical ones**. All public-tier data.
- **Subcritical branching inside the Oct 2025 crash** ([arXiv:2608.03616](https://arxiv.org/abs/2608.03616)) — n ≈ 0.1–0.2 at event level; margin-mechanics avalanche; 88% of forced selling within 30 min.
- Funding-rate machinery: [NBER w32936](https://www.nber.org/system/files/working_papers/w32936/w32936.pdf).

### Data availability — the crypto advantage
- **[data.binance.vision](https://data.binance.vision/)** ([GitHub](https://github.com/binance/binance-public-data)): free CSVs of **every tick** — spot & futures `trades`/`aggTrades` (**taker-side pre-signed — no Lee-Ready needed**), klines; USDT-M futures also: `bookTicker` (best bid/ask tape), sampled `bookDepth`, `metrics` (open interest, long/short ratios), `liquidationSnapshot`.
- Caveats: full L2/L3 book *history* not in the dump (record websockets yourself or use a vendor); since mid-2021 the **liquidation stream is throttled to ≤1 msg/sec/symbol** — a *sample*, not a census; treat accordingly.
- **An undergraduate has better raw data access in crypto than most equity academics had in 2005.**

**The gap in one sentence:** crypto has equity-grade tape data for free, one clean-but-obsolete metaorder study (Mt. Gox 2015), a proprietary industry model (Talos 2025), scattered Hawkes/OFI papers mostly BTC-only — and almost nothing systematically applying the propagator/response program, the OFI program, or metaorder synthesis across the modern crypto universe.

## Part 3 — Gap synthesis: feasible projects with free public tick data

(Difficulty: 1 = undergrad weekend; 5 = publishable-with-effort, a semester.)

### Q1. Propagator/response-function atlas of crypto (cross-sectional universality test)
Estimate R(ℓ), kernel G(ℓ), sign-autocorrelation exponent γ; test β ≈ (1−γ)/2 across 100+ Binance pairs spanning 4+ decades of liquidity, spot vs perp. **Why:** nobody has tested whether the equity fine-tuning holds in retail-dominated 24/7 markets; the cross-sectional collapse (or failure) vs liquidity is real evidence about mechanism. **Data:** aggTrades + bookTicker, free. **Difficulty 2–3.** Open: only BTC/ETH touched, never the liquidity cross-section.

### Q2. Does the square-root law hold on modern Binance? (metaorder synthesis)
Port [arXiv:2503.18199](https://arxiv.org/pdf/2503.18199) to Binance tape; measure δ; quantify the bias of naive run-based reconstruction (known to give δ ≈ 0.1). **Why:** whether the law survived HFT-ification/perps/100x leverage is open and checkable. **Difficulty 3–4** (validate the method on equity ground truth first). A careful bias study alone is a contribution.

### Q3. Liquidations as free, labeled, uninformed metaorders ★
Use futures `liquidationSnapshot` records as *marked forced trades — uninformed by construction* — and measure (a) impact scaling with size (√ or linear?), (b) post-liquidation decay vs matched ordinary aggressive flow. **Why:** attacks the mechanical-vs-informational debate with a natural experiment equities cannot offer (no ANcerno file labels forced metaorders). Same √ + same decay plateau ⇒ mechanical camp; full reversion ⇒ informational. **Caveat:** ≥2021 stream throttling (≤1 msg/sec/symbol) — fine for impact-per-observed-liquidation, dangerous for cascade totals; document it. **Difficulty 3.** Genuinely underexplored — **best novelty-per-effort on this list.**

### Q4. State-dependent OFI: does Cont's linear law bend under stress?
Replicate OFI linearity on Binance (Silantyev did BitMEX), then map where it *breaks*: extreme imbalance, 8h funding windows, cascade windows, depleted depth. LLOB predicts impact steepens as revealed liquidity vanishes — Oct 2025 (multi-year depth lows) is the test bed. **Difficulty 2 replication / 4 stress analysis.** Baseline well-covered; breakdown mapping open.

### Q5. Hawkes endogeneity across the crypto cross-section and through time
Branching ratios (power-law kernels, robust estimators) across many pairs, 2019–2026: has crypto endogeneity risen toward equity/FX near-criticality as the market professionalized? Does n̂ *forecast* fragility, given cascades run subcritically? **Why:** reconciling "steady state near-critical" (EJF 2022) with "cascades are first-order, subcritical" ([arXiv:2608.03616](https://arxiv.org/abs/2608.03616)) is a live tension. **Difficulty 4** (real pitfalls; lean on `tick` + the misspecification literature). Panel version open.

### Q6. Spot–perp cross-impact via multivariate OFI
Cross-impact matrix between BTC spot, BTC perp, ETH, alts on one venue; has price discovery migrated spot→perp over time? Perps are ~93% of crypto futures volume. **Data:** same-venue time-synced tapes, free. **Difficulty 3.** The within-venue spot-perp panel and "discovery migration" series would be new.

### Q7. Realized aggregate impact on DEXs vs CEXs
Uniswap v3 fills (on-chain, free) — per-trade slippage is mechanical, but what's the *realized* response including arb reversion? Discriminates order-book-ecology vs aggregate-liquidity theories of the √ law. **Difficulty 4** (on-chain plumbing; MEV/sandwich filtering essential). Realized-response framing open.

### Q8. Impact law under liquidity collapse: a direct LLOB test at the extremes ★
Using the seven 2022–2025 cascade windows as high-stress samples: does impact steepen as 1/√(liquidity) as LLOB predicts, or discontinuously as first-order-transition papers suggest? **Why:** connects the impact-law and cascade literatures — currently disjoint — and is THE practical risk question in crypto. **Data:** aggTrades, bookTicker, bookDepth, metrics — free. **Difficulty 3–4.** Genuinely open.

### Honest triage
- **Well-covered, don't oversell:** baseline OFI linearity on a major pair; BTC-only Hawkes endogeneity; mechanical AMM slippage; BTC cross-exchange lead-lag.
- **Open with method-in-hand:** Q1, Q2, Q6.
- **Most novel per unit effort:** **Q3 (liquidations as labeled uninformed metaorders)** and **Q8 (impact law under liquidity collapse)** — both exploit data only crypto publishes free, both plug into the field's biggest open debates (mechanical vs informational impact; critical vs first-order fragility), both executable in Python on data.binance.vision downloads.
