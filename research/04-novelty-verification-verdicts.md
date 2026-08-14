# Adversarial Novelty Verification — Verdicts on the Three Claimed Gaps

Three independent agents were tasked with *refuting* the gap claims from `03-empirical-impact-crypto-gaps.md`. All three claims failed in strong form. Summary + key prior art below. Net conclusion for this project: novelty-hunting is the wrong selection criterion; benchmark-rich replication is the right one for a learning project.

## Claim 1: "Hawkes-based optimal execution calibrated to crypto — full loop unpublished"
**Verdict: narrowly true, oversold.** The exact conjunction (Hawkes model + crypto data + backtest) appears unpublished as of Aug 2026, but:
- The identical full loop exists for **intraday energy markets** — Chatziandreou & Karbach 2025, [arXiv:2504.10282](https://arxiv.org/abs/2504.10282) (calibration → reactive strategy → backtest vs TWAP/VWAP).
- Alfonsi & Blanc themselves did the full loop on **CAC40 equities** ([arXiv:1506.08740](https://arxiv.org/abs/1506.08740)); Warwick PhD thesis (Li 2013) on FX ([WRAP 57916](https://wrap.warwick.ac.uk/id/eprint/57916/)).
- Crypto already has full execution-backtest loops with non-Hawkes models: RL-Exec on BTC LOB replays beating TWAP/VWAP ([arXiv:2511.07434](https://arxiv.org/html/2511.07434)), Almgren-Chriss family on BNB ([arXiv:2303.10043](https://arxiv.org/abs/2303.10043)), Schnaubelt RL on ~300M trades ([EJOR 2022](https://www.sciencedirect.com/science/article/abs/pii/S0377221721003854)), Claremont senior thesis 2020 ([link](https://scholarship.claremont.edu/cmc_theses/2387/)).
- What remains: an asset-class transfer, not an open problem.

## Claim 2: "Liquidations as labeled uninformed metaorders — underexplored"
**Verdict: strong form false.**
- Garcia Seuma 2026 ([arXiv:2608.03616](https://arxiv.org/abs/2608.03616)) measures impact on labeled forced fills (Binance + Hyperliquid's per-user-attributed on-chain log). Bank of Canada 2025 measured DeFi liquidation impacts ([SWP 2025-12](https://www.bankofcanada.ca/2025/03/staff-working-paper-2025-12/)).
- **Donier & Bonart 2015** ([arXiv:1412.4503](https://arxiv.org/abs/1412.4503)) already showed uninformed flow's impact decays almost completely in Bitcoin — the headline mechanical-vs-informational result, a decade ago.
- Identification template (forced sales = uninformed) is standard equity fire-sale methodology (Coval–Stafford 2007).
- **Data landmine:** Binance liquidation feed censored since Apr 2021 — only largest liquidation per symbol per second ([Binance dev forum](https://dev.binance.vision/t/forceorder-data-missing-in-snapshots-since-april-27-2021/5151)). Size-dependent censoring biases any scaling estimate. Clean alternative (Hyperliquid) already being mined.
- Surviving sliver: per-liquidation size-scaling + matched-control decay on CEX records; isolated quiet-period liquidations. Crowded, fast-moving (3 papers in 18 months).

## Claim 3a: "Impact-law deformation under liquidity stress in crypto is open"
**Verdict: partially covered; strong form refuted.**
- Donier & Bouchaud 2015 ([arXiv:1503.06704](https://arxiv.org/abs/1503.06704)): impact-derived liquidity metric on full Mt.Gox flow; crash magnitude conditioned on liquidity — the program's spirit, on crypto, a decade ago.
- LLOB liquidity-dependence tested quantitatively in equities (Bucci et al. PRL 2019, [arXiv:1811.05230](https://arxiv.org/abs/1811.05230)); latent-liquidity calibration on 100+ assets ([arXiv:1808.09677](https://arxiv.org/abs/1808.09677)); endogenous liquidity crises ([arXiv:1912.00359](https://arxiv.org/abs/1912.00359)).
- A GitHub project already runs calm-vs-stress impact-form comparison on Binance BTC/ETH ([SLMolenaar](https://github.com/SLMolenaar/crypto-market-impact)); state-dependent order-flow effects on Binance futures ([arXiv:2607.09230](https://arxiv.org/abs/2607.09230)); cascade impact spikes measured ([arXiv:2608.03616](https://arxiv.org/abs/2608.03616)); industry quantification by Kaiko/Amberdata.
- Residual: peer-reviewed metaorder-level exponent/prefactor-vs-liquidity (1/√L) test through cascades across the coin cross-section.

## Claim 3b: "Propagator program never applied across crypto cross-section; only BTC/ETH touched"
**Verdict: "only BTC/ETH" factually false; narrow methodological gap survives.**
- Hyperliquid study 2026 ([arXiv:2606.15715](https://arxiv.org/abs/2606.15715)): **201 perp markets**, 641M fills, 4.3M hidden metaorders — impact curves (δ≈0.30), decay trajectories, permanent impact. Talos 2025: 50k metaorders across 60 pairs.
- Ingredients on crypto: OFI regressions (Silantyev 2019), cross-venue cross-impact (Albers et al.), LOB stylized facts (Schnaubelt et al. 2019), **Charles University master's thesis doing impact-of-book-events on Binance BTC** ([Erben 2023](https://dspace.cuni.cz/handle/20.500.11956/185017)) — precedent that this is student-feasible.
- Methodology exists cross-sectionally in FX ([arXiv:2104.09309](https://arxiv.org/pdf/2104.09309)).
- Residual: joint R(ℓ) + kernel G(ℓ) + sign-ACF + critical-balance verification across many crypto pairs on a CEX — unexecuted as a package, but an incremental transfer.

## Implication for this project
Everything in the planned core (sign autocorrelation, response function, OFI linearity on Binance) is **well-trodden** — which is the desired property for a learning project: published benchmarks exist at every step, including a master's thesis and practitioner replications to check against. Novelty is explicitly not the goal.
