# A Survey of the Classical Optimal Execution Literature

**Scope.** This survey traces the canonical lineage of "how should a large trader break up a big order?" — from Bertsimas & Lo's dynamic programming (1998) through Almgren–Chriss (2000), Obizhaeva–Wang's order-book resilience model (2013, circulated from 2005), Gatheral's no-dynamic-arbitrage constraints (2010), and the Bouchaud-school propagator models — framed by the industry practice (VWAP, TWAP, implementation shortfall) that motivated all of it.

---

## 0. The industry context: implementation shortfall, VWAP, and TWAP

**Perold (1988), "The Implementation Shortfall: Paper versus Reality," *Journal of Portfolio Management* 14(3), 4–9.**
Sources: [Journal of Portfolio Management](https://www.pm-research.com/content/iijpormgmt/14/3/4), [Quantitative Brokers history of IS](https://www.quantitativebrokers.com/blog/a-brief-history-of-implementation-shortfall)

**Berkowitz, Logue & Noser (1988), "The Total Cost of Transactions on the NYSE," *Journal of Finance* 43(1), 97–112**, and **Madhavan (2002), "VWAP Strategies," *Journal of Trading*** ([PDF](https://www.smallake.kr/wp-content/uploads/2016/03/TP_Spring_2002_Madhavan.pdf)).

**Plain-English summary.** Compare your *real* portfolio against an imaginary "paper portfolio" that could trade the full amount instantly at the decision price with no costs. The difference — the **implementation shortfall** — is what execution actually cost you, including the *opportunity cost* of shares never filled because the price ran away. Like comparing your actual road-trip time to the Google Maps estimate: the gap captures traffic, wrong turns, and the detour you gave up on. Separately, **VWAP** (volume-weighted average price) is the humbler benchmark: did you trade at least as well as the "average trader" that day? VWAP algorithms slice orders in proportion to the market's typical intraday volume pattern (the U-shape: heavy at open and close, light at lunch); **TWAP** slices evenly through time.

**Assumptions.** These are benchmarks, not impact models. VWAP quietly assumes your own trading doesn't move the VWAP itself — fails for large orders (if you're 30% of daily volume you drag the benchmark toward your own fills). Implementation shortfall makes no impact assumption; it's a *measurement*. TWAP implicitly assumes impact and liquidity are constant through the day.

**What the next work fixed.** Perold defined the cost but gave no recipe for minimizing it. Bertsimas & Lo supplied the first rigorous optimization answer.

---

## 1. Bertsimas & Lo (1998): dynamic programming for execution

**"Optimal Control of Execution Costs," *Journal of Financial Markets* 1, 1–50.** ([MIT PDF](https://www.mit.edu/~dbertsim/papers/Finance/Optimal%20control%20of%20execution%20costs.pdf))

**Plain-English summary.** You must buy S shares over T periods; every share pushes the price up. Posed as textbook **dynamic programming** (work backwards from the last period). Headline result, beautifully anticlimactic: under linear impact and a random-walk price, the optimal strategy is to **trade the same amount every period** — plain TWAP is provably optimal. Intuition: impact costs are convex (trading 2x in one period costs more than 2×1x spread over two, like congestion pricing), and since the future price is a fair coin flip, there's no reason to front- or back-load. With a predictive signal, the schedule tilts.

**Key assumptions.**
- **Linear impact** — price concession proportional to trade size.
- **Permanent impact** — impact stays in the price forever; no decay component.
- **Deterministic impact coefficients**; additive arithmetic random walk; **risk neutrality** (minimize expected cost only); discrete time, hard deadline.

**What the next paper fixed.** A risk-neutral trader is happy to trade arbitrarily slowly — but slow trading means longer exposure to price *risk*. Bertsimas–Lo has no way to express "I'd pay more impact to be done sooner." Almgren–Chriss added that, and separated temporary from permanent impact.

---

## 2. Almgren & Chriss (2000): the canonical risk/impact tradeoff

**"Optimal Execution of Portfolio Transactions," *Journal of Risk* 3(2), 5–39.** Follow-on calibration: [Almgren, Thum, Hauptmann & Li (2005), "Direct Estimation of Equity Market Impact"](https://www.cis.upenn.edu/~mkearns/finread/costestim.pdf)

**Plain-English summary.** Sell fast → crash the price (huge impact, no risk). Sell slow → tiny impact but a month of exposure to bad news (huge risk). Like a giant tub of ice cream: gulp it (brain freeze) or eat slowly (it melts). Formalized as **mean–variance**: minimize *expected cost + λ × variance of cost*, λ = risk aversion. The solutions trace an **efficient frontier of execution**. For a risk-averse trader the optimal trajectory is **front-loaded** — fastest at the start, decaying exponentially, with an "urgency" half-life that grows with volatility and risk aversion and shrinks with liquidity. λ = 0 recovers Bertsimas–Lo's TWAP. Closed forms, three interpretable parameters — still the skeleton of most brokers' IS algos 25 years later.

**Key assumptions.**
- **Two-component impact:** a **permanent** part (shifts price forever) plus a **temporary** part that penalizes only the current slice and vanishes instantly. Impact decay is a step function — all-or-nothing.
- **Linear** in trade size/rate for both parts. (Linear *permanent* impact is exactly what rules out manipulation — [Huberman & Stanzl 2004](https://www.econometricsociety.org/publications/econometrica/2004/07/01/price-manipulation-and-quasi%E2%80%90arbitrage).)
- **Deterministic, stationary parameters** → the whole schedule is computable at 9:30am and never revised ("static").
- Arithmetic Brownian motion, no order book, no strategic traders.

**What the next papers fixed.** (i) Step-function decay is empirically false — impact decays *gradually* (→ Obizhaeva–Wang resilience). (ii) Linearity is empirically false — Almgren's own 2005 Citigroup calibration found concave (~3/5 power) impact. (iii) Static schedules ignore intraday information → adaptive/stochastic-control literature.

---

## 3. Obizhaeva & Wang (2013): limit order book resilience

**"Optimal Trading Strategy and Supply/Demand Dynamics," *Journal of Financial Markets* 16, 1–32** (circulated 2005). ([MIT PDF](https://web.mit.edu/wangj/www/pap/ObizhaevaWang13.pdf), [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=752022))

**Plain-English summary.** Model the actual **limit order book**: a standing wall of orders. Aggressive buying eats a chunk of the wall; the best price jumps. But the market is **resilient** — like a sponge with a thumb pressed in, new limit orders refill the dent, exponentially at rate ρ. Optimal strategy is strikingly different from the smooth AC curve: **a big block at the start** (eat deep once), a **steady trickle** calibrated to consume refilling liquidity (keep the dent at constant depth), and **a final block at the deadline**. The "bucket with two spikes." Infinitely fast resilience recovers AC-style temporary impact; zero resilience recovers permanent impact — the model *nests* the classical dichotomy.

**Key assumptions.** Block-shaped book with constant density (→ linear impact, now microfounded); **exponential resilience** at constant rate; deterministic book dynamics; no strategic traders detecting your pattern; martingale fundamental price.

**What the next paper fixed.** Gatheral asked: which (impact shape, decay shape) pairs are even logically permissible? Answer: exponential decay is compatible *only* with linear instantaneous impact. Since impact is empirically concave, exponential resilience is untenable → power-law decay (Bouchaud propagator world). Later work: general book shapes (Alfonsi & Schied), stochastic depth/resilience, manipulation-free kernel conditions.

---

## 4. Gatheral (2010): no-dynamic-arbitrage constraints

**"No-Dynamic-Arbitrage and Market Impact," *Quantitative Finance* 10(7), 749–759.** ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1292353); [Baruch lecture slides](https://mfe.baruch.cuny.edu/wp-content/uploads/2012/09/Chicago2016OptimalExecution.pdf))

**Plain-English summary.** A **consistency police** for the zoo of impact models: which (shape, decay) pairs allow making money from nothing via a round trip (buy, sell back, end flat, profit purely from your own mechanical price dynamics)? Any model permitting that is broken — like checking a proposed physics law by asking if it allows perpetual motion. Results: (i) **nonlinear permanent impact admits arbitrage** — only linear permanent survives; (ii) **exponential decay requires linear instantaneous impact** — so concave impact cannot decay exponentially; (iii) for power-law impact f(v) ∝ v^δ and power-law decay G(τ) ∝ τ^(−γ), no-arbitrage requires roughly **γ + δ ≥ 1** — and empirical estimates (δ ≈ 0.6, γ ≈ 0.4) sit almost exactly *on* the boundary. Markets appear to self-organize to the edge of arbitrage.

**Assumptions of the framework.** Price = unaffected martingale + sum over past trades of f(rate) × G(elapsed time) — a transient-impact/propagator representation with a universal kernel; deterministic, symmetric, no spread/fees.

**What followed.** The Bouchaud school supplied empirical content (measured propagators, order-flow memory, latent-liquidity theory). Refinements: transaction-triggered manipulation (Alfonsi–Schied–Slynko), cross-impact no-arbitrage ([arXiv:1612.07742](https://arxiv.org/abs/1612.07742)).

---

## 5. The Bouchaud school: transient impact and propagator models

**Bouchaud, Gefen, Potters & Wyart (2004), "Fluctuations and Response in Financial Markets," *Quantitative Finance* 4(2), 176–190** ([SSRN](https://doi.org/10.2139/ssrn.507322)). Modern treatment: **Bouchaud, Bonart, Donier & Gould, *Trades, Quotes and Prices* (Cambridge, 2018)**. Square-root law: [Bouchaud Substack explainer](https://bouchaud.substack.com/p/the-square-root-law-of-market-impact), [double square-root law on TSE data (arXiv 2502.16246)](https://arxiv.org/abs/2502.16246).

**Plain-English summary.** Empirical puzzle: trade *signs* are astonishingly **persistent** (a buy predicts more buys for days — institutions split metaorders into long streams), yet prices are nearly perfect random walks. If each trade permanently kicked the price, persistent buying would make prices trend — free money. Resolution: **each trade's impact decays as a power law precisely tuned to offset the power-law memory of order flow.** Picture pushing a boat: many pushes in the same direction, yet for the path to look like a drunkard's walk each push must fade just so. The **propagator model**: today's price = sum over all past signed trades, each weighted by decaying kernel G(τ) ∝ τ^(−β) — "impact echoes with a fade." The market sits at a self-organized **critical point** between trending and overshooting.

Crown jewel: the **square-root law** — a metaorder of size Q moves price by ~ σ√(Q/V), nearly independent of execution speed or slicing; among the most robust regularities in finance; explained via "latent liquidity" (Tóth et al. 2011: liquidity near the price is locally linear and vanishing, so eating it gives √ impact, of which roughly one-third to one-half persists).

**Key assumptions.** Transient power-law decay (β ≈ 0.2–0.5, critical balance β ≈ (1−γ_flow)/2); linear superposition of trade impacts; **mechanical, statistical impact** (mostly not information content — supported by [mechanical-impact evidence](https://arxiv.org/abs/2502.16246)); near-universality across assets.

**What later work fixed.** Plain propagator over-predicts long-horizon diffusion and can't match both single-trade response and metaorder concavity → history-dependent impact (TIM vs HDIM), latent-liquidity LLOB reaction–diffusion theory, cross-impact. Fed back into execution: optimal schedules under general decay kernels recover OW-like bucket shapes.

---

## Synthesis

### Most-falsified classical assumptions (ranked)

1. **Linear impact in size** — decisively falsified; metaorder impact is concave (square-root law); Almgren's own 2005 data rejected linearity. Nuance: *permanent* impact being linear is *required* (Huberman–Stanzl; Gatheral); nonlinearity must live in the transient part.
2. **All-or-nothing impact decay** (AC's split) — real impact decays gradually, power-law-like, with a partial permanent plateau; exponential resilience (OW) inconsistent with concave impact per Gatheral.
3. **Deterministic, stationary parameters** — depth/spread/volatility vary hugely intraday; static schedules dominated by adaptive policies.
4. **No strategic interaction** — other participants detect order-flow persistence and adjust (predatory trading, back-running).
5. **Mean–variance risk over fixed horizon** — crude but largely rehabilitable.
6. **Arithmetic Brownian price** — benign over hours, poor over weeks.

Aged well: the **risk/impact tradeoff** itself, and **no-dynamic-arbitrage as model selection** — with markets sitting near the arbitrage boundary.

### What's testable with public trade data only

**Directly testable:** long memory of order flow (sign autocorrelation); near-zero return autocorrelation alongside persistent flow; the **single-trade response function / propagator kernel** R(τ) (fit G(τ), adjudicate exponential vs power-law decay — OW vs Bouchaud — with zero proprietary data); concavity of aggregate impact (imbalance regressions); **Gatheral's boundary γ + δ ≈ 1** (both exponents estimable publicly); intraday stationarity; absence of round-trip money pumps.

**The hard ceiling:** the metaorder square-root law itself and post-metaorder decay require knowing *which trades belong to one parent order* — public prints don't say. Clean measurements (Tóth 2011, Almgren 2005, ANcerno) all used proprietary records. Workarounds (sign-run reconstruction, child-order aggregation) are noisy proxies.

**Bottom line.** A 25-year dialectic: benchmarks posed the measurement problem → Bertsimas–Lo solved risk-neutral scheduling (TWAP) → Almgren–Chriss added risk (the frontier) → Obizhaeva–Wang restored the book's memory → Gatheral proved which models are coherent → Bouchaud showed what impact empirically is: transient, concave, power-law-fading, mostly mechanical. The assumptions that died, died at the hands of measurements mostly replicable by anyone with tick data.
