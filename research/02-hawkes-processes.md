# Hawkes Processes in Market Microstructure: A Literature Survey

## 1. What is a Hawkes process?

A Poisson process is the standard model for "events at random over time" — its defining feature is amnesia. Markets are emphatically not like that: trades cluster; one trade begets more trades. The **Hawkes process** (Hawkes, 1971) is the minimal fix — a point process whose event rate (**intensity** λ(t)) rises with every event and decays back:

λ(t) = μ + Σ over past events tᵢ < t of φ(t − tᵢ)

- **μ**: baseline (exogenous) rate — events "from outside" (news, fundamental traders).
- **φ(·)**: the kernel — how much an event boosts the rate later, and for how long.

**Analogies:** earthquakes/aftershocks (Hawkes is the backbone of the ETAS seismology model — a big trade is the mainshock, algorithmic reactions are aftershocks); popcorn where each pop jostles neighboring kernels into popping — bursts and lulls instead of a steady patter.

### Branching interpretation and criticality

Exactly equivalent (Hawkes & Oakes, 1974): "immigrant" events arrive at rate μ; each event produces a Poisson number of "children" with mean **n = ∫φ(t)dt — the branching ratio**.

- n = 0.3 → ~30% of events are endogenous echoes; total activity amplified ×1/(1−n).
- n → 1 → **criticality**: cascades arbitrarily long; endogenous fraction → 100%; like a reactor where each fission triggers exactly one more.
- n > 1 → explosive/non-stationary.

The loaded empirical claim: **modern markets sit uncomfortably close to n = 1** — most activity is markets reacting to themselves. The formalization of Soros's "reflexivity."

### Kernel choices

- **Exponential** φ(t) = αβe^(−βt): one timescale; makes the process Markovian — why nearly all tractable control/execution papers use it; MLE is O(N).
- **Power-law** φ(t) ~ t^(−(1+ε)): long memory across seconds-to-days — what data actually favors. Common compromise: **sum of exponentials**.
- Kernel choice is not cosmetic: it changes the estimated branching ratio dramatically and (Jaisson–Rosenbaum) determines whether the large-scale price limit is ordinary or *rough* volatility.

Multivariate versions have a matrix of kernels (buys exciting sells, up-moves exciting down-moves); stability requires spectral radius of the kernel-integral matrix < 1.

## 2. Hawkes applied to order flow

- **Bacry, Mastromatteo & Muzy (2015), "Hawkes processes in finance"** ([arXiv:1502.04592](https://arxiv.org/abs/1502.04592)) — THE standard entry point: toolbox, estimation, applications. Read this first.
- **Bacry & Muzy (2014)** ([arXiv:1301.1135](https://arxiv.org/abs/1301.1135)) — 4-kernel system linking trades and mid-price: reproduces microstructure stylized facts *and* concave square-root-like metaorder impact with relaxation. The fitted kernels are an empirical "who pokes whom" causality map of the matching engine.
- Early fits: Large (2007) order-book resilience; Bowsher (2007); Lu & Abergel; Achab et al. (NPHC cumulant method).

### Reflexivity: Filimonov & Sornette (2012)
([arXiv:1201.3572](https://arxiv.org/abs/1201.3572)) Fit Hawkes to E-mini S&P mid-price changes; read off n as the endogenous fraction. **Findings:** endogeneity rose from ~30% (1998) to ~70%+ (2007–2010), tracking algorithmic/HFT growth; branching ratio spiked around the May 6, 2010 Flash Crash → n as a real-time instability barometer.

### Criticality: Hardiman, Bercot & Bouchaud (2013)
([arXiv:1302.1405](https://arxiv.org/abs/1302.1405)) **Direct rebuttal.** With a **power-law kernel** on the same data: kernel decays t^(−1.15) → t^(−1.45), and **integrates to ≈1 in every year 1998–2011**. Markets "are and always have been" near-critical; the apparent trend was an artifact of short-memory kernels. Follow-up: **Hardiman & Bouchaud (2014)** ([arXiv:1403.5227](https://arxiv.org/abs/1403.5227)) — model-free branching-ratio estimator from count mean/variance alone.

Why n ≈ 1 is appealing: activity is hugely amplified vs news but doesn't explode; long-range correlated flow arises naturally; near-criticality + heavy-tailed kernels ⇒ rough volatility (below).

### The calibration counterattack
**Filimonov & Sornette (2015)** ([arXiv:1308.6756](https://arxiv.org/abs/1308.6756)): n̂ ≈ 1 can be **manufactured from nothing** — fitting Hawkes to a pure Poisson process with regime switches (intraday U-shape, news bursts) produces spurious near-criticality. Edge effects, outliers, kernel regularization all bias n̂ up. The "is the market critical?" debate is partly econometrics and remains unsettled.

**Limitations of the whole program:** (i) linear Hawkes can't produce *inhibition* or state-dependence (a huge sell can freeze the market, not excite it); (ii) "exogenous vs endogenous" is model-relative — misspecification masquerades as endogeneity (the endo–exo problem); (iii) quadratic/nonlinear Hawkes (Blanc–Donier–Bouchaud 2017; [arXiv:2005.05730](https://arxiv.org/abs/2005.05730)) show the linear model misses feedback from *returns*, needed for fat tails and the Zumbach effect.

## 3. Hawkes-based market impact and optimal execution

**Why connect them:** classical execution treats the market as a static backdrop. If order flow is self-exciting, *your child orders provoke other people's orders*, and others' bursts are briefly predictable (that's what the kernel says). A schedule should react to the tape: stand aside during hostile bursts, lean in after favorable flow. Surfing aftershocks rather than walking at constant speed through an earthquake.

### The anchor paper
**Alfonsi & Blanc (2016), "Dynamic optimal execution in a mixed-market-impact Hawkes price model"** ([arXiv:1404.0648](https://arxiv.org/abs/1404.0648), *Finance and Stochastics*).
- **Model:** Obizhaeva–Wang-style linear transient impact + other liquidity takers arriving as a Hawkes process, each trade with mixed permanent/transient impact.
- **Results:** solved **in closed form**; optimal strategy reacts to observed flow via the current Hawkes intensity (a state variable). Strikingly: Poisson flow admits **price manipulation** (profitable round trips); a knife-edge condition on Hawkes parameters — self-excitation exactly balancing resilience ("MIH" martingale conditions) — kills manipulation. *Self-exciting order flow is not a pathology; a precise amount of it is what makes prices near-martingales.*

**Alfonsi & Blanc (2016), calibration companion** ([arXiv:1506.08740](https://arxiv.org/abs/1506.08740)): calibrated on CAC40 stocks; propagator decays over one–two timescales of a few seconds; martingale conditions *violated* in data; mid-price round trips profitable on paper (bid-ask presumably eats it). Real markets sit near, not on, the no-manipulation manifold.

### The line that followed
- **Cartea, Jaimungal & Ricci (2014/2018)** ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1964781), *SIAM Review* SIGEST) — market making via HJB with mutually exciting buy/sell flow; clustering materially changes quotes and P&L stability vs Poisson.
- **Jusselin & Rosenbaum (2020), "No-arbitrage implies power-law market impact and rough volatility"** ([arXiv:1805.07134](https://arxiv.org/abs/1805.07134)) — builds the market *from* near-critical Hawkes flow: no-arbitrage forces square-root-like power-law impact AND rough volatility, with the exponents tied one-to-one. The deepest conceptual bridge.
- Portfolio liquidation games with self-exciting flow ([arXiv:2011.05589](https://arxiv.org/abs/2011.05589)); intraday energy markets ([arXiv:2504.10282](https://ideas.repec.org/p/arx/papers/2504.10282.html)); Hawkes OTC market making ([arXiv:2608.02002](https://arxiv.org/html/2608.02002)).

**Limitations:** closed forms need exponential kernels while data wants power laws; linear-impact, symmetric, ignores book state; "react to aggressor flow" assumes real-time classification and is latency-fragile; the manipulation-free set is measure-zero.

### Side door: rough volatility
**Jaisson & Rosenbaum (2015, 2016)** ([arXiv:1310.2033](https://arxiv.org/abs/1310.2033), [arXiv:1504.03100](https://arxiv.org/abs/1504.03100)): Hawkes flow with n → 1 and heavy-tailed kernel of index α ∈ (½,1) ⇒ rescaled price converges to rough Heston with Hurst H = α − ½. **Near-criticality is precisely the regime that generates the empirically observed rough volatility (H ≈ 0.1).** Two independent stylized facts, one mechanism. See also [arXiv:1609.05177](https://arxiv.org/pdf/1609.05177) (leverage effect microfoundations).

## 4. Estimation in practice

### Methods
- **Parametric MLE** (Ogata 1978 likelihood): O(N) recursion for exponential kernels; O(N²) naïvely otherwise. Goodness-of-fit via **time-rescaling**: transformed event times should be unit-rate Poisson (Q–Q/KS).
- **EM** via the branching representation (latent parentage) — underlies nonparametric histogram-kernel estimators.
- **Nonparametric second-order**: Bacry–Muzy Wiener–Hopf inversion; NPHC cumulant matching.
- **Model-free n̂**: Hardiman–Bouchaud count-variance estimator — fast, kernel-robust, window-sensitive.
- **Software:** the **`tick` library** ([JMLR 2018](https://www.jmlr.org/papers/v18/17-381.html)) — HawkesExpKern, HawkesSumExpKern, HawkesEM, NPHC; maintenance has slowed (Python-version friction). Hand-rolled exponential-kernel MLE is ~50 lines and often the sanest baseline.

### Pitfalls (most of the craft)
1. **Timestamp resolution/ties.** Ms-truncated feeds → many simultaneous events → broken likelihoods and fake instantaneous excitation. **Deduplicate trades sharing an aggressor** (one market order sweeping several levels prints multiple times) or you'll "discover" ferocious self-excitation that's an artifact. Dither within the tick if needed.
2. **Non-stationary baseline → spurious criticality** (intraday U-shape, news). Remedies: time-varying μ(t), short quasi-stationary windows, volume-time. Symmetric warning: short windows *truncate long-memory kernels* and bias n̂ down. You're always navigating between these two biases.
3. **Kernel misspecification.** Exponential fits to power-law data underestimate n; power-law fits are sensitive to short-time regularization. Fit sums of exponentials spanning decades of timescales; check n̂ stability.
4. **Edge effects** (events near window start have invisible parents).
5. **Near-unidentifiability at n ≈ 1** (small-μ/high-n looks like big-μ/low-n locally). Multi-start; report profile likelihoods.
6. **"Everything looks self-exciting."** Any latent clustering (Cox, regime switching) fits decently as Hawkes. A KS pass is weak evidence *for* the mechanism.

## 5. Crypto specifically

- **Mark, Sila & Weber (2022), "Quantifying endogeneity of cryptocurrency markets"** (*Eur. J. Finance*, [open access](https://www.tandfonline.com/doi/full/10.1080/1351847X.2020.1791925)) — BTC power-law kernels; criticality level similar to fiat FX; reflexivity ports to crypto essentially unchanged.
- Binance fits: Shallow Neural Hawkes ([arXiv:2006.02460](https://arxiv.org/abs/2006.02460)); marked multivariate Hawkes on BTC/ETH ([arXiv:2402.04740](https://arxiv.org/abs/2402.04740)) — volume marks matter.
- LOB + Hawkes forecasting: [arXiv:2312.16190](https://arxiv.org/abs/2312.16190); [Springer 2026](https://link.springer.com/article/10.1007/s10203-026-00570-z).
- Tail asymmetry: crashes excite future extremes more than rallies ([arXiv:2011.12291](https://arxiv.org/abs/2011.12291)).
- **Liquidation cascades (young, fast-moving):** early-warning signals across seven BTC cascades 2022–2025 ([arXiv:2607.27070](https://arxiv.org/abs/2607.27070)) — critical-slowing-down precedes *endogenous-buildup* cascades, absent in *news-shock* ones; **subcritical branching inside the Oct 2025 crash** ([arXiv:2608.03616](https://arxiv.org/abs/2608.03616)) — event-level n ≈ 0.1–0.2, a margin-mechanics avalanche (first-order transition), NOT a Hawkes-critical chain reaction — an interesting negative result. DeFi cross-protocol liquidation clustering ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6508318)) — spectral radius ≈ 0.73, significant cross-venue excitation. Also: stablecoin depegs ([arXiv:2205.06338](https://ideas.repec.org/p/arx/papers/2205.06338.html)), Hawkes(p,q) flash crashes ([QF 2022](https://www.tandfonline.com/doi/full/10.1080/14697688.2021.1941212)), spoofing detection ([arXiv:2502.04027](https://arxiv.org/abs/2502.04027)).

**Crypto caveats:** sloppy ms timestamps, batched prints (aggregate by aggressor), wash trading on small venues injects fake clustering, 24/7 removes open/close seasonality but adds 8h funding-clock seasonality on perps; liquidations are threshold-triggered forced flow that violates linear-Hawkes in spirit.

## 6. What fitting actually requires

- **Resolution:** ms timestamps are the floor for trade-arrival Hawkes on liquid crypto; **aggregate prints into aggressor-level orders first** or short-lag excitation is pure artifact. Rule of thumb: no kernel structure below ~10× timestamp granularity. Full L2/L3 needs self-recorded websockets (crypto public dumps give sampled depth + bookTicker).
- **Sample size:** exponential-kernel univariate MLE stabilizes with ~10⁴–10⁵ events (hours of BTC trades). Long-memory branching-ratio claims need windows ≫ longest kernel timescale (Hardiman needed years of E-mini). Days → defensible short-memory fits; weeks-months → power-law tail claims. Keep dimension ≤ 4–8 for parametric multivariate.
- **Non-negotiables:** deseasonalize μ(t); multi-start; time-rescaling residuals; report n̂ sensitivity to window and kernel family — that sensitivity IS the finding in much of this literature.

## 7. Open / thinly covered questions (esp. crypto)

1. **Is crypto near-critical, and stably so?** No Hardiman-style multi-year multi-venue power-law study across BTC/ETH/alts, regimes, CEX-vs-DEX. The n̂-vs-time series for crypto basically doesn't exist.
2. **Liquidations as marked/nonlinear Hawkes** — threshold-triggered, size-dependent; linear model misspecified by construction; a state-dependent (leverage/OI-conditioned) treatment is not yet written.
3. **Hawkes-based optimal execution calibrated to crypto** — the full loop (calibrate mixed-impact Hawkes on Binance → derive reactive schedule → backtest vs TWAP/POV) is **unpublished**, despite crypto being the one market with free data.
4. **Cross-exchange excitation** (Binance↔Coinbase↔perp↔spot) at event level — thin; low-hanging fruit with free multi-venue feeds.
5. **Endo–exo identification** remains unsolved; crypto's news torrent makes it harder and more interesting; marked models with news covariates are rare.
6. **Rough volatility from crypto microstructure:** does H ≈ α − ½ hold quantitatively for BTC, with both sides measured on the same venue? Not done cleanly.
7. **Nonlinear/inhibitory effects** (liquidity withdrawal after large events, spread excitation + depth inhibition, [arXiv:1912.00359](https://arxiv.org/abs/1912.00359)) — unexplored in crypto books.
