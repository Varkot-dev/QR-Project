# LEARNING.md

This document explains every concept, every estimator, and every judgment call behind the
results in `results/`. It is written to be defended out loud. Every number here comes
from this repository's actual output — `results/q1_results.json`, `q2_results.json`,
`q3_results.json` for Phase 1, `q4_cross_section.json`, `q5_kernel_panel.json` for Phase 2, and
`q6_endogeneity.json`, `q7_execution.json` for Phase 3 —
not from the literature and not from memory. Where a result is uncertain or
sample-limited, the sentence containing it says so.

Read it alongside the code it describes. Each section names the file it explains.

**Contents**

1. [The order book and aggressor trades](#1-the-order-book-and-aggressor-trades)
2. [Order-flow memory](#2-order-flow-memory)
3. [Price impact and the response function](#3-price-impact-and-the-response-function)
4. [OFI and linear impact](#4-ofi-and-linear-impact)
5. [Statistics used honestly](#5-statistics-used-honestly)
6. [Phase 2: the cross-section and the kernel](#6-phase-2-the-cross-section-and-the-kernel)
7. [Phase 3: self-excitation and execution](#7-phase-3-self-excitation-and-execution)
8. [Interview drill](#8-interview-drill)

---

## 1. The order book and aggressor trades

*Code: `src/microstructure/data/events.py`, `src/microstructure/data/ingest.py`*

### The book

A limit order book is two queues. On the bid side sit resting buy orders, sorted by price
descending; on the ask side, resting sell orders, sorted ascending. The best bid is the highest
price someone will pay right now; the best ask is the lowest price someone will sell at. The gap
between them is the spread, and the midpoint

```
m = (best_bid + best_ask) / 2
```

is the conventional "price" — it is not a price anyone traded at, it is the center of the
quotes. Every mid-price in this project comes from Binance's `bookTicker` feed, which publishes
best bid/ask price and size on every change to the top of book.

Two ways to interact with the book:

- **Post a limit order.** You join a queue and wait. You supply liquidity. You do not move
  the price by posting; you move it only when you improve the best quote.
- **Send a market order.** You cross the spread and consume resting orders. You demand
  liquidity, you pay the spread, and you are the one who moves the price.

The party who crosses the spread is the **aggressor** (or **taker**). The party whose resting
order got hit is the **maker**. Every trade has exactly one of each. Which side was the
aggressor is the single most important fact in this entire project, because "did a buyer or a
seller initiate this trade?" is the sign that drives order-flow memory (Q1), the response
function (Q2), and the intuition behind OFI (Q3).

### Why `is_buyer_maker` encodes the aggressor

On many datasets you must *infer* the aggressor with a classification rule — Lee-Ready, tick
rule, quote rule — and those rules misclassify a few percent of trades, which then contaminates
every downstream sign statistic. Binance hands it to you exactly. Each `aggTrades` row carries a
boolean `is_buyer_maker`, and the logic is a two-step deduction:

- `is_buyer_maker == True` → the **buyer** was the maker (resting) → therefore the **seller**
  crossed the spread → seller-initiated → **sign = −1**.
- `is_buyer_maker == False` → the buyer was *not* the maker → the buyer was the taker →
  buyer-initiated → **sign = +1**.

The field names the passive side; the aggressor is whoever is left over. That inversion is where
sign-convention bugs are born, and a flipped sign would silently negate every response function
in the project while leaving the sign ACF completely unchanged (the ACF of `−s` equals the ACF of
`s`) — so Q1 would look fine while Q2 came out upside down. In this repo the convention is
pinned in one place, `events.py`, and asserted in `tests/data/test_events.py`:

```python
pl.when(pl.col("is_buyer_maker")).then(pl.lit(-1)).otherwise(pl.lit(1))
```

Choosing "no trade-sign classification needed" is why the design spec picked Binance USDT-M
futures `aggTrades` as the primary dataset. It removes an entire class of measurement error
before it starts.

### Why prints ≠ orders

Here is the failure that this project's first data-layer task exists to prevent.

One market order does not produce one row. A trader sends a single order to buy 40 ETH. The best
ask has 12 available, the next level 15, the next 13. The matching engine walks down the book and
fills against all three, and the tape prints **three** rows — three different prices, three
`agg_trade_id`s, but one identical millisecond timestamp and one identical `is_buyer_maker`
value, because it was all one decision by one aggressor.

Naively, that tape says three buyers arrived in a row. In truth, one did. If you compute a sign
autocorrelation on raw prints, you are measuring the mechanics of the matching engine walking
the book — not the behavior of traders. And because a sweep is by construction same-signed, the
artifact is enormous and always points the same way: toward spurious positive autocorrelation at
short lags.

This is not hypothetical. Measured on this repo's ETHUSDT 2023-06 file:

| quantity | raw prints | aggressor events |
|---|---|---|
| rows / events | 24,368,924 | 14,239,099 |
| lag-1 sign ACF | 0.4290 | 0.0284 |
| fitted γ̂ (lags 10–500) | 0.7077 | 0.2858 |

Raw prints average 1.71 per aggressor event, with the largest single sweep in that month
printing 104 times. Skipping aggregation inflates the lag-1 autocorrelation by a factor of about
15 and would have reported γ̂ ≈ 0.71 instead of ≈ 0.29 — i.e. it would have placed ETH
comfortably inside the equity range of 0.3–0.7 for an entirely mechanical reason. **The
headline finding of Q1 would have been an artifact of not merging prints.** That is the single
best answer available to "what breaks if you skip this step".

This contrast is no longer only a fact stated in prose here: [`results/q0_aggregation_effect.md`
→](results/q0_aggregation_effect.md) is a committed, re-runnable artifact (`microstructure.analyses.q0_aggregation_effect`)
that recomputes the raw-vs-aggregated gamma table above for all four symbol-month cells (BTC and
ETH, 2023-06 and 2023-07). The pattern generalizes: raw-print γ̂ is inflated by +0.29 to +0.50
relative to aggregated γ̂ in every cell, and whether the inflated number lands inside [0.3, 0.7]
or overshoots past it depends on the symbol-month — BTC's raw γ̂ reaches as high as 0.96.

### What the aggregation does

`to_aggressor_events` groups by `(ts, is_buyer_maker)` and produces one row per aggressor
decision:

- `qty` — summed across the prints.
- `price` — the **notional-weighted** average, `Σ(price·qty) / Σqty`, not the arithmetic mean.
  A sweep that takes 1 unit at 100 and 3 units at 101 has an effective price of 100.75, not
  100.5. This is why the function fails loudly on `qty <= 0` instead of quietly producing a NaN
  price via division by zero.
- `n_prints` — how many rows were merged, retained so the aggregation's effect stays auditable.
- `sign` — ±1 as above.

Two details worth defending:

- **Same-timestamp, opposite-side events are NOT merged.** A buy sweep and a sell sweep in the
  same millisecond are two genuinely different decisions. They are kept separate and sorted
  deterministically by `(ts, sign)`, so sells precede buys, and the output does not depend on
  input row order. Reproducibility is a correctness property, not a nicety.
- **Merging is by key, not by adjacency.** Every row sharing `(ts, side)` anywhere in the frame
  merges, so a shuffled input yields identical output.

The honest limitation: Binance timestamps are **milliseconds**. Two genuinely independent
traders who both buy within the same millisecond get merged into one event, and one trader whose
sweep straddles a millisecond boundary gets split into two. Both errors exist in the data and
neither is fixable at this resolution. The design spec draws the line explicitly: no kernel or
ACF claims below roughly 10ms lags.

---

## 2. Order-flow memory

*Code: `src/microstructure/estimators/acf.py`, `src/microstructure/analyses/q1_orderflow_memory.py`*
*Results: `results/q1_results.md`*

### What an ACF is

Take the sign series `s_1, s_2, …, s_n` — one ±1 per aggressor event, in event time (not clock
time; "lag 1" means "the next trade", not "one second later"). The autocorrelation at lag k asks:
knowing the sign now, how much does that tell you about the sign k events later?

```
ACF(k) = Cov(s_t, s_{t+k}) / Var(s_t)
```

ACF(0) = 1 always. ACF(k) = 0 means no linear predictability at that horizon. For a fair coin,
every positive lag is 0 in expectation.

Intuitively: if buys tended to be followed by buys, the products `s_t · s_{t+k}` would be +1 more
often than −1, and the average would be positive. That is all the ACF is — an average of
products, normalized.

### Why FFT

The naive computation costs O(n · max_lag). Here n ≈ 38 million events for BTC and max_lag =
1000, which is roughly 4×10¹⁰ multiply-adds — minutes to hours, per symbol, and you must redo it
whenever you change a parameter.

The **Wiener-Khinchin theorem** says the autocovariance function is the inverse Fourier
transform of the power spectrum. So instead of sliding the series against itself once per lag,
you transform once, multiply by the conjugate, and transform back:

```python
f = np.fft.rfft(x, nfft)
acov = np.fft.irfft(f * np.conj(f), nfft)[: max_lag + 1]
```

That is O(n log n) — seconds instead of hours, and mathematically identical, not an
approximation. `tests/estimators/test_acf.py::test_acf_matches_naive_computation` asserts the
FFT path equals the textbook definition to 1e-6 on a series small enough to compute both ways.
The zero-padding to `nfft = 1 << (2n−1).bit_length()` matters: the FFT computes *circular*
correlation, so without padding the end of the series would wrap around and correlate with the
beginning. Padding to more than 2n makes the circular result equal the linear one.

### What long memory means

Two series can both have decaying autocorrelation and still be profoundly different.

**Short memory** decays *exponentially*: `ACF(k) ~ φ^k`. There is a characteristic timescale;
past it, correlation is numerically dead. The sum `Σ ACF(k)` converges. This project's
`markov_signs` generator is exactly this case, with theoretical `ACF(k) = (2p−1)^k`.

**Long memory** decays as a *power law*: `ACF(k) ~ k^(−γ)` with 0 < γ < 1. There is no
characteristic timescale — the shape looks the same at every zoom level — and crucially the sum
`Σ k^(−γ)` **diverges** for γ ≤ 1. Correlation at lag 1000 is small but the *accumulated*
correlation never stops mattering. That divergence is not a technicality; it is the entire
reason the Q2 response function rises rather than falls, and it is why long memory is a
qualitative regime change rather than "slightly more correlated".

On a log-log plot a power law is a straight line with slope −γ, which is why `q1_acf_loglog.png`
is drawn log-log and why γ is estimated by OLS on `log(ACF)` vs `log(lag)` over lags [10, 500].
The window starts at 10 to avoid microstructure effects at the shortest lags and stops at
`max_lag/2` because ACF estimates get noisy where few pairs contribute.

### Our numbers

| symbol | n_events | γ̂ | OLS stderr | equity range 0.3–0.7 |
|---|---|---|---|---|
| BTCUSDT | 38,046,362 | **0.3803** | 0.0010 | inside |
| ETHUSDT | 25,773,466 | **0.2380** | 0.0015 | below |

Sample: 2023-06 and 2023-07, Binance USDT-M perpetual futures.

Both symbols show clear long memory — γ̂ well below 1, so the correlation sum diverges in both
cases. The literature benchmark is Bouchaud et al. (2004) and the equities/futures tradition
after it: γ ≈ 0.3–0.7, persisting over thousands of trades. **BTC at 0.380 sits inside that
range. ETH at 0.238 sits below it**, meaning ETH's order flow is *more* persistent than
typical equities, since a smaller exponent is a slower decay.

Say that clearly in an interview, because the direction is easy to get backwards: **lower γ =
slower decay = stronger memory.**

Falling outside the equity range is a finding, not a failure. There is no reason a crypto perp
must reproduce an exponent measured on a different asset class, in a different decade, under a
different market structure.

### Why might ETH's memory be stronger? Candidate explanations

Four candidates, none of which this project's data can currently distinguish. Naming them and
naming the test that would separate them is more valuable in an interview than picking a favorite.

**1. Order splitting.** The standard explanation in the equities literature (Lillo, Mike &
Farmer; Tóth et al.). Large traders slice a metaorder into many child orders over minutes or
hours; each child inherits the parent's sign, and a heavy-tailed distribution of metaorder sizes
mechanically produces a power-law sign ACF. If ETH's metaorders are relatively larger or more
finely split than BTC's, ETH's γ would be lower.
*Test:* this needs metaorder identification. Binance public dumps have no account IDs, so it
cannot be tested here. A venue with per-user attribution (Hyperliquid's on-chain log — see
`research/04-novelty-verification-verdicts.md`) could.

**2. Retail herding.** Genuinely distinct traders correlating with each other rather than one
trader splitting an order — momentum-chasing, social-media-driven flow, liquidation cascades.
Crypto's retail share is far higher than equities', and ETH plausibly higher than BTC's, since
BTC carries more institutional and basis-trade flow.
*Test:* herding and splitting predict different **relationships between correlation and volume**.
Splitting predicts sign correlation concentrated within a single trader's execution horizon;
herding predicts it across many small independent participants. Absent account IDs, a partial
discriminator is conditioning the sign ACF on trade size: if the memory is driven by many small
retail orders, the ACF for small-trade subsamples should stay long-memory; if by splitting of
institutional metaorders, memory should concentrate in the mid-size buckets where child orders
live. That analysis is not in this repo.

**3. Liquidity and the depth profile.** Thinner books force *everyone* to split more, whatever
their motive. ETH's book is thinner in dollar terms than BTC's, so identical trading intentions
produce more child orders and more sign persistence. This is mechanical, not behavioral.
*Test:* the Q3 machinery already measures depth. Estimating γ within depth quintiles, or
across a cross-section of pairs spanning decades of volume, would show whether γ varies
systematically with liquidity. That is precisely the Phase-2 plan in the design spec.

**4. Regime.** Both estimates come from the *same* two months, 2023-06 and 2023-07. If ETH
happened to see a distinctive regime in that window, the difference could be period-specific
rather than a stable property of the symbol.
*Test:* the cheapest and most honest one — re-run Q1 on non-overlapping periods. The estimator
is already CLI-driven with a `--periods` flag, so this is a single command per period; it has
simply not been run yet. Until it is, "BTC and ETH genuinely differ" and "these two months
differ" are not separated by this data.

The last one deserves emphasis because it undercuts the other three: with a single two-month
window per symbol and no confidence interval that accounts for autocorrelation, the honest
statement is that ETH's γ̂ is lower **in this sample**, and the mechanism is open.

### Phase-1.5 diagnostics: the short-lag zigzag, and answering candidate #4 above

Two follow-up checks, both committed as re-runnable artifacts rather than left as prose claims.

**Is the short-lag ACF zigzag a tie-break artifact?** Q1's log-log ACF plot shows a visible
odd/even alternation at short lags — negative ACF(1), strongly positive ACF(2) — and
`to_aggressor_events` breaks same-millisecond, opposite-side ties deterministically (sells
before buys). [`results/q1b_zigzag.md` →](results/q1b_zigzag.md) tests whether that deterministic
sort manufactures the pattern, on BTCUSDT 2023-06. Three variants of the sign series are
compared at lags 1–10: the baseline; the same series with same-timestamp pairs randomly
reordered (p=0.5, fixed seed); and the same series with same-timestamp opposite-sign groups
netted into one event. **Verdict: real structure, not an artifact.** The zigzag amplitude
(mean ACF at even lags minus mean ACF at odd lags) is 0.138174 at baseline, moves to 0.137872
under randomization (a 0.22% change), and stays at 0.128335 under netting (a 7.12% change) —
it survives both perturbations essentially intact. A sanity check explains why the tie-break
can't be the cause: only 1.20% of consecutive event pairs share a millisecond timestamp in the
first place, far too few to produce a ~0.14 alternation. This is consistent with genuine market
structure (e.g. bid-ask bounce or interleaved liquidity-taking reversals), not a sorting
convention. One thing this analysis does **not** establish: whether the alternation persists
*past* lag 10 into Q1's [10, 500] fit window — it only measures lags 1–10, entirely at or before
the window's start, so "the fit is unaffected" is a plausible inference from where the fit
window begins, not a measurement extending into it.

**Is ETH's low γ̂ = 0.238 a stable property of ETH, or a regime artifact?** Candidate #4 above
names this as the cheapest open test — re-run on non-overlapping periods — and it has now been
run. Splitting ETHUSDT into ISO weeks across June–July 2023 gives:

| period | γ̂ | stderr | n_events |
|---|---|---|---|
| week 22 (Jun 1–4, pre-SEC-suit) | 0.1976 | 0.0010 | 1,485,486 |
| week 23 | 0.2906 | 0.0022 | 3,433,795 |
| week 24 | 0.3074 | 0.0026 | 3,141,900 |
| week 25 | 0.3059 | 0.0025 | 3,513,709 |
| week 26 (Jun 26–Jul 2) | 0.3156 | 0.0018 | 3,406,889 |
| week 27 | 0.2275 | 0.0012 | 2,765,846 |
| week 28 | 0.2041 | 0.0016 | 2,978,800 |
| week 29 | 0.2138 | 0.0016 | 2,517,441 |
| week 30 | 0.1795 | 0.0011 | 2,225,679 |

(Week 31 — July 31 only, 303,921 events — was skipped for falling under the 1M-event
threshold.) **Verdict: regime-driven, not robust.** The weekly γ̂ values split cleanly in time:
weeks 24–26 (mid/late June, after the Jun 5–6 SEC filings against Binance and Coinbase) sit at
0.306–0.316, straddling the 0.3 equity-range boundary, while every July week (27–30) falls to
0.180–0.228 and pre-suit week 22 sits at 0.198. Month-level BTC confirms the same market-wide
June elevation (γ̂ = 0.462 in June vs 0.326 in July). So the headline ETH γ̂ = 0.238 is a blend of
two different regimes, not a single stable number — the "regime" explanation from candidate #4 is
the one the data actually supports, at least directionally; volatility and volume were not
controlled for, so the SEC-filing link is suggestive, not established causally. This weekly
breakdown was produced by a scratch script (not yet a committed `q1_orderflow_memory`-style CLI
analysis with its own results artifact); it reuses `sign_acf` and `fit_power_law` exactly as Q1
does, over the same [10, 500] fit window, so the numbers are directly comparable to Q1's
headline γ̂ values.

---

## 3. Price impact and the response function

*Code: `src/microstructure/estimators/response.py`, `src/microstructure/analyses/q2_response.py`*
*Results: `results/q2_results.md`, `results/q2_response.png`*

This is the most interesting part of the project, and the part with the best interview story,
because we initially got the interpretation wrong and the review caught it.

### Temporary vs permanent impact: the bathtub

You buy. The price goes up. Two questions follow, and they have different answers: how far up,
and how much of that is still there later?

Think of the book as a bathtub of resting liquidity. Your market order scoops water out of one
end. Immediately the level there drops — the ask side is thinner and the price has moved. Two
things then happen at once:

- **Refill.** Market makers, seeing a gap, post new orders. Water flows back. The part of the
  move that refills away is **temporary impact** — you paid it, but it does not persist in the
  price.
- **A permanently changed level.** If your buy carried information — you knew something — then
  everyone else revises their view of fair value and posts their new quotes higher. The tub's
  resting level has genuinely risen. That part is **permanent impact**, and it does not decay.

The distinction is the whole ballgame for execution: temporary impact is a cost you can reduce
by trading slower, permanent impact is a cost you cannot avoid because it is the market learning
what you know.

The bathtub analogy has a limit worth stating: real refill is not passive physics. Market makers
choose whether to replenish, and in stress they widen or step away — which is exactly why impact
is state-dependent and why the "liquidity stress" literature in `research/` exists.

### What R(ℓ) measures

The response function is the empirical handle on this:

```
R(ℓ) = E[ s_t · (m_{t+ℓ} − m_t) ]
```

In words: over all events, take the sign of the event, multiply by how far the mid moved in the
ℓ events that followed, and average. Multiplying by `s_t` folds buys and sells together — a buy
followed by a rise and a sell followed by a fall both contribute positively. R(ℓ) is the average
price move in the direction of the aggressor, ℓ events later.

Two implementation details that matter:

- **`m_t` is the mid strictly BEFORE event t.** This is enforced in
  `signals/load.py::events_with_prior_mid` by shifting the event key back 1ms before an
  as-of backward join, because polars' `join_asof(backward)` matches `<=` and we need strict
  `<`. If you use the mid *after* the trade, you have baked the trade's own immediate effect into
  your baseline and R(ℓ) measures something else. The function hard-fails unless both frames are
  `Datetime("ms","UTC")`, because a silent precision mismatch would corrupt every join.
- **Event time, not clock time.** ℓ counts trades, not seconds.

In our run, 6,396,387 events joined to a prior mid with **0 dropped (0.0000%)** — the analysis
refuses to proceed if more than 1% fail to join, since that would signal an alignment problem
between the trade tape and the quote tape.

### What we measured

| quantity | value |
|---|---|
| R(1) | 0.010402 |
| R(500) | 0.056107 |
| R(500)/R(1) | **5.3939×** |
| power-law exponent γ̂ | −0.0831 (stderr 0.0034) |
| exponential rate λ̂ | −0.0009 (stderr 0.0001) |
| log-RSS, power law | **0.2161** |
| log-RSS, exponential | 0.4718 |

Sample: ETHUSDT, 2023-06-01 to 2023-06-14, 6,396,387 events.

Two readings:

**The shape.** Over lags 10–200 the power law fits better — log-scale RSS 0.2161 versus 0.4718,
roughly half the residual. In the classical framing this is the propagator/power-law family
(Bouchaud et al. 2004) fitting better than a single-exponential decay
(Obizhaeva-Wang-style). Note the caveat honestly: this is one window, one symbol, one 14-day
sample, and RSS on the log scale over a fixed window — a different window or a linear-scale
comparison could favor the other shape, since power laws and exponentials with matched
short-lag behavior often diverge only at large lag.

**The sign.** Both fitted exponents are **negative**. A negative γ̂ in `R ~ ℓ^(−γ̂)` means
`R ~ ℓ^(+0.083)` — R(ℓ) is *growing*. Over lags 1–500 the response rises by a factor of 5.39
before flattening. It does not decay at all in this window.

### The arc: how we initially misread this, and what fixed it

**What we first wrote.** The first version of the Q2 write-up reported the rise as a departure
from the classical picture — the verdict sentence carried a parenthetical, "(not the classic
decaying-impact case)", and the caveats framed the growth as our result differing from
Bouchaud (2004). The measurement was right. The code was right. The *interpretation* was
backwards, and it was backwards in the most seductive way: it looked like we had found something
that disagreed with a famous paper.

**What the review caught.** Impact is supposed to decay — but the thing that decays in the
literature is the **kernel** `G(ℓ)`, and the thing we measured is the **response** `R(ℓ)`.
They are different objects, related by

```
R(ℓ) ≈ G(ℓ) + Σ_{n<ℓ} G(ℓ−n) · C(n)
```

where `C(n)` is the sign autocorrelation from Q1. `G(ℓ)` is the bare impact of a single event
in isolation — that decays. But every trade is followed by *correlated* trades, and each of
those contributes its own impact. The second term accumulates. When order flow has long memory,
`C(n)` decays so slowly that its sum diverges, and the accumulation term **dominates** the decay
of `G`. So `R` keeps climbing well past the point where `G` alone would have faded.

Bouchaud's own equity response functions show exactly this: a rise to a maximum somewhere around
10²–10³ trades, then a slow decline. Our rising R was never a contradiction of the literature.
It was the literature's own predicted shape, and we had misfiled it as a disagreement.

**The quantitative check that closed it.** This is the part worth memorizing, because it is what
turns a narrative correction into evidence. Q1 and Q2 are not independent — Q1's measured γ
predicts Q2's plateau ratio.

Take ETH's measured sign-ACF exponent from Q1: **γ ≈ 0.24**. Bouchaud's diffusivity argument
says the kernel exponent must satisfy `β = (1−γ)/2` — this is the condition that keeps prices
from being trivially predictable, i.e. that keeps `R(ℓ) ~ ℓ^(1−2β)` growing no faster than
diffusively. At γ = 0.24 that gives β ≈ 0.38, and a pure power-law accumulation predicts

```
R(500)/R(1) = 500^(1−2β) = 500^γ ≈ 500^0.24 ≈ 4.4×
```

(ETH's exact measured γ = 0.23796 gives 4.39×.)

Letting the lag-1 sign autocorrelation range over a plausible 0.2–0.4 widens that point
prediction to roughly **3.5×–6.9×**. Be precise about where those endpoints come from, because
it is easy to overstate their authority: they correspond to *effective exponents*
γ_eff ≈ 0.202 and γ_eff ≈ 0.310, since `500^0.202 ≈ 3.5×` and `500^0.310 ≈ 6.9×`. The 0.2–0.4
autocorrelation range is a **heuristic input** used to motivate that spread of effective
exponents — it is not propagated through the model analytically. So the band is a plausibility
envelope, not a derived confidence interval, and "inside the band" is a weaker statement than a
statistical test.

**We measured 5.3939×** — inside the band, above the 4.4× point prediction. A number derived
from Q1's exponent on two months of *trades* predicts the scale of a ratio measured in Q2 from
14 days of *quotes* — two different datasets, two different estimators, one consistent theory.
That is stronger evidence that both analyses are correct than either result is on its own.

One important limit on what that agreement proves: the toy band predicts the **magnitude** of
the rise, not its **shape**. The calculation assumes `R(ℓ) ~ ℓ^γ` all the way out to ℓ = 500, but
this document already notes that R plateaus — measured R(500)/R(100) = 1.056×, whereas a pure
power law with γ ≈ 0.24 would give `5^0.24 ≈ 1.47×` over that same stretch. So the pure
power-law form overstates late-lag growth, and the plateau means the accumulation term has
largely saturated by ℓ ≈ 100. The 5.39× landing in-band is consistent with the mechanism, but it
is not evidence for the functional form.

**What we still cannot claim.** We measured `R`, not `G`. Separating the kernel from the flow
memory requires propagator deconvolution, which is out of scope here. So the honest verdict is:
*the response function's rising portion is better described by a power law than an exponential
over lags 10–200, and its magnitude is consistent with the kernel exponent implied by Q1's γ.*
It is **not** "we measured a power-law impact kernel". Also note R(ℓ) plateaus around ℓ ≈
300–500, outside the fitted [10, 200] window, so the fits describe only the rising portion and
say nothing about behavior at or past the plateau.

**Why this is the story to tell.** An interviewer probing for self-correction is asking whether
you can tell the difference between a discovery and a misunderstanding of your own measurement.
The honest arc — measured it, misread it as a novel disagreement, had the framing corrected, then
found an independent quantitative check that confirmed the corrected reading — demonstrates
exactly that. The temptation to keep the "we contradicted Bouchaud" headline was real, and it
was wrong.

---

## 4. OFI and linear impact

*Code: `src/microstructure/estimators/ofi.py`, `src/microstructure/analyses/q3_ofi.py`*
*Results: `results/q3_results.md`, `results/q3_ofi_scatter.png`*

### The queue intuition

Q2 asked what happens after a trade. Q3 asks a blunter question: over a short window, can you
predict the mid-price change from the net pressure at the top of the book — including orders
that were merely *posted and cancelled*, never traded?

Picture the best bid as a queue of people waiting to buy at that price. Four things change the
queue's length, and only two of them are trades:

- Someone joins the bid queue (a new limit buy) → **buying pressure, +**
- Someone leaves the bid queue (a cancel) → **buying pressure gone, −**
- Someone joins the ask queue → **selling pressure, −**
- Someone leaves the ask queue → **selling pressure gone, +**

Plus the price-improvement cases: if the bid ticks *up*, the entire old bid queue is irrelevant
and a new one has formed above it — unambiguous buying pressure, and you count the whole new
size. Cont, Kukanov & Stoikov (2014) formalize this into a single signed quantity per book
update:

```python
e += np.where(b_now >= b_prev, bid_q[1:], 0.0)   # bid improved or held: add new bid size
e -= np.where(b_now <= b_prev, bid_q[:-1], 0.0)  # bid worsened or held: remove old bid size
e -= np.where(a_now <= a_prev, ask_q[1:], 0.0)   # ask improved (down) or held: subtract new ask
e += np.where(a_now >= a_prev, ask_q[:-1], 0.0)  # ask worsened or held: add back old ask
```

Note the `>=` / `<=` on both branches: when the price is *unchanged*, both branches fire and you
get `bid_q_now − bid_q_prev`, the pure size change at that price. When the price moves, only one
fires and you get the full queue size. One vectorized expression handles both regimes.

Why this is a better predictor than trades alone: a cancelled order never prints on the trade
tape, but a large bid vanishing is real information about supply. OFI sees the whole top-of-book
event stream; signed trade flow sees only the subset that executed.

### Slope ≈ 1/depth

The theory is a queueing argument. To push the mid up by one tick you must exhaust the ask
queue. If that queue holds `D` units, you need roughly `D` units of net buying pressure. So

```
Δmid ≈ β · OFI,    with β ∝ 1/D
```

Deep book → each unit of imbalance moves the price less. Thin book → the same imbalance moves it
much more. This is the formal version of "liquidity absorbs impact", and it is why the same
order size costs far more in a thin market.

The regression is fit **through the origin** — no intercept. That is a deliberate modeling
choice, not an oversight: zero net order-flow imbalance should imply zero expected price change.
Fitting an intercept would let the model absorb a spurious drift term.

### Our numbers

| metric | value |
|---|---|
| slope β̂ | 0.000169 (stderr 0.000001) |
| R² | **0.4019** |
| n_windows | 120,960 |
| Cont equities R² | 0.65–0.70 |

Sample: ETHUSDT, 2023-06-01 to 2023-06-14, 10-second bars.

The relationship is real and strongly signed — β̂ is positive and about 285 standard errors from
zero, though see §5 on why that standard error is optimistic. But **R² = 0.4019 is well below
Cont's 65–70% on equities**. OFI explains about 40% of 10-second mid-price variance here, versus
about two-thirds in the original equities study.

Three candidate reasons, in rough order of how much I would bet on them:

**1. L1-only OFI.** Binance `bookTicker` publishes *only* the best bid and ask — one level. Cont
et al. had deeper book data. Pressure building at the second, third, and fifth levels is
completely invisible to us, so our OFI is a noisy proxy for true order-flow pressure. This is a
data limitation, not a modeling error, and it is the most likely single explanation. It biases
R² down mechanically: measurement error in a regressor attenuates fit.

**2. Bar length.** We use 10-second bars. The OFI-price relation is tightest at short horizons;
over 10 seconds, more unrelated price movement — news, cross-venue arbitrage, trades at prices
away from L1 — accumulates in the residual. Cont's strongest results come from short windows.
A bar-length sweep would settle this and has not been run.

**3. Crypto trade-flow dominance.** Silantyev (2019), studying BitMEX, found that **trade-flow
imbalance** — net signed traded volume — predicted price changes *better* than book-based OFI in
that venue. If crypto price formation is driven relatively more by aggressive trades and less by
passive quote revision than equities, book-based OFI should underperform its equities benchmark
by construction. That is a substantive market-structure claim, and it is directly testable with
data already in this repo: we have the signed trade series from Q1 and could regress the same
bars on signed volume instead of OFI. It has not been done.

Notice these are not mutually exclusive, and the first is not really a claim about crypto at all.
Resist the temptation to lead with the interesting story (#3) when the boring explanation (#1) is
more likely.

### Depth scaling: −0.774 vs −1

The sharper test of the theory. Split the 120,960 bars into five quintiles by mean depth, fit a
separate slope in each, and check that slope really does fall as depth rises:

| quintile | mean depth | slope | n_bars |
|---|---|---|---|
| 0 | 53.33 | 0.000303 | 24,192 |
| 1 | 71.50 | 0.000182 | 24,192 |
| 2 | 84.24 | 0.000153 | 24,192 |
| 3 | 100.05 | 0.000137 | 24,192 |
| 4 | 158.12 | 0.000125 | 24,192 |

The direction is unambiguous and monotone: the thinnest quintile's slope (0.000303) is **2.4×**
the thickest quintile's (0.000125). Regressing `log|slope|` on `log(mean depth)` gives an
exponent of **−0.7741**, against Cont's theoretical **−1**.

So: the sign is right, the monotonicity is right, the magnitude is short of theory. Impact falls
with depth more slowly than strict inverse proportionality.

**The caveat belongs in the same breath as the number.** That −0.7741 is a regression on **five
points**. Five. There is no reported confidence interval, and with five points spanning less than
a decade of depth (53 to 158, barely a factor of 3) the uncertainty is wide enough that −1 is not
obviously excluded. Treat it as suggestive of the right direction, not as a measurement that
rejects the theory. If asked "is −0.774 significantly different from −1?", the correct answer is
"I did not compute that, and with five points I would not trust it if I had — the fix is more
quintiles, a wider depth range, and a bootstrap."

The plausible substantive story, if the gap survives better estimation: L1 depth is a poor proxy
for *total* available liquidity. When L1 is thin, deeper levels often are not, so the true depth
varies less than measured L1 depth does — which flattens the fitted exponent toward zero. That
would make −0.774 an artifact of L1-only data rather than a real deviation from Cont, and it is
the same root cause as the low R².

---

## 5. Statistics used honestly

The theme: every number in this repo is accompanied by the reason it might be wrong. The
following are the specific ways these particular estimates can mislead.

### Why OLS standard errors lie under autocorrelation

Every standard error in this project — γ̂'s 0.0010, the OFI slope's 0.000001, the response
exponent's 0.0034 — comes from ordinary least squares, and OLS derives its standard errors under
the assumption that **residuals are independent**.

They are not. In every one of these regressions:

- **Q1:** we regress `log ACF(k)` on `log k`. But ACF(k) and ACF(k+1) are computed from almost
  the same overlapping pairs of the same series. They are massively correlated by construction.
- **Q2:** identically, R(ℓ) and R(ℓ+1) share nearly all their data.
- **Q3:** adjacent 10-second bars are autocorrelated — volatility clusters, order flow persists
  (that is Q1's whole finding).

The consequence is not subtle. OLS counts each observation as independent evidence. When your
1000 points really contain, say, 30 points' worth of independent information, the formula divides
by a sample size that is too large and the standard error comes out far too small. **It
understates the true uncertainty, and it always errs in the direction of overconfidence.**

So when Q1 reports γ̂ = 0.3803 ± 0.0010, do not read that as "γ is between 0.378 and 0.382".
Read it as "the line through these particular log-log points is well determined; the uncertainty
in γ as a property of ETH's or BTC's order flow is larger, and I have not measured it." The
implication matters for the headline claim: ETH's γ̂ = 0.2380 is nominally 40+ stderrs below the
0.3 boundary of the equity range, but that comparison uses a standard error known to be too
small, so "ETH is below the equity range" is a statement about the point estimate, not a
hypothesis test.

**The fix, not yet implemented:** a **block bootstrap**. Cut the series into long contiguous
blocks (long enough to contain the autocorrelation), resample blocks with replacement, re-run the
whole estimator on each resample, and take the spread of the resulting γ̂ distribution as the
real confidence interval. Resampling *blocks* rather than individual points preserves the
within-block dependence structure that makes i.i.d. resampling invalid here. The design spec
lists block bootstrap as required for autocorrelated data; Phase 1 shipped without it, it is
documented as a caveat in all three results files, and it is the first rigor upgrade queued for
Phase 2. That gap is a known, deliberate, documented debt — which is the only acceptable kind.

### Why synthetic ground-truth validation comes first

The rule from the design spec: **no estimator touches real data until it recovers a known answer
on synthetic data.**

The reasoning is that real data has no answer key. If `sign_acf` had an off-by-one in its lag
indexing, or `response_function` had a sign flip, the output on 38 million real events would
still be a plausible-looking curve. You would fit it, get a number in a believable range,
compare it to the literature, and publish a bug. There is no error message. Nothing crashes.

So `src/microstructure/synthetic.py` generates series whose properties are known analytically:

- `iid_signs` — fair coin flips. ACF must be **exactly 0** at every positive lag. Catches
  spurious correlation from padding or normalization errors.
- `markov_signs` — repeats the previous sign with probability p. Theoretical
  `ACF(k) = (2p−1)^k`. At p = 0.75 the test asserts ACF(1) = 0.5, ACF(2) = 0.25, ACF(3) = 0.125
  within 0.02. This pins down lag indexing exactly.
- `fractional_signs` — FARIMA(0,d,0) noise, whose sign series has a power-law ACF with known
  exponent `γ = 1 − 2d`. This is the one that validates the actual Q1 measurement.

The response estimator gets the same treatment, and its tests are the sharpest in the repo. With
**i.i.d. signs**, the accumulation term vanishes (`C(n) = 0`), so `R(ℓ)` collapses to exactly
`G(ℓ)`. That gives two exact tests:

- Permanent impact: every event moves the mid by `c·sign` forever → `R(ℓ) = c` for all ℓ. Test
  asserts a flat response at c = 0.7.
- Exponential kernel: build mids by convolving i.i.d. signs with a known kernel
  `g(ℓ) = 0.5 · 0.8^(ℓ−1)` → the estimator must return that kernel back. Test checks all lags
  1–10 to within 0.02.

That second test is why we can say the estimator is right even though the real-data result
looks nothing like it. On synthetic i.i.d. signs the estimator recovers a *decaying* kernel
perfectly; on real long-memory signs the same code returns a *rising* R. The difference is the
data's memory, not a bug — and we know that because the i.i.d. case was verified first.

### Two real bias episodes from this repo's history

Abstract warnings about bias are cheap. These two actually happened here, and both are in the
git log.

**Episode 1 — biased vs unbiased ACF normalization.** When you estimate the autocovariance at
lag k from n observations, only `n − k` pairs exist. Divide the sum by `n` and you get the
*biased* estimator, which shrinks every value toward zero by a factor of `(n−k)/n`. Divide by
`n − k` and you get the *unbiased* one.

For small lags on a long series this is irrelevant — at n = 38M and k = 1000, the factor is
0.999974. But the bias is **lag-dependent and monotone**: it shrinks large lags more than small
ones. And γ is estimated from the *slope* of log ACF against log lag. A multiplicative factor
that grows with lag adds a systematic tilt to exactly the quantity being measured, steepening the
apparent decay and biasing γ̂ upward. The choice matters far more for the slope than for any
individual ACF value, which is exactly the kind of trap that looks harmless in a spot check.
`acf.py` divides by `(n − lag)` and says so in a comment:

```python
# Use unbiased ACF: divide by (n - lag) for each lag
acov = acov / (n - lags)
```

**Episode 2 — truncation bias in the synthetic generator (commit `8ce4c54`).** This one is the
better story, because the bug was in the *test fixture*, not the estimator.

`fractional_signs` builds long-memory noise as a moving average with infinitely many terms,
truncated in practice. The first implementation used 2,000 terms. The tests passed — but they
only asserted loose bounds on ACF values, not a recovered exponent.

When a real exponent check was added, it failed: at d = 0.4 the theoretical γ is
`1 − 2(0.4) = 0.20`, but the generator produced **γ̂ ≈ 0.3116**. The truncated tail — the part
being thrown away — decays as `k^(d−1)`, which is itself so slow that discarding it at 2,000
terms removed a meaningful chunk of the long-range dependence. The generator was producing
*less* memory than requested. Raising the truncation to 50,000 terms recovered **γ̂ ≈ 0.2498**,
and since `np.convolve` is O(n·n_lags) and too slow at that size, the convolution moved to
zero-padded FFT (verified bit-identical to `np.convolve` on matching inputs).

Two lessons, both worth stating in an interview:

1. **A validation fixture can be as wrong as the code it validates.** Had this gone unnoticed,
   the ACF estimator would have been "validated" against a target that was itself biased — and
   an estimator tuned to match a broken fixture is worse than no validation at all.
2. **A test that asserts loose bounds is barely a test.** The old test passed against the broken
   generator. The new one asserts the recovered exponent against theory within ±0.06 and was
   explicitly confirmed to **fail** against the old implementation before being accepted. If you
   have not seen a test fail, you do not know what it is testing.

### "In the literature range" ≠ "correct"

The most important epistemic point in this document, and the one most likely to be probed.

Q1 found BTC γ̂ = 0.3803, inside the equities range of 0.3–0.7. It is tempting to treat that as
validation. It is not, for three separate reasons:

**A range that wide is easy to hit.** 0.3–0.7 spans more than a factor of two. A moderately
broken estimator can land inside it by luck. Agreement with a wide interval is weak evidence.

**Bugs can push you *into* the range.** This is not hypothetical here — it is measured, in §1.
Skipping aggressor aggregation gives ETH γ̂ = 0.7077 instead of 0.2858. The broken number sits
*inside* the equity range; the correct one sits *outside* it. Had we used raw prints and stopped
at "matches the literature", we would have shipped an artifact and called it a replication. The
literature check would have actively concealed the bug.

**Crypto is not equities.** There is no law requiring a 2023 crypto perp to reproduce an exponent
from 2000s equity data. ETH's γ̂ = 0.2380 falls outside the range, and that is a finding to
investigate, not a failure to hide. The design spec states it directly: mismatches with published
benchmarks are findings, not failures.

The chain of reasoning that actually justifies confidence in these numbers runs the other way:

1. The estimator recovers known answers on synthetic data with analytically derived properties.
2. The data pipeline is verified independently — SHA-256 checksums against Binance's published
   hashes on every download, sequential `agg_trade_id` continuity checks confirming no missing
   trades within and across months, and a hard failure if more than 1% of events fail to join a
   prior mid (actual: 0 dropped).
3. Two independent analyses agree quantitatively: Q1's γ ≈ 0.24 predicts a plateau ratio of
   3.5–6.9×, and Q2 independently measured 5.39×, on a different dataset with a different
   estimator.
4. *Then* we compare to the literature — as context for interpreting the result, not as evidence
   that the code is right.

Literature comparison is the last step, and it is a sanity check on interpretation. It is never
the proof.

---

## 6. Phase 2: the cross-section and the kernel

*Code: `src/microstructure/estimators/propagator.py`,
`src/microstructure/analyses/q4_cross_section.py`,
`src/microstructure/analyses/q5_kernel_panel.py`.
Results: `results/q4_cross_section.md`, `results/q5_kernel_panel.md`.*

Phase 1 measured two symbols. Phase 2 does two things Phase 1 explicitly could not: it runs the
order-flow-memory statistics across **121 symbols**, and it separates the impact kernel from flow
memory using a **deconvolution estimator** — closing the "R, not G" gap that §3 and the Phase-1
limitations list both flagged.

### 6.1 The propagator model, and why deconvolution is necessary

§3 established the mixing identity: what we measure, `R(ℓ)`, is not the thing we want, `G(ℓ)`.
The propagator model states the mixing precisely, as a moving average on mid-price increments:

```
dm[t] = m[t+1] − m[t] = Σ_n κ[n] · signs[t−n] + noise[t]
```

Each signed event contributes `κ[n]` to the price change `n` steps later, and contributions from
all past events **superpose linearly**. Cross-correlate both sides with `signs[t−j]`:

```
b[j] = E[dm[t] · signs[t−j]] = Σ_n κ[n] · C[|j−n|]
```

where `C` is the sign ACF from Q1. That is a **Toeplitz linear system**: `b` and `C` are both
measurable, `κ` is the unknown, and recovering `κ` is a linear solve. This is the whole idea.
Reading the kernel off `b` directly — or off `R(ℓ)`, which is a partial sum of `b` — implicitly
assumes `C = δ`, i.e. i.i.d. signs. Q1 measured that assumption to be badly false. Deconvolution
is the correction.

**The plain-language version:** the response function is the kernel convolved with the crowd's
reaction to itself. One trade moves the price a little, but that trade also predicts the *next*
several trades, which move the price too, and the response function adds up all of it. Solving
the linear system un-mixes the two — it asks "what per-event impact, run through this particular
flow-memory structure, would produce the response I actually see?"

**Why the naive read is not a rough approximation but a wrong answer.** On synthetic data with a
*planted* kernel exponent of 0.35 (fractional signs at d = 0.35, mids built by convolving with
`G0(ℓ) = ℓ^(−0.35)`), the two methods on the same dataset, across the three seeds
`tests/estimators/test_propagator.py` runs:

| Method | recovered exponent (range across seeds 20-22) | error (range) |
|---|---|---|
| Deconvolved β̂ | **0.377–0.397** | 0.027–0.047 |
| Naive fit on the response function | **0.063–0.125** | 0.225–0.287 |

(Seed 20 alone: deconvolved 0.3766, error 0.0266; naive 0.0633, error 0.2867.)

The naive read is off by an order of magnitude in error — it returns something nearly flat,
because long-memory flow makes the raw response *rise* rather than track the decaying bare
kernel. That is the same phenomenon as Q2's rising R, seen from the estimator's side. This
contrast is a passing test in `tests/estimators/test_propagator.py`, not a claim.

**The honesty constraints built into the estimator.** Three, each earned:

1. **A samples-per-lag floor.** `deconvolve_kernel` requires `n_samples / L ≥ 100`. The reason is
   subtle and worth stating: the rank and condition number of the Toeplitz matrix are properties
   of the ACF *values*, not of how well those values were estimated. At n = 1,000 and L = 300,
   condition numbers of ~40–500 were observed — entirely normal-looking — while the recovered
   exponent ranged **0.15 to 0.65** against a planted 0.35 across 20 seeds. Pure estimation
   noise, invisible to every diagnostic the matrix itself offers.
2. **Block-bootstrap uncertainty, not OLS stderr.** On the same synthetic setup, OLS stderr was
   ≈ 0.00136 while the actual Monte-Carlo sd of β̂ across seeds was ≈ 0.0092 — **6.8× larger**.
   This is §5's autocorrelation problem again, and this time the fix shipped:
   `kernel_exponent_blocked` reports the sd across 5 contiguous blocks.
3. **A measured finite-L bias.** The mean recovered β̂ was ≈ 0.386 against a planted 0.35 — a
   systematic **+0.03 to +0.04** bias at L = 300. This one matters enormously downstream, and
   §6.3 is where it earns its keep.

### 6.2 Two cross-sectional laws

Q4 runs Q1's statistics on every symbol in a 207-symbol universe with ≥ 1M aggressor events in
2023-06. **121 symbols cleared the bar; 86 were skipped below it, 0 failed.** The retained set
spans **1.33 decades** of activity (1,009,205 to 21,816,890 events). Two results, pulling in
opposite directions.

**Law 1 — γ is liquidity-invariant.** Regressing γ̂ on log₁₀(activity): slope **−0.0112**
(stderr 0.0547), **R² = 0.0003**. That is not a weak relationship; it is the absence of one.
Across the full observed activity range the fitted line moves γ̂ by **−0.0149**, against a
cross-sectional standard deviation of **0.1674** — the fitted line's movement is about a ninth
of one standard deviation (R² says the trend explains about 0.03% of the variance). Meanwhile
γ̂ itself varies enormously: median **0.327**, range **0.065**
(BNXUSDT) to **1.429** (GALABUSD), IQR 0.278–0.376, with **79 of 121** landing inside the
0.3–0.7 equities range. So symbols differ a lot in long memory, and how much they trade predicts
essentially none of it.

**Law 2 — p_flip rises with activity.** `p_flip = P(sign_{t+1} ≠ sign_t)`, where 0.5 is the
coin-flip benchmark. Regressing on log₁₀(activity): slope **+0.1114** (stderr 0.0171),
**R² = 0.2632**. Over the observed range the fitted p_flip climbs from **0.414 to 0.563** —
crossing 0.5. It is not a subtle tilt. The cleanest way to see it: of the 20 most active symbols,
**8 are anti-persistent** (p_flip > 0.5, equivalently lag-1 ACF < 0 — the two sets are
identical); of the 20 least active, **zero** are. Across the whole cross-section **20 of 121**
symbols are anti-persistent at lag 1, and they are concentrated at the top of the activity
distribution.

**The interpretation — offered as hypothesis, not result.** The two laws split a statistic that
Phase 1 treated as one thing. Long-memory decay at lags 10–500 may reflect **order splitting**: a
large trader working a metaorder over hours leaves a persistent trail regardless of how liquid
the venue is, so γ is a property of *how institutions execute*, roughly universal. Lag-1
structure may instead be **mechanical and competitive**: in a busy book, one aggressive order
provokes an immediate opposite-side response — market makers refilling, arbitrageurs leaning
against — producing alternation, and this pressure scales with how contested the book is. Long
memory would then be a trader-behavior property and short-lag structure a market-structure
property, which is why one tracks liquidity and the other does not.

**What would falsify it — tested in Q4b.** The alternative was a **tick-size confound**: relative
tick size (tick divided by price) is a mechanical driver of bid-ask bounce and lag-1 alternation,
and it correlates with activity — high-activity Binance perps tend to be the ones where the tick
is small relative to price. If p_flip is really tracking relative tick size, "activity" is a
proxy and the competitive-response story is decoration on a bid-ask-bounce artifact. Q4b
(`results/q4b_tick_confound.md`) ran the discriminating regression on 111 of Q4's 121 symbols
(10 dropped for missing current tick-size data — mostly delisted BUSD pairs): `p_flip ~
log10(n_events) + log10(rel_tick)` jointly. **Both coefficients survive.** Activity: coefficient
**+0.1130** (t≈7.14); relative tick size: coefficient **+0.0192** (t≈2.09) — smaller and noisier,
but not indistinguishable from zero by the rough t-ratio this project uses elsewhere. The
collinearity motivating the test turned out weaker than assumed: corr(log-activity, log-rel-tick)
= **−0.21**, not the strong entanglement the hypothesis implied. Joint R² (0.322) barely beats
activity alone (0.294), while tick size alone explains almost nothing (R² = 0.002) — so the
verdict is **activity is the dominant driver, tick size is a real but minor second contributor**,
not the reverse. This does not fully clear the law: two caveats bite harder here than usual.
First, tick size came from Binance's *current* exchangeInfo, not June 2023's — a small,
uncorrected source of error if any symbol's tick changed since. Second, and larger: the mainnet
`fapi.binance.com` endpoint this project meant to hit returned HTTP 451 (geo-blocked) from the
execution environment; the numbers above come from the futures **testnet** exchangeInfo mirror
instead (schema-identical, spot-checked against BTCUSDT's known mainnet tick, but not verified
symbol-by-symbol against mainnet). Read this as a real result on a documented substitute data
source, not a fully clean confirmation. Neither this nor Q4 has ruled out a survivor bias — the
86 skipped symbols are all low-activity, so the low end of the regression is the most-active
slice of an otherwise-excluded population.

Also note that Q4's regression stderrs are worse than usually admitted: each symbol's γ̂ stderr
already understates its own uncertainty (§5), and the understatement is **heteroskedastic** —
it scales with each symbol's own n_events and ACF shape. The cross-sectional OLS therefore
violates homoskedasticity on top of everything else. Read those R² and stderr figures as
descriptive summaries, not confidence intervals. This is also why
`q4_gamma_vs_activity.png` deliberately carries **no per-symbol error bars**: drawing them would
imply a precision the estimates do not have.

### 6.3 The critical-balance test

**What the relation says.** Prices are approximately diffusive — variance grows roughly linearly
in time, with no strong trend or mean reversion at the event scale. But order flow is strongly
persistent (Q1). Persistent flow pushed through a non-decaying kernel would produce a trending,
super-diffusive price. So for prices to stay diffusive, the kernel's decay must be **fine-tuned**
against the flow's persistence. Bouchaud et al. (2004) make that precise:

```
β = (1 − γ) / 2
```

Faster-decaying memory (larger γ) permits a slower-decaying kernel, and vice versa. It is not a
modelling convenience; it is a constraint that a diffusive market has to satisfy. Phase 1 could
only check it indirectly — Q1's γ predicting Q2's 5.39× response ratio. Phase 2 measures β
directly, so the relation becomes a real test.

Q5 runs it on a **16-symbol panel** over one week (2023-06-01..07), computing `γ̂_week` and the
deconvolved `β̂` from the same week's data, then `Δ = β̂ − (1−γ̂_week)/2`.

**The verdict: 12 of 16 consistent, 4 violated** (1000PEPEUSDT, OPUSDT, SOLUSDT, ARBUSDT).
Judgement rule: `|Δ| ≤ 2·max(block_sd, 0.04)`.

**Why the bias floor makes this test conservative in exactly one direction.** The 0.04 floor is
not a safety margin picked to be generous — it is §6.1's *measured* finite-L deconvolution bias.
β̂ typically reads 0.03–0.04 **too high** purely from the method. Without the floor, a
low-noise symbol with a genuinely zero Δ could be flagged "violated" by nothing but the
estimator's own known bias — a false positive the code would have manufactured. (The floor binds
for only **3 of the 16** symbols; the rest have block_sd above 0.04 and are judged on their own
noise.)

But the bias is **signed**, and that asymmetry is the most important sentence in this section.
Since β̂ is biased *upward*, Δ = β̂ − (1−γ)/2 is biased upward too. So:

- A **positive** Δ is suspect — some of it may be bias rather than signal.
- A **negative** Δ is *understated* — the true departure is larger than measured.

And **11 of 16 deltas are negative**, including **all 4 violations**, all of which are negative
(−0.155 to −0.087). Correcting for the bias would push the deltas further negative and make more
symbols violate, not fewer. The 12/16 "consistent" verdict is therefore the **most favourable**
reading the data supports; a bias-corrected test would be harsher. Anyone quoting "75% consistent"
without that sentence is quoting a number the method flatters.

**What "kernels decay slower than critical" would mean if real.** β below `(1−γ)/2` means the
kernel does not decay fast enough to offset the flow's persistence, so impact accumulates —
mildly **super-diffusive**, trending prices at the event scale. Predictable directional drift
after an event, which in principle is tradeable, which is why one should be suspicious of it
surviving in a liquid market.

**And here the two Phase-2 results are in tension, which is worth sitting with.** Q4's headline
was that the most active symbols are the ones flipping *anti-persistently* — lag-1 ACF below zero
— which is a **sub**-diffusive, mean-reverting pressure at short lags. Q5 says the kernels of
high-activity symbols decay too slowly, a **super**-diffusive pressure. All four Q5 violations
are mid-to-high activity symbols. Both cannot be the dominant effect at the same scale in the
same book. Three ways to read it, and this data does not settle between them:

1. **Different lags, both real.** Anti-persistence is a lag-1 phenomenon; the kernel exponent is
   fit over lags 5–150. A book can bounce at one event and trend over a hundred. If so, the two
   findings are describing different regions of the same curve and the tension is only apparent.
2. **The linear model is wrong for exactly these symbols.** The deconvolution assumes impacts
   superpose linearly. If real impact saturates or is state-dependent on spread and depth — most
   plausibly in the busiest, most contested books — then β̂ for those symbols is a
   linear-model artifact and the violation says the model failed, not that the market trends.
3. **Both are estimator artifacts of the same underlying alternation.** Strong lag-1 alternation
   distorts the ACF that feeds the Toeplitz system. The four violating symbols being high-activity
   is exactly what you would see if short-lag zigzag were corrupting the deconvolution input.

I lean toward (1) or (2), and I would not assert either. The honest statement is that Phase 2
produced two results that do not sit comfortably together, and separating them needs the
lag-resolved work Phase 3 would have to do.

**A structural caveat, stated plainly:** a "violated" verdict is equally consistent with the true
impact process being **nonlinear** and with the linear model holding but with a genuinely
different β–γ relationship. The balance relation `β = (1−γ)/2` is *itself* a linear/diffusive
propagator prediction. This analysis cannot distinguish "the market violates critical balance"
from "the linear propagator is the wrong model here."

### 6.4 What Phase 2 does not establish

Every number in §6.2 and §6.3 is bounded by:

- **One week, one month, one regime.** Q4 is a single month (2023-06); Q5 a single **7-day**
  window. Phase 1.5 (§2) *measured* that these statistics are regime-dependent, so this is a
  documented risk, not a hypothetical one. Nothing here shows the two cross-sectional laws
  survive into another month.
- **L1 mids only.** Every mid is a best-bid/best-ask midpoint. Impact through queue depletion or
  hidden liquidity at depth is invisible to all of it — the same limitation that plausibly
  explains Q3's low R².
- **The linear-propagator assumption**, load-bearing for every β̂ in Q5, and untestable within
  this analysis (see §6.3).
- **Uncertainty is block_sd plus a bias floor, not a confidence interval.** `block_sd` comes from
  only **5 contiguous blocks** per symbol — a noisy estimate of noise. The 0.04 floor is the
  measured bias at L = 300, and the true bias at this panel's actual max_lag may differ from the
  synthetic measurement it is based on. There is no formal CI anywhere in Phase 2.
- **Survivorship.** The Q4 cross-section is the 121 symbols that cleared 1M events, not the
  universe.

### 6.5 On the novelty question

`research/04-novelty-verification-verdicts.md` records three agents tasked with *refuting* this
project's novelty claims. On the Phase-2 claim specifically ("the propagator program has never
been applied across a crypto cross-section"), the verdict was that the strong form is **factually
false** — a 2026 Hyperliquid study covers 201 perp markets and 641M fills with impact curves and
decay trajectories, and cross-sectional propagator methodology already exists in FX. What survived
was narrow: the **joint** package — R(ℓ) plus deconvolved G(ℓ) plus sign-ACF plus critical-balance
verification, together, across many pairs on a centralized exchange — appears unexecuted, but as
an incremental asset-class transfer, not an open problem. That is the claim this phase supports,
with those caveats attached, and it is the only one worth making out loud.

---

## 7. Phase 3: self-excitation and execution

*Code: `src/microstructure/estimators/hawkes.py`,
`src/microstructure/signals/eventtime.py`,
`src/microstructure/execution/simulator.py`,
`src/microstructure/analyses/q6_endogeneity.py`,
`src/microstructure/analyses/q7_execution.py`.
Results: `results/q6_endogeneity.md`, `results/q7_execution.md`.*

Phases 1 and 2 asked what order flow *looks like* — how it decays, how it maps into price.
Phase 3 asks a different question: **how much of the flow is the market reacting to itself?**
Then it asks the only question a trader actually cares about: given all this structure, does
knowing it change what an execution schedule should do?

### 7.1 Self-excitation, the branching ratio, and what 0.707 means

**The popcorn picture.** A Poisson process is a steady patter of rain — each event arrives
independently of every other, with no memory. Markets are not like that. Trades cluster: a burst,
then a lull, then another burst. The **Hawkes process** (Hawkes 1971) is the minimal fix. Its
event rate — the **intensity** λ(t) — jumps up after every event and decays back:

```
λ(t) = μ + Σ_{t_i < t} φ(t − t_i)
```

`μ` is the baseline: events arriving "from outside" — news, a fundamental trader deciding to buy.
`φ(·)` is the **kernel**: how much an event lifts the rate afterward, and for how long. This is
popcorn, not rain. Each pop jostles its neighbours into popping, so you get bursts and silences
rather than an even patter. It is also exactly the ETAS model seismologists use for aftershocks —
a big trade is the mainshock, the algorithmic reactions are the aftershocks.

**The branching ratio is the whole point.** Hawkes & Oakes (1974) showed the model above is
*exactly equivalent* to a branching process: "immigrant" events arrive from outside at rate μ, and
each event — immigrant or not — produces a random number of "children", with mean

```
n = ∫ φ(t) dt        ← the branching ratio
```

That single number is interpretable in a way almost nothing else in this project is:

- `n` is the **fraction of events that are endogenous** — reactions to other events rather than
  arrivals from outside. `n = 0.3` means about 30% of activity is the market echoing itself.
- Total activity is amplified by `1/(1 − n)` relative to the news flow driving it. At a round
  `n = 0.7` (illustrative, not the panel figure below) the market is doing 3.3× the volume the
  outside world actually justifies.
- `n → 1` is **criticality**: cascades of unbounded length, endogenous fraction → 100%. Like a
  reactor where each fission triggers exactly one more.
- `n ≥ 1` is explosive — non-stationary, activity diverges. (This repo's simulator raises on
  `alpha ≥ 1` rather than hanging forever in the thinning loop, which is what it did before Task 1's
  review caught it.)

In this repo's parameterization, `φ(t) = α·β·exp(−βt)`, whose integral is exactly `α` — so
**alpha *is* the branching ratio**, and the whole of Q6 is a cross-section of one number.

**Our measurement.** Q6 fits the exponential-kernel Hawkes MLE to a **41-symbol** panel of
2023-06 aggressor flow (the union of the 16-symbol Phase-2 panel and the 40 most-active universe
symbols; they overlap 15/16, which is why the union is 41 and not ~56). Each symbol's month is
split into 6 contiguous business-time sub-windows, fit independently, and summarized by the
median:

| | |
|---|---|
| Median α̂ across 41 symbols | **0.707** |
| Range | **0.370** (KEYUSDT) to **0.879** (LINAUSDT) |
| Distance from criticality (1 − α̂) | **0.293** |
| Symbols with α̂ ≥ 0.9 | **0** |
| Symbols with α̂ ≥ 0.95 | **0** |

Read the headline out loud: **roughly 70% of aggressor events on a typical Binance perp are
reactions to other aggressor events, not independent arrivals from outside.** The market is
mostly talking to itself. Only about three trades in ten are "news" in any sense the model can
see.

That is a high endogeneity level. Whether it is near-critical depends entirely on which estimator
you ask. Under the exponential-kernel MLE, **zero of 41 symbols** land above 0.9, let alone at the
`n ≈ 1` the reflexivity literature argues about — that reading says "high, not critical." But the
model-free count-variance estimator (§7.3) says the opposite: **all 41 of 41 symbols** land above
0.9 (median 0.959), which reads as squarely near-critical. The two estimators disagree
one-directionally on every symbol (§7.3), and this project's own honesty doctrine is precisely
that a criticality verdict must not be asserted off one estimator when the other contradicts it.
So the correct statement is: this panel does **not** resolve the near-criticality question in
either direction — it establishes high endogeneity (median ≈0.71–0.96 depending on estimator) and
leaves "how close to critical" open, pending the power-law refit or window-sensitivity sweep that
would attribute the gap between the two estimators.

**Engaging the literature honestly.** The reflexivity program runs through two opposed papers.
Filimonov & Sornette (2012) fit exponential-ish kernels to E-mini S&P and found endogeneity
rising from ~30% in 1998 to ~70% by 2010, tracking the growth of algorithmic trading — n as an
instability barometer. Hardiman, Bercot & Bouchaud (2013) refit the same data with **power-law**
kernels and found `n ≈ 1` in *every* year, 1998–2011: markets "are and always have been"
near-critical, and the apparent trend was an artifact of short-memory kernels. On crypto
specifically, **Mark, Šíla & Weber (2022, *European Journal of Finance*)** fit BTC with power-law
kernels and find a criticality level comparable to fiat FX — reflexivity ports to crypto
essentially unchanged.

Our 0.707 does **not** contradict that. It is a **lower bound**, and the reason is structural, not
statistical. An exponential kernel has one timescale and finite memory; if the true kernel is a
slowly-decaying power law, the exponential fit truncates the long-range excitation it cannot
represent, and the missing kernel mass shows up as a systematically **understated** branching
ratio. That is precisely the mechanism Hardiman et al. used to explain away Filimonov &
Sornette's trend. So the correct sentence is: *this panel measures α̂ ≈ 0.707 under an
exponential kernel, and a power-law refit would likely push every number in the table upward,
potentially materially so.* Whether it would push them to 1.0 — whether crypto is near-critical —
is a question this phase set up but did not answer. Quoting 0.707 as a point estimate comparable
to the EJF 2022 power-law numbers would be a category error.

**And the panel says endogeneity is liquidity-invariant.** Regressing α̂ on log₁₀(activity)
across the 41 symbols: slope **+0.0286** (stderr 0.1094), **R² = 0.0017**. The stderr is nearly
four times the slope. That is not a weak relationship; it is the absence of one, in the same shape
Q4 found for γ. §7.6 takes that pattern seriously.

### 7.2 The trap, the cure, and why measuring a near-zero correction was not wasted work

This is the part of Phase 3 that generalizes beyond Hawkes.

**The trap.** Filimonov & Sornette (2015) made the sharpest possible criticism of the whole
program: **`n̂ ≈ 1` can be manufactured from nothing.** Fit a Hawkes model to a process that has
*no self-excitation whatsoever* — just a Poisson process whose rate switches between two levels on
a fixed clock — and the estimator reports severe endogeneity. The reason is that a likelihood fit
cannot distinguish "events cause more events" from "the rate happened to be high just then". Both
produce clustering. Both look identical in count statistics.

This repo does not take that on faith. `tests/estimators/test_hawkes.py::
test_regime_switching_poisson_produces_spurious_endogeneity_trap` builds the trap explicitly —
rate alternating between 0.5 and 2.0 every 5000 seconds, zero self-excitation by construction —
and measures what our own estimators say about it:

| estimator | reports on a process with true n = 0 |
|---|---|
| `branching_count_variance` (model-free) | **n̂ = 0.934** |
| `fit_hawkes_exp` (MLE) | **α̂ = 0.976** |

Both estimators report near-criticality on a process containing no excitation at all. If we had
run Q6 on raw clock time and reported "crypto is critical," that table is the entire refutation,
and it comes from our own code.

**The cure: business time.** The fix is a deterministic time change. Estimate the intraday rate
profile — a 48-bin, mean-1 histogram of time-of-day activity — and integrate it to define

```
τ(t) = ∫₀ᵗ rate(s) ds
```

Under τ, a Poisson process with a seasonal rate `μ̄ · rate(tod(t))` becomes a **homogeneous**
Poisson process at rate `μ̄` (the standard time-change theorem for point processes). The seasonal
clustering is flattened out of existence *before* the Hawkes fit ever runs, so whatever excitation
the fit then finds is not the daily cycle in disguise. `signals/eventtime.py` implements this, and
Q6 applies it unconditionally, to every symbol, before any fitting.

**Does the cure work?** Measured, not asserted. `tests/signals/test_eventtime.py` simulates a
Hawkes process with a genuinely seasonal baseline (`simulate_seasonal_hawkes_exp`) at a known
true α = 0.35, fits it both ways, at two seasonal amplitudes:

| planted seasonal amplitude | raw clock-time α̂ | error | business-time α̂ | error |
|---|---|---|---|---|
| 1.4 (shallow trough) | **0.5698** | +0.220 | **0.3406** | −0.009 |
| 1.05 (deep trough) | **0.8981** | +0.548 | **0.3423** | −0.008 |

Clock time inflates α by +0.22 to +0.55. Business time recovers truth to within 0.01 — errors 6–8×
smaller than the test's ±0.06 tolerance, and stable at a second seed. The armor works.

**Now the punchline, which is the actually interesting result.** Q6 measures the size of that
bias on the *real* data too. Each symbol gets one extra fit on raw clock time (`raw_delta =
α_raw − α_rescaled`), and across the panel:

**Median raw_delta = −0.0003.** Largest magnitude anywhere in the 41 symbols: **0.0484.**

Essentially zero. The correction that saved us from a fake 0.93 on synthetic data changed the real
answer by three ten-thousandths.

**This is not a wasted effort, and understanding why is the point.** Three things are true at once
and they do not conflict:

1. The threat is real and severe — 0.934 and 0.976 on a process with no excitation, measured here.
2. The correction works — +0.22/+0.55 bias removed to within 0.01, measured here.
3. The threat is small **in this particular market** — median bias −0.0003, measured here.

Point 3 is a *finding about crypto*, not a verdict on the method. Crypto perps trade 24/7. There is
no open, no close, no lunch lull, no dominant regional session. The 48-bin intraday profile
recovered from a month of these symbols is close to flat — and a flat profile makes
`rescale_to_business_time` nearly the identity map, so there is very little seasonal confound left
to remove. An equity or FX panel would not look like this.

And here is the epistemics that matters: **you cannot know the correction was unnecessary until
you build it and measure it.** Skipping business time because "crypto is 24/7 so seasonality is
probably small" is a guess. Building it, validating it against a planted bias, running it on all
41 symbols, and *measuring* the bias at −0.0003 is knowledge. The correction is applied
unconditionally in `q6_endogeneity.py` for exactly this reason: not knowing in advance how flat a
given symbol's profile will be is the argument for correcting always rather than deciding
per-symbol on a hunch. **The armor was necessary to prove the threat was small here.** That is not
wasted work — that is how you know.

### 7.3 Two estimators, one disagreement, and why disagreement is information

Q6 runs a second, completely independent branching-ratio estimator alongside the MLE: the
Hardiman & Bouchaud (2014) **count-variance** estimator. For a stationary Hawkes process, as the
counting window grows much larger than the kernel timescale,

```
var(N_W) / mean(N_W) → 1 / (1 − n)²      ⟹      n̂ = 1 − sqrt(mean / var)
```

Look at what that formula needs: event counts. That is all. **No kernel shape is assumed at all** —
not exponential, not power law, nothing. Clustering amplifies count variance above the Poisson
`var = mean` benchmark, and the amplification factor pins down `n`.

Two estimators, two different assumption sets:

| | `fit_hawkes_exp` (MLE) | `branching_count_variance` |
|---|---|---|
| assumes kernel shape | **yes** — exponential, one timescale | **no** |
| assumes stationarity | yes | yes |
| uses | full event-time likelihood | count mean/variance in windows |
| main weakness | misspecification if the true kernel is power-law | window-choice sensitivity; large-window asymptotic is approximate at any finite window |

They disagree, and the disagreement is not small:

| | |
|---|---|
| Median \|α̂_MLE − n̂_CV\| across 41 symbols | **0.2395** |
| Pearson correlation between them | **0.2597** |
| Symbols where n̂_CV > α̂_MLE | **41 of 41 (100%)** |
| Median n̂_CV | **0.959** vs median α̂_MLE **0.707** |

A median gap of 0.24 on a quantity bounded in [0, 1] is enormous — a third of the usable range.
A correlation of 0.26 means they barely rank the cross-section the same way. If you wanted a
comfortable result you would report whichever number suited your thesis, or average them. Neither
is defensible.

**Why the disagreement is information rather than failure.** The key fact is the *direction*.
The gap is not noise scattered both ways — it is **one-directional, 41 out of 41 symbols**, with
count-variance reading higher every single time. Noise does not do that. A systematic,
unanimous, one-directional gap between two estimators is a **misspecification signal**, and the
sign points somewhere specific.

Recall §7.1: an exponential kernel truncates power-law memory and therefore **understates** the
branching ratio. The count-variance estimator assumes no kernel shape at all and is free of that
particular bias. So "count-variance reads higher than exponential-kernel MLE, on every symbol"
is *exactly* the pattern you would predict if the true kernel is a slowly-decaying power law. The
disagreement is not the two estimators failing — it is the two estimators jointly telling us the
exponential kernel is the wrong model.

**But the honest version does not stop there,** because there is a second explanation that fits
the same evidence. `branching_count_variance` uses one fixed 200-second (business-time) window per
symbol, and its own docstring warns that the large-window asymptotic is an approximation at any
finite window. A window choice that is systematically too small or too large relative to the
kernel timescale would also produce a systematic gap. This analysis **cannot separate the two**.
Attribution would need either a power-law-kernel MLE refit or a window-sensitivity sweep on n̂_CV,
and neither is in this phase. It is logged as a known open item, not resolved.

The takeaway worth carrying into an interview: **when two estimators with different assumptions
disagree systematically, that is a measurement, not a bug.** It localizes which assumption is
doing the damage. Papering over it — averaging, or picking the publishable one — throws away the
single most informative thing the analysis produced.

**This is also why §7.1's "not near-critical" framing needed qualifying.** The MLE's zero-of-41
above 0.9 and the count-variance estimator's 41-of-41 above 0.9 are the same disagreement viewed
through the near-criticality threshold specifically. Neither number gets to be *the* answer; the
honest position is that this panel does not resolve near-criticality either way.

### 7.4 The thinning-bias episode: what a rejected fix looks like

Like Phase 1's "R is not G" arc (§3), Phase 3 has an episode where the first attempt was wrong in
an instructive way, and the record is worth keeping.

**The setup.** Task 2 needed a *justification test*: prove that business-time rescaling actually
recovers the true branching ratio from seasonality-confounded data. That requires simulating a
Hawkes process with a seasonal baseline. No such simulator existed. The implementer approximated
one: simulate an ordinary constant-μ Hawkes process, then **thin** the realized events by a
time-of-day acceptance probability — keep more events during "busy" hours, fewer during "quiet"
ones. Realized event density then has the right daily shape.

**The problem.** The rescaled fit came in at α̂ ≈ 0.285 against a true 0.35 — outside the
brief's ±0.06 tolerance. The obvious move is to loosen the tolerance and move on. Instead the
implementer ran a control: thin the same series **uniformly at random**, at the same overall keep
fraction, with no seasonality at all. Fitted α dropped to ≈0.25 anyway. That isolates the cause
completely. **Thinning discards genuinely self-excited children along with baseline events**, and
a branching ratio estimated from a series with its children randomly deleted is biased downward
for reasons that have nothing to do with seasonality. The residual was a real, reproducible
construction bias (sd ≈ 0.001 across five thinning seeds), not seed noise — so the implementer
loosened the tolerance to ±0.09 and documented exactly why.

**The review.** The reviewer **confirmed the physics** — the diagnosis was correct, the control
experiment was the right experiment, the bias is real — and **rejected the remedy anyway**.
The argument: a loosened tolerance is a permanent, silent tax on the test's power. Once you accept
±0.09 you can no longer detect a rescaling bug that costs you 0.07. The correct move is not to
widen the goalposts around a broken construction; it is to fix the construction. The reviewer went
further and prototyped the fix — a genuine seasonal-baseline simulator — measuring 0.011 error and
demonstrating the tight ±0.06 tolerance was recoverable.

**The fix.** `simulate_seasonal_hawkes_exp` was built into `hawkes.py`: Ogata thinning where the
*baseline itself* is seasonal (`λ(t) = μ̄·shape(tod(t)) + excitation`), with the thinning bound
raised to `μ̄·max(shape) + excitation-peak` to stay valid. No events are discarded after the
fact, so no children go missing. The tolerance went back to ±0.06 and the errors came in at
−0.009 and −0.008 (§7.2's table). The thinning construction was **kept**, but relabeled: it is now
`test_thinning_construction_has_documented_downward_bias`, asserting a loose range rather than a
tolerance, because documenting a known bias and asserting recovery-within-tolerance are different
claims and should not share a test.

A bonus fell out. The old thinning construction, at deep seasonal troughs, drove the Nelder-Mead
search into a degenerate optimum — α ≈ 0.98 with β ≈ 0.006, a near-flat "excitation" fitting the
sparse-then-bursty gaps left by deleted events. The rebuilt simulator at the *same* deep amplitude
(1.05) produces a large but non-degenerate raw fit (α ≈ 0.898, β staying near the true kernel
timescale). That is direct evidence the degeneracy was a construction artifact, not an inherent
property of deep troughs — a fact nobody knew before the rebuild.

**The lesson, stated generally: a correct diagnosis does not license the first remedy that occurs
to you.** The implementer's analysis was right and the reviewer said so. What got rejected was the
inference from "this bias is real" to "therefore I should widen my tolerance." A measurement
apparatus with a known defect should be repaired, not accommodated — and the repair here made two
tests sharper and surfaced a new fact about the optimizer.

### 7.5 Execution: a risk/cost frontier the data actually resolves

Q7 asks whether any of this structure pays. Three schedules execute the same parent order against
replayed 2023-06 flow on 6 panel symbols, under one shared cost model
(`execution/simulator.py`): adverse drift vs. arrival mid, plus half-spread, plus own-impact from
the symbol's **own measured Q5 kernel**.

- **TWAP** — uniform children on an even event grid. The neutral benchmark.
- **Front-loaded** — same grid, exponentially decaying sizes; decay set from the symbol's measured
  kernel half-life. Almgren-Chriss-flavored: get done before the impact you cause decays away.
- **Flow-reactive** — TWAP's grid, but a child is deferred whenever trailing signed flow opposes
  the parent beyond a threshold. The Hawkes-motivated schedule: stand aside during hostile bursts.

**Calibration discipline first.** The reactive schedule has two free parameters (lookback,
pause threshold). They are grid-searched on **days 1–3 only** and then **frozen** for evaluation on
the disjoint window of **days 4–7**. Not one reported number includes calibration-day data. This
matters more than it sounds: a reactive schedule tuned and scored on the same days will always
look good, and the difference between "tuned in-sample" and "held out" is the difference between a
result and a story. The chosen params were lookback=50, threshold=0.2.

**The evaluation-window results (96 cells each — 6 symbols × 4 days × 2 sides × 2 parent sizes):**

| schedule | mean shortfall | sd | n |
|---|---|---|---|
| TWAP | +0.0161 | **5.209** | 96 |
| front-loaded | **+0.1306** | **0.340** | 96 |
| flow-reactive | **−0.0111** | **5.271** | 96 |

**The reactive schedule won on the mean — and that result is not resolved.** It came in at
−0.0111 versus TWAP's +0.0161, a gap of 0.0272. But both distributions carry a standard deviation
of about **5.2** across cells. The gap is roughly **1/190th** of the noise. A difference that
small against dispersion that large is entirely consistent with chance; this sample cannot
distinguish reactive's mean shortfall from TWAP's. The correct statement is "reactive had the
lower mean in this window," and the correct next sentence is "which does not establish it is
better." Claiming a Hawkes-informed schedule beats TWAP off this evidence would be exactly the
kind of thing this project exists to avoid.

**What *is* resolved is the variance.** Front-loading pays a consistently *higher* mean cost
(+0.1306 — it deliberately eats more own-impact by trading fast) but with a standard deviation of
**0.340** against ~5.2 for the other two — roughly a **15× reduction in dispersion**, and it holds
for **every symbol in the panel**, not just the pooled numbers. That is not a noise-scale effect;
that is the one clearly resolved finding in Q7.

The mechanism is clean and worth being able to state cold: **front-loading trades a known,
deterministic cost for an unknown, stochastic one.** Own-impact is charged by the model every time,
predictably, and front-loading incurs more of it by concentrating size. Adverse drift is the
opposite — it is the dominant and wildly noisy term for schedules that spread execution across the
full horizon, because the longer you are exposed to the market, the more the price can wander
against you. Compress the execution window and you shrink your exposure to drift while paying more
impact. That is not "better." That is **a risk/cost frontier**, and which point on it you want
depends on a risk preference this analysis deliberately does not take a position on. A desk that
must hit a benchmark within tight tracking error and a desk optimizing expected cost want opposite
ends of that table.

**What the simulator does not claim.** This is load-bearing and the results file leads with it:

- **This is not a trading recommendation and not a backtest of a tradable strategy.** It is a
  model-based cost comparison. The `NO-TRADING-CLAIM` header on `results/q7_execution.md` says so
  in those words.
- **Impact is linear.** `temp_impact(q) = G[1]·(q / typical_event_qty)` extrapolates the measured
  lag-1 kernel linearly in size. The square-root-law literature (Almgren et al. 2005; Bouchaud et
  al. 2018) finds temporary impact grows *sublinearly* at large child sizes. Q7's children stay at
  or below a few multiples of typical event size, where linear and sqrt curves are close — so the
  linearization is a defensible *local* approximation, and nothing more. It is not validated
  against large-order impact data and must not be extrapolated.
- **No queue, no latency, no partial fills.** Children execute instantaneously at the prevailing
  mid plus half-spread. Real execution has queue position, and queue position is often the whole
  game.
- **The market does not react to us.** The replayed flow is fixed historical data. Other
  participants never notice the parent order, never lean against a visible schedule, never adapt.
  This is the deepest limitation, and it cuts hardest against precisely the reactive schedule —
  whose entire premise is interacting with flow that, in this simulation, cannot interact back.
- **Four evaluation days, one regime**, on one panel, in one historical week.

### 7.6 Three liquidity-invariants — offered as a hypothesis, with its falsifiers

Something has now recurred across three phases, and it is worth naming even though it is not a
result.

| phase | quantity | what it measures | regression on log₁₀(activity) |
|---|---|---|---|
| Q4 | **γ̂** | long-memory decay exponent of the sign series | slope **−0.0112**, **R² = 0.0003**, n = 121 |
| Q6 | **α̂** | Hawkes branching ratio (endogenous fraction) | slope **+0.0286**, **R² = 0.0017**, n = 41 |

Two independent statistics, measured with entirely different estimators (FFT autocorrelation
power-law fit vs. Hawkes likelihood), on different panels (121 symbols vs. 41), both showing
essentially **zero** relationship with how much a symbol trades. Both R² values are under 0.2%.
Meanwhile both quantities vary a great deal *across* symbols — γ̂ from 0.065 to 1.429, α̂ from
0.370 to 0.879. Symbols differ enormously in these properties; activity predicts none of the
difference.

**The hypothesis:** these are properties of *how participants behave* — how metaorders get split,
how strongly traders react to each other — and that behavior is roughly constant across venues of
wildly different liquidity, because it is set by the trading population and the algorithms it
runs, not by the depth of any particular book. Contrast this with the quantities that *do* track
liquidity: Q4's `p_flip` (slope +0.1114, R² = 0.2632) is mechanical lag-1 structure — market
makers refilling, arbitrageurs leaning against — and it scales sharply with how contested the
book is. The proposed split is **behavioral statistics are liquidity-invariant; mechanical
statistics are not.**

**A candidate third invariant that failed, and it is reported because it failed.** Q5's
`balance_delta` (the departure from critical balance, `β̂ − (1−γ)/2`) was an obvious candidate for
this pattern. It is not one. Regressed on log₁₀(activity) across Q5's 16 panel symbols: slope
**−0.124**, **R² = 0.189**, correlation **−0.435**. That is a visible negative relationship —
weak, on only 16 symbols, but an order of magnitude more structure than γ̂ or α̂ show. And it is
consistent with §6.3's own observation that all four balance violations are mid-to-high-activity
symbols. So the pattern is **two invariants, not three**, and the third candidate points the other
way.

**What would falsify the two-invariant hypothesis** (stating this is the point of offering a
hypothesis at all):

1. **A wider activity range.** Q4 spans 1.33 decades and Q6's panel is deliberately activity-tilted
   (it is a union with the *top* 40 symbols). A null slope over a narrow range is weak evidence.
   Add genuinely illiquid symbols; if γ̂ or α̂ start moving, invariance was a range artifact.
2. **A power-law-kernel refit of Q6.** §7.1 says α̂ is a lower bound whose tightness depends on how
   power-law-ish each symbol's true kernel is. If kernel shape *itself* varies with activity, the
   exponential bias varies with activity too — and a flat α̂-vs-activity line could be a real slope
   cancelled by a compensating bias gradient. This is the falsifier I would run first, because it
   attacks the measurement rather than the sample.
3. **A different month.** Every number above is 2023-06. Phase 1.5 *measured* these statistics to
   be regime-dependent. An invariance that holds in June and breaks in October is a June fact.
4. **Non-activity conditioning.** Activity is one axis. If γ̂ or α̂ track volatility, venue tier,
   or asset category while ignoring activity, "liquidity-invariant" is the wrong description of
   what is going on.

Until at least (1) and (3) are done, this is a pattern noticed across three phases, offered as a
hypothesis with named ways to kill it. It is not a law.

### 7.7 What Phase 3 does not establish

- **One month, one week, one regime.** Q6 is 2023-06; Q7 evaluates on four days of it.
- **Exponential kernel only.** Every α̂ in Q6 is a lower bound (§7.1), and the 41/41
  estimator disagreement (§7.3) is unattributed between kernel misspecification and window
  sensitivity.
- **`converged=True` is weaker than it sounds.** It means Nelder-Mead stopped improving locally —
  not that μ and α are identified. Near α ≈ 1 the likelihood has a shallow μ–α ridge, so the flag
  is a weaker signal near the boundary than away from it. `fit_hawkes_exp`'s docstring says this
  at length.
- **A 250,000-event/window runtime cap** means the largest windows are fit on a truncated prefix.
- **The intraday profile is estimated on the same month it corrects**, so genuine self-excitation
  clustering at a ~30-minute scale could in principle leak into the profile and be removed with
  the seasonality.
- **Q7 is a cost model, not a market.** No queue, no latency, no reaction to our own schedule —
  and the reactive schedule is the one that assumption hurts most.
- **Q7's reactive-vs-TWAP mean difference is unresolved noise.** Only the front-loaded variance
  reduction is a resolved effect.

---

## 8. Interview drill

Eighteen questions with answers grounded in this project's actual numbers.

---

**1. Why might your γ differ from equities?**

First, the numbers: BTC γ̂ = 0.3803, inside the equities range of 0.3–0.7; ETH γ̂ = 0.2380,
below it — meaning ETH's flow is *more* persistent than typical equities, since lower γ is slower
decay.

Four candidate explanations, which my data cannot currently separate. Order splitting: crypto
metaorders sliced into more child orders, each inheriting the parent's sign. Retail herding:
crypto's retail share is much higher than equities', and correlated momentum-chasing produces
sign persistence without any single trader splitting anything. Liquidity: ETH's book is thinner
than BTC's in dollar terms, so identical intentions mechanically require more child orders.
Regime: both estimates come from the same two months, 2023-06 and 2023-07.

The one I would test first is the cheapest — re-run on non-overlapping periods, which is a
single CLI flag, to check the difference is a property of the symbol and not of that window. To
separate splitting from herding properly I would need per-account attribution, which Binance
public dumps do not have; a venue like Hyperliquid does. And I should say the estimates carry OLS
standard errors that understate the true uncertainty, so "ETH is below the range" is a statement
about point estimates, not a hypothesis test.

---

**2. What breaks if you don't merge same-timestamp prints?**

Your headline result becomes an artifact of the matching engine.

One market order sweeping several book levels prints as several `aggTrades` rows — same
millisecond, same `is_buyer_maker`, different prices. Those are one taker decision. Left
unmerged, the sign series contains runs of identical signs that reflect the engine walking the
book, not trader behavior, and since sweeps are always same-signed the artifact always inflates
short-lag autocorrelation.

I measured it on ETHUSDT 2023-06: 24,368,924 raw prints collapse to 14,239,099 aggressor events,
1.71 prints per event on average, largest sweep 104 prints. Lag-1 sign ACF goes from **0.4290 on
raw prints to 0.0284 on aggressor events** — a factor of about 15. The fitted γ̂ goes from
**0.7077 to 0.2858**.

The sharpest part: 0.7077 is *inside* the equity range of 0.3–0.7 and 0.2858 is outside it. So
the bug would have produced a number that looked like a successful replication. That is why I do
not treat literature agreement as validation.

---

**3. Why does your response function RISE — isn't impact supposed to decay?**

The thing that decays in the literature is the **kernel** G(ℓ). What I measured is the
**response** R(ℓ). They are different objects.

R(ℓ) ≈ G(ℓ) + Σ_{n<ℓ} G(ℓ−n)·C(n), where C is the sign autocorrelation from Q1. G is the impact
of one event in isolation and it does decay. But every trade is followed by correlated trades,
each contributing its own impact, and that accumulation term is the second piece. With long-memory
flow, C decays so slowly its sum diverges, so the accumulation dominates G's decay and R keeps
climbing. Bouchaud's own equity response functions show the same rise, peaking around 10²–10³
trades before a slow decline.

I measured R(1) = 0.0104 rising to a plateau near 0.056 around ℓ ≈ 300–500 — a ratio of 5.39×.
And here is the check that convinced me: taking Q1's ETH exponent γ ≈ 0.24 and the
diffusivity-consistent kernel exponent β = (1−γ)/2 ≈ 0.38, the predicted R(500)/R(1) is
500^γ ≈ 4.4×, or roughly 3.5–6.9× — endpoints corresponding to effective exponents of about
0.202 and 0.310, motivated heuristically by letting lag-1 sign autocorrelation range over
0.2–0.4. I measured 5.39×, inside that band and above the point prediction, from a different
dataset with a different estimator.

I'd flag two limits on that check rather than let it carry more weight than it should. The band
is a plausibility envelope, not a confidence interval — the autocorrelation range is a heuristic
input, not something propagated analytically. And it constrains the magnitude of the rise, not
its shape: R has largely plateaued by ℓ ≈ 100, with R(500)/R(100) = 1.056× against 1.47× for a
pure power law, so the functional form is not what the agreement supports.

I should also be honest that I initially wrote this up as a departure from Bouchaud. The
measurement was right and the framing was backwards; review caught it. And I still cannot claim
to have measured G — separating kernel from flow memory needs propagator deconvolution, which I
did not do.

---

**4. Why is your OFI R² lower than Cont's?**

I get R² = 0.4019 on 120,960 ten-second bars of ETHUSDT; Cont, Kukanov & Stoikov report 65–70% on
equities.

The most likely reason is boring: my OFI is **L1-only**. Binance `bookTicker` publishes just the
best bid and ask, so pressure at deeper levels is invisible. That is measurement error in my
regressor, which attenuates R² mechanically. Second, bar length — 10 seconds lets a lot of
unrelated price movement into the residual, and the OFI relation is tightest at shorter horizons.
I have not run a bar-length sweep, and I should have.

Third and most interesting, but the one I would bet on least: Silantyev (2019) found on BitMEX
that trade-flow imbalance beats book-based OFI in crypto. If crypto price formation leans more on
aggressive trades than passive quote revision, book OFI should underperform its equities
benchmark. That is testable with data I already have — I have the signed trade series from Q1 and
could regress the same bars on signed volume — and I have not done it.

I would resist leading with the interesting market-structure story when the data limitation
explains it more simply.

---

**5. How do you know your estimators aren't buggy?**

Real data has no answer key, so a sign flip or an off-by-one in lag indexing produces a
plausible-looking curve and no error message. The rule in this project is that no estimator
touches real data until it recovers a known answer on synthetic data.

Concretely: i.i.d. signs, where the ACF must be exactly zero at every positive lag; a Markov chain
with theoretical ACF (2p−1)^k, where at p = 0.75 the test asserts 0.5, 0.25, 0.125 at lags 1–3,
which pins lag indexing exactly; and FARIMA noise with a known power-law exponent γ = 1 − 2d.

The response estimator gets a sharper test. With i.i.d. signs the accumulation term vanishes and
R(ℓ) collapses to exactly G(ℓ), so I can build mids by convolving i.i.d. signs with a known
kernel 0.5·0.8^(ℓ−1) and require the estimator to return that kernel back, lags 1–10 within 0.02.
That is why I trust the rising real-data R: the same code recovers a decaying kernel exactly when
the signs are i.i.d., so the rise comes from the data's memory, not from a bug.

Beyond estimators: SHA-256 checksum verification on every download against Binance's published
hashes, sequential `agg_trade_id` continuity checks confirming no missing trades within and across
months, and Q2 hard-fails if more than 1% of events fail to join a prior mid — actual was 0
dropped out of 6,396,387.

One caveat I would volunteer: a validation fixture can itself be wrong. That happened here — see
question 9.

---

**6. What's the biggest weakness of this study?**

Three, and I would rank them.

**Sample.** Q1 is two months (2023-06, 2023-07); Q2 and Q3 are a single 14-day window, one
symbol, ETHUSDT. That is one market regime. I cannot distinguish "this is a property of ETH" from
"this is what June 2023 looked like." Every generalization is unsupported, and re-running on
disjoint periods is cheap and simply has not been done.

**Standard errors.** Every stderr I report is OLS, which assumes independent residuals. Nothing
here has independent residuals — ACF values at adjacent lags share nearly all their data, and
adjacent 10-second bars are autocorrelated. So all my stated uncertainties are too small, in the
overconfident direction. The fix is a block bootstrap, it is specified in the design spec, and it
is not implemented. That is the first thing I would add.

**L1-only book data.** `bookTicker` gives one level. This plausibly explains both the low OFI R²
(0.4019 vs Cont's 0.65–0.70) and the depth-scaling exponent falling short of theory (−0.7741 vs
−1), since L1 depth is a poor proxy for total available liquidity.

Smaller ones: the depth-scaling regression uses five points; and I measured R(ℓ) but never
separated the kernel G from flow memory C.

---

**7. What is OFI, and why is the slope supposed to go like 1/depth?**

OFI counts net pressure at the top of the book, including orders that are posted and cancelled
without ever trading. Joining the bid queue or cancelling from the ask is buying pressure;
cancelling from the bid or joining the ask is selling pressure. When the best price moves you
count the whole new queue; when it holds you count the size change. It captures information the
trade tape cannot, because a cancelled order never prints but a large bid vanishing is real
information about supply.

The 1/depth prediction is a queueing argument: to move the mid you must exhaust the queue at the
best quote, so if it holds D units you need about D units of net imbalance, hence Δmid ≈ β·OFI
with β ∝ 1/D. I fit through the origin deliberately — zero imbalance should mean zero expected
price change, and an intercept would absorb spurious drift.

My data supports the direction clearly. Across five depth quintiles the slope falls monotonically
from 0.000303 at mean depth 53.3 to 0.000125 at mean depth 158.1 — a factor of 2.4. Regressing
log|slope| on log(depth) gives −0.7741 against a theoretical −1.

---

**8. Is −0.774 significantly different from −1?**

I did not compute that, and with the data I have I would not trust the answer if I had.

That exponent is a regression on **five points** — one per depth quintile — spanning mean depths
of 53 to 158, barely a factor of three. There is no confidence interval on it, and with five
points over less than a decade of depth the uncertainty is wide enough that I would not claim −1
is excluded. It is suggestive of the right direction, not a rejection of Cont's theory.

To answer it properly I would use more depth buckets over a wider range, bootstrap the
quintile-level slopes to get a real interval, and ideally use deeper book data — because my
suspicion is that the gap is an L1 artifact. L1 depth is a poor proxy for total liquidity: when
L1 is thin, deeper levels often are not, so true depth varies less than measured L1 depth,
which flattens the fitted exponent toward zero.

---

**9. Tell me about a bug you found in your own work.**

The best one was in a test fixture rather than in the estimator.

The synthetic generator `fractional_signs` builds long-memory noise as a truncated
infinite-order moving average, originally at 2,000 terms. The tests passed — but they only
asserted loose bounds on ACF values, not a recovered exponent. When I added a real exponent
check, it failed: at d = 0.4 theory says γ = 1 − 2d = 0.20, and the generator was producing
γ̂ ≈ 0.3116. The discarded tail decays as k^(d−1), slowly enough that truncating at 2,000 terms
removed a real chunk of the long-range dependence, so the generator produced *less* memory than
requested. At 50,000 terms it recovers γ̂ ≈ 0.2498, and the convolution had to move to FFT
because `np.convolve` is O(n·n_lags) and too slow at that size.

Two things I took from it. A validation fixture can be as wrong as the code it validates — if
this had gone unnoticed, the ACF estimator would have been "validated" against a biased target,
which is worse than no validation. And a test asserting loose bounds is barely a test: the old one
passed against the broken generator, so I replaced it with an exponent check and confirmed it
*fails* against the old implementation before accepting it. If you have not watched a test fail,
you do not know what it is testing.

The related judgment call is the biased-vs-unbiased ACF normalization: dividing by n rather than
n − k shrinks large lags more than small ones, which tilts the log-log slope and biases γ̂ upward.
This project divides by (n − lag).

---

**10. Where would you take this next?**

Three tiers.

**Rigor first, before any new results.** Block bootstrap confidence intervals on γ̂ and the OFI
slope, so my error bars stop being fictional. Re-run Q1 on disjoint periods to separate symbol
effects from regime effects. Sweep the Q3 bar length, and regress on signed trade volume
alongside OFI to test Silantyev's crypto claim directly with data already on disk. None of that
needs new data.

**The measurement I stopped short of — since done.** Phase 1 measured R(ℓ) but never separated
the kernel G from flow memory C. Phase 2 built the deconvolution estimator (§6.1) and ran the
balance test on a 16-symbol panel (§6.3), so β = (1−γ)/2 is now a measurement rather than the
consistency check Q2 used.

**Phase 2, the cross-section — done, with the results in §6.** 121 symbols on the trades side,
16 on the kernel panel. I checked the novelty question adversarially first — three agents tasked
with refuting the gap claims, written up in `research/04-novelty-verification-verdicts.md`. The
honest verdict is that everything in Phase 1 is well-trodden, which is exactly what I wanted for
a learning project, since published benchmarks exist at every step. For Phase 2, a Hyperliquid
study has already done 201 perp markets and 641M fills, so the surviving gap is narrow: the joint
R(ℓ) + G(ℓ) + sign-ACF + critical-balance package across a CEX cross-section, which is an
incremental transfer rather than an open problem. I would rather state that accurately than
oversell it.

**Phase 3, what §6 leaves open.** Three things, in order of how much they would change the
conclusions. Run the tick-size regression that would settle whether the p_flip law is real or a
bid-ask-bounce proxy (§6.2). Resolve the sub-vs-super-diffusive tension between Q4's
anti-persistence and Q5's slow kernels by fitting β over disjoint lag windows (§6.3). And repeat
both on a second, disjoint week — because Phase 1.5 already measured that these statistics move
with regime, and a single week cannot distinguish a law from a June.

---

**11. Walk me through how you separate the kernel from flow memory.**

The problem is that the response function is not the kernel. `R(ℓ)` is what a signed trade is
*followed* by, which mixes the trade's own impact with the impact of the correlated trades it
predicts. With long-memory flow those correlated trades dominate, which is why Q2's response
rises instead of decaying.

The propagator model makes the mixing explicit: `dm[t] = Σ_n κ[n]·signs[t−n] + noise`, impacts
superposing linearly. Cross-correlating with `signs[t−j]` gives `b[j] = Σ_n κ[n]·C[|j−n|]`, where
C is the sign ACF I already measure in Q1. That is a Toeplitz system — b measurable, C
measurable, κ unknown — so recovering the bare kernel is a linear solve. Reading κ straight off
R implicitly sets C = δ, i.e. assumes i.i.d. signs, which Q1 measured to be badly false.

I validated it before trusting it. On synthetic data with a planted exponent of 0.35, the
deconvolution recovers 0.3767 while a naive power-law fit to the response function returns
0.0633 — nearly flat, off by an order of magnitude in error. Same dataset, both methods; the
gap is the flow memory.

Three practical constraints in the implementation. I solve with least-squares rather than a
direct inverse, because long-memory ACFs make the Toeplitz matrix near-singular. I enforce
n_samples/L ≥ 100, because rank and condition number are properties of the ACF *values* and are
blind to how noisily those values were estimated — at n = 1,000 and L = 300 I measured
condition numbers of 40–500, which look completely fine, while the recovered exponent ranged
0.15 to 0.65 across seeds. And I report uncertainty as a block-bootstrap sd, never the OLS
stderr, which I measured to understate the true spread by 6.8×.

---

**12. Your balance test says consistent for 12 of 16. How much of that is your tolerance?**

A fair amount, and the direction of the slack is the part that matters.

The band is `|Δ| ≤ 2·max(block_sd, 0.04)`. The 0.04 is not a margin I chose for comfort — it is
the measured finite-L bias of my own estimator: on synthetic data at L = 300, β̂ came back
+0.03 to +0.04 too high. Without the floor, a low-noise symbol with a genuinely zero Δ could be
flagged "violated" by nothing but that bias, which would be a false positive my code manufactured.
It binds for 3 of the 16 symbols; the other 13 are judged against their own block_sd.

But here is the asymmetry I would want to volunteer rather than be asked. The bias is *signed* —
upward — so Δ = β̂ − (1−γ)/2 is biased upward too. A positive Δ is therefore suspect, and a
negative Δ is understated. Eleven of my sixteen deltas are negative, and all four violations are
negative, from −0.087 to −0.155. Bias-correcting would push everything further negative and
produce *more* violations, not fewer. So 12/16 is the most favourable reading the data supports,
not a robust one — the tolerance is doing real work in exactly the direction that flatters the
result. If I quoted "75% consistent" without that sentence I would be overselling it.

The remaining honesty gap is that block_sd itself comes from only 5 contiguous blocks, so it is a
noisy estimate of noise, and none of this is a formal confidence interval.

---

**13. What confounds your p_flip-versus-activity law?**

The finding first: p_flip rises with log-activity, slope +0.111 per decade, R² = 0.26 across 121
symbols, and it crosses 0.5 — of the 20 most active symbols 8 are anti-persistent at lag 1, and
of the 20 least active, none are.

The confound was **relative tick size**. Tick divided by price is a mechanical driver of bid-ask
bounce and lag-1 alternation, and it correlates with activity on Binance perps — the busiest
contracts tend to have a small tick relative to price. If p_flip is really tracking relative tick
size, then activity is just a proxy and my competitive-response story is decoration on an
artifact.

I ran the joint regression (`p_flip ~ log10(n_events) + log10(rel_tick)`, Q4b,
`results/q4b_tick_confound.md`) on 111 of the 121 symbols. Both coefficients survive: activity
+0.1130 (t≈7.14), relative tick size +0.0192 (t≈2.09) — smaller and noisier, but not zero by the
rough t-ratio I use elsewhere. The two variables turned out less collinear than I assumed
(corr = −0.21, not the strong entanglement the hypothesis implied), and univariate R² tells the
same story: 0.294 for activity alone versus 0.002 for tick size alone. So the verdict is activity
dominates, tick size is a real but minor second contributor — not the reverse, and not a clean
acquittal either. Two things stop me calling it fully settled. Tick size came from Binance's
*current* exchangeInfo, not June 2023's — a small, uncorrected error if any symbol's tick moved
since. And the mainnet endpoint I meant to hit returned HTTP 451 (geo-blocked) from my execution
environment, so this ran against the futures testnet mirror instead — schema-identical and
spot-checked against BTCUSDT's known mainnet tick, but not verified symbol-by-symbol. Real result,
documented substitute data source, not a fully clean mainnet confirmation.

Two more I would flag unprompted. Survivorship: I dropped 86 symbols below 1M events, all of them
low-activity, so the bottom of my regression is the most-active slice of an excluded population —
which could flatten or steepen the slope, and I do not know which. And single-regime: this is one
month, and Phase 1.5 measured these statistics to be regime-dependent, so I would want a disjoint
month before calling it a law rather than a June fact.

---

**14. You found kernels decaying slower than critical. What would that mean for price diffusivity,
and do you believe it?**

What it would mean: β below (1−γ)/2 says the kernel does not decay fast enough to offset the
flow's persistence, so impact accumulates and prices are mildly super-diffusive at the event
scale — predictable directional drift after a trade. That is in principle tradeable, which is
itself a reason for suspicion: it should not survive in the most liquid books.

Do I believe it? Not as stated, for two reasons.

First, it is in tension with my own other result. Q4 found the most active symbols flipping
*anti*-persistently at lag 1 — sub-diffusive, mean-reverting pressure — and all four of my
balance violations are mid-to-high-activity symbols. Both cannot be the dominant effect in the
same book at the same scale. The most likely reconciliation is that they are different scales:
anti-persistence is lag-1, and I fit β over lags 5–150, so a book can bounce at one event and
trend over a hundred. The alternative I take seriously is that strong lag-1 alternation is
corrupting the ACF that feeds my Toeplitz system, in which case both numbers are partly artifacts
of the same zigzag.

Second, and more fundamental: the relation β = (1−γ)/2 is *itself* a linear-propagator
prediction, and my β̂ comes from a linear deconvolution. So a violation is equally consistent with
"the market is super-diffusive" and with "impact is nonlinear here and my model is wrong" — most
plausibly in the busiest, most contested books, which is exactly where the violations are. My
analysis cannot distinguish those, and I would not claim it can.

The honest summary is that Phase 2 produced two results that do not sit comfortably together. The
next measurement is fitting β over disjoint lag windows to see whether the tension is scale
separation or estimator contamination.

---

**15. What's a branching ratio, and what did you measure?**

A Hawkes process models clustered events: the arrival rate jumps after every event and decays
back, `λ(t) = μ + Σ φ(t − t_i)`. By the Hawkes-Oakes branching equivalence, that is identical to a
branching process where outside "immigrant" events arrive at rate μ and each event spawns a mean
of `n = ∫φ` children. So `n` — the branching ratio — is the **fraction of activity that is the
market reacting to itself**, and total volume is amplified `1/(1−n)` over the news flow driving
it. `n → 1` is criticality: cascades of unbounded length. In my parameterization
`φ(t) = αβe^(−βt)`, whose integral is exactly α, so alpha *is* the branching ratio.

I measured it on 41 Binance perps, one month of aggressor flow, six sub-windows per symbol, median
of the per-window fits. **Median α̂ = 0.707**, range 0.370 to 0.879. So roughly 70% of trades are
reactions to other trades, and about three in ten are exogenous. That is high endogeneity.

Whether it's near-critical, I would not answer unprompted with just this number, and I'd say so:
under my exponential-kernel MLE, zero of 41 symbols exceed 0.9 (distance from criticality 0.293 at
the median), which looks like "high, not critical." But I ran a second, model-free estimator on
the same panel and it says the opposite — every one of the 41 symbols exceeds 0.9 there. The two
disagree one-directionally on every symbol, which is exactly the situation my own doctrine says
not to resolve by picking one estimator's verdict. So the honest answer is that this panel
establishes high endogeneity but leaves near-criticality open — under an exponential kernel, which
is the caveat that makes question 16 necessary, and the estimator disagreement is question 17.

---

**16. Why is your n̂ a lower bound?**

Because of the kernel family I chose, not because of sample size.

An exponential kernel has a single timescale and finite memory. If the true kernel is a slowly
decaying power law — which is what most of the data-driven literature finds — then the exponential
fit simply cannot represent the long-range excitation, and the kernel mass it cannot represent
does not go into α. It goes missing. The result is a systematically **understated** branching
ratio.

This is not my speculation; it is the mechanism at the center of the field's main dispute.
Filimonov & Sornette (2012) fit short-memory kernels to E-mini and found endogeneity rising from
~30% to ~70% over a decade. Hardiman, Bercot & Bouchaud (2013) refit the same data with power-law
kernels and got `n ≈ 1` in *every* year — the trend was an artifact of the kernel. On crypto, Mark,
Šíla & Weber (2022, *EJF*) fit BTC with power-law kernels and land near fiat-FX criticality levels.

So my 0.707 does not contradict them and is not comparable to them as a point estimate. The honest
framing is that a power-law refit would push every number in my table upward, potentially
materially, and whether crypto is genuinely near-critical is a question I set up but did not
answer. I have a second piece of evidence pointing the same way — see question 17.

---

**17. Your two estimators disagree by 0.24. Is your result broken?**

No — the disagreement is the most informative thing I measured, and I would lead with it rather
than bury it.

I ran two estimators with different assumption sets. The exponential-kernel MLE uses the full
event-time likelihood and assumes a kernel shape. The Hardiman-Bouchaud count-variance estimator
uses only count mean and variance in windows — `n̂ = 1 − sqrt(mean/var)` — and assumes **no kernel
shape at all**. Median absolute difference across 41 symbols: **0.2395**. Correlation: **0.2597**.
On a quantity bounded in [0,1], that is a third of the usable range.

The key fact is not the size, it is the **direction**: count-variance reads higher on **41 of 41
symbols**. One hundred percent, unanimous. Noise does not do that. A systematic one-directional
gap between two estimators is a misspecification signal, and here it points at a specific
assumption — the one the MLE makes and the count-variance estimator does not. If the true kernel
is power-law, the exponential MLE understates α while the model-free estimator does not carry that
bias. A gap in exactly this direction is what that hypothesis predicts.

What stops me claiming I have proven kernel misspecification: there is a competing explanation
that fits the same evidence. My count-variance estimate uses one fixed 200-second business-time
window per symbol, and the estimator's large-window asymptotic is approximate at any finite
window — a systematically bad window choice would also produce a systematic gap. My data cannot
separate the two. Resolving it needs a power-law-kernel MLE refit, or a window-sensitivity sweep
on n̂_CV, and ideally both. I logged it as an open item rather than picking whichever explanation
sounded better.

The general point: two estimators disagreeing systematically is a *measurement* that localizes
which assumption is doing damage. Averaging them, or quoting the more publishable one, would throw
that away.

---

**18. Your reactive schedule won on average. Why don't you claim it's better?**

Because the win is smaller than the noise, and I checked.

Over my held-out evaluation window the flow-reactive schedule had the lowest mean shortfall,
**−0.0111** against TWAP's **+0.0161** — a gap of 0.0272. But both distributions have a standard
deviation of about **5.2** across the 96 cells. The gap is roughly 1/190th of the dispersion. This
sample cannot distinguish those two means. "Reactive had the lower mean in this window" is what
happened; "reactive is better" is not something my evidence supports, and reporting a
Hawkes-motivated schedule beating TWAP off a difference that small would be the exact failure mode
this project is built to avoid.

I would say what *did* resolve. Front-loading paid a **higher** mean cost, +0.1306, with a standard
deviation of **0.340** against ~5.2 — roughly a **15× variance reduction**, holding for every
symbol in the panel, not just pooled. That is well outside noise. The mechanism is that
front-loading swaps a deterministic cost for a stochastic one: own-impact is charged predictably
every time and front-loading incurs more of it, while adverse drift — the dominant, noisy term —
scales with how long you stay exposed. Compress the window, shrink the drift exposure, pay more
impact. That is a risk/cost frontier, not a winner, and which end you want is a risk preference I
don't take a position on.

Two more things I'd flag unprompted. First, the parameters were grid-searched on days 1–3 and
frozen for evaluation on days 4–7 — no leakage — which is why I trust the ranking as a description
even though I can't call the mean gap real. Second, and worse for the reactive schedule
specifically: my replayed flow is fixed historical data that cannot react to my order. The whole
premise of a reactive schedule is interacting with flow, and in this simulator the flow can't
interact back. That assumption cuts hardest against exactly the schedule that looked best.

---

## Literature

- **Bouchaud, Gefen, Potters & Wyart (2004)**, *Fluctuations and response in financial markets:
  the subtle nature of "random" price changes* — the response function R(ℓ), the propagator
  framing, the kernel-vs-response distinction that section 3 turns on, and the diffusivity
  relation β = (1−γ)/2.
- **Cont, Kukanov & Stoikov (2014)**, *The price impact of order book events* — the OFI
  construction in `estimators/ofi.py`, the linear relation, the 1/depth scaling, and the 65–70%
  equities R² benchmark.
- **Tóth et al. (2011)**, *Anomalous price impact and the critical nature of liquidity in
  financial markets* — the square-root impact law and latent-liquidity picture behind the
  temporary/permanent framing.
- **Silantyev (2019)** — BitMEX order flow; trade-flow imbalance outperforming book-based OFI in
  crypto, the third candidate explanation for our lower R².
- **Lillo, Mike & Farmer**; **Tóth et al.** — order-splitting as the standard mechanism behind
  long-memory order flow.
- **Hawkes (1971)**; **Hawkes & Oakes (1974)** — the self-exciting point process and its branching
  equivalence, which is what makes the branching ratio `n = ∫φ` interpretable as an endogenous
  fraction. The basis of §7.1 and `estimators/hawkes.py`.
- **Filimonov & Sornette (2012)** — reflexivity: fitting Hawkes to E-mini and reading `n` as the
  endogenous fraction; endogeneity rising with algorithmic trading.
- **Hardiman, Bercot & Bouchaud (2013)** — the power-law-kernel rebuttal: `n ≈ 1` in every year,
  the short-memory-kernel trend an artifact. The reason §7.1 reports our 0.707 as a lower bound.
- **Hardiman & Bouchaud (2014)** — the model-free count-variance branching-ratio estimator,
  `branching_count_variance`, and the second half of §7.3's disagreement.
- **Filimonov & Sornette (2015)** — the calibration counterattack: near-critical `n̂` manufactured
  from a non-self-exciting regime-switching Poisson process. The trap §7.2 reproduces at
  n̂ = 0.934 / α̂ = 0.976 in this repo's own tests, and the reason business-time rescaling is
  applied unconditionally.
- **Mark, Šíla & Weber (2022)**, *Quantifying endogeneity of cryptocurrency markets*
  (*European Journal of Finance*) — BTC power-law kernels, criticality comparable to fiat FX. The
  crypto benchmark Q6 engages with, and cannot be directly compared to under an exponential kernel.
- **Alfonsi & Blanc (2016)** — optimal execution under Hawkes order flow; the "react to the tape"
  intuition behind Q7's flow-reactive schedule.
- **Almgren et al. (2005)**; **Bouchaud et al. (2018)**, *Trades, Quotes and Prices* — the
  square-root impact law, which is why Q7's linear own-impact scaling is flagged as a local
  approximation valid only at the small child sizes it actually uses.

The full annotated library, including the adversarial novelty verification, is in `research/`.
