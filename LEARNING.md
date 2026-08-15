# LEARNING.md

This document explains every concept, every estimator, and every judgment call behind the
Phase-1 results in `results/`. It is written to be defended out loud. Every number here comes
from this repository's actual output — `results/q1_results.json`, `q2_results.json`,
`q3_results.json` — not from the literature and not from memory. Where a result is uncertain or
sample-limited, the sentence containing it says so.

Read it alongside the code it describes. Each section names the file it explains.

**Contents**

1. [The order book and aggressor trades](#1-the-order-book-and-aggressor-trades)
2. [Order-flow memory](#2-order-flow-memory)
3. [Price impact and the response function](#3-price-impact-and-the-response-function)
4. [OFI and linear impact](#4-ofi-and-linear-impact)
5. [Statistics used honestly](#5-statistics-used-honestly)
6. [Interview drill](#6-interview-drill)

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
R(500)/R(1) = 500^(1−2β) = 500^γ ≈ 500^0.24 ≈ 4.8×
```

Letting the lag-1 sign autocorrelation range over a plausible 0.2–0.4 (a band, not something fit
to this data) widens that prediction to roughly **3.5×–6.9×**.

**We measured 5.3939×.** Inside the band, near its middle. A number derived from Q1's exponent
on two months of *trades* predicts a ratio measured in Q2 from 14 days of *quotes* — two
different datasets, two different estimators, one consistent theory. That is much stronger
evidence that both analyses are correct than either result is on its own.

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

## 6. Interview drill

Ten questions with answers grounded in this project's actual numbers.

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
500^γ ≈ 4.8×, or roughly 3.5–6.9× allowing lag-1 sign autocorrelation in 0.2–0.4. I measured
5.39×, inside that band — from a different dataset with a different estimator.

I should be honest that I initially wrote this up as a departure from Bouchaud. The measurement
was right and the framing was backwards; review caught it. And I still cannot claim to have
measured G — separating kernel from flow memory needs propagator deconvolution, which I did not do.

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

**The measurement I stopped short of.** I measured R(ℓ) but never separated the kernel G from
flow memory C. Propagator deconvolution would give me G directly, and then the critical-balance
relation β = (1−γ)/2 becomes a real test rather than the consistency check I used — right now
Q1's γ predicting Q2's 5.39× ratio is suggestive, not a measurement of β.

**Phase 2, the cross-section.** Run Q1, Q2 and the balance check as one package across 50–100
Binance perp pairs spanning at least three decades of daily volume, and ask whether γ, β and
response amplitude vary systematically with liquidity. I checked the novelty question
adversarially first — three agents tasked with refuting the gap claims, written up in
`research/04-novelty-verification-verdicts.md`. The honest verdict is that everything in Phase 1
is well-trodden, which is exactly what I wanted for a learning project, since published
benchmarks exist at every step. For Phase 2, a Hyperliquid study has already done 201 perp
markets and 641M fills, so the surviving gap is narrow: the joint R(ℓ) + G(ℓ) + sign-ACF +
critical-balance package across a CEX cross-section, which is an incremental transfer rather
than an open problem. I would rather state that accurately than oversell it.

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

The full annotated library, including the adversarial novelty verification, is in `research/`.
