# Phase-1 Research Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and validate the three Phase-1 analyses (order-flow memory, response function, OFI linearity) on Binance data, with every estimator proven against synthetic known answers first.

**Architecture:** `signals/` turns cached Parquet into analysis-ready series; `estimators/` holds pure statistical functions (numpy in / numpy out); `analyses/` are runnable scripts producing figures + a results markdown with a benchmark-comparison table; `LEARNING.md` explains everything.

**Tech Stack:** Python 3.12, polars, numpy, matplotlib (added in Task 6), pytest.

## Global Constraints

- Estimators are pure functions on numpy arrays — no I/O, no polars inside `estimators/`.
- Every estimator gets a synthetic-ground-truth test BEFORE it touches real data (spec §5.1). Statistical tolerances are stated in each test; never loosen one to pass.
- Sign convention everywhere: +1 = buy aggressor (`is_buyer_maker == False`), −1 = sell.
- Mid-price convention: the mid prevailing strictly BEFORE an event (join_asof backward on `ts`, bookTicker row must be earlier than the event's ts).
- Real-data analyses fail loudly (nonzero exit, clear message) if required Parquet months/days are missing — never analyze partial data silently.
- Data on disk: aggTrades monthly Parquet at `data/parquet/aggTrades/{SYMBOL}/{YYYY-MM}.parquet` (BTCUSDT+ETHUSDT 2023-06, 2023-07 synced separately). bookTicker arrives as DAILY files (Task 5) at `data/parquet/bookTicker/{SYMBOL}/{YYYY-MM-DD}.parquet`.
- Files ≤400 lines; no mutation of inputs; `uv run pytest -m "not network" -q` and `uv run ruff check src/ tests/` must be green at every commit.

---

### Task 1: Sign autocorrelation + power-law fit estimators

**Files:**
- Create: `src/microstructure/estimators/__init__.py` (empty), `src/microstructure/estimators/acf.py`
- Test: `tests/estimators/__init__.py` (empty), `tests/estimators/test_acf.py`

**Interfaces:**
- Produces: `sign_acf(signs: np.ndarray, max_lag: int) -> np.ndarray` (length max_lag+1, acf[0]==1, FFT-based, O(n log n)); `fit_power_law(y: np.ndarray, lo: int, hi: int) -> PowerLawFit` where `PowerLawFit` is a frozen dataclass with fields `exponent: float` (positive γ for decay y ~ lag^(−γ)), `intercept: float`, `stderr: float`, and lags with y<=0 inside [lo,hi] are excluded from the fit.

- [ ] **Step 1: Write the failing tests**

```python
# tests/estimators/test_acf.py
import numpy as np

from microstructure.estimators.acf import fit_power_law, sign_acf
from microstructure.synthetic import fractional_signs, iid_signs, markov_signs


def test_acf_lag0_is_one_and_iid_is_zero():
    a = sign_acf(iid_signs(200_000, seed=1), max_lag=100)
    assert a.shape == (101,)
    assert a[0] == 1.0
    assert np.all(np.abs(a[1:]) < 0.02)


def test_acf_matches_naive_computation():
    x = markov_signs(5_000, p_repeat=0.7, seed=4).astype(float)
    a = sign_acf(x, max_lag=5)
    xc = x - x.mean()
    for k in range(1, 6):
        naive = (xc[:-k] * xc[k:]).mean() / (xc * xc).mean()
        assert abs(a[k] - naive) < 1e-6  # FFT path must equal the definition


def test_acf_markov_known_answer():
    a = sign_acf(markov_signs(400_000, p_repeat=0.75, seed=2), max_lag=3)
    for k, expected in [(1, 0.5), (2, 0.25), (3, 0.125)]:
        assert abs(a[k] - expected) < 0.02


def test_power_law_fit_exact():
    lags = np.arange(201, dtype=float)
    y = np.zeros(201)
    y[1:] = 2.0 * lags[1:] ** -0.4
    fit = fit_power_law(y, lo=10, hi=200)
    assert abs(fit.exponent - 0.4) < 1e-9
    assert fit.stderr < 1e-9


def test_power_law_fit_recovers_fractional_gamma():
    a = sign_acf(fractional_signs(400_000, d=0.4, seed=3), max_lag=200)
    fit = fit_power_law(a, lo=10, hi=200)
    assert 0.14 < fit.exponent < 0.26  # theory: gamma = 1 - 2d = 0.2


def test_power_law_fit_skips_nonpositive_values():
    y = np.zeros(101)
    y[1:] = 1.5 * np.arange(1, 101, dtype=float) ** -0.3
    y[50] = -0.001  # one noisy negative point must not crash or poison the fit
    fit = fit_power_law(y, lo=10, hi=100)
    assert abs(fit.exponent - 0.3) < 0.02
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/estimators/test_acf.py -v` — expected: ImportError.

- [ ] **Step 3: Implement**

```python
# src/microstructure/estimators/acf.py
"""Autocorrelation and power-law decay estimation for sign series.

sign_acf uses the FFT (Wiener-Khinchin): O(n log n) vs O(n*max_lag) naive.
fit_power_law is OLS on log(y) vs log(lag) — standard in the order-flow
memory literature; its stderr is the OLS slope standard error, which
understates true uncertainty for autocorrelated data (documented caveat,
addressed with block bootstrap at the analysis level if needed).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sign_acf(signs: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized autocorrelation of a ±1 (or real) series, lags 0..max_lag."""
    x = signs.astype(np.float64) - signs.mean()
    n = x.size
    if max_lag >= n:
        raise ValueError(f"max_lag {max_lag} must be < series length {n}")
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, nfft)
    acov = np.fft.irfft(f * np.conj(f), nfft)[: max_lag + 1] / n
    return acov / acov[0]


@dataclass(frozen=True)
class PowerLawFit:
    exponent: float  # gamma in y ~ lag^(-gamma)
    intercept: float
    stderr: float


def fit_power_law(y: np.ndarray, lo: int, hi: int) -> PowerLawFit:
    """OLS fit of log y vs log lag over [lo, hi], skipping y <= 0 points."""
    lags = np.arange(len(y))
    mask = (lags >= lo) & (lags <= hi) & (y > 0)
    if mask.sum() < 3:
        raise ValueError("fewer than 3 positive points in fit window")
    lx, ly = np.log(lags[mask]), np.log(y[mask])
    (slope, intercept), cov = np.polyfit(lx, ly, 1, cov=True)
    return PowerLawFit(exponent=-slope, intercept=intercept, stderr=float(np.sqrt(cov[0, 0])))
```

- [ ] **Step 4: Run tests to verify they pass** — `uv run pytest tests/estimators/test_acf.py -v`, 6 PASS.

- [ ] **Step 5: Commit** — `git add src/microstructure/estimators tests/estimators && git commit -m "feat: FFT sign-ACF and power-law fit estimators, synthetic-validated"`

---

### Task 2: Response function estimator

**Files:**
- Create: `src/microstructure/estimators/response.py`
- Test: `tests/estimators/test_response.py`

**Interfaces:**
- Produces: `response_function(signs: np.ndarray, mids: np.ndarray, max_lag: int) -> np.ndarray` — R(ℓ) = mean over t of signs[t] * (mids[t+ℓ] − mids[t]) for ℓ = 0..max_lag (R(0)==0 by construction). `mids[t]` is the mid prevailing BEFORE event t. Arrays must be same length.

- [ ] **Step 1: Write the failing tests**

```python
# tests/estimators/test_response.py
import numpy as np
import pytest

from microstructure.estimators.response import response_function
from microstructure.synthetic import iid_signs


def _mids_from_kernel(signs: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """mid before event t = sum over k<t of signs[k] * kernel[t-k]."""
    n = signs.size
    full = np.convolve(signs.astype(float), kernel, mode="full")[:n]
    mids = np.zeros(n)
    mids[1:] = full[:-1]  # strictly-before convention
    return mids


def test_permanent_impact_gives_flat_response():
    signs = iid_signs(200_000, seed=5)
    c = 0.7  # each event permanently moves mid by c*sign, forever
    mids = c * np.concatenate(([0.0], np.cumsum(signs)[:-1]))  # mid strictly before event t
    r = response_function(signs, mids, max_lag=20)
    assert r[0] == 0.0
    assert np.all(np.abs(r[1:] - c) < 0.02)  # theory: R(l) = c for all l >= 1


def test_exponential_kernel_recovered():
    signs = iid_signs(400_000, seed=6)
    g0, phi = 0.5, 0.8
    taus = np.arange(1, 400)
    kernel = np.concatenate(([0.0], g0 * phi ** (taus - 1)))
    mids = _mids_from_kernel(signs, kernel)
    r = response_function(signs, mids, max_lag=10)
    for lag in range(1, 11):
        assert abs(r[lag] - g0 * phi ** (lag - 1)) < 0.02  # R(l) = g(l) for iid signs


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        response_function(np.ones(10), np.ones(9), max_lag=2)
```

- [ ] **Step 2: Run to verify fail** — ImportError expected.

- [ ] **Step 3: Implement**

```python
# src/microstructure/estimators/response.py
"""Average price response to a signed event (Bouchaud et al. 2004).

R(l) = E[ sign_t * (m_{t+l} - m_t) ], with m_t the mid strictly BEFORE
event t. For uncorrelated signs, R(l) recovers the impact kernel itself;
for real (long-memory) signs it mixes kernel and flow memory — that
distinction is the point of the analysis comparing both.
"""
from __future__ import annotations

import numpy as np


def response_function(signs: np.ndarray, mids: np.ndarray, max_lag: int) -> np.ndarray:
    if signs.shape != mids.shape:
        raise ValueError(f"signs {signs.shape} and mids {mids.shape} must match")
    n = signs.size
    if max_lag >= n:
        raise ValueError(f"max_lag {max_lag} must be < series length {n}")
    s = signs.astype(np.float64)
    out = np.zeros(max_lag + 1)
    for lag in range(1, max_lag + 1):
        out[lag] = np.mean(s[:-lag] * (mids[lag:] - mids[:-lag]))
    return out
```

- [ ] **Step 4: Run to verify pass** — 3 PASS. (The loop is O(n·max_lag); at n=4e5, max_lag≈500 this is ~2e8 float ops — acceptable; do NOT prematurely optimize.)

- [ ] **Step 5: Commit** — `git commit -m "feat: response-function estimator with kernel-recovery synthetic validation"`

---

### Task 3: OFI construction + regression estimator

**Files:**
- Create: `src/microstructure/estimators/ofi.py`
- Test: `tests/estimators/test_ofi.py`

**Interfaces:**
- Produces:
  - `ofi_events(bid_p, bid_q, ask_p, ask_q: np.ndarray) -> np.ndarray` — Cont–Kukanov–Stoikov per-update flow: e_n = 1{b_n≥b_{n−1}}·q^b_n − 1{b_n≤b_{n−1}}·q^b_{n−1} − 1{a_n≤a_{n−1}}·q^a_n + 1{a_n≥a_{n−1}}·q^a_{n−1}; length n−1 (first update has no predecessor).
  - `ols_through_origin(x, y: np.ndarray) -> OLSFit` frozen dataclass with `slope, stderr, r2`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/estimators/test_ofi.py
import numpy as np

from microstructure.estimators.ofi import OLSFit, ofi_events, ols_through_origin


def test_ofi_hand_computed_cases():
    # update 1: bid price rises (add q^b_n), ask unchanged (both ask terms fire: -q^a_n + q^a_{n-1} = 0)
    bid_p = np.array([100.0, 100.1, 100.1, 100.0])
    bid_q = np.array([5.0, 3.0, 7.0, 2.0])
    ask_p = np.array([100.2, 100.2, 100.3, 100.2])
    ask_q = np.array([4.0, 4.0, 6.0, 9.0])
    e = ofi_events(bid_p, bid_q, ask_p, ask_q)
    # n=1: b up: +3; a equal: -4+4=0                       -> 3
    # n=2: b equal: +7-3=4; a up: +4 (only q^a_{n-1} term) -> 8
    # n=3: b down: -7; a down: -9 (only -q^a_n term)       -> -16
    assert np.allclose(e, [3.0, 8.0, -16.0])


def test_ols_through_origin_recovers_slope():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 50, 20_000)
    y = 0.003 * x + rng.normal(0, 0.02, x.size)
    fit = ols_through_origin(x, y)
    assert isinstance(fit, OLSFit)
    assert abs(fit.slope - 0.003) < 2e-4
    assert fit.r2 > 0.9


def test_ols_pure_noise_r2_near_zero():
    rng = np.random.default_rng(8)
    fit = ols_through_origin(rng.normal(size=10_000), rng.normal(size=10_000))
    assert abs(fit.slope) < 0.05
    assert fit.r2 < 0.01
```

- [ ] **Step 2: Run to verify fail** — ImportError expected.

- [ ] **Step 3: Implement**

```python
# src/microstructure/estimators/ofi.py
"""Order-flow imbalance (Cont, Kukanov & Stoikov 2014) and its regression.

OFI counts liquidity-consuming and liquidity-adding events at the best
quotes: bid improvements/size-adds are buying pressure (+), ask
improvements/size-adds are selling pressure (-). Price change over a
window is theorized (and empirically found) linear in the window's
summed OFI with slope ~ 1/depth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ofi_events(
    bid_p: np.ndarray, bid_q: np.ndarray, ask_p: np.ndarray, ask_q: np.ndarray
) -> np.ndarray:
    if not (bid_p.shape == bid_q.shape == ask_p.shape == ask_q.shape):
        raise ValueError("all four L1 arrays must have identical shape")
    e = np.zeros(bid_p.size - 1)
    b_now, b_prev = bid_p[1:], bid_p[:-1]
    a_now, a_prev = ask_p[1:], ask_p[:-1]
    e += np.where(b_now >= b_prev, bid_q[1:], 0.0)
    e -= np.where(b_now <= b_prev, bid_q[:-1], 0.0)
    e -= np.where(a_now <= a_prev, ask_q[1:], 0.0)
    e += np.where(a_now >= a_prev, ask_q[:-1], 0.0)
    return e


@dataclass(frozen=True)
class OLSFit:
    slope: float
    stderr: float
    r2: float


def ols_through_origin(x: np.ndarray, y: np.ndarray) -> OLSFit:
    sxx = float(x @ x)
    if sxx == 0.0:
        raise ValueError("x has zero variance")
    slope = float(x @ y) / sxx
    resid = y - slope * x
    dof = max(x.size - 1, 1)
    stderr = float(np.sqrt((resid @ resid) / dof / sxx))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    return OLSFit(slope=slope, stderr=stderr, r2=r2)
```

- [ ] **Step 4: Run to verify pass** — 3 PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: Cont OFI construction and through-origin OLS, hand-validated"`

---

### Task 4: Signals layer — Parquet to analysis-ready series

**Files:**
- Create: `src/microstructure/signals/__init__.py` (empty), `src/microstructure/signals/load.py`
- Test: `tests/signals/__init__.py` (empty), `tests/signals/test_load.py`

**Interfaces:**
- Consumes: `parquet_path` (catalog), `to_aggressor_events` (events).
- Produces:
  - `load_events(root: Path, symbol: str, periods: list[str]) -> pl.DataFrame` — lazily scans the aggTrades Parquets for the given periods (monthly "YYYY-MM" or daily "YYYY-MM-DD" — whatever exists at parquet_path), raises FileNotFoundError naming ALL missing paths, concats, returns aggressor events (schema from to_aggressor_events).
  - `load_book_ticker(root, symbol, periods) -> pl.DataFrame` — same discipline; adds `mid = (bid_price + ask_price) / 2` column.
  - `events_with_prior_mid(events: pl.DataFrame, bt: pl.DataFrame) -> pl.DataFrame` — join_asof backward with `strategy="backward"` on ts using **strict inequality** (bookTicker row must precede the event: pre-shift bt ts by +1ms or filter equality post-join — implement by joining on `ts` with backward strategy after subtracting 1ms from event ts copy, keeping original ts in output). Output: events columns + `mid: Float64`, rows with no prior mid dropped, count of dropped rows returned via second element of a tuple `(df, n_dropped)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/signals/test_load.py
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from microstructure.data.catalog import parquet_path
from microstructure.signals.load import events_with_prior_mid, load_book_ticker, load_events


def _write_agg(root: Path, symbol: str, period: str, rows):
    p = parquet_path(root, symbol, "aggTrades", period)
    p.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        rows,
        schema={
            "agg_trade_id": pl.Int64, "price": pl.Float64, "qty": pl.Float64,
            "first_trade_id": pl.Int64, "last_trade_id": pl.Int64,
            "ts": pl.Datetime("ms", "UTC"), "is_buyer_maker": pl.Boolean,
        },
        orient="row",
    ).write_parquet(p)


def _ts(ms: int):
    return datetime(2023, 6, 1, 0, 0, 0, ms * 1000, tzinfo=timezone.utc)


def test_load_events_concats_periods_in_order(tmp_path: Path):
    _write_agg(tmp_path, "BTCUSDT", "2023-06", [[1, 100.0, 1.0, 1, 1, _ts(1), False]])
    _write_agg(tmp_path, "BTCUSDT", "2023-07", [[2, 101.0, 1.0, 2, 2, _ts(2), True]])
    ev = load_events(tmp_path, "BTCUSDT", ["2023-06", "2023-07"])
    assert ev.height == 2
    assert ev["sign"].to_list() == [1, -1]


def test_load_events_missing_period_raises_naming_all(tmp_path: Path):
    _write_agg(tmp_path, "BTCUSDT", "2023-06", [[1, 100.0, 1.0, 1, 1, _ts(1), False]])
    with pytest.raises(FileNotFoundError) as ei:
        load_events(tmp_path, "BTCUSDT", ["2023-06", "2023-07", "2023-08"])
    assert "2023-07" in str(ei.value) and "2023-08" in str(ei.value)


def test_events_with_prior_mid_strictly_before(tmp_path: Path):
    events = pl.DataFrame(
        {"ts": [_ts(5), _ts(10)], "sign": [1, -1], "qty": [1.0, 1.0],
         "price": [100.0, 100.0], "n_prints": [1, 1]},
        schema_overrides={"ts": pl.Datetime("ms", "UTC"), "sign": pl.Int8, "n_prints": pl.UInt32},
    )
    bt = pl.DataFrame(
        {"update_id": [1, 2], "bid_price": [99.0, 99.5], "bid_qty": [1.0, 1.0],
         "ask_price": [101.0, 100.5], "ask_qty": [1.0, 1.0], "ts": [_ts(3), _ts(10)]},
        schema_overrides={"ts": pl.Datetime("ms", "UTC")},
    ).with_columns(((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid"))
    out, n_dropped = events_with_prior_mid(events, bt)
    # event at ms 10 must NOT see the ms-10 quote (not strictly before) -> mid from ms 3
    assert out["mid"].to_list() == [100.0, 100.0]
    assert n_dropped == 0


def test_events_with_prior_mid_drops_events_before_first_quote(tmp_path: Path):
    events = pl.DataFrame(
        {"ts": [_ts(1)], "sign": [1], "qty": [1.0], "price": [100.0], "n_prints": [1]},
        schema_overrides={"ts": pl.Datetime("ms", "UTC"), "sign": pl.Int8, "n_prints": pl.UInt32},
    )
    bt = pl.DataFrame(
        {"update_id": [1], "bid_price": [99.0], "bid_qty": [1.0],
         "ask_price": [101.0], "ask_qty": [1.0], "ts": [_ts(2)]},
        schema_overrides={"ts": pl.Datetime("ms", "UTC")},
    ).with_columns(((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid"))
    out, n_dropped = events_with_prior_mid(events, bt)
    assert out.height == 0
    assert n_dropped == 1
```

- [ ] **Step 2: Run to verify fail** — ImportError expected.

- [ ] **Step 3: Implement**

```python
# src/microstructure/signals/load.py
"""Parquet cache -> analysis-ready frames. All scans lazy until the end."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl

from microstructure.data.catalog import parquet_path
from microstructure.data.events import to_aggressor_events


def _existing_paths(root: Path, symbol: str, data_type: str, periods: list[str]) -> list[Path]:
    paths = [parquet_path(root, symbol, data_type, p) for p in periods]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing {data_type} parquet for {symbol}: {missing}")
    return paths


def load_events(root: Path, symbol: str, periods: list[str]) -> pl.DataFrame:
    paths = _existing_paths(root, symbol, "aggTrades", periods)
    lf = pl.concat([pl.scan_parquet(p) for p in paths])
    return to_aggressor_events(lf)


def load_book_ticker(root: Path, symbol: str, periods: list[str]) -> pl.DataFrame:
    paths = _existing_paths(root, symbol, "bookTicker", periods)
    lf = pl.concat([pl.scan_parquet(p) for p in paths]).with_columns(
        ((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid")
    )
    return lf.collect()


def events_with_prior_mid(
    events: pl.DataFrame, bt: pl.DataFrame
) -> tuple[pl.DataFrame, int]:
    """Attach the mid prevailing STRICTLY before each event's ts.

    join_asof(backward) matches <=; shifting the event key back 1ms turns
    that into strict <, honoring the 'mid before the event' convention at
    the data's ms resolution.
    """
    ev = events.with_columns((pl.col("ts") - timedelta(milliseconds=1)).alias("_key")).sort("_key")
    quotes = bt.select("ts", "mid").sort("ts").rename({"ts": "_key"})
    joined = ev.join_asof(quotes, on="_key", strategy="backward").drop("_key")
    n_dropped = int(joined["mid"].null_count())
    return joined.drop_nulls("mid"), n_dropped
```

- [ ] **Step 4: Run to verify pass** — 4 PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: signals layer - events, book ticker, strictly-prior mid join"`

---

### Task 5: Daily-file support in the data layer

**Files:**
- Modify: `src/microstructure/data/binance.py` (add `DAILY_BASE`, `day_files`), `src/microstructure/data/catalog.py` (add `sync_days`)
- Test: `tests/data/test_daily.py`

**Interfaces:**
- Consumes: existing `DumpFile`, `download`, ingesters, `parquet_path`.
- Produces: `day_files(symbol, data_type, start, end) -> list[DumpFile]` (inclusive "YYYY-MM-DD" range; DumpFiles whose `.url` uses the daily base — implement via a `base` field on DumpFile defaulting to the monthly BASE, so existing behavior is unchanged); `sync_days(root, symbol, data_type, start, end, client=None) -> list[Path]` mirroring `sync` (verify-before-rename included) with daily periods.

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_daily.py
import datetime as dt

import pytest

from microstructure.data.binance import DAILY_BASE, DumpFile, day_files


def test_day_files_inclusive_and_daily_url():
    files = day_files("ETHUSDT", "bookTicker", "2023-06-29", "2023-07-02")
    assert [f.period for f in files] == ["2023-06-29", "2023-06-30", "2023-07-01", "2023-07-02"]
    assert files[0].url == (
        f"{DAILY_BASE}/bookTicker/ETHUSDT/ETHUSDT-bookTicker-2023-06-29.zip"
    )
    assert files[0].checksum_url == files[0].url + ".CHECKSUM"


def test_monthly_dumpfile_url_unchanged():
    f = DumpFile(symbol="BTCUSDT", data_type="aggTrades", period="2023-06")
    assert f.url == (
        "https://data.binance.vision/data/futures/um/monthly/"
        "aggTrades/BTCUSDT/BTCUSDT-aggTrades-2023-06.zip"
    )


def test_day_files_rejects_bad_date():
    with pytest.raises(ValueError):
        day_files("ETHUSDT", "bookTicker", "2023-06-31", "2023-07-02")
```

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement** — in `binance.py`: add `DAILY_BASE = "https://data.binance.vision/data/futures/um/daily"`; give `DumpFile` a `base: str = BASE` field (frozen dataclass — add with default, keeping the existing positional construction valid) and change `url` to use `self.base`; implement `day_files` with `datetime.date.fromisoformat` iteration (raises ValueError on bad dates natively):

```python
def day_files(symbol: str, data_type: str, start: str, end: str) -> list[DumpFile]:
    """All daily DumpFiles from start to end inclusive ("YYYY-MM-DD")."""
    import datetime as _dt

    d0, d1 = _dt.date.fromisoformat(start), _dt.date.fromisoformat(end)
    out: list[DumpFile] = []
    while d0 <= d1:
        out.append(DumpFile(symbol, data_type, d0.isoformat(), base=DAILY_BASE))
        d0 += _dt.timedelta(days=1)
    return out
```

In `catalog.py`: `sync_days` identical to `sync` but iterating `day_files`; factor the shared per-file logic (download → ingest → verify → rename → unlink) into a private `_sync_one(f, root, ingest, client) -> Path` used by both — do NOT duplicate the verify-before-rename block. Update Task 8's live-smoke `DailyFile` subclass usage is unaffected (tests untouched).

- [ ] **Step 4: Run full offline suite** — `uv run pytest -m "not network" -q`, all green (existing + 3 new).

- [ ] **Step 5: Commit** — `git commit -m "feat: daily dump-file support with shared verified-sync path"`

---

### Task 6: Q1 analysis — order-flow memory on real data

**Files:**
- Create: `src/microstructure/analyses/__init__.py` (empty), `src/microstructure/analyses/q1_orderflow_memory.py`
- Modify: `pyproject.toml` (add `matplotlib>=3.9` dependency via `uv add`)
- Test: `tests/analyses/__init__.py` (empty), `tests/analyses/test_q1.py`

**Interfaces:**
- Produces: `run_q1(root: Path, out_dir: Path, symbols: list[str], periods: list[str], max_lag: int = 1000) -> dict` returning `{symbol: {"gamma": float, "stderr": float, "n_events": int, "acf": list}}`, writing `out_dir/q1_acf_loglog.png` (log-log ACF for all symbols with fitted lines) and `out_dir/q1_results.md` (markdown: methodology paragraph, results table, benchmark table comparing to literature values, caveats). CLI: `python -m microstructure.analyses.q1_orderflow_memory --root data --out results` (argparse; defaults symbols=BTCUSDT,ETHUSDT periods=2023-06,2023-07).
- Benchmark table content (cite as literature ranges, sourced in research/01+03): equities/futures sign-ACF exponent γ ≈ 0.3–0.7 (Bouchaud et al. 2004); persistence horizon thousands of trades. The script states whether our γ falls in that range — either outcome is a documented finding, not a pass/fail.

- [ ] **Step 1: `uv add "matplotlib>=3.9"`**

- [ ] **Step 2: Write the failing test** (uses synthetic parquet, not real data — real runs are manual/CI-excluded):

```python
# tests/analyses/test_q1.py
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from microstructure.analyses.q1_orderflow_memory import run_q1
from microstructure.data.catalog import parquet_path
from microstructure.synthetic import markov_signs


def test_run_q1_on_synthetic_month_recovers_markov_memory(tmp_path: Path):
    signs = markov_signs(50_000, p_repeat=0.75, seed=9)
    t0 = datetime(2023, 6, 1, tzinfo=timezone.utc)
    df = pl.DataFrame({
        "agg_trade_id": np.arange(50_000),
        "price": np.full(50_000, 100.0),
        "qty": np.ones(50_000),
        "first_trade_id": np.arange(50_000),
        "last_trade_id": np.arange(50_000),
        "ts": [t0 + timedelta(milliseconds=3 * i) for i in range(50_000)],
        "is_buyer_maker": signs < 0,
    }, schema_overrides={"ts": pl.Datetime("ms", "UTC")})
    p = parquet_path(tmp_path, "TESTUSDT", "aggTrades", "2023-06")
    p.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p)

    out = tmp_path / "results"
    res = run_q1(tmp_path, out, symbols=["TESTUSDT"], periods=["2023-06"], max_lag=50)
    assert (out / "q1_acf_loglog.png").exists()
    assert (out / "q1_results.md").exists()
    assert res["TESTUSDT"]["n_events"] == 50_000
    acf = np.array(res["TESTUSDT"]["acf"])
    assert abs(acf[1] - 0.5) < 0.03  # markov ACF(1) = 2p-1
```

- [ ] **Step 3: Run to verify fail; implement.** Structure of `q1_orderflow_memory.py` (write exactly this logic; matplotlib with `Agg` backend set before pyplot import):

```python
# src/microstructure/analyses/q1_orderflow_memory.py
"""Q1: Is Binance aggressor order flow long-memory?

Method: aggressor-event sign series per symbol -> FFT ACF to max_lag ->
log-log power-law fit over lags [10, max_lag//2] -> compare exponent to
the equity/futures literature range (gamma ~ 0.3-0.7, Bouchaud 2004).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from microstructure.estimators.acf import fit_power_law, sign_acf
from microstructure.signals.load import load_events

LIT_RANGE = (0.3, 0.7)  # equity/futures sign-ACF exponent range, research/01+03


def run_q1(root: Path, out_dir: Path, symbols: list[str], periods: list[str],
           max_lag: int = 1000) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    fig, ax = plt.subplots(figsize=(7, 5))
    for sym in symbols:
        ev = load_events(root, sym, periods)
        signs = ev["sign"].to_numpy()
        acf = sign_acf(signs, max_lag)
        fit = fit_power_law(acf, lo=10, hi=max_lag // 2)
        results[sym] = {"gamma": fit.exponent, "stderr": fit.stderr,
                        "n_events": int(signs.size), "acf": acf.tolist()}
        lags = np.arange(1, max_lag + 1)
        ax.loglog(lags, np.clip(acf[1:], 1e-6, None), label=f"{sym} (γ̂={fit.exponent:.3f})")
    ax.set_xlabel("lag (events)"); ax.set_ylabel("sign ACF"); ax.legend()
    ax.set_title(f"Order-flow sign memory, periods {periods[0]}..{periods[-1]}")
    fig.savefig(out_dir / "q1_acf_loglog.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _write_results_md(out_dir, results, periods)
    (out_dir / "q1_results.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "acf"} for k, v in results.items()}, indent=2))
    return results
```

plus `_write_results_md` producing: methodology paragraph, per-symbol table (symbol | n_events | γ̂ | OLS stderr), benchmark table (our γ̂ vs literature 0.3–0.7 vs in-range yes/no), and a caveats section that MUST include: OLS stderr understates uncertainty under autocorrelation; ms timestamp ties merged by aggressor aggregation; sample is 2 months of a specific regime. And the `if __name__ == "__main__":` argparse block matching the CLI contract.

- [ ] **Step 4: Run tests; run ruff; ALL green.**

- [ ] **Step 5: RUN THE REAL ANALYSIS**: `uv run python -m microstructure.analyses.q1_orderflow_memory --root data --out results` (requires the synced aggTrades months; if missing, the FileNotFoundError from load_events is the correct loud failure — report it rather than working around). Record the actual γ̂ values in the commit message body.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: Q1 order-flow memory analysis with real-data results"` (results/ may be committed — small md/png/json only).

---

### Task 7: bookTicker slice + Q2 response-function analysis

**Files:**
- Create: `src/microstructure/analyses/q2_response.py`
- Test: `tests/analyses/test_q2.py`

**Interfaces:**
- Produces: `run_q2(root, out_dir, symbol, periods: list[str], max_lag=500) -> dict` with keys `response` (list), `decay_exponent`, `decay_stderr`, `n_events`; writes `q2_response.png` (R(ℓ) log-log + both candidate fits) and `q2_results.md` (methodology, exponential-vs-power-law comparison via residual sum of squares on log scale over lags 10–200, benchmark note: Bouchaud 2004 found slow power-law-like decay; verdict sentence states which shape fits Binance better — either answer is a finding). CLI mirrors Q1 with `--symbol` and daily `--periods`.
- Data step INSIDE this task: sync a 14-day ETHUSDT bookTicker slice (2023-06-01..2023-06-14, ETH chosen as the smaller files, ~1.5GB zipped total) via `sync_days`, plus the matching aggTrades daily files are NOT needed (monthly 2023-06 already covers those events — filter events to the slice's date range in the analysis).

- [ ] **Step 1: Write the failing test** — synthetic: build events + bookTicker parquet where mids follow a known exponential kernel from the signs (reuse the Task-2 `_mids_from_kernel` construction, written into daily-period parquet files at the right paths); assert `run_q2` recovers the kernel decay within tolerance and writes both outputs. Include the exact test code in the implementation commit (same pattern as test_q1 — synthetic parquet through the real pipeline).

- [ ] **Step 2: Implement `run_q2`**: `load_events` (monthly period "2023-06", filtered to ts within the daily slice range) → `load_book_ticker` (daily periods) → `events_with_prior_mid` → numpy arrays → `response_function` → fit power-law (reuse `fit_power_law`) AND exponential (linear fit of log R vs lag over the same window) → RSS comparison → figure + md. Loud failure if n_dropped / n_events > 0.01 (data alignment problem — raise, do not warn).

- [ ] **Step 3: Sync the slice**: `uv run python -c "from pathlib import Path; from microstructure.data.catalog import sync_days; print(len(sync_days(Path('data'), 'ETHUSDT', 'bookTicker', '2023-06-01', '2023-06-14')))"` (long download — run it, it is required for Step 4).

- [ ] **Step 4: Run tests (synthetic), then the REAL analysis**; record R(1), the decay exponent, and the shape verdict in the commit body.

- [ ] **Step 5: Commit** — `git commit -m "feat: Q2 response function analysis with real-data results"`.

---

### Task 8: Q3 OFI-linearity analysis

**Files:**
- Create: `src/microstructure/analyses/q3_ofi.py`
- Test: `tests/analyses/test_q3.py`

**Interfaces:**
- Produces: `run_q3(root, out_dir, symbol, periods: list[str], window: str = "10s") -> dict` with `slope`, `stderr`, `r2`, `n_windows`, `depth_scaling_check` (slope of log|beta| vs log depth across 5 depth quintiles — Cont predicts ≈ −1); writes `q3_ofi_scatter.png` (binned scatter + fit line) and `q3_results.md` (benchmark table: our R² vs Cont's 65–70% equities vs Silantyev's BitMEX finding). Method: from daily bookTicker parquet, compute per-update OFI (`ofi_events`), bucket by `window` bars (`pl.group_by_dynamic` on ts), sum OFI and take last-mid minus first-mid per bar, drop empty bars, regress with `ols_through_origin`; depth quintiles via mean (bid_qty+ask_qty)/2 per bar.
- Uses the same 14-day ETHUSDT bookTicker slice from Task 7 (no new downloads).

- [ ] **Step 1: Write the failing test** — synthetic bookTicker parquet engineered so that Δmid per window = 0.002 × ΣOFI + small noise (construct L1 paths whose ofi and mid moves are linked by construction; simplest: random walk bid/ask quantities, mid stepped by 0.002×that bar's summed OFI at bar end); assert recovered slope within 15% and r2 > 0.8, and both output files exist.

- [ ] **Step 2: Implement; tests green; ruff green.**

- [ ] **Step 3: Run the REAL analysis** on the ETHUSDT slice; record slope, R², and the depth-scaling exponent in the commit body.

- [ ] **Step 4: Commit** — `git commit -m "feat: Q3 OFI linearity analysis with real-data results"`.

---

### Task 9: LEARNING.md + results README

**Files:**
- Create: `LEARNING.md`, modify `README.md` (create if absent)

**Interfaces:** none — documentation of everything above, written for the project owner (an undergrad with Python skills, intro-stats level, preparing for quant interviews).

- [ ] **Step 1: Write LEARNING.md** with these mandatory sections, each written from the ACTUAL code and ACTUAL results in results/ (read them first; no generic filler):
  1. The order book and aggressor trades (why is_buyer_maker encodes the aggressor; why prints ≠ orders and how our aggregation fixes it)
  2. Order-flow memory (what an ACF is, why FFT computes it, what long memory means, what OUR γ̂ came out to and how it compares to equities — with the actual numbers)
  3. Price impact and the response function (bathtub analogy: temporary vs permanent; what R(ℓ) measures; what our decay-shape verdict was and what it says about Obizhaeva-Wang vs propagator models)
  4. OFI and linear impact (the queue intuition; slope ≈ 1/depth meaning; our slope/R² vs Cont's)
  5. Statistics used honestly (why OLS stderr lies under autocorrelation; why we validate on synthetic data; the difference between "in the literature range" and "correct")
  6. Interview drill: 10 likely questions about this project with strong answers (e.g., "why might your γ differ from equities?", "what breaks if you don't merge same-timestamp prints?")
- [ ] **Step 2: Write README.md**: project summary, results gallery (embed the three PNGs), how to reproduce every figure from a fresh clone (sync commands + analysis commands), repo map, literature references.
- [ ] **Step 3: Commit** — `git commit -m "docs: LEARNING.md and results README"`.

---

## Out of scope (deliberate)
- Phase-2 cross-section, Hawkes fitting, execution simulation — next plan.
- CI — being handled as a cloud-session errand by the user.
- Block-bootstrap CIs for γ̂ — documented as a caveat in Q1; candidate first Phase-2 rigor upgrade.
