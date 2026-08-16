"""Tests for the execution-cost replay simulator (Phase 3, Task 4).

Contract under test (see task-4-brief.md):
1. `replay_day` builds a `ReplayData` from events + bookTicker: event ts,
   signs, qtys, prior-mids, half-spreads (asof from bt, "strictly before"
   convention matching `events_with_prior_mid`), and typical_event_qty
   (median qty).
2. `simulate_schedule` costs each child at
   (mid_at_child - arrival_mid)*side + half_spread_at_child + temp_impact(q),
   temp_impact(q) = side * G[1] * (q / typical_event_qty) (LINEAR scaling of
   the lag-1 kernel value; documented caveat vs. sqrt-law literature).
   shortfall_per_unit = sum(cost_i * q_i) / sum(q_i). Verified by hand for a
   2-child case (exact equality, see module docstring below).
3. Schedule generators are pure and satisfy invariants: child sizes sum to
   the parent qty, indices lie in [0, horizon_events), indices non-decreasing
   (children never reordered).
4. Planted-advantage test: a replay where adverse flow clusters precede
   price moves against the parent side -> the reactive schedule defers into
   those windows and comes out ahead of TWAP. A shuffled/no-signal replay
   (flow uncorrelated with future price) -> reactive and TWAP shortfalls
   are statistically indistinguishable (their across-seed dispersions
   overlap).

Hand-computed 2-child shortfall (exact equality):
    side=+1, typical_event_qty=10, G=[0.0, 2.0], arrival_mid=100.0
    child 1: q=10 (1x typical) @ mid=100.0, half_spread=0.5
        cost = (100.0-100.0)*1 + 0.5 + 2.0*(10/10) = 0.0 + 0.5 + 2.0 = 2.5
    child 2: q=20 (2x typical) @ mid=101.0, half_spread=0.3
        cost = (101.0-100.0)*1 + 0.3 + 2.0*(20/10) = 1.0 + 0.3 + 4.0 = 5.3
    shortfall_per_unit = (2.5*10 + 5.3*20) / (10+20) = (25 + 106) / 30
                        = 131/30 = 4.366666...
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from microstructure.data.catalog import parquet_path
from microstructure.execution.simulator import (
    ReplayData,
    ScheduleResult,
    frontloaded_schedule,
    kernel_half_life_lag,
    reactive_schedule,
    replay_day,
    simulate_schedule,
    twap_schedule,
)

FRACTION_TOLERANCE = 1e-9


# --------------------------------------------------------------------------
# replay_day
# --------------------------------------------------------------------------


def _write_day_parquets(root: Path, symbol: str, month: str, day: str) -> None:
    """5 aggTrades prints -> 3 aggressor events, plus matching bookTicker quotes."""
    t0 = datetime(2023, 6, 1, 0, 0, 0, tzinfo=UTC)
    raw = pl.DataFrame(
        {
            "agg_trade_id": [1, 2, 3, 4, 5],
            "price": [100.0, 100.0, 100.5, 100.5, 101.0],
            "qty": [1.0, 1.0, 2.0, 1.0, 3.0],
            "first_trade_id": [1, 2, 3, 4, 5],
            "last_trade_id": [1, 2, 3, 4, 5],
            "ts": [
                t0,
                t0,
                t0 + timedelta(milliseconds=10),
                t0 + timedelta(milliseconds=10),
                t0 + timedelta(milliseconds=20),
            ],
            # buyer-taker (+1): rows 0,1 -> merge to sign +1 qty=2 @ ts=t0
            # buyer-taker (+1): rows 2,3 -> merge to sign +1 qty=3 @ ts=t0+10
            # seller-taker (-1): row 4 -> sign -1 qty=3 @ ts=t0+20
            "is_buyer_maker": [False, False, False, False, True],
        },
        schema_overrides={"ts": pl.Datetime("ms", "UTC")},
    )
    agg_path = parquet_path(root, symbol, "aggTrades", month)
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    raw.write_parquet(agg_path)

    bt = pl.DataFrame(
        {
            "update_id": [1, 2, 3],
            "bid_price": [99.9, 100.4, 100.8],
            "bid_qty": [1.0, 1.0, 1.0],
            "ask_price": [100.1, 100.6, 101.2],
            "ask_qty": [1.0, 1.0, 1.0],
            "ts": [
                t0 - timedelta(milliseconds=1),
                t0 + timedelta(milliseconds=9),
                t0 + timedelta(milliseconds=19),
            ],
        },
        schema_overrides={"ts": pl.Datetime("ms", "UTC")},
    )
    bt_path = parquet_path(root, symbol, "bookTicker", day)
    bt_path.parent.mkdir(parents=True, exist_ok=True)
    bt.write_parquet(bt_path)


def test_replay_day_builds_expected_arrays(tmp_path: Path):
    symbol = "TESTUSDT"
    _write_day_parquets(tmp_path, symbol, "2023-06", "2023-06-01")

    from microstructure.signals.load import load_book_ticker, load_events

    events = load_events(tmp_path, symbol, ["2023-06"])
    bt = load_book_ticker(tmp_path, symbol, ["2023-06-01"])

    rd = replay_day(events, bt)

    assert isinstance(rd, ReplayData)
    assert rd.signs.tolist() == [1, 1, -1]
    assert rd.qtys.tolist() == [2.0, 3.0, 3.0]
    # prior mids: event 0 -> bt row 0 mid=(99.9+100.1)/2=100.0
    #             event 1 -> bt row 1 mid=(100.4+100.6)/2=100.5
    #             event 2 -> bt row 2 mid=(100.8+101.2)/2=101.0
    np.testing.assert_allclose(rd.prior_mids, [100.0, 100.5, 101.0])
    # half-spreads: (ask-bid)/2 = 0.1, 0.1, 0.2
    np.testing.assert_allclose(rd.half_spreads, [0.1, 0.1, 0.2])
    # typical_event_qty = median([2.0, 3.0, 3.0]) = 3.0
    assert rd.typical_event_qty == pytest.approx(3.0)
    assert rd.ts.dtype == np.int64


def test_replay_day_frozen_dataclass_is_immutable(tmp_path: Path):
    symbol = "TESTUSDT"
    _write_day_parquets(tmp_path, symbol, "2023-06", "2023-06-01")
    from microstructure.signals.load import load_book_ticker, load_events

    events = load_events(tmp_path, symbol, ["2023-06"])
    bt = load_book_ticker(tmp_path, symbol, ["2023-06-01"])
    rd = replay_day(events, bt)

    with pytest.raises(Exception):  # noqa: B017 - dataclass(frozen=True) raises FrozenInstanceError
        rd.typical_event_qty = 99.0


# --------------------------------------------------------------------------
# simulate_schedule: hand-verified exact arithmetic
# --------------------------------------------------------------------------


def _tiny_replay(n: int = 5) -> ReplayData:
    """5-event replay: prior_mids rise by 1.0 per event, half_spread alternates."""
    return ReplayData(
        ts=np.arange(n, dtype=np.int64) * 10,
        signs=np.array([1, 1, -1, 1, -1][:n], dtype=np.int8),
        qtys=np.full(n, 10.0),
        prior_mids=100.0 + np.arange(n, dtype=np.float64),
        half_spreads=np.array([0.5, 0.3, 0.5, 0.3, 0.5][:n]),
        typical_event_qty=10.0,
    )


def test_simulate_schedule_two_child_exact_shortfall():
    rd = _tiny_replay(n=5)
    kernel_g = np.array([0.0, 2.0, 2.0, 2.0])  # G[1] = 2.0

    # Parent = 3.0 typical-event-units = 30 units total (typical_event_qty=10).
    # Two children as FRACTIONS of the parent: 1/3 (-> 10 units) at event
    # index 0 (mid=100.0, half_spread=0.5), 2/3 (-> 20 units) at event index 1
    # (mid=101.0, half_spread=0.3).
    child_times = np.array([0, 1])
    child_sizes = np.array([1.0 / 3.0, 2.0 / 3.0])

    result = simulate_schedule(
        rd, side=1, parent_qty_events=3.0, horizon_events=5,
        child_times=child_times, child_sizes=child_sizes, kernel_g=kernel_g,
        schedule_name="hand_check",
    )

    assert isinstance(result, ScheduleResult)
    expected = 131.0 / 30.0
    assert result.shortfall_per_unit == pytest.approx(expected, abs=1e-6)
    assert result.n_children == 2
    assert result.filled is True
    assert result.schedule_name == "hand_check"


def test_simulate_schedule_sell_side_flips_sign_of_drift_and_impact():
    """side=-1: drift and impact terms flip sign vs. the side=+1 hand-check."""
    rd = _tiny_replay(n=5)
    kernel_g = np.array([0.0, 2.0, 2.0, 2.0])

    child_times = np.array([0, 1])
    child_sizes = np.array([1.0 / 3.0, 2.0 / 3.0])

    result = simulate_schedule(
        rd, side=-1, parent_qty_events=3.0, horizon_events=5,
        child_times=child_times, child_sizes=child_sizes, kernel_g=kernel_g,
        schedule_name="sell_check",
    )
    # child 1: (100-100)*-1 + 0.5 + 2.0*(10/10) = 0.0 + 0.5 + 2.0 = 2.5
    #   (drift is zero here regardless of side; impact always costs the trader)
    # child 2: (101-100)*-1 + 0.3 + 2.0*(20/10) = -1.0 + 0.3 + 4.0 = 3.3
    expected = (2.5 * 10 + 3.3 * 20) / 30.0
    assert result.shortfall_per_unit == pytest.approx(expected, abs=1e-6)


# --------------------------------------------------------------------------
# kernel_half_life_lag
# --------------------------------------------------------------------------


def test_kernel_half_life_lag_finds_post_peak_half_decay():
    # peak=10.0 at lag 3; decays to <=5.0 at lag 6 -> half-life = 6
    g = np.array([0.0, 4.0, 8.0, 10.0, 9.0, 6.0, 4.0, 2.0])
    assert kernel_half_life_lag(g) == 6


def test_kernel_half_life_lag_falls_back_when_monotone_increasing():
    # never decays within recorded lags -> defensive fallback: len // 4
    g = np.arange(20, dtype=np.float64)
    assert kernel_half_life_lag(g) == 5


# --------------------------------------------------------------------------
# Schedule generator invariants
# --------------------------------------------------------------------------


@pytest.mark.parametrize("horizon,n_children", [(2000, 20), (100, 5)])
def test_twap_schedule_invariants(horizon, n_children):
    times, sizes = twap_schedule(horizon, n_children)
    assert len(times) == n_children
    assert len(sizes) == n_children
    assert np.all(times >= 0) and np.all(times < horizon)
    assert np.all(np.diff(times) >= 0)  # monotone non-decreasing
    assert np.all(sizes >= 0)
    # sizes are FRACTIONS of the parent order, summing to 1.0
    assert sizes.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("decay", [0.1, 0.5, 2.0])
def test_frontloaded_schedule_invariants(decay):
    horizon, n_children = 2000, 20
    times, sizes = frontloaded_schedule(horizon, n_children, decay=decay)
    assert len(times) == n_children
    assert len(sizes) == n_children
    assert np.all(times >= 0) and np.all(times < horizon)
    assert np.all(np.diff(times) >= 0)
    assert np.all(sizes >= 0)
    assert sizes.sum() == pytest.approx(1.0)


def test_frontloaded_schedule_puts_more_size_early():
    horizon, n_children = 2000, 20
    _times, sizes = frontloaded_schedule(horizon, n_children, decay=1.0)
    first_half = sizes[: n_children // 2].sum()
    second_half = sizes[n_children // 2 :].sum()
    assert first_half > second_half


def test_reactive_schedule_invariants():
    rd = _make_flow_replay(n_events=2200, seed=1)
    times, sizes, n_deferrals = reactive_schedule(
        rd, side=1, horizon_events=2000, n_children=20, lookback=50, pause_threshold=0.2
    )
    assert len(times) == 20
    assert len(sizes) == 20
    assert np.all(times >= 0) and np.all(times < 2000)
    assert np.all(np.diff(times) >= 0)
    assert n_deferrals >= 0
    # all must fill by horizon end -> last child forced at/near horizon-1 at worst
    assert times[-1] <= 1999
    assert sizes.sum() == pytest.approx(1.0)


def _make_flow_replay(n_events: int, seed: int, adverse_at: np.ndarray | None = None) -> ReplayData:
    """Baseline i.i.d. random-sign replay with flat mids and spreads.

    If `adverse_at` is given, those event indices carry a burst of signed
    flow opposing the parent's assumed +1 side, followed immediately by a
    price move in that same adverse direction — the "planted adverse
    cluster" the reactive schedule should learn to avoid.
    """
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_events)
    mids = np.full(n_events, 100.0)
    if adverse_at is not None:
        for start in adverse_at:
            burst = slice(start, start + 30)
            signs[burst] = -1  # opposing flow (parent side assumed +1)
            # price drops shortly after the burst -> adverse to a +1 parent
            mids[start + 30 :] -= 0.5
    return ReplayData(
        ts=np.arange(n_events, dtype=np.int64) * 10,
        signs=signs,
        qtys=np.full(n_events, 10.0),
        prior_mids=mids,
        half_spreads=np.full(n_events, 0.05),
        typical_event_qty=10.0,
    )


# --------------------------------------------------------------------------
# Planted-advantage test
# --------------------------------------------------------------------------


def test_reactive_schedule_beats_twap_when_adverse_flow_precedes_price_moves():
    """Adverse flow clusters planted right before the child-slot times TWAP
    would use -> reactive defers into the post-move (better) price and beats
    TWAP's mean shortfall for a +1 (buy) parent."""
    horizon, n_children = 2000, 20
    kernel_g = np.array([0.0] + [0.001] * 50)  # tiny impact so drift dominates

    # TWAP child slots land at roughly horizon/n_children apart; plant an
    # adverse burst immediately before several of them.
    twap_times, _ = twap_schedule(horizon, n_children)
    adverse_at = np.unique(np.clip(twap_times[2:18] - 35, 0, horizon - 40))

    n_seeds = 12
    twap_shortfalls = []
    reactive_shortfalls = []
    for seed in range(n_seeds):
        rd = _make_flow_replay(n_events=horizon + 200, seed=seed, adverse_at=adverse_at)

        t_times, t_sizes = twap_schedule(horizon, n_children)
        twap_res = simulate_schedule(
            rd, side=1, parent_qty_events=n_children * 1.0, horizon_events=horizon,
            child_times=t_times, child_sizes=t_sizes, kernel_g=kernel_g,
            schedule_name="twap",
        )
        twap_shortfalls.append(twap_res.shortfall_per_unit)

        r_times, r_sizes, _ = reactive_schedule(
            rd, side=1, horizon_events=horizon, n_children=n_children,
            lookback=50, pause_threshold=0.2,
        )
        reactive_res = simulate_schedule(
            rd, side=1, parent_qty_events=n_children * 1.0, horizon_events=horizon,
            child_times=r_times, child_sizes=r_sizes, kernel_g=kernel_g,
            schedule_name="reactive",
        )
        reactive_shortfalls.append(reactive_res.shortfall_per_unit)

    mean_twap = float(np.mean(twap_shortfalls))
    mean_reactive = float(np.mean(reactive_shortfalls))
    assert mean_reactive < mean_twap, (
        f"reactive ({mean_reactive}) should beat twap ({mean_twap}) "
        "when adverse flow reliably precedes price moves against the parent"
    )


def test_reactive_and_twap_tie_when_no_signal():
    """No relationship between flow and future price -> reactive's deferrals
    are noise, not signal; across-seed dispersions of reactive and TWAP
    shortfall should overlap (no reliable separation)."""
    horizon, n_children = 2000, 20
    kernel_g = np.array([0.0] + [0.001] * 50)

    n_seeds = 12
    twap_shortfalls = []
    reactive_shortfalls = []
    for seed in range(n_seeds):
        rd = _make_flow_replay(n_events=horizon + 200, seed=seed, adverse_at=None)

        t_times, t_sizes = twap_schedule(horizon, n_children)
        twap_res = simulate_schedule(
            rd, side=1, parent_qty_events=n_children * 1.0, horizon_events=horizon,
            child_times=t_times, child_sizes=t_sizes, kernel_g=kernel_g,
            schedule_name="twap",
        )
        twap_shortfalls.append(twap_res.shortfall_per_unit)

        r_times, r_sizes, _ = reactive_schedule(
            rd, side=1, horizon_events=horizon, n_children=n_children,
            lookback=50, pause_threshold=0.2,
        )
        reactive_res = simulate_schedule(
            rd, side=1, parent_qty_events=n_children * 1.0, horizon_events=horizon,
            child_times=r_times, child_sizes=r_sizes, kernel_g=kernel_g,
            schedule_name="reactive",
        )
        reactive_shortfalls.append(reactive_res.shortfall_per_unit)

    twap_arr = np.array(twap_shortfalls)
    reactive_arr = np.array(reactive_shortfalls)

    # Overlap check: the two distributions' [mean-sd, mean+sd] ranges intersect.
    t_lo, t_hi = twap_arr.mean() - twap_arr.std(), twap_arr.mean() + twap_arr.std()
    r_lo, r_hi = reactive_arr.mean() - reactive_arr.std(), reactive_arr.mean() + reactive_arr.std()
    overlap = max(t_lo, r_lo) <= min(t_hi, r_hi)
    assert overlap, (
        f"expected overlapping dispersions with no signal: twap=[{t_lo},{t_hi}] "
        f"reactive=[{r_lo},{r_hi}]"
    )


# --------------------------------------------------------------------------
# reactive_schedule: pause/defer behavior in isolation
# --------------------------------------------------------------------------


def test_reactive_schedule_defers_on_opposing_flow():
    """Construct a replay where the lookback window immediately before the
    SECOND TWAP slot has strongly opposing flow (all -1 signs against a +1
    parent) -> that child slot must be deferred later than its TWAP
    counterpart. (Slot 0's lookback window is empty -- there is nothing
    before event index 0 -- so it can never trigger a deferral; slot 1, at
    t=100 for this horizon/n_children grid, has a full lookback window and
    is the one this test targets.)"""
    horizon, n_children, lookback = 2000, 20, 50
    n = horizon + 100
    signs = np.ones(n, dtype=np.int8)
    twap_times, _ = twap_schedule(horizon, n_children)
    slot1 = int(twap_times[1])
    signs[slot1 - lookback : slot1] = -1  # opposing flow right before slot 1
    rd = ReplayData(
        ts=np.arange(n, dtype=np.int64) * 10,
        signs=signs,
        qtys=np.full(n, 10.0),
        prior_mids=np.full(n, 100.0),
        half_spreads=np.full(n, 0.05),
        typical_event_qty=10.0,
    )
    r_times, _r_sizes, n_deferrals = reactive_schedule(
        rd, side=1, horizon_events=horizon, n_children=n_children,
        lookback=lookback, pause_threshold=0.5,
    )
    assert n_deferrals >= 1
    assert r_times[1] > twap_times[1]
    assert np.all(r_times < horizon)
    assert np.all(np.diff(r_times) >= 0)
