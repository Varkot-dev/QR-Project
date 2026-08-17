"""Tests for Q6: branching-ratio (Hawkes endogeneity) panel (Phase 3, Task 3).

Contract under test (see task-3-brief.md and its binding elaboration):
1. Two "planted-Hawkes" synthetic symbols, built by running
   `simulate_hawkes_exp` with known (mu, alpha, beta) and synthesizing
   aggTrades-schema parquet event timestamps (ms) + alternating signs/qtys
   from the simulated event times. alpha=0.3 and alpha=0.6 respectively.
   `run_q6` must recover alpha_median within +-0.05 of the planted value
   for each.
2. One symbol with no parquet on disk at all -> must land in `failures`,
   never abort the run.
3. Output files (.json, .md, .parquet, .png) must exist and the json's
   per-symbol records must carry all the fields the brief specifies.

Fixture sizing: t_end is tuned per alpha so each simulation lands in the
~50-100k event range (see module docstring comments below at each
simulate_hawkes_exp call) -- large enough for 6 sub-windows to each have a
statistically meaningful sample, small enough to keep the full test file
under the ~120s runtime budget.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from microstructure.analyses.q6_endogeneity import run_q6
from microstructure.data.catalog import parquet_path
from microstructure.estimators.hawkes import simulate_hawkes_exp, simulate_seasonal_hawkes_exp

MU, BETA = 1.0, 2.0
# t_end tuned so each planted symbol lands roughly in [50_000, 100_000]
# events (measured via a one-off probe at seed=1): alpha=0.3 -> ~80k events
# at t_end=55_000s; alpha=0.6 -> ~90k events at t_end=35_000s (higher alpha
# needs a shorter horizon for a similar event count, since self-excitation
# amplifies the base Poisson rate).
ALPHA_LOW, T_END_LOW = 0.3, 55_000.0
ALPHA_HIGH, T_END_HIGH = 0.6, 35_000.0
ALPHA_TOL = 0.05


def _write_planted_hawkes_fixture(
    root: Path, symbol: str, mu: float, alpha: float, beta: float, t_end: float, seed: int,
    month: str = "2023-06",
) -> int:
    """Simulate a Hawkes process and write it as an aggTrades-schema parquet.

    Event times (float seconds from `simulate_hawkes_exp`) are converted to
    int64 epoch-ms starting at 2023-06-01 00:00 UTC, deduplicated at ms
    resolution (`to_aggressor_events` merges same-(ts, sign) rows, which
    would silently drop/merge distinct simulated events if two landed in
    the same millisecond -- resolved here by nudging any collision forward
    by 1ms, preserving event COUNT and approximate inter-event structure
    rather than losing events to aggregation). Signs alternate +1/-1 (the
    brief's "alternating signs/qtys") so `to_aggressor_events`' tie-break
    sort is exercised without materially disturbing the timing statistics
    the Hawkes fit depends on. Returns the final event count actually
    written (after dedup nudging).
    """
    times_s = simulate_hawkes_exp(mu, alpha, beta, t_end, seed=seed)
    t0 = datetime(2023, 6, 1, 0, 0, 0, tzinfo=UTC)
    ts_ms = (times_s * 1000.0).astype(np.int64)
    # Nudge forward any ms collisions to a strictly increasing sequence,
    # preserving event count.
    ts_ms = np.maximum.accumulate(ts_ms)
    for i in range(1, ts_ms.size):
        if ts_ms[i] <= ts_ms[i - 1]:
            ts_ms[i] = ts_ms[i - 1] + 1
    n = ts_ms.size
    event_ts = [t0 + timedelta(milliseconds=int(ms)) for ms in ts_ms]
    signs = np.where(np.arange(n) % 2 == 0, 1, -1)
    qty = np.full(n, 1.0) + 0.01 * (np.arange(n) % 5)  # alternating-ish qty, all positive

    agg = pl.DataFrame(
        {
            "agg_trade_id": np.arange(n),
            "price": np.full(n, 100.0),
            "qty": qty,
            "first_trade_id": np.arange(n),
            "last_trade_id": np.arange(n),
            "ts": event_ts,
            "is_buyer_maker": signs < 0,
        },
        schema_overrides={"ts": pl.Datetime("ms", "UTC")},
    )
    agg_path = parquet_path(root, symbol, "aggTrades", month)
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg.write_parquet(agg_path)
    return n


def _write_event_times_fixture(
    root: Path, symbol: str, times_s: np.ndarray, month: str = "2023-06",
) -> int:
    """Write a raw array of event times (float seconds from an arbitrary
    origin) as an aggTrades-schema parquet, same dedup/nudge/sign/qty
    convention as `_write_planted_hawkes_fixture` but taking already-
    simulated times directly (used for the regime-switching/seasonal trap
    fixture, which is not a plain `simulate_hawkes_exp` output). Returns the
    final event count actually written (after dedup nudging).
    """
    t0 = datetime(2023, 6, 1, 0, 0, 0, tzinfo=UTC)
    ts_ms = (times_s * 1000.0).astype(np.int64)
    ts_ms = np.maximum.accumulate(ts_ms)
    for i in range(1, ts_ms.size):
        if ts_ms[i] <= ts_ms[i - 1]:
            ts_ms[i] = ts_ms[i - 1] + 1
    n = ts_ms.size
    event_ts = [t0 + timedelta(milliseconds=int(ms)) for ms in ts_ms]
    signs = np.where(np.arange(n) % 2 == 0, 1, -1)
    qty = np.full(n, 1.0) + 0.01 * (np.arange(n) % 5)

    agg = pl.DataFrame(
        {
            "agg_trade_id": np.arange(n),
            "price": np.full(n, 100.0),
            "qty": qty,
            "first_trade_id": np.arange(n),
            "last_trade_id": np.arange(n),
            "ts": event_ts,
            "is_buyer_maker": signs < 0,
        },
        schema_overrides={"ts": pl.Datetime("ms", "UTC")},
    )
    agg_path = parquet_path(root, symbol, "aggTrades", month)
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg.write_parquet(agg_path)
    return n


@pytest.fixture
def planted_root(tmp_path: Path) -> Path:
    _write_planted_hawkes_fixture(
        tmp_path, "LOWUSDT", MU, ALPHA_LOW, BETA, T_END_LOW, seed=101
    )
    _write_planted_hawkes_fixture(
        tmp_path, "HIGHUSDT", MU, ALPHA_HIGH, BETA, T_END_HIGH, seed=102
    )
    # MISSINGUSDT deliberately has no parquet on disk -> must land in failures.
    return tmp_path


def test_run_q6_recovers_planted_alphas_and_reports_missing(planted_root: Path):
    out_dir = planted_root / "results"
    result = run_q6(
        planted_root, out_dir,
        symbols=["LOWUSDT", "HIGHUSDT", "MISSINGUSDT"],
        month="2023-06", windows=6, top_n=40,
    )

    by_symbol = {r["symbol"]: r for r in result["records"]}
    assert "LOWUSDT" in by_symbol
    assert "HIGHUSDT" in by_symbol

    low = by_symbol["LOWUSDT"]
    high = by_symbol["HIGHUSDT"]

    assert abs(low["alpha_median"] - ALPHA_LOW) < ALPHA_TOL, (
        f"LOWUSDT alpha_median={low['alpha_median']} vs planted {ALPHA_LOW}"
    )
    assert abs(high["alpha_median"] - ALPHA_HIGH) < ALPHA_TOL, (
        f"HIGHUSDT alpha_median={high['alpha_median']} vs planted {ALPHA_HIGH}"
    )

    for rec in (low, high):
        assert rec["n_events"] > 0
        assert len(rec["alphas"]) == 6
        assert "alpha_iqr" in rec
        assert "n_converged" in rec
        assert 0 <= rec["n_converged"] <= 6
        assert "alpha_cv" in rec
        assert "raw_delta" in rec
        assert "median_beta" in rec
        assert "median_mu" in rec

    # --- missing symbol lands in failures, never aborts the run ---
    failures = {f["symbol"]: f for f in result["failures"]}
    assert "MISSINGUSDT" in failures
    assert failures["MISSINGUSDT"]["reason"]
    assert "MISSINGUSDT" not in by_symbol

    # --- metadata ---
    assert result["month"] == "2023-06"
    assert result["windows"] == 6
    assert result["n_symbols_successful"] == 2
    assert result["n_symbols_failed"] == 1

    # --- cross-section regression present with >= 2 points ---
    assert "activity_regression" in result
    assert "agreement" in result

    # --- output files exist ---
    assert (out_dir / "q6_endogeneity.json").exists()
    assert (out_dir / "q6_endogeneity.md").exists()
    assert (out_dir / "q6_endogeneity.parquet").exists()
    assert (out_dir / "q6_endogeneity.png").exists()

    df = pl.read_parquet(out_dir / "q6_endogeneity.parquet")
    assert df.height == 2
    assert set(df.columns) >= {
        "symbol", "n_events", "alpha_median", "alpha_iqr", "n_converged",
        "alpha_cv", "raw_delta", "median_beta", "median_mu",
    }


def test_run_q6_never_aborts_on_all_failures(tmp_path: Path):
    out_dir = tmp_path / "results"
    result = run_q6(
        tmp_path, out_dir, symbols=["GHOSTUSDT"], month="2023-06", windows=6, top_n=40,
    )
    assert result["records"] == []
    assert len(result["failures"]) == 1
    assert result["failures"][0]["symbol"] == "GHOSTUSDT"
    assert (out_dir / "q6_endogeneity.md").exists()
    assert (out_dir / "q6_endogeneity.json").exists()
    assert (out_dir / "q6_endogeneity.parquet").exists()
    assert (out_dir / "q6_endogeneity.png").exists()


def test_run_q6_records_raw_vs_rescaled_delta(planted_root: Path):
    """The raw-clock-time fit on the first sub-window must be recorded
    separately from the business-time window-1 fit, and raw_delta must be
    their difference (documents the seasonality-bias measurement itself,
    not just that it runs)."""
    out_dir = planted_root / "results"
    result = run_q6(
        planted_root, out_dir, symbols=["LOWUSDT"], month="2023-06", windows=6, top_n=40,
    )
    rec = result["records"][0]
    assert isinstance(rec["raw_delta"], float)
    assert np.isfinite(rec["raw_delta"])


def test_run_q6_regime_switching_fixture_flags_inflated_raw_alpha_via_delta(tmp_path: Path):
    """Plan-mandated regression test (task-3 plan, "regime-switching fixture
    lands with documented inflated n̂ flagged by the raw-vs-rescaled delta"):
    a symbol with NO genuine self-excitation (alpha=0) but a strongly
    seasonal (time-of-day) baseline rate must, when run through the full Q6
    pipeline end-to-end, show the trap and the fix in the SAME per-symbol
    record. `simulate_seasonal_hawkes_exp` with alpha=0 (see
    `tests/estimators/test_hawkes.py::
    test_regime_switching_poisson_produces_spurious_endogeneity_trap` and
    `tests/signals/test_eventtime.py`'s justification tests for the
    mechanism) is the genuine seasonal-baseline generative model this repo
    settled on -- deliberately amplitude=1.05 (deep trough) so the
    seasonality-driven inflation is large.

    Q6's own `raw_delta` field is exactly the tool built to flag this: a
    positive raw_delta (raw clock-time alpha minus business-time-rescaled
    window-1 alpha) means clock time would have overstated endogeneity
    relative to the business-time-corrected fit. On a planted-zero-alpha
    seasonal symbol, raw_delta must be large and positive, and the final
    business-time-corrected alpha_median must land far below the raw
    clock-time estimate, near the planted true_alpha=0 -- i.e. the pipeline
    both manufactures and then substantially corrects the trap.
    """
    true_mu_bar, true_alpha, true_beta = 0.4, 0.0, 1.5
    # 45 days: alpha=0 leaves beta unidentified/degenerate (see
    # test_hawkes.py's Poisson-refutation docstring), which can push
    # branching_count_variance's window wide via its median_beta fallback;
    # a longer horizon keeps the business-time span >= 20x that window.
    t_end_s = 45 * 24 * 3600.0
    n_bins = 48
    bin_centers = (np.arange(n_bins) + 0.5) / n_bins
    amplitude = 1.05  # deep trough, matches test_eventtime.py's justification case
    shape = amplitude + np.cos(2 * 2 * np.pi * bin_centers)
    shape = shape / shape.mean()

    times_s = simulate_seasonal_hawkes_exp(
        true_mu_bar, true_alpha, true_beta, t_end_s, shape, seed=7
    )
    assert times_s.size > 1000, "need enough events for a stable 6-window panel"

    _write_event_times_fixture(tmp_path, "SEASONUSDT", times_s)

    out_dir = tmp_path / "results"
    result = run_q6(
        tmp_path, out_dir, symbols=["SEASONUSDT"], month="2023-06", windows=6, top_n=40,
    )

    by_symbol = {r["symbol"]: r for r in result["records"]}
    assert "SEASONUSDT" in by_symbol, (
        f"SEASONUSDT unexpectedly failed: {result['failures']}"
    )
    rec = by_symbol["SEASONUSDT"]

    # The trap: naive raw clock-time fitting on a purely seasonal (no
    # self-excitation) process is spuriously inflated well above true_alpha.
    assert rec["raw_alpha_window1"] > true_alpha + 0.3, (
        f"expected seasonality to inflate raw clock-time alpha above "
        f"{true_alpha + 0.3}, got {rec['raw_alpha_window1']}"
    )

    # raw_delta = raw - rescaled_window1 must be large and positive: this is
    # the pipeline's own documented flag for exactly this failure mode.
    assert rec["raw_delta"] > 0.3, (
        f"expected raw_delta to flag the inflated raw alpha (>0.3), got "
        f"{rec['raw_delta']}"
    )

    # The fix: business-time rescaling recovers alpha_median close to the
    # planted true_alpha=0, far below the raw clock-time estimate.
    assert rec["alpha_median"] < rec["raw_alpha_window1"] - 0.3, (
        f"expected business-time correction to land well below the raw "
        f"estimate; alpha_median={rec['alpha_median']}, "
        f"raw_alpha_window1={rec['raw_alpha_window1']}"
    )
    assert rec["alpha_median"] < 0.3, (
        f"expected business-time-corrected alpha_median near true_alpha=0, "
        f"got {rec['alpha_median']}"
    )
