"""Tests for Q7: execution-cost schedule comparison (Phase 3, Task 4).

Contract under test (see task-4-brief.md):
1. Symbol selection: 6 symbols at ranks 1,4,7,10,13,16 of the panel's Q5
   n_events ordering (spanning the panel's activity range), chosen
   programmatically from the kernels json — never hardcoded.
2. `run_q7` runs each symbol x day x side x parent_qty_events combination
   through TWAP / front-loaded / reactive, with reactive's (lookback,
   pause_threshold) calibrated ONLY on days 1-3 and evaluated ONLY on days
   4-7. The chosen params are exactly the calibration-window argmax over
   the grid {50,200}x{0.2,0.4} -- verified independently in this test by
   recomputing the argmax from the same synthetic fixtures and asserting
   equality with what's recorded in the output json. This is the
   no-evaluation-leakage check the brief requires.
3. Output files (.md, .json, .png) exist; the json carries both a
   "calibration" and an "evaluation" block, never conflating the two.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from microstructure.analyses.q7_execution import (
    REACTIVE_GRID,
    _pick_panel_symbols,
    calibrate_reactive_params,
    run_q7,
)
from microstructure.data.catalog import parquet_path

DAYS = [f"2023-06-0{d}" for d in range(1, 8)]
CALIB_DAYS = DAYS[:3]
EVAL_DAYS = DAYS[3:]


# --------------------------------------------------------------------------
# _pick_panel_symbols: pure, rank-based selection
# --------------------------------------------------------------------------


def test_pick_panel_symbols_selects_ranks_1_4_7_10_13_16():
    # 16 fake records with distinct n_events, symbols named by descending rank.
    records = [{"symbol": f"SYM{i:02d}", "n_events": 16 - i} for i in range(16)]
    kernels = {"records": records}
    picked = _pick_panel_symbols(kernels, n_pick=6)
    # sorted by n_events descending: SYM00 (rank1) .. SYM15 (rank16)
    expected_ranks = [1, 4, 7, 10, 13, 16]
    expected = [f"SYM{r - 1:02d}" for r in expected_ranks]
    assert picked == expected


def test_pick_panel_symbols_handles_fewer_than_16_gracefully():
    records = [{"symbol": f"SYM{i:02d}", "n_events": 10 - i} for i in range(8)]
    kernels = {"records": records}
    picked = _pick_panel_symbols(kernels, n_pick=6)
    assert len(picked) == 6
    assert len(set(picked)) == 6


# --------------------------------------------------------------------------
# calibrate_reactive_params: grid search, no leakage
# --------------------------------------------------------------------------


def test_calibrate_reactive_params_picks_grid_argmax():
    """Feed a hand-scored advantage table and confirm the argmax is returned."""

    def fake_scorer(lookback: int, pause_threshold: float) -> float:
        # Planted maximum at (200, 0.4).
        table = {
            (50, 0.2): 0.01,
            (50, 0.4): 0.02,
            (200, 0.2): 0.015,
            (200, 0.4): 0.05,
        }
        return table[(lookback, pause_threshold)]

    lookback, pause_threshold, scores = calibrate_reactive_params(fake_scorer)
    assert (lookback, pause_threshold) == (200, 0.4)
    assert scores[(200, 0.4)] == pytest.approx(0.05)
    assert set(scores.keys()) == set(REACTIVE_GRID)


# --------------------------------------------------------------------------
# run_q7: full synthetic pipeline through parquet fixtures
# --------------------------------------------------------------------------


def _write_symbol_week(root: Path, symbol: str, seed: int, n_per_day: int = 3000) -> None:
    """One week (2023-06-01..07) of synthetic aggTrades + bookTicker.

    Flat-ish random-walk mids, i.i.d. random signs -- no planted structure,
    just enough real data flow for the full pipeline (replay -> schedules
    -> shortfall) to execute end-to-end without error.
    """
    rng = np.random.default_rng(seed)
    month = "2023-06"

    all_ts: list[datetime] = []
    all_signs: list[int] = []
    all_qty: list[float] = []
    all_price: list[float] = []
    for day_idx, day in enumerate(DAYS):
        day_start = datetime(2023, 6, 1 + day_idx, 0, 0, 0, tzinfo=UTC)
        signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_per_day)
        qtys = rng.uniform(0.5, 2.0, size=n_per_day)
        price_walk = 100.0 + np.cumsum(rng.normal(0, 0.001, size=n_per_day))
        for i in range(n_per_day):
            all_ts.append(day_start + timedelta(milliseconds=5 * i))
            all_signs.append(int(signs[i]))
            all_qty.append(float(qtys[i]))
            all_price.append(float(price_walk[i]))

    n = len(all_ts)
    agg = pl.DataFrame(
        {
            "agg_trade_id": np.arange(n),
            "price": all_price,
            "qty": all_qty,
            "first_trade_id": np.arange(n),
            "last_trade_id": np.arange(n),
            "ts": all_ts,
            "is_buyer_maker": [s < 0 for s in all_signs],
        },
        schema_overrides={"ts": pl.Datetime("ms", "UTC")},
    )
    agg_path = parquet_path(root, symbol, "aggTrades", month)
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg.write_parquet(agg_path)

    for day_idx, day in enumerate(DAYS):
        day_start = datetime(2023, 6, 1 + day_idx, 0, 0, 0, tzinfo=UTC)
        quote_ts = [day_start + timedelta(milliseconds=5 * i - 1) for i in range(n_per_day)]
        day_prices = all_price[day_idx * n_per_day : (day_idx + 1) * n_per_day]
        bt = pl.DataFrame(
            {
                "update_id": np.arange(n_per_day),
                "bid_price": [p - 0.02 for p in day_prices],
                "bid_qty": np.ones(n_per_day),
                "ask_price": [p + 0.02 for p in day_prices],
                "ask_qty": np.ones(n_per_day),
                "ts": quote_ts,
            },
            schema_overrides={"ts": pl.Datetime("ms", "UTC")},
        )
        bt_path = parquet_path(root, symbol, "bookTicker", day)
        bt_path.parent.mkdir(parents=True, exist_ok=True)
        bt.write_parquet(bt_path)


def _fake_kernels_json(tmp_path: Path, symbols: list[str]) -> Path:
    """Minimal Q5-shaped kernels json: G rises then decays (like the real panel)."""
    records = []
    for i, sym in enumerate(symbols):
        lags = np.arange(100)
        peak_lag = 5
        g = np.where(
            lags <= peak_lag,
            0.01 * lags,
            0.01 * peak_lag * np.exp(-(lags - peak_lag) / 10.0),
        )
        g[0] = 0.0
        records.append(
            {
                "symbol": sym,
                "n_events": 1_000_000 - i * 10_000,
                "G": g.tolist(),
            }
        )
    path = tmp_path / "fake_kernels.json"
    path.write_text(json.dumps({"records": records}))
    return path


def test_run_q7_end_to_end_synthetic(tmp_path: Path):
    symbols = [f"SYM{i:02d}" for i in range(6)]
    for i, sym in enumerate(symbols):
        _write_symbol_week(tmp_path, sym, seed=i, n_per_day=3000)

    kernels_path = _fake_kernels_json(tmp_path, symbols)
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("\n".join(symbols))

    out_dir = tmp_path / "results"
    result = run_q7(
        root=tmp_path,
        out_dir=out_dir,
        symbols_file=symbols_file,
        kernels_path=kernels_path,
        month="2023-06",
        days=DAYS,
        horizon_events=500,
        n_children=10,
        parent_qty_events_list=[2.0, 10.0],
    )

    # --- calibration / evaluation split present and distinct ---
    assert "calibration" in result
    assert "evaluation" in result
    assert result["calibration"]["days"] == CALIB_DAYS
    assert result["evaluation"]["days"] == EVAL_DAYS
    assert "lookback" in result["calibration"]["chosen_params"]
    assert "pause_threshold" in result["calibration"]["chosen_params"]

    # --- no-leakage: recompute the calibration argmax independently from
    # the same on-disk fixtures and assert it matches what's in the json ---
    from microstructure.analyses.q7_execution import _build_calibration_scorer

    scorer = _build_calibration_scorer(
        root=tmp_path, symbols=symbols, kernels_path=kernels_path,
        month="2023-06", calib_days=CALIB_DAYS,
        horizon_events=500, n_children=10,
        parent_qty_events_list=[2.0, 10.0],
    )
    independent_lookback, independent_threshold, _ = calibrate_reactive_params(scorer)
    assert result["calibration"]["chosen_params"]["lookback"] == independent_lookback
    assert result["calibration"]["chosen_params"]["pause_threshold"] == independent_threshold

    # --- evaluation results carry per-schedule mean/sd shortfall ---
    eval_summary = result["evaluation"]["summary"]
    for schedule_name in ("twap", "frontloaded", "reactive"):
        assert schedule_name in eval_summary
        assert "mean_shortfall" in eval_summary[schedule_name]
        assert "sd_shortfall" in eval_summary[schedule_name]

    # --- per-symbol table present ---
    assert len(result["evaluation"]["per_symbol"]) == 6

    # --- output files exist ---
    assert (out_dir / "q7_execution.json").exists()
    assert (out_dir / "q7_execution.md").exists()
    assert (out_dir / "q7_execution.png").exists()

    md_text = (out_dir / "q7_execution.md").read_text()
    assert "NO-TRADING-CLAIM" in md_text or "no trading claim" in md_text.lower()
    assert "cost model" in md_text.lower()


def test_run_q7_symbol_failure_does_not_abort_run(tmp_path: Path):
    symbols = [f"SYM{i:02d}" for i in range(6)]
    # Only build data for 5 of the 6 symbols; the 6th has no parquet at all.
    for i, sym in enumerate(symbols[:5]):
        _write_symbol_week(tmp_path, sym, seed=i, n_per_day=1500)

    kernels_path = _fake_kernels_json(tmp_path, symbols)
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("\n".join(symbols))

    out_dir = tmp_path / "results"
    result = run_q7(
        root=tmp_path,
        out_dir=out_dir,
        symbols_file=symbols_file,
        kernels_path=kernels_path,
        month="2023-06",
        days=DAYS,
        horizon_events=300,
        n_children=10,
        parent_qty_events_list=[2.0, 10.0],
    )
    assert len(result["failures"]) >= 1
    failed_symbols = {f["symbol"] for f in result["failures"]}
    assert "SYM05" in failed_symbols
    assert (out_dir / "q7_execution.json").exists()
