from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from microstructure.analyses.q4_cross_section import run_q4
from microstructure.data.catalog import parquet_path
from microstructure.synthetic import markov_signs


def _write_synthetic_agg_trades(root: Path, symbol: str, signs: np.ndarray, period: str = "2023-06") -> None:
    n = signs.size
    t0 = datetime(2023, 6, 1, tzinfo=UTC)
    qty = np.ones(n)
    df = pl.DataFrame(
        {
            "agg_trade_id": np.arange(n),
            "price": np.full(n, 100.0),
            "qty": qty,
            "first_trade_id": np.arange(n),
            "last_trade_id": np.arange(n),
            "ts": [t0 + timedelta(milliseconds=3 * i) for i in range(n)],
            "is_buyer_maker": signs < 0,
        },
        schema_overrides={"ts": pl.Datetime("ms", "UTC")},
    )
    p = parquet_path(root, symbol, "aggTrades", period)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p)


def test_run_q4_recovers_distinct_acf1_skips_and_reports_failures(tmp_path: Path):
    period = "2023-06"
    min_events = 20_000
    max_lag = 200

    # Three symbols with distinct known lag-1 ACF via markov_signs:
    # theoretical ACF(1) = 2*p_repeat - 1. Distinct n_events per symbol so
    # the cross-sectional regression's log10(n_events) has nonzero variance.
    p_repeats = {"AAAUSDT": 0.6, "BBBUSDT": 0.75, "CCCUSDT": 0.9}
    expected_acf1 = {sym: 2 * p - 1 for sym, p in p_repeats.items()}
    n_events_by_symbol = {"AAAUSDT": 50_000, "BBBUSDT": 80_000, "CCCUSDT": 120_000}
    for i, (sym, p_repeat) in enumerate(p_repeats.items()):
        signs = markov_signs(n_events_by_symbol[sym], p_repeat=p_repeat, seed=100 + i)
        _write_synthetic_agg_trades(tmp_path, sym, signs, period)

    # A symbol with too few events -> must be skipped, not failed.
    small_signs = markov_signs(500, p_repeat=0.5, seed=200)
    _write_synthetic_agg_trades(tmp_path, "SMALLUSDT", small_signs, period)

    # A symbol deliberately absent from disk -> must land in failures.
    missing_symbol = "MISSINGUSDT"

    symbols = [*p_repeats.keys(), "SMALLUSDT", missing_symbol]
    out_dir = tmp_path / "results"

    result = run_q4(
        tmp_path, out_dir, symbols=symbols, period=period,
        min_events=min_events, max_lag=max_lag,
    )

    # --- recovered ACF(1) within tolerance for the three good symbols ---
    by_symbol = {r["symbol"]: r for r in result["symbols"]}
    for sym, expected in expected_acf1.items():
        assert sym in by_symbol, f"{sym} missing from successful results"
        got = by_symbol[sym]["acf1"]
        assert abs(got - expected) < 0.03, f"{sym}: acf1={got}, expected~{expected}"
        assert by_symbol[sym]["n_events"] == n_events_by_symbol[sym]
        assert "gamma" in by_symbol[sym]
        assert "stderr" in by_symbol[sym]
        assert "p_flip" in by_symbol[sym]
        assert "zigzag_amplitude" in by_symbol[sym]
        assert "total_qty" in by_symbol[sym]

    # --- skip record for too-few-events symbol ---
    skips = {s["symbol"]: s for s in result["skips"]}
    assert "SMALLUSDT" in skips
    assert skips["SMALLUSDT"]["reason"] == "below min_events"
    assert "SMALLUSDT" not in by_symbol

    # --- failure record for the missing-parquet symbol ---
    failures = {f["symbol"]: f for f in result["failures"]}
    assert missing_symbol in failures
    assert failures[missing_symbol]["reason"]
    assert missing_symbol not in by_symbol

    # --- cross-sectional regressions present on the successful set ---
    assert "regressions" in result
    assert "gamma_vs_activity" in result["regressions"]
    assert "p_flip_vs_activity" in result["regressions"]
    for reg_name in ("gamma_vs_activity", "p_flip_vs_activity"):
        reg = result["regressions"][reg_name]
        assert "slope" in reg
        assert "intercept" in reg
        assert "r2" in reg

    # --- run metadata ---
    assert result["period"] == period
    assert result["min_events"] == min_events

    # --- output files exist ---
    assert (out_dir / "q4_cross_section.json").exists()
    assert (out_dir / "q4_cross_section.md").exists()
    assert (out_dir / "q4_gamma_vs_activity.png").exists()
    assert (out_dir / "q4_flip_vs_activity.png").exists()

    # --- parquet output round-trips with the right height and columns ---
    parquet_out = out_dir / "q4_cross_section.parquet"
    assert parquet_out.exists()
    pq = pl.read_parquet(parquet_out)
    assert pq.height == len(result["symbols"])
    assert set(pq.columns) == {
        "symbol", "n_events", "gamma", "gamma_stderr", "acf1", "p_flip",
        "zigzag_amplitude", "total_qty",
    }
    pq_by_symbol = {row["symbol"]: row for row in pq.to_dicts()}
    for sym, expected in expected_acf1.items():
        assert sym in pq_by_symbol
        assert abs(pq_by_symbol[sym]["acf1"] - expected) < 0.03
        assert pq_by_symbol[sym]["n_events"] == n_events_by_symbol[sym]
        assert pq_by_symbol[sym]["gamma_stderr"] == by_symbol[sym]["stderr"]


def test_run_q4_never_aborts_on_all_failures(tmp_path: Path):
    """If every symbol fails/skips, run_q4 must still return cleanly (no raise)."""
    out_dir = tmp_path / "results"
    result = run_q4(
        tmp_path, out_dir, symbols=["GHOSTUSDT"], period="2023-06",
        min_events=1_000_000, max_lag=1000,
    )
    assert result["symbols"] == []
    assert len(result["failures"]) == 1
    assert result["failures"][0]["symbol"] == "GHOSTUSDT"
    assert (out_dir / "q4_cross_section.md").exists()
    assert (out_dir / "q4_cross_section.json").exists()

    parquet_out = out_dir / "q4_cross_section.parquet"
    assert parquet_out.exists()
    pq = pl.read_parquet(parquet_out)
    assert pq.height == 0
    assert set(pq.columns) == {
        "symbol", "n_events", "gamma", "gamma_stderr", "acf1", "p_flip",
        "zigzag_amplitude", "total_qty",
    }
