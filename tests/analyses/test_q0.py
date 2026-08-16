from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from microstructure.analyses.q0_aggregation_effect import run_q0
from microstructure.data.catalog import parquet_path
from microstructure.synthetic import markov_signs


def _write_sweep_fixture(tmp_path: Path, symbol: str, period: str, n_events: int, sweep_size: int, seed: int) -> None:
    """One aggressor event per Markov-chain sign, each event printed `sweep_size` times.

    Every print in a sweep shares (ts, is_buyer_maker), so `to_aggressor_events`
    merges them back into `n_events` aggressor events, while a raw read sees
    n_events * sweep_size individual same-signed rows -- the fragmentation this
    analysis is built to expose.
    """
    event_signs = markov_signs(n_events, p_repeat=0.75, seed=seed)
    t0 = datetime(2023, 6, 1, tzinfo=UTC)

    ts_col: list[datetime] = []
    sign_col: list[int] = []
    for i, s in enumerate(event_signs):
        ts_col.extend([t0 + timedelta(milliseconds=3 * i)] * sweep_size)
        sign_col.extend([s] * sweep_size)

    n_rows = len(ts_col)
    df = pl.DataFrame(
        {
            "agg_trade_id": np.arange(n_rows),
            "price": np.full(n_rows, 100.0),
            "qty": np.ones(n_rows),
            "first_trade_id": np.arange(n_rows),
            "last_trade_id": np.arange(n_rows),
            "ts": ts_col,
            "is_buyer_maker": np.array(sign_col) < 0,
        },
        schema_overrides={"ts": pl.Datetime("ms", "UTC")},
    )
    p = parquet_path(tmp_path, symbol, "aggTrades", period)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p)


def test_run_q0_shows_raw_gamma_inflated_relative_to_aggregated(tmp_path: Path):
    n_events = 20_000
    sweep_size = 4
    _write_sweep_fixture(tmp_path, "TESTUSDT", "2023-06", n_events, sweep_size, seed=3)

    out = tmp_path / "results"
    res = run_q0(tmp_path, out, symbols=["TESTUSDT"], periods=["2023-06"])

    assert (out / "q0_aggregation_effect.md").exists()
    assert (out / "q0_aggregation_effect.json").exists()

    cell = res["TESTUSDT_2023-06"]
    assert cell["raw"]["n"] == n_events * sweep_size
    assert cell["aggregated"]["n"] == n_events
    assert cell["prints_per_event"] == sweep_size

    # every raw sweep is 4 identical-signed prints back-to-back -> raw lag-1
    # ACF is inflated well above the aggregated (true Markov) lag-1 ACF.
    assert cell["raw"]["acf1"] > cell["aggregated"]["acf1"]
    # aggregated series recovers the underlying Markov ACF(1) = 2p - 1 = 0.5
    assert abs(cell["aggregated"]["acf1"] - 0.5) < 0.05
    # gamma_inflation = raw gamma - agg gamma should be positive: the raw
    # (fragmented) series decays more steeply on a log-log plot.
    assert cell["gamma_inflation"] > 0


def test_run_q0_multiple_symbols_and_periods_all_present(tmp_path: Path):
    for sym in ("AAAUSDT", "BBBUSDT"):
        for period in ("2023-06", "2023-07"):
            _write_sweep_fixture(tmp_path, sym, period, 5_000, 2, seed=hash((sym, period)) % 1000)

    out = tmp_path / "results"
    res = run_q0(tmp_path, out, symbols=["AAAUSDT", "BBBUSDT"], periods=["2023-06", "2023-07"])

    assert set(res.keys()) == {
        "AAAUSDT_2023-06", "AAAUSDT_2023-07", "BBBUSDT_2023-06", "BBBUSDT_2023-07",
    }
    for cell in res.values():
        assert cell["raw"]["n"] == 10_000
        assert cell["aggregated"]["n"] == 5_000
