from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from microstructure.analyses.q1_orderflow_memory import run_q1
from microstructure.data.catalog import parquet_path
from microstructure.synthetic import markov_signs


def test_run_q1_on_synthetic_month_recovers_markov_memory(tmp_path: Path):
    signs = markov_signs(50_000, p_repeat=0.75, seed=9)
    t0 = datetime(2023, 6, 1, tzinfo=UTC)
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
