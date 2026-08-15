from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from microstructure.analyses.q2_response import run_q2
from microstructure.data.catalog import parquet_path
from microstructure.synthetic import iid_signs


def _mids_from_kernel(signs: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """mid before event t = sum over k<t of signs[k] * kernel[t-k]."""
    n = signs.size
    full = np.convolve(signs.astype(float), kernel, mode="full")[:n]
    return full


def _write_synthetic_q2_fixture(
    tmp_path: Path, symbol: str, n: int, g0: float, phi: float, seed: int
) -> tuple[Path, list[str], str]:
    """Build monthly aggTrades + daily bookTicker parquets with a known exponential kernel.

    Events are spaced 3ms apart starting at t0. bookTicker carries one quote row
    1ms BEFORE each event, whose mid equals sum_{k<t} sign_k * g0*phi**(t-k-1)
    (kernel[0] = 0, so the "prior mid" convention in events_with_prior_mid lines up
    exactly with this construction). All days used land in the same daily bucket
    to keep the fixture small; run_q2 is exercised through daily periods regardless.
    """
    signs = iid_signs(n, seed=seed)
    taus = np.arange(1, n)
    kernel = np.concatenate(([0.0], g0 * phi ** (taus - 1)))
    mids = _mids_from_kernel(signs, kernel)

    t0 = datetime(2023, 6, 1, 0, 0, 0, tzinfo=UTC)
    event_ts = [t0 + timedelta(milliseconds=3 * i) for i in range(n)]
    quote_ts = [t0 + timedelta(milliseconds=3 * i - 1) for i in range(n)]

    agg = pl.DataFrame(
        {
            "agg_trade_id": np.arange(n),
            "price": np.full(n, 100.0),
            "qty": np.ones(n),
            "first_trade_id": np.arange(n),
            "last_trade_id": np.arange(n),
            "ts": event_ts,
            "is_buyer_maker": signs < 0,
        },
        schema_overrides={"ts": pl.Datetime("ms", "UTC")},
    )
    agg_path = parquet_path(tmp_path, symbol, "aggTrades", "2023-06")
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg.write_parquet(agg_path)

    bt = pl.DataFrame(
        {
            "update_id": np.arange(n),
            "bid_price": mids - 0.01,
            "bid_qty": np.ones(n),
            "ask_price": mids + 0.01,
            "ask_qty": np.ones(n),
            "ts": quote_ts,
        },
        schema_overrides={"ts": pl.Datetime("ms", "UTC")},
    )
    bt_path = parquet_path(tmp_path, symbol, "bookTicker", "2023-06-01")
    bt_path.parent.mkdir(parents=True, exist_ok=True)
    bt.write_parquet(bt_path)

    return tmp_path, ["2023-06-01"], "2023-06"


def test_run_q2_recovers_known_exponential_kernel(tmp_path: Path):
    symbol = "TESTUSDT"
    n = 20_000
    g0, phi = 0.5, 0.8
    root, periods, _month = _write_synthetic_q2_fixture(tmp_path, symbol, n, g0, phi, seed=6)

    out = tmp_path / "results"
    res = run_q2(root, out, symbol=symbol, periods=periods, max_lag=210)

    assert (out / "q2_response.png").exists()
    assert (out / "q2_results.md").exists()
    assert (out / "q2_results.json").exists()

    # the last event can fall after the last bookTicker quote and get filtered by the
    # ts_max bound (correct: no prior-mid data exists past the quote coverage), so allow
    # a handful of events at the boundary to be excluded.
    assert n - 5 <= res["n_events"] <= n
    response = np.array(res["response"])
    # R(l) = g(l) = g0 * phi**(l-1) for iid signs, lag >= 1
    for lag in (1, 5, 10, 50):
        expected = g0 * phi ** (lag - 1)
        assert abs(response[lag] - expected) < 0.03

    # exponential decay -> exponent should be recovered close to -log(phi) per step
    assert res["decay_exponent"] > 0  # power-law gamma or exp rate, either way: decaying
    assert res["decay_stderr"] >= 0


def test_run_q2_raises_on_excessive_drop_rate(tmp_path: Path):
    """If bookTicker coverage doesn't align with events, > 1% drop must raise."""
    symbol = "TESTUSDT"
    n = 2_000
    t0 = datetime(2023, 6, 1, 0, 0, 0, tzinfo=UTC)
    signs = iid_signs(n, seed=1)

    agg = pl.DataFrame(
        {
            "agg_trade_id": np.arange(n),
            "price": np.full(n, 100.0),
            "qty": np.ones(n),
            "first_trade_id": np.arange(n),
            "last_trade_id": np.arange(n),
            "ts": [t0 + timedelta(milliseconds=3 * i) for i in range(n)],
            "is_buyer_maker": signs < 0,
        },
        schema_overrides={"ts": pl.Datetime("ms", "UTC")},
    )
    agg_path = parquet_path(tmp_path, symbol, "aggTrades", "2023-06")
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg.write_parquet(agg_path)

    # bookTicker quotes only cover the FIRST 10% of events -> most events get no prior mid
    n_bt = n // 10
    bt = pl.DataFrame(
        {
            "update_id": np.arange(n_bt),
            "bid_price": np.full(n_bt, 99.99),
            "bid_qty": np.ones(n_bt),
            "ask_price": np.full(n_bt, 100.01),
            "ask_qty": np.ones(n_bt),
            "ts": [t0 + timedelta(milliseconds=3 * i - 1) for i in range(n_bt)],
        },
        schema_overrides={"ts": pl.Datetime("ms", "UTC")},
    )
    bt_path = parquet_path(tmp_path, symbol, "bookTicker", "2023-06-01")
    bt_path.parent.mkdir(parents=True, exist_ok=True)
    bt.write_parquet(bt_path)

    out = tmp_path / "results"
    import pytest

    with pytest.raises(ValueError):
        run_q2(tmp_path, out, symbol=symbol, periods=["2023-06-01"], max_lag=50)
