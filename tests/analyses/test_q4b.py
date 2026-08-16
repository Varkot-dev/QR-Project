import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import polars as pl
import pytest

from microstructure.analyses.q4b_tick_confound import (
    _assemble_records,
    fit_bivariate,
    fit_univariate,
    mean_trade_price,
    run_q4b,
    tick_sizes_from_exchange_info,
)
from microstructure.data.catalog import parquet_path

# --------------------------------------------------------------------------
# Regression math on synthetic inputs (offline, no network)
# --------------------------------------------------------------------------


def test_fit_univariate_recovers_known_linear_relationship():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 5, 200)
    true_slope, true_intercept = 0.111, -0.25
    y = true_intercept + true_slope * x + rng.normal(0, 1e-6, x.size)

    reg = fit_univariate(x, y, "x")

    coefs = dict(zip(reg.names, reg.coefs, strict=True))
    assert coefs["x"] == pytest.approx(true_slope, abs=1e-4)
    assert coefs["intercept"] == pytest.approx(true_intercept, abs=1e-4)
    assert reg.r2 > 0.999
    assert reg.n == 200


def test_fit_univariate_exact_fit_has_zero_residual_r2_one():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x + 1.0  # exact line, no noise

    reg = fit_univariate(x, y, "x")

    coefs = dict(zip(reg.names, reg.coefs, strict=True))
    assert coefs["x"] == pytest.approx(2.0, abs=1e-8)
    assert coefs["intercept"] == pytest.approx(1.0, abs=1e-8)
    assert reg.r2 == pytest.approx(1.0, abs=1e-8)


def test_fit_bivariate_recovers_known_coefficients_with_correlated_regressors():
    """Two correlated regressors: lstsq must still recover each true coefficient."""
    rng = np.random.default_rng(1)
    n = 500
    x1 = rng.normal(0, 1, n)
    # x2 correlated with x1 but with independent variance so the design
    # matrix stays full rank and both coefficients are identifiable.
    x2 = 0.6 * x1 + rng.normal(0, 1, n)
    true_b0, true_b1, true_b2 = -0.1, 0.3, -0.05
    y = true_b0 + true_b1 * x1 + true_b2 * x2 + rng.normal(0, 1e-6, n)

    reg = fit_bivariate(x1, x2, y, "x1", "x2")

    coefs = dict(zip(reg.names, reg.coefs, strict=True))
    assert coefs["intercept"] == pytest.approx(true_b0, abs=1e-3)
    assert coefs["x1"] == pytest.approx(true_b1, abs=1e-3)
    assert coefs["x2"] == pytest.approx(true_b2, abs=1e-3)
    assert reg.r2 > 0.999


def test_fit_bivariate_isolates_the_true_driver_when_other_is_pure_noise():
    """If y depends only on x1, x2's coefficient should be near zero and x1's near the truth,
    even though x1 and x2 are correlated with each other (the discriminating-regression case
    this module exists for)."""
    rng = np.random.default_rng(2)
    n = 1000
    x1 = rng.normal(0, 1, n)
    x2 = 0.9 * x1 + rng.normal(0, 0.3, n)  # collinear with x1
    true_slope = 0.5
    y = true_slope * x1 + rng.normal(0, 0.01, n)  # y depends only on x1

    reg = fit_bivariate(x1, x2, y, "x1", "x2")
    coefs = dict(zip(reg.names, reg.coefs, strict=True))

    assert coefs["x1"] == pytest.approx(true_slope, abs=0.05)
    assert abs(coefs["x2"]) < 0.05


def test_ols_raises_on_insufficient_observations():
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="observations"):
        # 2 points, 3 parameters (intercept + x1 + x2) via fit_bivariate
        fit_bivariate(x, x, y, "x1", "x2")


# --------------------------------------------------------------------------
# tick_sizes_from_exchange_info: parsing the Binance PRICE_FILTER shape
# --------------------------------------------------------------------------


def test_tick_sizes_from_exchange_info_extracts_price_filter():
    raw = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "filters": [
                    {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                    {"filterType": "PRICE_FILTER", "tickSize": "0.10", "minPrice": "1", "maxPrice": "1000000"},
                ],
            },
            {
                "symbol": "NOFILTERUSDT",
                "filters": [{"filterType": "LOT_SIZE", "stepSize": "1"}],
            },
        ]
    }
    ticks = tick_sizes_from_exchange_info(raw)
    assert ticks == {"BTCUSDT": 0.10}
    assert "NOFILTERUSDT" not in ticks


def test_tick_sizes_from_exchange_info_handles_empty_symbols():
    assert tick_sizes_from_exchange_info({"symbols": []}) == {}
    assert tick_sizes_from_exchange_info({}) == {}


# --------------------------------------------------------------------------
# mean_trade_price: lazy scan over synthetic parquet
# --------------------------------------------------------------------------


def _write_synthetic_agg_trades(root: Path, symbol: str, prices: np.ndarray, period: str = "2023-06") -> None:
    n = prices.size
    t0 = datetime(2023, 6, 1, tzinfo=UTC)
    df = pl.DataFrame(
        {
            "agg_trade_id": np.arange(n),
            "price": prices,
            "qty": np.ones(n),
            "first_trade_id": np.arange(n),
            "last_trade_id": np.arange(n),
            "ts": [t0 + timedelta(milliseconds=3 * i) for i in range(n)],
            "is_buyer_maker": np.zeros(n, dtype=bool),
        },
        schema_overrides={"ts": pl.Datetime("ms", "UTC")},
    )
    p = parquet_path(root, symbol, "aggTrades", period)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p)


def test_mean_trade_price_returns_none_for_missing_parquet(tmp_path: Path):
    assert mean_trade_price(tmp_path, "MISSINGUSDT") is None


def test_mean_trade_price_computes_correct_mean(tmp_path: Path):
    prices = np.array([100.0, 200.0, 300.0])
    _write_synthetic_agg_trades(tmp_path, "AAAUSDT", prices)

    result = mean_trade_price(tmp_path, "AAAUSDT")

    assert result == pytest.approx(200.0)


# --------------------------------------------------------------------------
# _assemble_records: join Q4 symbols with tick sizes and mean prices
# --------------------------------------------------------------------------


def test_assemble_records_computes_rel_tick_and_skips_missing_data(tmp_path: Path):
    _write_synthetic_agg_trades(tmp_path, "AAAUSDT", np.full(10, 100.0))
    # BBBUSDT has no parquet on disk -> should be skipped for missing price.
    q4_symbols = [
        {"symbol": "AAAUSDT", "n_events": 50_000, "p_flip": 0.55},
        {"symbol": "BBBUSDT", "n_events": 60_000, "p_flip": 0.45},
        {"symbol": "CCCUSDT", "n_events": 70_000, "p_flip": 0.50},  # no tick size below
    ]
    tick_sizes = {"AAAUSDT": 0.01, "BBBUSDT": 0.01}  # CCCUSDT missing

    records, skipped = _assemble_records(tmp_path, q4_symbols, tick_sizes, "2023-06")

    assert len(records) == 1
    assert records[0]["symbol"] == "AAAUSDT"
    assert records[0]["rel_tick"] == pytest.approx(0.01 / 100.0)
    assert records[0]["mean_price"] == pytest.approx(100.0)

    skipped_symbols = {s["symbol"]: s["reason"] for s in skipped}
    assert skipped_symbols["BBBUSDT"] == "missing parquet or non-positive mean price"
    assert skipped_symbols["CCCUSDT"] == "no tickSize in exchangeInfo snapshot"


# --------------------------------------------------------------------------
# run_q4b end-to-end: network call mocked via httpx.MockTransport, all else
# synthetic and offline.
# --------------------------------------------------------------------------


def _fake_exchange_info_response(symbols_and_ticks: dict[str, float]) -> dict:
    return {
        "symbols": [
            {
                "symbol": sym,
                "filters": [{"filterType": "PRICE_FILTER", "tickSize": f"{tick:.8f}"}],
            }
            for sym, tick in symbols_and_ticks.items()
        ]
    }


def _mock_client(payload: dict) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


def test_run_q4b_end_to_end_with_mocked_network_and_synthetic_data(tmp_path: Path):
    root = tmp_path / "data"
    out_dir = tmp_path / "results"
    root.mkdir()

    # Build a synthetic q4_cross_section.json with a real p_flip-vs-activity
    # relationship baked in, plus tick sizes engineered to be UNcorrelated
    # with activity in a way that lets the discriminating regression show
    # activity surviving. Prices are set so rel_tick has real variance.
    rng = np.random.default_rng(42)
    n_symbols = 30
    symbols_data = []
    tick_sizes = {}
    for i in range(n_symbols):
        sym = f"SYM{i:02d}USDT"
        n_events = 100_000 * (i + 1)
        # p_flip driven by log10(n_events) with a small positive slope.
        p_flip = 0.45 + 0.02 * np.log10(n_events) + rng.normal(0, 0.005)
        p_flip = float(np.clip(p_flip, 0.05, 0.95))
        symbols_data.append({"symbol": sym, "n_events": n_events, "p_flip": p_flip})

        price = float(rng.uniform(1.0, 1000.0))
        tick = float(rng.uniform(0.0001, 0.01)) * price  # tick size roughly independent driver
        tick_sizes[sym] = tick
        prices = np.full(1000, price)
        _write_synthetic_agg_trades(root, sym, prices)

    q4_json = {"symbols": symbols_data}
    q4_json_path = tmp_path / "q4_cross_section.json"
    q4_json_path.write_text(json.dumps(q4_json))

    payload = _fake_exchange_info_response(tick_sizes)
    client = _mock_client(payload)

    result = run_q4b(root, out_dir, q4_json_path=q4_json_path, period="2023-06", client=client)

    assert result["n_q4_symbols"] == n_symbols
    assert result["n_usable"] == n_symbols
    assert result["n_skipped"] == 0

    regs = result["regressions"]
    assert regs["note"] is None
    assert regs["reg_activity"] is not None
    assert regs["reg_tick"] is not None
    assert regs["reg_joint"] is not None
    assert "log10_n_events" in regs["reg_joint"]["coefficients"]
    assert "log10_rel_tick" in regs["reg_joint"]["coefficients"]
    assert regs["corr_log_n_log_rel_tick"] is not None

    # Output files exist.
    assert (out_dir / "q4b_tick_confound.json").exists()
    assert (out_dir / "q4b_tick_confound.md").exists()
    assert (out_dir / "q4b_flip_vs_rel_tick.png").exists()
    assert (out_dir / "exchangeinfo_snapshot.json").exists()

    cached = json.loads((out_dir / "exchangeinfo_snapshot.json").read_text())
    assert cached == payload

    md_text = (out_dir / "q4b_tick_confound.md").read_text()
    assert "Verdict" in md_text
    assert "current, not June-2023" in md_text


def test_run_q4b_handles_symbols_missing_from_exchange_info(tmp_path: Path):
    root = tmp_path / "data"
    out_dir = tmp_path / "results"
    root.mkdir()

    _write_synthetic_agg_trades(root, "AAAUSDT", np.full(500, 50.0))
    _write_synthetic_agg_trades(root, "BBBUSDT", np.full(500, 60.0))

    q4_json = {
        "symbols": [
            {"symbol": "AAAUSDT", "n_events": 100_000, "p_flip": 0.5},
            {"symbol": "BBBUSDT", "n_events": 200_000, "p_flip": 0.52},
            {"symbol": "DELISTEDUSDT", "n_events": 300_000, "p_flip": 0.48},
        ]
    }
    q4_json_path = tmp_path / "q4_cross_section.json"
    q4_json_path.write_text(json.dumps(q4_json))

    # exchangeInfo only has AAAUSDT and BBBUSDT — DELISTEDUSDT is absent.
    payload = _fake_exchange_info_response({"AAAUSDT": 0.01, "BBBUSDT": 0.01})
    client = _mock_client(payload)

    result = run_q4b(root, out_dir, q4_json_path=q4_json_path, period="2023-06", client=client)

    assert result["n_usable"] == 2
    assert result["n_skipped"] == 1
    assert result["skipped"][0]["symbol"] == "DELISTEDUSDT"
    assert result["skipped"][0]["reason"] == "no tickSize in exchangeInfo snapshot"
    # Below the n>=4 threshold for regressions -> reported as not estimable.
    assert result["regressions"]["note"] is not None


def test_run_q4b_too_few_usable_symbols_reports_note_not_crash(tmp_path: Path):
    root = tmp_path / "data"
    out_dir = tmp_path / "results"
    root.mkdir()

    _write_synthetic_agg_trades(root, "AAAUSDT", np.full(10, 50.0))

    q4_json = {"symbols": [{"symbol": "AAAUSDT", "n_events": 100_000, "p_flip": 0.5}]}
    q4_json_path = tmp_path / "q4_cross_section.json"
    q4_json_path.write_text(json.dumps(q4_json))

    payload = _fake_exchange_info_response({"AAAUSDT": 0.01})
    client = _mock_client(payload)

    result = run_q4b(root, out_dir, q4_json_path=q4_json_path, period="2023-06", client=client)

    assert result["n_usable"] == 1
    assert result["regressions"]["reg_activity"] is None
    assert "fewer than 4" in result["regressions"]["note"]
    assert (out_dir / "q4b_tick_confound.md").exists()
    assert (out_dir / "q4b_tick_confound.json").exists()
