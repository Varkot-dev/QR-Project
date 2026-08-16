"""Tests for Q5: kernel panel analysis (Phase 2, Task 3).

Contract under test (see task-3-brief.md):
1. Two "good" synthetic symbols, each built from `fractional_signs(d=0.35)`
   convolved with a planted power-law kernel G0(l) = l^(-0.35), so both the
   kernel exponent beta and the sign-memory exponent gamma_week are close to
   their theoretical values (beta=0.35, gamma=1-2*d=0.30) and the critical
   balance relation beta = (1-gamma)/2 holds BY CONSTRUCTION (0.35 =
   (1-0.30)/2 = 0.35 exactly, in the infinite-sample limit). n=500,000 events
   per symbol is used (not the brief's illustrative "~60_000") because at
   smaller n the fit_power_law(sign_acf(signs, 1000), 10, 500) estimate of
   gamma_week is itself too noisy/biased (see probe measurements in the
   task-3 report) for the balance verdict to land "consistent" reliably;
   500,000 keeps runtime in the ~1-2s range via FFT convolution while giving
   both exponent estimates enough samples to land near their planted values.
2. One symbol with no parquet on disk at all -> must land in `failures`,
   never abort the run.
3. Output files (.json, .md, .png) must exist and the json's per-symbol
   records must carry all the fields the brief specifies.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from microstructure.analyses.q5_kernel_panel import run_q5
from microstructure.data.catalog import parquet_path
from microstructure.synthetic import fractional_signs

MAX_LAG = 100
N_EVENTS = 500_000
PLANTED_BETA = 0.35
PLANTED_D = 0.35
PLANTED_GAMMA = 1 - 2 * PLANTED_D  # = 0.30, theoretical sign-ACF exponent


def _fft_convolve_full(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Linear ("full") convolution via zero-padded FFT (mirrors synthetic.py)."""
    out_len = len(a) + len(b) - 1
    size = 1 << (out_len - 1).bit_length()
    fa = np.fft.rfft(a, n=size)
    fb = np.fft.rfft(b, n=size)
    return np.fft.irfft(fa * fb, n=size)[:out_len]


def _mids_from_kernel(signs: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """mid strictly before event t = sum_{k<t} signs[k] * kernel[t-k]."""
    n = signs.size
    return _fft_convolve_full(signs.astype(float), kernel)[:n]


def _write_synthetic_kernel_fixture(
    root: Path, symbol: str, n: int, seed: int, month: str = "2023-06"
) -> None:
    """Monthly aggTrades + daily bookTicker parquets with a planted G0(l)=l^-0.35 kernel.

    Signs are fractional_signs(d=0.35) (long-memory, gamma_week ~ 0.30 in
    theory); mids are built by convolving signs with G0(l)=l^(-0.35)
    directly (Task 1's pattern), so the deconvolved kernel exponent recovers
    ~0.35 and the balance relation beta = (1-gamma)/2 holds by construction.
    Events are spaced 3ms apart starting at day 1 00:00 UTC so all events and
    the one week of bookTicker periods fall inside 2023-06-01..07. bookTicker
    carries one quote row 1ms before each event (events_with_prior_mid's
    'strictly before' convention).
    """
    signs = fractional_signs(n, d=PLANTED_D, seed=seed)

    n_lags = 2000
    ell = np.arange(1, n_lags + 1, dtype=float)
    G0 = ell**-PLANTED_BETA
    mids = _mids_from_kernel(signs, np.concatenate(([0.0], G0)))

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
    agg_path = parquet_path(root, symbol, "aggTrades", month)
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg.write_parquet(agg_path)

    bt_schema = {
        "update_id": pl.Int64,
        "bid_price": pl.Float64,
        "bid_qty": pl.Float64,
        "ask_price": pl.Float64,
        "ask_qty": pl.Float64,
        "ts": pl.Datetime("ms", "UTC"),
    }
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
    bt_path = parquet_path(root, symbol, "bookTicker", "2023-06-01")
    bt_path.parent.mkdir(parents=True, exist_ok=True)
    bt.write_parquet(bt_path)

    # All planted events/quotes land on day 1 (n*3ms << 1 day); the remaining
    # days of the requested week still need a (schema-correct, empty)
    # bookTicker parquet on disk since run_q5 requests the full daily range.
    empty_bt = pl.DataFrame(schema=bt_schema)
    for day in ("2023-06-02", "2023-06-03", "2023-06-04", "2023-06-05", "2023-06-06", "2023-06-07"):
        p = parquet_path(root, symbol, "bookTicker", day)
        p.parent.mkdir(parents=True, exist_ok=True)
        empty_bt.write_parquet(p)


def test_run_q5_recovers_planted_kernel_and_balance_and_reports_failure(tmp_path: Path):
    symbols = ["AAAUSDT", "BBBUSDT", "MISSINGUSDT"]
    # Both seeds independently checked (see task-3 report) to land the balance
    # verdict "consistent" at n=500_000, d=0.35, planted beta=0.35.
    _write_synthetic_kernel_fixture(tmp_path, "AAAUSDT", N_EVENTS, seed=2)
    _write_synthetic_kernel_fixture(tmp_path, "BBBUSDT", N_EVENTS, seed=3)
    # MISSINGUSDT deliberately has no parquet on disk -> must land in failures.

    out_dir = tmp_path / "results"
    result = run_q5(
        tmp_path, out_dir, symbols=symbols,
        month="2023-06", start_day="2023-06-01", end_day="2023-06-07",
        max_lag=MAX_LAG,
    )

    by_symbol = {r["symbol"]: r for r in result["records"]}
    assert "AAAUSDT" in by_symbol
    assert "BBBUSDT" in by_symbol

    for sym in ("AAAUSDT", "BBBUSDT"):
        rec = by_symbol[sym]
        assert abs(rec["beta"] - PLANTED_BETA) < 0.1, f"{sym}: beta={rec['beta']}"
        assert rec["verdict"] == "consistent", (
            f"{sym}: verdict={rec['verdict']} delta={rec['balance_delta']} "
            f"beta_block_sd={rec['beta_block_sd']}"
        )
        assert rec["n_events"] > 0
        assert "gamma_week" in rec
        assert "gamma_week_stderr" in rec
        assert "beta_block_sd" in rec
        assert "balance_delta" in rec
        assert "R1" in rec
        assert len(rec["response"]) == 100
        assert len(rec["G"]) == 100
        assert 0.0 <= rec["drop_rate"] <= 1.0

    # --- missing symbol lands in failures, never aborts the run ---
    failures = {f["symbol"]: f for f in result["failures"]}
    assert "MISSINGUSDT" in failures
    assert failures["MISSINGUSDT"]["reason"]
    assert "MISSINGUSDT" not in by_symbol

    # --- metadata ---
    assert result["month"] == "2023-06"
    assert result["max_lag"] == MAX_LAG
    assert result["n_consistent"] + result["n_violated"] == 2
    assert result["n_consistent"] == 2

    # --- output files exist ---
    assert (out_dir / "q5_kernel_panel.json").exists()
    assert (out_dir / "q5_kernel_panel.md").exists()
    assert (out_dir / "q5_kernel_panel.png").exists()


def test_run_q5_never_aborts_on_all_failures(tmp_path: Path):
    out_dir = tmp_path / "results"
    result = run_q5(
        tmp_path, out_dir, symbols=["GHOSTUSDT"],
        month="2023-06", start_day="2023-06-01", end_day="2023-06-07",
        max_lag=MAX_LAG,
    )
    assert result["records"] == []
    assert len(result["failures"]) == 1
    assert result["failures"][0]["symbol"] == "GHOSTUSDT"
    assert (out_dir / "q5_kernel_panel.md").exists()
    assert (out_dir / "q5_kernel_panel.json").exists()
    assert (out_dir / "q5_kernel_panel.png").exists()


def test_run_q5_thin_symbol_fails_gracefully_not_crashes(tmp_path: Path):
    """A symbol too thin to satisfy the n/L>=100 guard must land in failures,
    not raise out of run_q5 (per-symbol try/except is the point)."""
    thin_n = 5_000  # n/max_lag = 50, below MIN_SAMPLES_PER_LAG=100 at max_lag=100
    _write_synthetic_kernel_fixture(tmp_path, "THINUSDT", thin_n, seed=7)

    out_dir = tmp_path / "results"
    result = run_q5(
        tmp_path, out_dir, symbols=["THINUSDT"],
        month="2023-06", start_day="2023-06-01", end_day="2023-06-07",
        max_lag=MAX_LAG,
    )
    assert result["records"] == []
    failures = {f["symbol"]: f for f in result["failures"]}
    assert "THINUSDT" in failures
    assert failures["THINUSDT"]["reason"]
