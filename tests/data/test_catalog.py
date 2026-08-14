import hashlib
import io
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import httpx
import polars as pl
import pytest

from microstructure.data.catalog import continuity_report, integrity_report, parquet_path, sync

AGG_ROW = "100,50000.5,0.010,200,201,1687392000123,true"

_AGG_SCHEMA = {
    "agg_trade_id": pl.Int64, "price": pl.Float64, "qty": pl.Float64,
    "first_trade_id": pl.Int64, "last_trade_id": pl.Int64,
    "ts": pl.Datetime("ms", "UTC"), "is_buyer_maker": pl.Boolean,
}


def _agg_trades_month(agg_trade_ids: list[int], year: int, month: int) -> pl.DataFrame:
    """Build a synthetic aggTrades month frame: one row per id, timestamps
    spaced evenly through the month, all within bounds."""
    base = datetime(year, month, 2, tzinfo=UTC)  # comfortably inside the month
    rows = [
        [tid, 100.0 + i, 1.0, tid, tid, base.replace(hour=i % 23), False]
        for i, tid in enumerate(agg_trade_ids)
    ]
    return pl.DataFrame(rows, schema=_AGG_SCHEMA, orient="row")


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.csv", AGG_ROW + "\n")
    return buf.getvalue()


def make_client() -> httpx.Client:
    payload = _zip_bytes()
    sha = hashlib.sha256(payload).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".CHECKSUM"):
            name = request.url.path.rsplit("/", 1)[-1].removesuffix(".CHECKSUM")
            return httpx.Response(200, text=f"{sha}  {name}\n")
        return httpx.Response(200, content=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_sync_downloads_ingests_and_cleans_up(tmp_data_dir: Path):
    paths = sync(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-06", "2023-07", client=make_client())
    assert len(paths) == 2
    assert all(p.exists() for p in paths)
    assert paths[0] == parquet_path(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-06")
    assert pl.read_parquet(paths[0]).height == 1
    assert not list((tmp_data_dir / "raw").glob("*.zip"))  # zips removed after ingest


def test_sync_is_idempotent(tmp_data_dir: Path):
    client = make_client()
    sync(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-06", "2023-06", client=client)
    p = parquet_path(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-06")
    mtime = p.stat().st_mtime_ns
    sync(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-06", "2023-06", client=client)
    assert p.stat().st_mtime_ns == mtime  # untouched second time


def test_integrity_report_flags_missing(tmp_data_dir: Path):
    sync(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-06", "2023-06", client=make_client())
    rep = integrity_report(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-06", "2023-07")
    assert rep["present"].to_list() == [True, False]
    assert rep["rows"].to_list() == [1, None]


def test_sync_rejects_corrupt_ingest_output(tmp_data_dir: Path):
    """Verify that sync validates parquet before canonical rename and fails cleanly."""
    def bad_ingester(zip_path: Path, out_dir: Path) -> Path:
        # Write garbage bytes to the expected parquet filename
        produced = out_dir / "2023-06.parquet"
        produced.write_bytes(b"garbage data")
        return produced

    client = make_client()
    with patch("microstructure.data.catalog._INGESTERS", {"aggTrades": bad_ingester}):
        with pytest.raises(ValueError, match="Parquet verification failed"):
            sync(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-06", "2023-06", client=client)
        # Verify canonical dest path does not exist
        dest = parquet_path(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-06")
        assert not dest.exists()


def test_sync_unknown_data_type_raises_value_error(tmp_data_dir: Path):
    """Verify that sync raises ValueError for unknown data_type."""
    with pytest.raises(ValueError, match="unknown data_type"):
        sync(tmp_data_dir, "BTCUSDT", "nope", "2023-06", "2023-06")


def test_continuity_report_clean_pair_all_checks_pass(tmp_data_dir: Path):
    """Two consecutive months with fully sequential agg_trade_ids and
    in-bounds timestamps: id_gaps==0, ts_in_bounds True, joins_previous True."""
    june = _agg_trades_month(list(range(1, 11)), 2023, 6)  # ids 1..10
    july = _agg_trades_month(list(range(11, 21)), 2023, 7)  # ids 11..20, joins june

    june_path = parquet_path(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-06")
    july_path = parquet_path(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-07")
    june_path.parent.mkdir(parents=True, exist_ok=True)
    july_path.parent.mkdir(parents=True, exist_ok=True)
    june.write_parquet(june_path)
    july.write_parquet(july_path)

    rep = continuity_report(tmp_data_dir, "BTCUSDT", "2023-06", "2023-07")
    assert rep["period"].to_list() == ["2023-06", "2023-07"]
    assert rep["rows"].to_list() == [10, 10]
    assert rep["id_gaps"].to_list() == [0, 0]
    assert rep["ts_in_bounds"].to_list() == [True, True]
    assert rep["joins_previous"].to_list() == [None, True]


def test_continuity_report_deleted_row_and_broken_join_detected(tmp_data_dir: Path):
    """June is missing id 5 (a deleted middle row) -> id_gaps==1 for June.
    July does not start right after June's last id -> joins_previous False."""
    june_ids = [i for i in range(1, 11) if i != 5]  # ids 1..10 minus 5 -> gap of 1
    june = _agg_trades_month(june_ids, 2023, 6)
    july = _agg_trades_month(list(range(50, 60)), 2023, 7)  # does not join june

    june_path = parquet_path(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-06")
    july_path = parquet_path(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-07")
    june_path.parent.mkdir(parents=True, exist_ok=True)
    july_path.parent.mkdir(parents=True, exist_ok=True)
    june.write_parquet(june_path)
    july.write_parquet(july_path)

    rep = continuity_report(tmp_data_dir, "BTCUSDT", "2023-06", "2023-07")
    assert rep["rows"].to_list() == [9, 10]
    assert rep["id_gaps"].to_list() == [1, 0]
    assert rep["ts_in_bounds"].to_list() == [True, True]
    assert rep["joins_previous"].to_list() == [None, False]
