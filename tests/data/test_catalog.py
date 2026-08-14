import hashlib
import io
import zipfile
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import polars as pl

from microstructure.data.catalog import integrity_report, parquet_path, sync

AGG_ROW = "100,50000.5,0.010,200,201,1687392000123,true"


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
        with pytest.raises(Exception):  # Should raise on parquet validation failure
            sync(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-06", "2023-06", client=client)
        # Verify canonical dest path does not exist
        dest = parquet_path(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-06")
        assert not dest.exists()


def test_sync_unknown_data_type_raises_value_error(tmp_data_dir: Path):
    """Verify that sync raises ValueError for unknown data_type."""
    with pytest.raises(ValueError, match="unknown data_type"):
        sync(tmp_data_dir, "BTCUSDT", "nope", "2023-06", "2023-06")
