from pathlib import Path

import httpx
import pytest

from microstructure.data.binance import ChecksumError, DumpFile, download, month_files


def test_dumpfile_url_monthly_aggtrades():
    f = DumpFile(symbol="BTCUSDT", data_type="aggTrades", period="2023-06")
    assert f.filename == "BTCUSDT-aggTrades-2023-06.zip"
    assert f.url == (
        "https://data.binance.vision/data/futures/um/monthly/"
        "aggTrades/BTCUSDT/BTCUSDT-aggTrades-2023-06.zip"
    )


def test_dumpfile_checksum_url():
    f = DumpFile(symbol="BTCUSDT", data_type="bookTicker", period="2023-06")
    assert f.checksum_url == f.url + ".CHECKSUM"


def test_month_files_inclusive_range():
    files = month_files("ETHUSDT", "aggTrades", "2023-11", "2024-02")
    assert [f.period for f in files] == ["2023-11", "2023-12", "2024-01", "2024-02"]


def test_month_files_rejects_bad_month():
    with pytest.raises(ValueError):
        month_files("ETHUSDT", "aggTrades", "2023-13", "2024-02")


def test_download_empty_checksum_body_raises_checksum_error(tmp_path: Path):
    """An empty/whitespace CHECKSUM response body must raise ChecksumError
    naming the file, not crash with an unrelated IndexError."""
    f = DumpFile(symbol="BTCUSDT", data_type="aggTrades", period="2023-06")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, text="   \n")  # empty/whitespace-only body
        return httpx.Response(200, content=b"irrelevant")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ChecksumError, match=f.filename):
        download(f, tmp_path, client=client)
