import pytest

from microstructure.data.binance import DAILY_BASE, DumpFile, day_files


def test_day_files_inclusive_and_daily_url():
    files = day_files("ETHUSDT", "bookTicker", "2023-06-29", "2023-07-02")
    assert [f.period for f in files] == ["2023-06-29", "2023-06-30", "2023-07-01", "2023-07-02"]
    assert files[0].url == (
        f"{DAILY_BASE}/bookTicker/ETHUSDT/ETHUSDT-bookTicker-2023-06-29.zip"
    )
    assert files[0].checksum_url == files[0].url + ".CHECKSUM"


def test_monthly_dumpfile_url_unchanged():
    f = DumpFile(symbol="BTCUSDT", data_type="aggTrades", period="2023-06")
    assert f.url == (
        "https://data.binance.vision/data/futures/um/monthly/"
        "aggTrades/BTCUSDT/BTCUSDT-aggTrades-2023-06.zip"
    )


def test_day_files_rejects_bad_date():
    with pytest.raises(ValueError):
        day_files("ETHUSDT", "bookTicker", "2023-06-31", "2023-07-02")
