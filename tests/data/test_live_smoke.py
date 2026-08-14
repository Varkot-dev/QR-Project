"""Downloads ONE real daily file (~a few MB) and validates every assumption.

Run explicitly with:  uv run pytest -m network -v
"""
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest

from microstructure.data.binance import DumpFile, download
from microstructure.data.events import to_aggressor_events
from microstructure.data.ingest import ingest_agg_trades

DAILY_BASE = "https://data.binance.vision/data/futures/um/daily"


@dataclass(frozen=True)
class DailyFile(DumpFile):
    @property
    def url(self) -> str:  # daily layout differs only in base + date-length
        return f"{DAILY_BASE}/{self.data_type}/{self.symbol}/{self.filename}"


@pytest.mark.network
def test_real_daily_aggtrades_roundtrip(tmp_path: Path):
    f = DailyFile(symbol="ETHUSDT", data_type="aggTrades", period="2023-06-15")
    zip_path = download(f, tmp_path)          # checksum-verified against Binance
    pq = ingest_agg_trades(zip_path, tmp_path)
    df = pl.read_parquet(pq)
    assert df.height > 100_000                # a normal ETH day has ~783k+ prints (observed)
    assert df["ts"].is_sorted()
    assert df["ts"][0].date().isoformat() == "2023-06-15"
    assert df["price"].min() > 100            # sanity: ETH was ~$1.6-1.9k mid-2023
    assert df["price"].max() < 10_000
    ev = to_aggressor_events(df)
    assert 0 < ev.height <= df.height
    assert set(ev["sign"].unique().to_list()) == {-1, 1}
    # both sides active on any real day
    frac_buy = (ev["sign"] == 1).mean()
    assert 0.2 < frac_buy < 0.8
