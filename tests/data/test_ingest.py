import zipfile
from pathlib import Path

import polars as pl

from microstructure.data.ingest import ingest_agg_trades, ingest_book_ticker

AGG_HEADER = "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker"
AGG_ROWS = [
    "100,50000.5,0.010,200,201,1687392000123,true",
    "101,50000.0,0.020,202,204,1687392000456,false",
]
BT_HEADER = (
    "update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time"
)
BT_ROWS = ["9001,49999.9,1.5,50000.1,2.0,1687392000123,1687392000125"]


def _zip_csv(path: Path, name: str, lines: list[str]) -> Path:
    z = path.with_suffix(".zip")
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(name, "\n".join(lines) + "\n")
    return z


def test_agg_trades_with_header(tmp_path: Path):
    z = _zip_csv(tmp_path / "a", "a.csv", [AGG_HEADER, *AGG_ROWS])
    out = ingest_agg_trades(z, tmp_path)
    df = pl.read_parquet(out)
    assert df.columns == [
        "agg_trade_id", "price", "qty", "first_trade_id", "last_trade_id", "ts", "is_buyer_maker",
    ]
    assert df["is_buyer_maker"].to_list() == [True, False]
    assert df["ts"].dtype == pl.Datetime("ms", "UTC")


def test_agg_trades_without_header(tmp_path: Path):
    z = _zip_csv(tmp_path / "b", "b.csv", AGG_ROWS)
    df = pl.read_parquet(ingest_agg_trades(z, tmp_path))
    assert df.height == 2
    assert df["price"][0] == 50000.5


def test_agg_trades_microsecond_timestamps_normalized(tmp_path: Path):
    row_us = "100,50000.5,0.010,200,201,1687392000123456,true"  # 16-digit epoch µs
    z = _zip_csv(tmp_path / "c", "c.csv", [row_us])
    df = pl.read_parquet(ingest_agg_trades(z, tmp_path))
    assert df["ts"].dtype == pl.Datetime("ms", "UTC")
    assert df["ts"][0].year == 2023


def test_book_ticker_roundtrip(tmp_path: Path):
    z = _zip_csv(tmp_path / "d", "d.csv", [BT_HEADER, *BT_ROWS])
    df = pl.read_parquet(ingest_book_ticker(z, tmp_path))
    assert df.columns == ["update_id", "bid_price", "bid_qty", "ask_price", "ask_qty", "ts"]
    assert df["ask_price"][0] == 50000.1


def test_agg_trades_header_only_file_yields_empty_frame_with_correct_schema(tmp_path: Path):
    z = _zip_csv(tmp_path / "e", "e.csv", [AGG_HEADER])
    df = pl.read_parquet(ingest_agg_trades(z, tmp_path))
    assert df.height == 0
    assert df.schema == {
        "agg_trade_id": pl.Int64,
        "price": pl.Float64,
        "qty": pl.Float64,
        "first_trade_id": pl.Int64,
        "last_trade_id": pl.Int64,
        "ts": pl.Datetime("ms", "UTC"),
        "is_buyer_maker": pl.Boolean,
    }


def test_agg_trades_quoted_numeric_first_cell_not_treated_as_header(tmp_path: Path):
    row_quoted = '"100","50000.5","0.010","200","201","1687392000123","true"'
    z = _zip_csv(tmp_path / "f", "f.csv", [row_quoted])
    df = pl.read_parquet(ingest_agg_trades(z, tmp_path))
    assert df.height == 1
    assert df["price"][0] == 50000.5


def test_multi_member_zip_raises(tmp_path: Path):
    import pytest
    z = tmp_path / "multi.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("a.csv", "data1\n")
        zf.writestr("b.csv", "data2\n")
    with pytest.raises(ValueError):
        ingest_agg_trades(z, tmp_path)
