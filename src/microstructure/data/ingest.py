"""Zip-CSV → Parquet with schema normalization.

Binance dump quirks handled here so nothing downstream ever sees them:
- some eras include a CSV header row, some don't (sniffed per file);
- epoch timestamps are ms in some eras, µs in others (sniffed by magnitude);
- booleans appear as true/false strings.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import polars as pl

_US_THRESHOLD = 100_000_000_000_000  # 1e14: epoch-ms values are ~1.7e12, epoch-µs ~1.7e15

_AGG_COLS = ["agg_trade_id", "price", "qty", "first_trade_id", "last_trade_id", "ts_raw", "is_buyer_maker"]
_BT_COLS = ["update_id", "bid_price", "bid_qty", "ask_price", "ask_qty", "ts_raw", "event_time_raw"]


def _read_zipped_csv(zip_path: Path, columns: list[str]) -> pl.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        inner = zf.namelist()[0]
        raw = zf.read(inner)
    first_cell = raw.split(b",", 1)[0].split(b"\n", 1)[0]
    has_header = not first_cell.strip().lstrip(b"-").isdigit()
    df = pl.read_csv(io.BytesIO(raw), has_header=has_header, infer_schema_length=1000)
    if df.width < len(columns):
        raise ValueError(f"{zip_path.name}: expected >= {len(columns)} cols, got {df.width}")
    df = df.select(df.columns[: len(columns)])
    return df.rename(dict(zip(df.columns, columns)))


def _epoch_to_ms_utc(col: pl.Expr) -> pl.Expr:
    ms = pl.when(col > _US_THRESHOLD).then(col // 1000).otherwise(col)
    return ms.cast(pl.Datetime("ms")).dt.replace_time_zone("UTC")


def ingest_agg_trades(zip_path: Path, out_dir: Path) -> Path:
    df = _read_zipped_csv(zip_path, _AGG_COLS)
    out = (
        df.with_columns(
            pl.col("is_buyer_maker").cast(pl.Utf8).str.to_lowercase().eq("true").alias("is_buyer_maker"),
            _epoch_to_ms_utc(pl.col("ts_raw").cast(pl.Int64)).alias("ts"),
            pl.col("price").cast(pl.Float64),
            pl.col("qty").cast(pl.Float64),
        )
        .select("agg_trade_id", "price", "qty", "first_trade_id", "last_trade_id", "ts", "is_buyer_maker")
    )
    dest = out_dir / (zip_path.stem + ".parquet")
    out.write_parquet(dest)
    return dest


def ingest_book_ticker(zip_path: Path, out_dir: Path) -> Path:
    df = _read_zipped_csv(zip_path, _BT_COLS)
    out = (
        df.with_columns(
            _epoch_to_ms_utc(pl.col("ts_raw").cast(pl.Int64)).alias("ts"),
            pl.col("bid_price").cast(pl.Float64),
            pl.col("ask_price").cast(pl.Float64),
            pl.col("bid_qty").cast(pl.Float64),
            pl.col("ask_qty").cast(pl.Float64),
        )
        .select("update_id", "bid_price", "bid_qty", "ask_price", "ask_qty", "ts")
    )
    dest = out_dir / (zip_path.stem + ".parquet")
    out.write_parquet(dest)
    return dest
