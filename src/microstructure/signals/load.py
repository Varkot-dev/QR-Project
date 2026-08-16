"""Parquet cache -> analysis-ready frames. All scans lazy until the end."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl

from microstructure.data.catalog import parquet_path
from microstructure.data.events import to_aggressor_events


def _existing_paths(root: Path, symbol: str, data_type: str, periods: list[str]) -> list[Path]:
    paths = [parquet_path(root, symbol, data_type, p) for p in periods]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing {data_type} parquet for {symbol}: {missing}")
    return paths


def load_events(root: Path, symbol: str, periods: list[str]) -> pl.DataFrame:
    paths = _existing_paths(root, symbol, "aggTrades", periods)
    lf = pl.concat([pl.scan_parquet(p) for p in paths])
    return to_aggressor_events(lf)


def load_book_ticker(root: Path, symbol: str, periods: list[str]) -> pl.DataFrame:
    """Load bookTicker parquets and compute mid price.

    Output row order follows the given period order; no global ts sort is applied.
    """
    paths = _existing_paths(root, symbol, "bookTicker", periods)
    lf = pl.concat([pl.scan_parquet(p) for p in paths]).with_columns(
        ((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid")
    )
    return lf.collect()


def events_with_prior_mid(
    events: pl.DataFrame, bt: pl.DataFrame
) -> tuple[pl.DataFrame, int]:
    """Attach the mid prevailing STRICTLY before each event's ts.

    join_asof(backward) matches <=; shifting the event key back 1ms turns
    that into strict <, honoring the 'mid before the event' convention at
    the data's ms resolution. Output is sorted by ts regardless of input order.

    Raises ValueError if events or bt have ts precision other than ms-UTC.
    """
    # Validate ms-UTC precision on both frames
    events_ts_dtype = events.schema["ts"]
    bt_ts_dtype = bt.schema["ts"]

    if events_ts_dtype != pl.Datetime("ms", "UTC"):
        raise ValueError(
            f"events ts must be Datetime('ms', 'UTC'), got {events_ts_dtype}"
        )
    if bt_ts_dtype != pl.Datetime("ms", "UTC"):
        raise ValueError(
            f"bt ts must be Datetime('ms', 'UTC'), got {bt_ts_dtype}"
        )

    ev = events.with_columns((pl.col("ts") - timedelta(milliseconds=1)).alias("_key")).sort("_key")
    quotes = bt.select("ts", "mid").sort("ts").rename({"ts": "_key"})
    joined = ev.join_asof(quotes, on="_key", strategy="backward").drop("_key")
    n_dropped = int(joined["mid"].null_count())
    return joined.drop_nulls("mid"), n_dropped
