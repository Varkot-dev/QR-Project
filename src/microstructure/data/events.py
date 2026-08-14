"""Collapse exchange prints into aggressor-level events.

One market order sweeping several book levels prints as several aggTrades
rows with identical (ts, is_buyer_maker). Analyses of order-flow memory or
impact must see ONE event per aggressor decision, or self-excitation at
0-1ms lags is pure artifact (see research/02-hawkes-processes.md, pitfalls).

Sign convention: is_buyer_maker == False -> buyer was the taker -> +1.
"""
from __future__ import annotations

import polars as pl


def to_aggressor_events(df: pl.DataFrame) -> pl.DataFrame:
    """Merge consecutive same-(ts, side) prints; returns a new frame."""
    return (
        df.group_by("ts", "is_buyer_maker", maintain_order=True)
        .agg(
            (pl.col("price") * pl.col("qty")).sum().alias("_notional"),
            pl.col("qty").sum().alias("qty"),
            pl.len().cast(pl.UInt32).alias("n_prints"),
        )
        .with_columns(
            pl.when(pl.col("is_buyer_maker")).then(pl.lit(-1)).otherwise(pl.lit(1))
            .cast(pl.Int8).alias("sign"),
            (pl.col("_notional") / pl.col("qty")).alias("price"),
        )
        .select("ts", "sign", "qty", "price", "n_prints")
        .sort("ts")
    )
