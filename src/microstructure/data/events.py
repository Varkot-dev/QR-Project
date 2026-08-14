"""Collapse exchange prints into aggressor-level events.

One market order sweeping several book levels prints as several aggTrades
rows with identical (ts, is_buyer_maker). Analyses of order-flow memory or
impact must see ONE event per aggressor decision, or self-excitation at
0-1ms lags is pure artifact (see research/02-hawkes-processes.md, pitfalls).

Sign convention: is_buyer_maker == False -> buyer was the taker -> +1.
"""
from __future__ import annotations

import polars as pl


def to_aggressor_events(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Merge ALL same-(ts, side) prints in the frame; returns a new frame.

    The group_by merges every row sharing the same (ts, is_buyer_maker) pair
    anywhere in the input, not just adjacent/consecutive rows — input row
    order does not affect which rows get merged or the resulting output.

    Raises ValueError if any aggregated group has qty <= 0 (prevents NaN prices).
    Tie-break for same-ts opposite-side events: sort by (ts, sign) so sells (-1)
    precede buys (+1) — deterministic and reproducible across input orderings.

    Accepts either a DataFrame or a LazyFrame (e.g. from pl.scan_parquet); the
    group_by/agg/sort pipeline runs lazily and is only collected once, after
    aggregation, so passing a LazyFrame avoids materializing the raw input.
    """
    lf = df.lazy()
    result_lf = (
        lf.group_by("ts", "is_buyer_maker", maintain_order=True)
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
        .sort("ts", "sign")
    )
    result = result_lf.collect()

    # Fail fast: check for zero or negative qty (prevents NaN prices)
    bad_rows = result.filter(pl.col("qty") <= 0)
    if bad_rows.height > 0:
        bad_ts = bad_rows.select("ts").unique()["ts"].to_list()
        raise ValueError(
            f"Found group(s) with qty <= 0 at timestamp(s): {bad_ts}. "
            "Check input data for zero or negative quantities."
        )

    return result
