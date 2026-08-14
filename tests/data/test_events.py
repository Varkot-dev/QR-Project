from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from microstructure.data.events import to_aggressor_events


def _df(rows):
    return pl.DataFrame(
        rows,
        schema={
            "agg_trade_id": pl.Int64, "price": pl.Float64, "qty": pl.Float64,
            "first_trade_id": pl.Int64, "last_trade_id": pl.Int64,
            "ts": pl.Datetime("ms", "UTC"), "is_buyer_maker": pl.Boolean,
        },
        orient="row",
    )


T0 = datetime(2023, 6, 22, 0, 0, 0, 123000, tzinfo=UTC)
T1 = datetime(2023, 6, 22, 0, 0, 0, 456000, tzinfo=UTC)


def test_same_ts_same_side_merged_into_one_event():
    df = _df([
        [1, 100.0, 1.0, 1, 1, T0, False],   # buy aggressor sweep, print 1
        [2, 101.0, 3.0, 2, 2, T0, False],   # same order sweeping next level
        [3, 100.5, 2.0, 3, 3, T1, True],    # later sell aggressor
    ])
    ev = to_aggressor_events(df)
    assert ev.height == 2
    first = ev.row(0, named=True)
    assert first["sign"] == 1
    assert first["qty"] == 4.0
    assert abs(first["price"] - (100.0 * 1.0 + 101.0 * 3.0) / 4.0) < 1e-12
    assert first["n_prints"] == 2
    assert ev.row(1, named=True)["sign"] == -1


def test_same_ts_opposite_sides_not_merged():
    df = _df([
        [1, 100.0, 1.0, 1, 1, T0, False],
        [2, 100.0, 1.0, 2, 2, T0, True],
    ])
    ev = to_aggressor_events(df)
    assert ev.height == 2


def test_input_not_mutated():
    df = _df([[1, 100.0, 1.0, 1, 1, T0, False]])
    before = df.clone()
    to_aggressor_events(df)
    assert df.equals(before)


def test_zero_qty_group_raises():
    """A group with total qty == 0 should raise ValueError."""
    df = _df([[1, 100.0, 0.0, 1, 1, T0, False]])
    with pytest.raises(ValueError):
        to_aggressor_events(df)


def test_same_ts_opposite_sides_deterministic_order():
    """Same-ts opposite-side events must have deterministic order."""
    # Build frame with buy-first order
    df_buy_first = _df([
        [1, 100.0, 1.0, 1, 1, T0, False],  # buy aggressor
        [2, 100.0, 1.0, 2, 2, T0, True],   # sell aggressor
    ])
    # Build frame with sell-first order (inverted input row order)
    df_sell_first = _df([
        [2, 100.0, 1.0, 2, 2, T0, True],   # sell aggressor
        [1, 100.0, 1.0, 1, 1, T0, False],  # buy aggressor
    ])

    ev_buy_first = to_aggressor_events(df_buy_first)
    ev_sell_first = to_aggressor_events(df_sell_first)

    # Both should produce the same output: sell (-1) before buy (+1)
    assert ev_buy_first.equals(ev_sell_first)
    # Verify sell row comes first (sign -1 < +1)
    assert ev_buy_first.row(0, named=True)["sign"] == -1
    assert ev_buy_first.row(1, named=True)["sign"] == 1


def test_lazyframe_input_matches_dataframe_input(tmp_path: Path):
    """to_aggressor_events must accept pl.scan_parquet output (a LazyFrame)
    and produce the same result as calling it on the equivalent DataFrame."""
    df = _df([
        [1, 100.0, 1.0, 1, 1, T0, False],
        [2, 101.0, 3.0, 2, 2, T0, False],
        [3, 100.5, 2.0, 3, 3, T1, True],
    ])
    parquet_path = tmp_path / "trades.parquet"
    df.write_parquet(parquet_path)

    ev_from_df = to_aggressor_events(df)
    ev_from_lazy = to_aggressor_events(pl.scan_parquet(parquet_path))

    assert ev_from_df.equals(ev_from_lazy)
