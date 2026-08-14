from datetime import datetime, timezone

import polars as pl

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


T0 = datetime(2023, 6, 22, 0, 0, 0, 123000, tzinfo=timezone.utc)
T1 = datetime(2023, 6, 22, 0, 0, 0, 456000, tzinfo=timezone.utc)


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
