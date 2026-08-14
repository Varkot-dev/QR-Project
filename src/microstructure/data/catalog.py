"""Local dataset registry: what's on disk, and one entry point to fill gaps."""
from __future__ import annotations

from pathlib import Path

import httpx
import polars as pl

from microstructure.data.binance import download, month_files
from microstructure.data.ingest import ingest_agg_trades, ingest_book_ticker

_INGESTERS = {"aggTrades": ingest_agg_trades, "bookTicker": ingest_book_ticker}


def parquet_path(root: Path, symbol: str, data_type: str, period: str) -> Path:
    return root / "parquet" / data_type / symbol / f"{period}.parquet"


def sync(
    root: Path, symbol: str, data_type: str, start: str, end: str,
    client: httpx.Client | None = None,
) -> list[Path]:
    """Ensure Parquet exists for every month in [start, end]; return the paths."""
    if data_type not in _INGESTERS:
        raise ValueError(
            f"unknown data_type {data_type!r}; expected one of {sorted(_INGESTERS)}"
        )
    ingest = _INGESTERS[data_type]
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for f in month_files(symbol, data_type, start, end):
        dest = parquet_path(root, symbol, data_type, f.period)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            zip_path = download(f, raw_dir, client=client)
            produced = ingest(zip_path, dest.parent)
            # Verify parquet is valid before canonical rename: existence == validity
            # only because nothing reaches the canonical path unverified.
            try:
                pl.scan_parquet(produced).select(pl.len()).collect()
            except Exception as e:
                produced.unlink()
                raise ValueError(
                    f"Parquet verification failed for {produced}. "
                    f"File was unlinked. Please re-run sync. Error: {e}"
                ) from e
            produced.rename(dest)
            zip_path.unlink()
        out.append(dest)
    return out


def integrity_report(root: Path, symbol: str, data_type: str, start: str, end: str) -> pl.DataFrame:
    rows = []
    for f in month_files(symbol, data_type, start, end):
        p = parquet_path(root, symbol, data_type, f.period)
        n = pl.scan_parquet(p).select(pl.len()).collect().item() if p.exists() else None
        rows.append({"period": f.period, "present": p.exists(), "rows": n})
    return pl.DataFrame(rows)


def continuity_report(root: Path, symbol: str, start: str, end: str) -> pl.DataFrame:
    """Real cross-checks for aggTrades continuity, month by month.

    Binance agg_trade_ids are sequential per symbol, so within a clean
    month (max_id - min_id + 1) must equal the row count, and the first
    agg_trade_id of month k+1 must be exactly one more than the last
    agg_trade_id of month k. All scans are lazy; only the small per-month
    summary rows are collected.

    Columns:
        period: "YYYY-MM"
        rows: row count for the month
        id_gaps: (max(agg_trade_id) - min(agg_trade_id) + 1) - rows;
            0 means no missing aggregate trades in that month.
        ts_in_bounds: True if every row's ts falls within [period start, period end).
        joins_previous: True if this month's first agg_trade_id is exactly
            the previous month's last agg_trade_id + 1; null for the first
            month (nothing to join against) or when either month is missing.
    """
    files = month_files(symbol, "aggTrades", start, end)
    summaries: list[dict] = []
    for f in files:
        p = parquet_path(root, symbol, "aggTrades", f.period)
        if not p.exists():
            summaries.append(
                {
                    "period": f.period,
                    "rows": None,
                    "id_gaps": None,
                    "ts_in_bounds": None,
                    "first_id": None,
                    "last_id": None,
                }
            )
            continue

        year, month = int(f.period[:4]), int(f.period[5:7])
        month_start = pl.datetime(year, month, 1, time_zone="UTC")
        next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
        month_end = pl.datetime(next_year, next_month, 1, time_zone="UTC")

        lf = pl.scan_parquet(p)
        stats = (
            lf.select(
                pl.len().alias("rows"),
                pl.col("agg_trade_id").min().alias("first_id"),
                pl.col("agg_trade_id").max().alias("last_id"),
                (
                    (pl.col("ts") >= month_start) & (pl.col("ts") < month_end)
                ).all().alias("ts_in_bounds"),
            )
            .collect()
            .row(0, named=True)
        )
        rows = stats["rows"]
        id_gaps = (stats["last_id"] - stats["first_id"] + 1) - rows if rows > 0 else None
        summaries.append(
            {
                "period": f.period,
                "rows": rows,
                "id_gaps": id_gaps,
                "ts_in_bounds": stats["ts_in_bounds"],
                "first_id": stats["first_id"],
                "last_id": stats["last_id"],
            }
        )

    joins_previous: list[bool | None] = [None]
    for prev, cur in zip(summaries, summaries[1:]):
        if prev["last_id"] is None or cur["first_id"] is None:
            joins_previous.append(None)
        else:
            joins_previous.append(cur["first_id"] == prev["last_id"] + 1)

    return pl.DataFrame(
        {
            "period": [s["period"] for s in summaries],
            "rows": [s["rows"] for s in summaries],
            "id_gaps": [s["id_gaps"] for s in summaries],
            "ts_in_bounds": [s["ts_in_bounds"] for s in summaries],
            "joins_previous": joins_previous,
        },
        schema={
            "period": pl.Utf8,
            "rows": pl.Int64,
            "id_gaps": pl.Int64,
            "ts_in_bounds": pl.Boolean,
            "joins_previous": pl.Boolean,
        },
    )
