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
