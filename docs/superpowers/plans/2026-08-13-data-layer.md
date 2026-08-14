# Binance Microstructure Data Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the verified data pipeline (download → checksum → Parquet → aggressor events) plus synthetic ground-truth generators, so research analysis can start on trustworthy data.

**Architecture:** A small Python package `microstructure` with a `data/` subpackage (download, ingest, catalog) and a `synthetic.py` module for known-answer test series. Everything TDD with pytest; network-touching tests are marked and use one tiny daily file.

**Tech Stack:** Python 3.12, uv, polars, httpx, pytest, ruff.

## Global Constraints

- Package layout: `src/microstructure/` (spec §4). Research code (`signals/`, `estimators/`, notebooks) is OUT OF SCOPE for this plan — built interactively with the user.
- Data source: `https://data.binance.vision/data/futures/um/{monthly,daily}/{dataType}/{SYMBOL}/...` (VERIFIED 2026-08-13). bookTicker exists only 2023-05→2024-04.
- Phase-1 window: 2023-06 → 2024-03, symbols BTCUSDT, ETHUSDT (spec §3).
- Every download must verify against Binance's published `.CHECKSUM` (SHA-256) file (spec §5).
- Sign convention everywhere: `is_buyer_maker == False` → buyer was taker → aggressor sign **+1**; `True` → **−1**. Document at every use site.
- Raw data dir: `data/raw/` (zips), processed: `data/parquet/`. Both gitignored.
- Files ≤ 400 lines; functions ≤ 50 lines; no mutation of input frames (return new).

---

### Task 1: Repo scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/microstructure/__init__.py`, `src/microstructure/data/__init__.py`, `tests/__init__.py`, `tests/conftest.py`

**Interfaces:**
- Produces: importable package `microstructure`; `uv run pytest` works; `TEST_DATA_DIR` fixture other tasks use.

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "microstructure"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["polars>=1.0", "httpx>=0.27"]

[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.5"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/microstructure"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["network: touches the internet (deselect with '-m \"not network\"')"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 2: Create .gitignore**

```
data/
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
.ruff_cache/
```

- [ ] **Step 3: Create empty package files and conftest**

`src/microstructure/__init__.py` and `src/microstructure/data/__init__.py`: empty files. `tests/__init__.py`: empty.

`tests/conftest.py`:
```python
from pathlib import Path

import pytest


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Isolated data directory for tests."""
    d = tmp_path / "data"
    d.mkdir()
    return d
```

- [ ] **Step 4: Verify environment**

Run: `uv sync && uv run pytest`
Expected: "no tests ran" (exit code 5 is fine at this stage), no import errors.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore src tests
git commit -m "chore: scaffold microstructure package with uv, polars, pytest"
```

---

### Task 2: URL construction and file naming (`binance.py` part 1)

**Files:**
- Create: `src/microstructure/data/binance.py`
- Test: `tests/data/test_binance.py` (create `tests/data/__init__.py` too)

**Interfaces:**
- Produces: `DumpFile(symbol: str, data_type: str, period: str)` frozen dataclass with `.url` and `.filename` properties; `month_files(symbol: str, data_type: str, start: str, end: str) -> list[DumpFile]` where start/end are "YYYY-MM" inclusive.

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_binance.py
from microstructure.data.binance import DumpFile, month_files


def test_dumpfile_url_monthly_aggtrades():
    f = DumpFile(symbol="BTCUSDT", data_type="aggTrades", period="2023-06")
    assert f.filename == "BTCUSDT-aggTrades-2023-06.zip"
    assert f.url == (
        "https://data.binance.vision/data/futures/um/monthly/"
        "aggTrades/BTCUSDT/BTCUSDT-aggTrades-2023-06.zip"
    )


def test_dumpfile_checksum_url():
    f = DumpFile(symbol="BTCUSDT", data_type="bookTicker", period="2023-06")
    assert f.checksum_url == f.url + ".CHECKSUM"


def test_month_files_inclusive_range():
    files = month_files("ETHUSDT", "aggTrades", "2023-11", "2024-02")
    assert [f.period for f in files] == ["2023-11", "2023-12", "2024-01", "2024-02"]


def test_month_files_rejects_bad_month():
    import pytest
    with pytest.raises(ValueError):
        month_files("ETHUSDT", "aggTrades", "2023-13", "2024-02")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_binance.py -v`
Expected: FAIL (ModuleNotFoundError / ImportError).

- [ ] **Step 3: Implement**

```python
# src/microstructure/data/binance.py
"""Binance public-dump file addressing.

Base layout (verified 2026-08-13 against the S3 bucket):
https://data.binance.vision/data/futures/um/monthly/{dataType}/{SYMBOL}/{SYMBOL}-{dataType}-{YYYY-MM}.zip
"""
from __future__ import annotations

import re
from dataclasses import dataclass

BASE = "https://data.binance.vision/data/futures/um/monthly"
_MONTH_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")


@dataclass(frozen=True)
class DumpFile:
    symbol: str
    data_type: str
    period: str  # "YYYY-MM"

    @property
    def filename(self) -> str:
        return f"{self.symbol}-{self.data_type}-{self.period}.zip"

    @property
    def url(self) -> str:
        return f"{BASE}/{self.data_type}/{self.symbol}/{self.filename}"

    @property
    def checksum_url(self) -> str:
        return self.url + ".CHECKSUM"


def month_files(symbol: str, data_type: str, start: str, end: str) -> list[DumpFile]:
    """All monthly DumpFiles from start to end inclusive ("YYYY-MM")."""
    for m in (start, end):
        if not _MONTH_RE.match(m):
            raise ValueError(f"bad month {m!r}, expected YYYY-MM")
    y, mo = int(start[:4]), int(start[5:7])
    ey, emo = int(end[:4]), int(end[5:7])
    out: list[DumpFile] = []
    while (y, mo) <= (ey, emo):
        out.append(DumpFile(symbol, data_type, f"{y:04d}-{mo:02d}"))
        mo += 1
        if mo == 13:
            y, mo = y + 1, 1
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_binance.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/microstructure/data/binance.py tests/data
git commit -m "feat: dump-file URL construction for Binance UM futures monthly data"
```

---

### Task 3: Download with checksum verification (`binance.py` part 2)

**Files:**
- Modify: `src/microstructure/data/binance.py`
- Test: `tests/data/test_download.py`

**Interfaces:**
- Consumes: `DumpFile` from Task 2.
- Produces: `download(file: DumpFile, dest_dir: Path, client: httpx.Client | None = None) -> Path` — downloads zip + CHECKSUM, verifies SHA-256, returns local path; raises `ChecksumError` on mismatch; skips download if file exists AND verifies.

- [ ] **Step 1: Write the failing tests (offline, using a stub transport)**

```python
# tests/data/test_download.py
import hashlib
from pathlib import Path

import httpx
import pytest

from microstructure.data.binance import ChecksumError, DumpFile, download

PAYLOAD = b"fake zip bytes"
GOOD_SHA = hashlib.sha256(PAYLOAD).hexdigest()
F = DumpFile(symbol="BTCUSDT", data_type="aggTrades", period="2023-06")


def make_client(checksum_line: str) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, text=checksum_line)
        return httpx.Response(200, content=PAYLOAD)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_download_writes_file_and_verifies(tmp_data_dir: Path):
    client = make_client(f"{GOOD_SHA}  {F.filename}\n")
    path = download(F, tmp_data_dir, client=client)
    assert path.read_bytes() == PAYLOAD


def test_download_raises_on_bad_checksum(tmp_data_dir: Path):
    client = make_client(f"{'0' * 64}  {F.filename}\n")
    with pytest.raises(ChecksumError):
        download(F, tmp_data_dir, client=client)
    assert not (tmp_data_dir / F.filename).exists()  # bad file not kept


def test_download_skips_when_cached(tmp_data_dir: Path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if request.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, text=f"{GOOD_SHA}  {F.filename}\n")
        return httpx.Response(200, content=PAYLOAD)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    download(F, tmp_data_dir, client=client)
    n_first = calls["n"]
    download(F, tmp_data_dir, client=client)
    assert calls["n"] == n_first + 1  # only CHECKSUM re-fetched, zip not re-downloaded
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_download.py -v`
Expected: FAIL (ImportError: cannot import `download`).

- [ ] **Step 3: Implement (append to binance.py)**

```python
import hashlib
from pathlib import Path

import httpx


class ChecksumError(RuntimeError):
    """Downloaded file does not match Binance's published SHA-256."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_expected_sha(file: DumpFile, client: httpx.Client) -> str:
    resp = client.get(file.checksum_url)
    resp.raise_for_status()
    return resp.text.split()[0].lower()


def download(file: DumpFile, dest_dir: Path, client: httpx.Client | None = None) -> Path:
    """Download file.url into dest_dir, verifying the published SHA-256.

    Cached files that still verify are not re-downloaded.
    """
    owns_client = client is None
    client = client or httpx.Client(timeout=120)
    try:
        dest = dest_dir / file.filename
        expected = _fetch_expected_sha(file, client)
        if dest.exists() and _sha256(dest) == expected:
            return dest
        with client.stream("GET", file.url) as resp:
            resp.raise_for_status()
            tmp = dest.with_suffix(".part")
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes(1 << 20):
                    fh.write(chunk)
        if _sha256(tmp) != expected:
            tmp.unlink()
            raise ChecksumError(f"{file.filename}: SHA-256 mismatch vs {file.checksum_url}")
        tmp.rename(dest)
        return dest
    finally:
        if owns_client:
            client.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_download.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/microstructure/data/binance.py tests/data/test_download.py
git commit -m "feat: checksum-verified download with caching"
```

---

### Task 4: CSV → Parquet ingestion (`ingest.py`)

**Files:**
- Create: `src/microstructure/data/ingest.py`
- Test: `tests/data/test_ingest.py`

**Interfaces:**
- Consumes: local zip paths (from Task 3).
- Produces:
  - `ingest_agg_trades(zip_path: Path, out_dir: Path) -> Path` → Parquet with schema `agg_trade_id: i64, price: f64, qty: f64, first_trade_id: i64, last_trade_id: i64, ts: datetime[ms, UTC], is_buyer_maker: bool`.
  - `ingest_book_ticker(zip_path: Path, out_dir: Path) -> Path` → Parquet with schema `update_id: i64, bid_price: f64, bid_qty: f64, ask_price: f64, ask_qty: f64, ts: datetime[ms, UTC]`.
  - Both must handle files WITH or WITHOUT a header row (Binance is inconsistent across eras — sniff: if the first cell is non-numeric, treat row 0 as header) and both ms and µs epoch timestamps (sniff magnitude: > 1e14 → µs).

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_ingest.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_ingest.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
# src/microstructure/data/ingest.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_ingest.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/microstructure/data/ingest.py tests/data/test_ingest.py
git commit -m "feat: schema-normalizing zip-CSV to Parquet ingestion for aggTrades and bookTicker"
```

---

### Task 5: Aggressor-event aggregation (`events.py`)

**Files:**
- Create: `src/microstructure/data/events.py`
- Test: `tests/data/test_events.py`

**Interfaces:**
- Consumes: aggTrades Parquet schema from Task 4.
- Produces: `to_aggressor_events(df: pl.DataFrame) -> pl.DataFrame` with schema `ts: datetime[ms, UTC], sign: i8 (+1 buy aggressor / −1 sell), qty: f64 (summed), price: f64 (qty-weighted avg), n_prints: u32`. Consecutive aggTrades rows sharing (ts, is_buyer_maker) are merged into ONE event — this is the one-market-order-many-prints fix from the spec (research/02 §4.2 pitfall 1).

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_events.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_events.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
# src/microstructure/data/events.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_events.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/microstructure/data/events.py tests/data/test_events.py
git commit -m "feat: aggressor-event aggregation (one market order = one event)"
```

---

### Task 6: Dataset catalog and pipeline entry point (`catalog.py`)

**Files:**
- Create: `src/microstructure/data/catalog.py`
- Test: `tests/data/test_catalog.py`

**Interfaces:**
- Consumes: `month_files`, `download` (Tasks 2–3), `ingest_agg_trades`, `ingest_book_ticker` (Task 4).
- Produces:
  - `parquet_path(root: Path, symbol: str, data_type: str, period: str) -> Path` — canonical location `{root}/parquet/{data_type}/{symbol}/{period}.parquet`.
  - `sync(root: Path, symbol: str, data_type: str, start: str, end: str, client=None) -> list[Path]` — for each month: skip if Parquet exists, else download zip to `{root}/raw/`, ingest, delete zip. Returns all Parquet paths.
  - `integrity_report(root: Path, symbol: str, data_type: str, start: str, end: str) -> pl.DataFrame` — columns `period, present: bool, rows: i64 | null` (row count read from Parquet metadata; null when absent).

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_catalog.py
import hashlib
import io
import zipfile
from pathlib import Path

import httpx
import polars as pl

from microstructure.data.catalog import integrity_report, parquet_path, sync

AGG_ROW = "100,50000.5,0.010,200,201,1687392000123,true"


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.csv", AGG_ROW + "\n")
    return buf.getvalue()


def make_client() -> httpx.Client:
    payload = _zip_bytes()
    sha = hashlib.sha256(payload).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".CHECKSUM"):
            name = request.url.path.rsplit("/", 1)[-1].removesuffix(".CHECKSUM")
            return httpx.Response(200, text=f"{sha}  {name}\n")
        return httpx.Response(200, content=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_sync_downloads_ingests_and_cleans_up(tmp_data_dir: Path):
    paths = sync(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-06", "2023-07", client=make_client())
    assert len(paths) == 2
    assert all(p.exists() for p in paths)
    assert paths[0] == parquet_path(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-06")
    assert pl.read_parquet(paths[0]).height == 1
    assert not list((tmp_data_dir / "raw").glob("*.zip"))  # zips removed after ingest


def test_sync_is_idempotent(tmp_data_dir: Path):
    client = make_client()
    sync(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-06", "2023-06", client=client)
    p = parquet_path(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-06")
    mtime = p.stat().st_mtime_ns
    sync(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-06", "2023-06", client=client)
    assert p.stat().st_mtime_ns == mtime  # untouched second time


def test_integrity_report_flags_missing(tmp_data_dir: Path):
    sync(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-06", "2023-06", client=make_client())
    rep = integrity_report(tmp_data_dir, "BTCUSDT", "aggTrades", "2023-06", "2023-07")
    assert rep["present"].to_list() == [True, False]
    assert rep["rows"].to_list() == [1, None]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_catalog.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

```python
# src/microstructure/data/catalog.py
"""Local dataset registry: what's on disk, and one entry point to fill gaps."""
from __future__ import annotations

from pathlib import Path

import httpx
import polars as pl

from microstructure.data.binance import DumpFile, download, month_files
from microstructure.data.ingest import ingest_agg_trades, ingest_book_ticker

_INGESTERS = {"aggTrades": ingest_agg_trades, "bookTicker": ingest_book_ticker}


def parquet_path(root: Path, symbol: str, data_type: str, period: str) -> Path:
    return root / "parquet" / data_type / symbol / f"{period}.parquet"


def sync(
    root: Path, symbol: str, data_type: str, start: str, end: str,
    client: httpx.Client | None = None,
) -> list[Path]:
    """Ensure Parquet exists for every month in [start, end]; return the paths."""
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_catalog.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/microstructure/data/catalog.py tests/data/test_catalog.py
git commit -m "feat: dataset catalog with idempotent sync and integrity report"
```

---

### Task 7: Synthetic ground-truth generators (`synthetic.py`)

**Files:**
- Create: `src/microstructure/synthetic.py`
- Test: `tests/test_synthetic.py`

**Interfaces:**
- Produces (used later to validate every estimator before real data):
  - `iid_signs(n: int, seed: int) -> np.ndarray` — ±1, zero autocorrelation by construction.
  - `markov_signs(n: int, p_repeat: float, seed: int) -> np.ndarray` — ±1 chain where next sign repeats with probability `p_repeat`; theoretical ACF at lag k is exactly `(2*p_repeat - 1)**k` (short-memory known answer).
  - `fractional_signs(n: int, d: float, seed: int) -> np.ndarray` — signs of FARIMA(0,d,0) Gaussian noise; asymptotic sign-ACF decays as a power law with exponent `1 - 2d` (long-memory known answer; d in (0, 0.5)).
- Requires adding `numpy>=1.26` to project dependencies.

- [ ] **Step 1: Add numpy dependency**

Run: `uv add "numpy>=1.26"`
Expected: pyproject and lock updated.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_synthetic.py
import numpy as np

from microstructure.synthetic import fractional_signs, iid_signs, markov_signs


def _acf(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    return float((x[:-lag] * x[lag:]).mean() / (x * x).mean())


def test_iid_signs_values_and_no_memory():
    s = iid_signs(200_000, seed=1)
    assert set(np.unique(s)) == {-1, 1}
    assert abs(_acf(s, 1)) < 0.01


def test_iid_signs_reproducible():
    assert np.array_equal(iid_signs(1000, seed=7), iid_signs(1000, seed=7))


def test_markov_signs_match_theoretical_acf():
    p = 0.75  # theoretical ACF(k) = (2p-1)^k = 0.5^k
    s = markov_signs(400_000, p_repeat=p, seed=2)
    for k, expected in [(1, 0.5), (2, 0.25), (3, 0.125)]:
        assert abs(_acf(s, k) - expected) < 0.02


def test_fractional_signs_long_memory_slower_than_markov():
    s = fractional_signs(400_000, d=0.4, seed=3)
    # long memory: ACF at lag 50 must remain clearly positive,
    # whereas any short-memory chain with same lag-1 ACF would be ~0 by lag 50
    assert _acf(s, 1) > 0.05
    assert _acf(s, 50) > 0.02
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_synthetic.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 4: Implement**

```python
# src/microstructure/synthetic.py
"""Series with KNOWN statistical properties, for validating estimators.

Rule (spec section 5): no estimator touches real data until it recovers the
known answers generated here within stated error bars.
"""
from __future__ import annotations

import numpy as np


def iid_signs(n: int, seed: int) -> np.ndarray:
    """±1 coin flips: ACF is exactly 0 at every positive lag."""
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=n)


def markov_signs(n: int, p_repeat: float, seed: int) -> np.ndarray:
    """±1 chain repeating previous sign w.p. p_repeat.

    Theoretical ACF(k) = (2*p_repeat - 1)**k  (geometric, short memory).
    """
    if not 0.0 < p_repeat < 1.0:
        raise ValueError("p_repeat must be in (0, 1)")
    rng = np.random.default_rng(seed)
    flips = rng.random(n) >= p_repeat  # True -> switch sign
    signs = np.empty(n, dtype=np.int8)
    signs[0] = 1
    switches = np.where(flips[1:], -1, 1)
    signs[1:] = np.cumprod(switches)
    return signs


def fractional_signs(n: int, d: float, seed: int) -> np.ndarray:
    """Signs of FARIMA(0, d, 0) noise: power-law (long-memory) sign ACF.

    Built by MA(inf) truncation with coefficients
    psi_k = Gamma(k + d) / (Gamma(k + 1) Gamma(d)), computed recursively:
    psi_0 = 1, psi_k = psi_{k-1} * (k - 1 + d) / k.
    """
    if not 0.0 < d < 0.5:
        raise ValueError("d must be in (0, 0.5)")
    rng = np.random.default_rng(seed)
    n_lags = 2000
    psi = np.empty(n_lags)
    psi[0] = 1.0
    for k in range(1, n_lags):
        psi[k] = psi[k - 1] * (k - 1 + d) / k
    eps = rng.standard_normal(n + n_lags)
    x = np.convolve(eps, psi, mode="full")[n_lags : n_lags + n]
    return np.where(x >= 0, 1, -1).astype(np.int8)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_synthetic.py -v`
Expected: 4 PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/microstructure/synthetic.py tests/test_synthetic.py
git commit -m "feat: synthetic sign series with known ACF for estimator validation"
```

---

### Task 8: Live smoke test against real Binance data (network-marked)

**Files:**
- Create: `tests/data/test_live_smoke.py`

**Interfaces:**
- Consumes: `DumpFile`, `download` (Tasks 2–3), `ingest_agg_trades` (Task 4), `to_aggressor_events` (Task 5).
- Purpose: prove the schema assumptions hold against ONE real (small, daily) file. This is the "not lying" gate between our fixtures and reality.

- [ ] **Step 1: Write the test**

```python
# tests/data/test_live_smoke.py
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
    assert df.height > 100_000                # a normal ETH day has ~1M+ prints
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
```

- [ ] **Step 2: Run the network test for real**

Run: `uv run pytest -m network -v`
Expected: PASS. If it FAILS, the failure is a REAL FINDING about schema assumptions — investigate, fix ingest, and document the discrepancy in the commit message. Do not weaken assertions to make it pass.

- [ ] **Step 3: Verify the default test run stays offline**

Run: `uv run pytest -m "not network" -q`
Expected: all previous tests pass, live test deselected.

- [ ] **Step 4: Commit**

```bash
git add tests/data/test_live_smoke.py
git commit -m "test: live smoke test validating schema assumptions against real dump file"
```

---

## Out of scope for this plan (deliberate)

- `signals/`, `estimators/`, notebooks, `LEARNING.md` content — built interactively with the user (spec §4 division of labor).
- `plots/` styling — deferred until the first real figure exists to style (YAGNI).
- Bulk download of the full 10-month window — user-run one-liner once the pipeline is trusted:
  `uv run python -c "from pathlib import Path; from microstructure.data.catalog import sync; [sync(Path('data'), s, t, '2023-06', '2024-03') for s in ('BTCUSDT','ETHUSDT') for t in ('aggTrades','bookTicker')]"`
- GitHub remote/CI — added when the user creates the remote.
