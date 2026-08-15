"""Binance public-dump file addressing.

Base layout (verified 2026-08-13 against the S3 bucket):
https://data.binance.vision/data/futures/um/monthly/{dataType}/{SYMBOL}/{SYMBOL}-{dataType}-{YYYY-MM}.zip
https://data.binance.vision/data/futures/um/daily/{dataType}/{SYMBOL}/{SYMBOL}-{dataType}-{YYYY-MM-DD}.zip
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

BASE = "https://data.binance.vision/data/futures/um/monthly"
DAILY_BASE = "https://data.binance.vision/data/futures/um/daily"
_MONTH_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")


@dataclass(frozen=True)
class DumpFile:
    symbol: str
    data_type: str
    period: str  # "YYYY-MM" or "YYYY-MM-DD"
    base: str = BASE

    @property
    def filename(self) -> str:
        return f"{self.symbol}-{self.data_type}-{self.period}.zip"

    @property
    def url(self) -> str:
        return f"{self.base}/{self.data_type}/{self.symbol}/{self.filename}"

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


def day_files(symbol: str, data_type: str, start: str, end: str) -> list[DumpFile]:
    """All daily DumpFiles from start to end inclusive ("YYYY-MM-DD")."""
    import datetime as _dt

    d0, d1 = _dt.date.fromisoformat(start), _dt.date.fromisoformat(end)
    out: list[DumpFile] = []
    while d0 <= d1:
        out.append(DumpFile(symbol, data_type, d0.isoformat(), base=DAILY_BASE))
        d0 += _dt.timedelta(days=1)
    return out


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
    parts = resp.text.split()
    if not parts:
        raise ChecksumError(
            f"{file.filename}: empty CHECKSUM body from {file.checksum_url}"
        )
    return parts[0].lower()


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
