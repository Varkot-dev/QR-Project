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
