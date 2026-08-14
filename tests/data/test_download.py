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
