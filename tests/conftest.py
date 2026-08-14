from pathlib import Path

import pytest


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Isolated data directory for tests."""
    d = tmp_path / "data"
    d.mkdir()
    return d
