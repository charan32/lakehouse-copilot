import os
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
DEFAULT_DB = ROOT / ".lakehouse" / "streaming-lakehouse" / "dbt" / "target" / "lakehouse.duckdb"


def pytest_configure(config):
    os.environ.setdefault("LAKEHOUSE_DB", str(DEFAULT_DB))
    if not Path(os.environ["LAKEHOUSE_DB"]).exists():
        raise pytest.UsageError(
            f"Lakehouse data not found at {os.environ['LAKEHOUSE_DB']}. Run `make demo-data` first."
        )


@pytest.fixture(scope="session")
def wh():
    from copilot.warehouse import Warehouse

    return Warehouse()
