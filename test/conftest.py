from pathlib import Path

import pytest


@pytest.fixture
def tests_data_dir() -> Path:
    tests_dir = Path(__file__).parent.resolve()
    tests_data_dir = tests_dir / "data"
    return tests_data_dir
