from __future__ import annotations

from pathlib import Path

import pytest

from utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config_path() -> Path:
    return PROJECT_ROOT / "configs/pa_hydrokan/subset150_main.xml"


@pytest.fixture(scope="session")
def config(config_path: Path) -> dict:
    return load_config(config_path)
