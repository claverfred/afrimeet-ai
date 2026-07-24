"""Config loading for the AfriMeet AI pipeline."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@cache
def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and cache the YAML config. Paths inside `paths:` are resolved to
    absolute paths relative to the project root."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for key, value in config.get("paths", {}).items():
        config["paths"][key] = str(PROJECT_ROOT / value)

    return config


def resolve_path(relative_path: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    return PROJECT_ROOT / relative_path
