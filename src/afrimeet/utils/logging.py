"""Shared logger setup (thin wrapper around loguru)."""

from __future__ import annotations

import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")

__all__ = ["logger"]
