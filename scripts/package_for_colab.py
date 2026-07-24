#!/usr/bin/env python
"""Zip up just the code (no data/models/venv) so it can be uploaded to Google
Drive and unpacked inside notebooks/finetune_whisper_colab.ipynb.

Usage: python scripts/package_for_colab.py
Output: dist/afrimeet-ai-code.zip
"""

from __future__ import annotations

import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "dist" / "afrimeet-ai-code.zip"

INCLUDE = [
    "src",
    "scripts",
    "configs",
    "requirements",
    "requirements.txt",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    ".env.example",
    "tests",
]

EXCLUDE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}


def iter_files():
    for name in INCLUDE:
        path = PROJECT_ROOT / name
        if not path.exists():
            continue
        if path.is_file():
            yield path
            continue
        for file_path in path.rglob("*"):
            if file_path.is_dir():
                continue
            if any(part in EXCLUDE_DIR_NAMES for part in file_path.parts):
                continue
            yield file_path


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    with zipfile.ZipFile(OUTPUT_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        n = 0
        for file_path in iter_files():
            zf.write(file_path, file_path.relative_to(PROJECT_ROOT))
            n += 1

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {n} files to {OUTPUT_PATH} ({size_mb:.2f} MB)")
    print("Upload this file to Google Drive at: My Drive/AfriMeet_AI/afrimeet-ai-code.zip")


if __name__ == "__main__":
    main()
