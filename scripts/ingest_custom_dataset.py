#!/usr/bin/env python
"""Custom-dataset helper, step 2 (Phase 2): after correcting the draft manifest from
scripts/segment_for_correction.py, copy it into data/raw/custom/<split>/ so it flows
through the normal cleaning pipeline (scripts/prepare_dataset.py) like every other
dataset.

    python scripts/ingest_custom_dataset.py \
        --manifest data/external/custom_review/meeting/manifest_draft.csv

After running this, make sure configs/config.yaml's data.datasets includes a `custom`
entry (see the printed reminder), bump data.schema_version, then run
scripts/prepare_dataset.py (or just re-run the Colab notebook -- its data cell will
detect the schema_version bump and rebuild automatically).
"""

from __future__ import annotations

import argparse

from afrimeet.data.custom_dataset import ingest_corrected_manifest
from afrimeet.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a corrected custom-dataset manifest.")
    parser.add_argument("--manifest", required=True, help="Path to the corrected manifest CSV")
    parser.add_argument("--split", default="train", help="Split to write into (default: train)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    ingest_corrected_manifest(args.manifest, config["paths"]["data_raw"], args.split)


if __name__ == "__main__":
    main()
