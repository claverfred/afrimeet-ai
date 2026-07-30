"""Phase 2 — Data Collection.

Downloads public multilingual speech datasets (Common Voice, FLEURS, ...) via the
Hugging Face `datasets` library and writes each split to `data/raw/<dataset>/<split>/`
as a manifest CSV (audio path relative to the manifest + transcript) plus the raw
audio files. Requires `requirements/ml.txt` to be installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf

from afrimeet.utils.config import load_config
from afrimeet.utils.logging import logger


def download_dataset(
    name: str, config_name: str, split: str, out_dir: Path, text_column: str | None = None
) -> Path:
    """Download one (dataset, config, split) and materialize it as
    `out_dir/<split>/audio/*.wav` + `out_dir/<split>/manifest.csv`.

    `text_column`, if given, overrides auto-detection -- needed for datasets like
    FLEURS that expose multiple text columns with different normalization (its
    `transcription` is lowercased with punctuation mostly stripped; `raw_transcription`
    keeps original casing and punctuation, which is what we want the model to learn to
    produce).

    Returns the path to the written manifest.
    """
    from datasets import Audio, load_dataset

    logger.info(f"Loading {name} ({config_name}, split={split}) ...")
    ds = load_dataset(name, config_name, split=split, trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    split_dir = out_dir / split
    audio_dir = split_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    text_column = text_column or _guess_text_column(ds.column_names)
    rows = []
    for i, example in enumerate(ds):
        audio = example["audio"]
        audio_path = audio_dir / f"{i:07d}.wav"
        sf.write(audio_path, audio["array"], audio["sampling_rate"])
        rows.append(
            {
                "audio_path": str(audio_path.relative_to(split_dir)),
                "transcript": example.get(text_column, ""),
                "duration_s": len(audio["array"]) / audio["sampling_rate"],
            }
        )
        if (i + 1) % 500 == 0:
            logger.info(f"  ... {i + 1} examples written")

    manifest_path = split_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    logger.info(f"Wrote {len(rows)} examples to {manifest_path}")
    return manifest_path


def _guess_text_column(column_names: list[str]) -> str:
    for candidate in ("sentence", "transcription", "text", "raw_transcription"):
        if candidate in column_names:
            return candidate
    raise ValueError(f"Could not find a transcript column among {column_names}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download configured speech datasets.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    raw_dir = Path(config["paths"]["data_raw"])

    failures: list[str] = []
    n_ok = 0
    for dataset_cfg in config["data"]["datasets"]:
        out_dir = raw_dir / dataset_cfg["name"].replace("/", "__")
        for split in dataset_cfg["splits"]:
            try:
                download_dataset(
                    dataset_cfg["name"],
                    dataset_cfg["config"],
                    split,
                    out_dir,
                    text_column=dataset_cfg.get("text_column"),
                )
                n_ok += 1
            except Exception as exc:  # dataset/network/auth errors shouldn't kill the whole run
                logger.error(f"Failed to download {dataset_cfg['name']}/{split}: {exc}")
                failures.append(f"{dataset_cfg['name']}/{split}")

    if n_ok == 0:
        raise RuntimeError(
            "Every dataset/split failed to download (see errors above) — nothing was "
            "written to data/raw. A common cause is a gated dataset (e.g. Common Voice) "
            "that needs `huggingface_hub.login()` plus accepting its terms on the "
            "dataset's Hugging Face page first. Failed: " + ", ".join(failures)
        )
    if failures:
        logger.warning(f"{len(failures)} split(s) failed and were skipped: {', '.join(failures)}")


if __name__ == "__main__":
    main()
