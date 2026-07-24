"""Phase 2 — Data cleaning, segmentation, and preparation.

Reads the raw manifests produced by `download.py`, cleans transcripts, trims
leading/trailing silence, drops clips outside a sane duration range, resamples to
the target sample rate, and writes the result to `data/processed/<dataset>/<split>/`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf

from afrimeet.utils.config import load_config
from afrimeet.utils.logging import logger

MIN_DURATION_S = 1.0


def clean_transcript(text: str) -> str:
    """Lowercase, strip punctuation noise, and collapse whitespace."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s'-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def process_split(
    raw_split_dir: Path,
    processed_split_dir: Path,
    sample_rate: int,
    max_duration_s: float,
) -> Path:
    manifest_path = raw_split_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found at {manifest_path}")

    df = pd.read_csv(manifest_path)
    audio_out_dir = processed_split_dir / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    kept_rows = []
    for _, row in df.iterrows():
        transcript = clean_transcript(str(row["transcript"]))
        if not transcript:
            continue

        src_audio_path = raw_split_dir / row["audio_path"]
        signal, sr = librosa.load(src_audio_path, sr=sample_rate, mono=True)
        signal, _ = librosa.effects.trim(signal, top_db=30)

        duration_s = len(signal) / sr
        if duration_s < MIN_DURATION_S or duration_s > max_duration_s:
            continue

        out_path = audio_out_dir / Path(row["audio_path"]).name
        sf.write(out_path, signal, sr)
        kept_rows.append(
            {
                "audio_path": str(out_path.relative_to(processed_split_dir)),
                "transcript": transcript,
                "duration_s": duration_s,
            }
        )

    out_manifest = processed_split_dir / "manifest.csv"
    pd.DataFrame(kept_rows).to_csv(out_manifest, index=False)
    logger.info(f"{raw_split_dir.name}: kept {len(kept_rows)}/{len(df)} examples -> {out_manifest}")
    return out_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and segment downloaded datasets.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    raw_dir = Path(config["paths"]["data_raw"])
    processed_dir = Path(config["paths"]["data_processed"])
    sample_rate = config["data"]["sample_rate"]
    max_duration_s = float(config["data"]["max_audio_seconds"])

    for dataset_cfg in config["data"]["datasets"]:
        dataset_dir_name = dataset_cfg["name"].replace("/", "__")
        for split in dataset_cfg["splits"]:
            raw_split_dir = raw_dir / dataset_dir_name / split
            processed_split_dir = processed_dir / dataset_dir_name / split
            if not raw_split_dir.exists():
                logger.warning(f"Skipping missing raw split: {raw_split_dir}")
                continue
            process_split(raw_split_dir, processed_split_dir, sample_rate, max_duration_s)


if __name__ == "__main__":
    main()
