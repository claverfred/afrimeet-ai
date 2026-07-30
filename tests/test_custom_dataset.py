from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")  # afrimeet.data.custom_dataset imports WhisperTranscriber
np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from afrimeet.data.custom_dataset import ingest_corrected_manifest  # noqa: E402


def test_ingest_corrected_manifest_skips_uncorrected_rows(tmp_path: Path):
    source_dir = tmp_path / "review"
    audio_dir = source_dir / "audio"
    audio_dir.mkdir(parents=True)

    sample_rate = 16_000
    for i in range(2):
        sf.write(audio_dir / f"{i:04d}.wav", np.zeros(sample_rate, dtype="float32"), sample_rate)

    pd.DataFrame(
        [
            {
                "audio_path": "audio/0000.wav",
                "start_s": 0.0,
                "end_s": 1.0,
                "draft_transcript": "hello",
                "corrected_transcript": "Hello, world.",
            },
            {
                "audio_path": "audio/0001.wav",
                "start_s": 1.0,
                "end_s": 2.0,
                "draft_transcript": "uncorrected draft",
                "corrected_transcript": "",  # not yet reviewed -- must be skipped
            },
        ]
    ).to_csv(source_dir / "manifest_draft.csv", index=False)

    raw_dir = tmp_path / "raw"
    manifest_path = ingest_corrected_manifest(
        source_dir / "manifest_draft.csv", raw_dir, split="train"
    )

    out_df = pd.read_csv(manifest_path)
    assert len(out_df) == 1
    assert out_df.iloc[0]["transcript"] == "Hello, world."
    assert (raw_dir / "custom" / "train" / "audio" / "0000.wav").exists()
    assert not (raw_dir / "custom" / "train" / "audio" / "0001.wav").exists()
