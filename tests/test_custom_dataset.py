from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")  # afrimeet.data.custom_dataset imports WhisperTranscriber
np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from afrimeet.data.custom_dataset import (  # noqa: E402
    _transcribe_full_coverage,
    ingest_corrected_manifest,
)


class _StoppingEarlyFakeTranscriber:
    """Simulates Whisper's observed behavior: a call can stop short of the audio it
    was given, requiring _transcribe_full_coverage to resume from where it left off."""

    def __init__(self, responses: list[list[dict]]):
        self._responses = list(responses)
        self.call_lengths: list[int] = []

    def transcribe_segments(self, audio, sample_rate):
        self.call_lengths.append(len(audio))
        return self._responses.pop(0)


def test_transcribe_full_coverage_resumes_after_early_stop():
    sample_rate = 16_000
    audio = np.zeros(10 * sample_rate, dtype="float32")  # 10s total

    transcriber = _StoppingEarlyFakeTranscriber(
        [
            [{"start": 0.0, "end": 3.0, "text": "first chunk"}],  # stops early, 3s of 10s
            [{"start": 0.0, "end": 7.0, "text": "second chunk"}],  # rest, relative to its own slice
        ]
    )

    segments = _transcribe_full_coverage(transcriber, audio)

    assert len(transcriber.call_lengths) == 2  # resumed exactly once, then reached the end
    assert segments == [
        {"start": 0.0, "end": 3.0, "text": "first chunk"},
        {"start": 3.0, "end": 10.0, "text": "second chunk"},  # offset by where call 1 stopped
    ]


def test_transcribe_full_coverage_skips_ahead_on_empty_response():
    sample_rate = 16_000
    audio = np.zeros(5 * sample_rate, dtype="float32")

    transcriber = _StoppingEarlyFakeTranscriber(
        [
            [],  # nothing returned at all -- must not loop forever
            [{"start": 0.0, "end": 4.0, "text": "recovered"}],
        ]
    )

    segments = _transcribe_full_coverage(transcriber, audio)

    assert len(transcriber.call_lengths) == 2
    assert segments == [{"start": 1.0, "end": 5.0, "text": "recovered"}]


def test_transcribe_full_coverage_backs_off_exponentially_on_repeated_stalls():
    sample_rate = 16_000
    audio = np.zeros(20 * sample_rate, dtype="float32")

    transcriber = _StoppingEarlyFakeTranscriber(
        [
            [],  # stall 1: skip 1.0s   (cursor -> 1.0s)
            [],  # stall 2: skip 2.0s   (cursor -> 3.0s)
            [],  # stall 3: skip 4.0s   (cursor -> 7.0s)
            [{"start": 0.0, "end": 13.0, "text": "recovered"}],  # covers the rest
        ]
    )

    segments = _transcribe_full_coverage(transcriber, audio)

    assert len(transcriber.call_lengths) == 4
    assert segments == [{"start": 7.0, "end": 20.0, "text": "recovered"}]


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
