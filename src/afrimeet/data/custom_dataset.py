"""Phase 2 — Custom Dataset workflow (real conference/institutional recordings, as
originally scoped in the concept note, alongside the public Common Voice/FLEURS data).

Two steps, since a long recording can't be corrected as one giant transcript and
can't be trained on as one giant clip (the pipeline expects short, per-utterance
audio like Common Voice/FLEURS provide):

1. `segment_for_correction()` — auto-segments a long audio file into short clips using
   the current model's own predicted segment boundaries, with a draft transcript per
   clip. A human reviews/corrects each short clip's text instead of transcribing the
   whole recording from scratch.
2. `ingest_corrected_manifest()` — takes the corrected manifest and copies it into
   data/raw/custom/<split>/ in the same format afrimeet.data.download produces, so it
   flows through the exact same cleaning pipeline (afrimeet.data.preprocess) as every
   other dataset.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import soundfile as sf

from afrimeet.models.inference import WHISPER_SAMPLE_RATE, WhisperTranscriber
from afrimeet.utils.logging import logger


def segment_for_correction(
    audio_path: str | Path,
    out_dir: str | Path,
    transcriber: WhisperTranscriber,
) -> Path:
    """Splits `audio_path` into per-segment clips under `out_dir/audio/` using the
    model's own long-form segment boundaries, and writes `out_dir/manifest_draft.csv`
    with columns (audio_path, start_s, end_s, draft_transcript, corrected_transcript).
    `corrected_transcript` is pre-filled with the draft so a reviewer only has to edit
    what's wrong, not retype everything -- open the CSV in a spreadsheet, listen to
    each clip, and fix that column.

    Returns the path to the written manifest.
    """
    audio_path = Path(audio_path)
    out_dir = Path(out_dir)
    audio_out_dir = out_dir / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    signal, sample_rate = sf.read(audio_path, dtype="float32")
    prepared = transcriber.prepare_audio(signal, sample_rate)  # downmixed + resampled to 16kHz

    logger.info(f"Transcribing {audio_path} to find segment boundaries (this can take a while) ...")
    segments = transcriber.transcribe_segments(prepared, WHISPER_SAMPLE_RATE)
    if not segments:
        raise RuntimeError(
            f"No segments detected in {audio_path} -- the model produced no timestamped "
            "output at all (e.g. silence, or a decoding failure). Nothing to correct."
        )

    rows = []
    for i, seg in enumerate(segments):
        start_sample = int(seg["start"] * WHISPER_SAMPLE_RATE)
        end_sample = int(seg["end"] * WHISPER_SAMPLE_RATE)
        clip = prepared[start_sample:end_sample]
        if len(clip) == 0:
            continue

        clip_path = audio_out_dir / f"{i:04d}.wav"
        sf.write(clip_path, clip, WHISPER_SAMPLE_RATE)
        rows.append(
            {
                "audio_path": str(clip_path.relative_to(out_dir)),
                "start_s": seg["start"],
                "end_s": seg["end"],
                "draft_transcript": seg["text"],
                "corrected_transcript": seg["text"],  # pre-filled; edit in place
            }
        )

    manifest_path = out_dir / "manifest_draft.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    logger.info(
        f"Wrote {len(rows)} segments to {audio_out_dir} and a draft manifest to {manifest_path}. "
        "Open that CSV, listen to each clip, and correct the 'corrected_transcript' column "
        "where the draft is wrong, then run scripts/ingest_custom_dataset.py on it."
    )
    return manifest_path


def ingest_corrected_manifest(
    corrected_manifest_path: str | Path,
    raw_dir: str | Path,
    split: str = "train",
) -> Path:
    """Copies a corrected manifest (from segment_for_correction, after review) and its
    audio clips into `raw_dir/custom/<split>/`, in the same (audio_path, transcript,
    duration_s) format afrimeet.data.download produces for every other dataset -- so
    afrimeet.data.preprocess's existing cleaning/trimming/duration-filtering applies
    uniformly, and afrimeet.models.train.build_dataset() picks it up automatically
    (it globs every dataset under data/processed/*, not just the ones named in
    configs/config.yaml).

    Rows with an empty corrected_transcript are skipped (not yet reviewed). Returns
    the path to the written manifest.
    """
    corrected_manifest_path = Path(corrected_manifest_path)
    source_dir = corrected_manifest_path.parent
    df = pd.read_csv(corrected_manifest_path)

    out_split_dir = Path(raw_dir) / "custom" / split
    out_audio_dir = out_split_dir / "audio"
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0
    for _, row in df.iterrows():
        transcript = str(row.get("corrected_transcript", "")).strip()
        if not transcript or transcript.lower() == "nan":
            skipped += 1
            continue

        src_audio = source_dir / row["audio_path"]
        dst_audio = out_audio_dir / Path(row["audio_path"]).name
        shutil.copy2(src_audio, dst_audio)

        signal, sample_rate = sf.read(dst_audio, dtype="float32")
        rows.append(
            {
                "audio_path": str(dst_audio.relative_to(out_split_dir)),
                "transcript": transcript,
                "duration_s": len(signal) / sample_rate,
            }
        )

    manifest_path = out_split_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    logger.info(
        f"Ingested {len(rows)} corrected segments to {manifest_path} ({skipped} skipped, "
        "empty corrected_transcript). Add a `custom` entry (local: true, splits: "
        f"[{split}]) to configs/config.yaml's data.datasets if not already there, bump "
        "data.schema_version, then re-run scripts/prepare_dataset.py."
    )
    return manifest_path
