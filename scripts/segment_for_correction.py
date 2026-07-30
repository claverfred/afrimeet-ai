#!/usr/bin/env python
"""Custom-dataset helper, step 1 (Phase 2): auto-segment a long recording into short
clips with draft transcripts, so you only have to review/correct short segments
instead of transcribing the whole thing from scratch.

    python scripts/segment_for_correction.py --audio path/to/meeting.wav

Writes <out-dir>/audio/0000.wav, 0001.wav, ... and <out-dir>/manifest_draft.csv.
Open that CSV in a spreadsheet, listen to each clip, and correct the
'corrected_transcript' column where the draft (from the model's own current
predictions) is wrong. Then run scripts/ingest_custom_dataset.py on the result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from afrimeet.data.custom_dataset import segment_for_correction
from afrimeet.models.inference import WhisperTranscriber, resolve_model_path
from afrimeet.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-segment a long recording for correction.")
    parser.add_argument("--audio", required=True, help="Path to the long audio file")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: data/external/custom_review/<audio filename>/)",
    )
    parser.add_argument("--model", default=None, help="Model id or path (default: auto-resolved)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    model_id = args.model
    if model_id is None:
        model_id, is_finetuned = resolve_model_path(config)
        print(f"Using {'fine-tuned' if is_finetuned else 'baseline'} model: {model_id}")

    out_dir = args.out_dir
    if out_dir is None:
        stem = Path(args.audio).stem
        out_dir = Path(config["paths"]["data_external"]) / "custom_review" / stem

    transcriber = WhisperTranscriber(model_id, language=config["whisper"]["language"])
    segment_for_correction(args.audio, out_dir, transcriber)


if __name__ == "__main__":
    main()
