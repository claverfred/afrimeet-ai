#!/usr/bin/env python
"""CLI entry point for Phases 3 & 5: evaluate a Whisper model's WER/CER on a manifest.

Named evaluate_model.py rather than evaluate.py deliberately: when this script runs,
Python puts its own directory (scripts/) first on sys.path, so a file literally named
evaluate.py here would shadow the third-party `evaluate` package (used by
afrimeet.models.train) for any other script run from this same directory.

Used for both the baseline evaluation (Phase 3) and the fine-tuned model evaluation
(Phase 5) — just point --model / --run-name at the model and label you want:

    python scripts/evaluate_model.py --manifest data/processed/.../test/manifest.csv \\
        --model openai/whisper-small --run-name baseline

    python scripts/evaluate_model.py --manifest data/processed/.../test/manifest.csv \\
        --model models/finetuned/afrimeet-whisper-small-sw-en --run-name finetuned
"""

from afrimeet.models.evaluate import main

if __name__ == "__main__":
    main()
