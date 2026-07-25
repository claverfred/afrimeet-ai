"""Phase 3 & 5 — Evaluate a Whisper model (baseline or fine-tuned) on a manifest.

Computes per-example and aggregate Word Error Rate (WER) and Character Error Rate
(CER) using `jiwer`, and writes results to `reports/metrics/`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jiwer
import pandas as pd

from afrimeet.data.dataset import ManifestAudioDataset
from afrimeet.models.inference import WhisperTranscriber
from afrimeet.utils.config import load_config
from afrimeet.utils.logging import logger

NORMALIZATION = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def evaluate_model(
    model_id_or_path: str,
    manifest_path: str | Path,
    language: str = "sw",
    max_examples: int | None = None,
) -> pd.DataFrame:
    dataset = ManifestAudioDataset(manifest_path)
    transcriber = WhisperTranscriber(model_id_or_path, language=language)

    n = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    results = []
    for i in range(n):
        example = dataset[i]
        start = time.perf_counter()
        hypothesis = transcriber.transcribe(example["audio"], example["sample_rate"])
        latency_s = time.perf_counter() - start

        reference = example["transcript"]
        wer = jiwer.wer(
            reference,
            hypothesis,
            reference_transform=NORMALIZATION,
            hypothesis_transform=NORMALIZATION,
        )
        cer = jiwer.cer(reference, hypothesis)

        results.append(
            {
                "audio_path": example["audio_path"],
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": wer,
                "cer": cer,
                "latency_s": latency_s,
            }
        )
        if (i + 1) % 20 == 0:
            logger.info(f"  evaluated {i + 1}/{n}")

    return pd.DataFrame(results)


def summarize(results_df: pd.DataFrame) -> dict:
    return {
        "n_examples": len(results_df),
        "mean_wer": float(results_df["wer"].mean()),
        "mean_cer": float(results_df["cer"].mean()),
        "mean_latency_s": float(results_df["latency_s"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Whisper model's WER/CER.")
    parser.add_argument(
        "--model", default=None, help="Model id or path (defaults to config baseline_model)"
    )
    parser.add_argument("--manifest", required=True, help="Path to a processed manifest.csv")
    parser.add_argument("--run-name", default="baseline", help="Label used for output files")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    model_id = args.model or config["whisper"]["baseline_model"]
    language = config["whisper"]["language"]

    logger.info(f"Evaluating {model_id} on {args.manifest}")
    results_df = evaluate_model(model_id, args.manifest, language, args.max_examples)
    summary = summarize(results_df)
    logger.info(f"Results: {summary}")

    metrics_dir = Path(config["paths"]["reports_metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(metrics_dir / f"{args.run_name}_per_example.csv", index=False)
    with (metrics_dir / f"{args.run_name}_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"model": model_id, "manifest": str(args.manifest), **summary}, f, indent=2)


if __name__ == "__main__":
    main()
