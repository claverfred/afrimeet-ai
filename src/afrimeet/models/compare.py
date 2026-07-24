"""Phase 5 — Compare baseline vs. fine-tuned evaluation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from afrimeet.utils.config import load_config
from afrimeet.utils.logging import logger


def compare_runs(summary_paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for run_name, path in summary_paths.items():
        with Path(path).open("r", encoding="utf-8") as f:
            summary = json.load(f)
        summary["run"] = run_name
        rows.append(summary)
    df = pd.DataFrame(rows).set_index("run")
    if "baseline" in df.index:
        for col in ("mean_wer", "mean_cer"):
            if col in df.columns:
                df[f"{col}_relative_improvement"] = (df.loc["baseline", col] - df[col]) / df.loc[
                    "baseline", col
                ]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs. fine-tuned eval runs.")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help=(
            "name=path pairs, e.g. baseline=reports/metrics/baseline_summary.json "
            "finetuned=reports/metrics/finetuned_summary.json"
        ),
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    summary_paths = dict(pair.split("=", 1) for pair in args.runs)
    df = compare_runs({name: Path(p) for name, p in summary_paths.items()})

    logger.info(f"\n{df.to_string()}")
    out_path = Path(config["paths"]["reports_metrics"]) / "comparison.csv"
    df.to_csv(out_path)
    logger.info(f"Comparison written to {out_path}")


if __name__ == "__main__":
    main()
