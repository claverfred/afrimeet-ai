import json
from pathlib import Path

from afrimeet.models.compare import compare_runs


def test_compare_runs_computes_relative_improvement(tmp_path: Path):
    baseline_path = tmp_path / "baseline_summary.json"
    finetuned_path = tmp_path / "finetuned_summary.json"
    baseline_path.write_text(json.dumps({"mean_wer": 0.4, "mean_cer": 0.2}))
    finetuned_path.write_text(json.dumps({"mean_wer": 0.2, "mean_cer": 0.1}))

    df = compare_runs({"baseline": baseline_path, "finetuned": finetuned_path})

    assert df.loc["finetuned", "mean_wer_relative_improvement"] == 0.5
    assert df.loc["finetuned", "mean_cer_relative_improvement"] == 0.5
    assert df.loc["baseline", "mean_wer_relative_improvement"] == 0.0
