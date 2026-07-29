from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("soundfile")

from fastapi.testclient import TestClient  # noqa: E402

from afrimeet.api.main import app, resolve_model_path  # noqa: E402


def _fake_config(tmp_path: Path) -> dict:
    return {
        "paths": {"models_finetuned": str(tmp_path / "finetuned")},
        "whisper": {
            "finetuned_model_name": "afrimeet-whisper-small-sw-en",
            "baseline_model": "openai/whisper-small",
        },
    }


def test_resolve_model_path_falls_back_to_baseline_when_no_weights(tmp_path: Path):
    config = _fake_config(tmp_path)
    model_id, is_finetuned = resolve_model_path(config)
    assert model_id == "openai/whisper-small"
    assert is_finetuned is False


def test_resolve_model_path_uses_finetuned_when_weights_present(tmp_path: Path):
    config = _fake_config(tmp_path)
    finetuned_dir = tmp_path / "finetuned" / "afrimeet-whisper-small-sw-en"
    finetuned_dir.mkdir(parents=True)
    (finetuned_dir / "model.safetensors").write_bytes(b"fake weights")

    model_id, is_finetuned = resolve_model_path(config)
    assert model_id == str(finetuned_dir)
    assert is_finetuned is True


def test_health_endpoint_reports_status_and_model():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model" in body
    assert "finetuned" in body
