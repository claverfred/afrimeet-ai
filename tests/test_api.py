import pytest

pytest.importorskip("fastapi")
pytest.importorskip("soundfile")

from fastapi.testclient import TestClient  # noqa: E402

from afrimeet.api.main import app  # noqa: E402


def test_health_endpoint_returns_ok():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
