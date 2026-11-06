import pytest
from fastapi.testclient import TestClient

from burnish.server import create_app


@pytest.fixture
def client(cfg, harness):
    return TestClient(create_app(cfg, harness.pipeline), base_url="http://127.0.0.1:8765")


def test_refine_shape_and_default_mode_is_all(client, harness):
    harness.engine.replies = ["She goes home.", "She goes home now."]
    response = client.post("/refine", json={"text": "She go home."})
    assert response.status_code == 200
    assert set(response.json()) == {"text", "diff", "model", "issues"}
    assert len(harness.engine.requests) == 2


def test_bad_mode_is_rejected(client):
    assert client.post("/refine", json={"text": "x", "mode": "nope"}).status_code == 422


def test_too_long_text_is_rejected(client, cfg):
    response = client.post("/refine", json={"text": "a" * (cfg.server.max_chars + 1)})
    assert response.status_code == 413


def test_service_failure_is_503(client, harness):
    harness.engine.down = True
    response = client.post("/refine", json={"text": "She go home."})
    assert response.status_code == 503
    assert "engine down" in response.json()["detail"]


def test_foreign_host_header_is_rejected(cfg, harness):
    foreign = TestClient(create_app(cfg, harness.pipeline), base_url="http://evil.example")
    assert foreign.post("/refine", json={"text": "x"}).status_code == 400


def test_health_reports_each_service(client, harness):
    harness.lt_down = True
    body = client.get("/health").json()
    assert body["gpu_offload"] is True
    assert body["model_downloaded"] is True
    assert "LanguageTool" in body["languagetool"]
