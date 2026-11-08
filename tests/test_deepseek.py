import json

import httpx
import pytest
from conftest import Harness
from fastapi.testclient import TestClient

from burnish.deepseek import DeepSeekClient, load_key
from burnish.server import create_app


class FakeApi:
    def __init__(self):
        self.replies = []
        self.down = False
        self.requests = []
        self.models = ["deepseek-flash", "deepseek-v4-pro"]

    def __call__(self, request):
        if self.down:
            raise httpx.ConnectError("down")
        if request.url.path == "/models":
            return httpx.Response(200, json={"data": [{"id": m} for m in self.models]})
        self.requests.append(request)
        content, finish = self.replies.pop(0)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}]})


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def api():
    return FakeApi()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def remote_harness(cfg, api, clock):
    h = Harness(cfg)
    h.pipeline.remote = DeepSeekClient(cfg.deepseek, "sk-test", transport=httpx.MockTransport(api))
    h.pipeline.clock = clock
    return h


def test_deepseek_rewrites_first_and_the_local_model_is_not_used(remote_harness, api):
    api.replies = [("She goes home.", "stop")]
    result = remote_harness.pipeline.refine("She go home.", "fix")
    assert result.text == "She goes home."
    assert result.model == "deepseek:deepseek-flash"
    assert remote_harness.engine.requests == []
    body = json.loads(api.requests[0].content)
    assert api.requests[0].headers["Authorization"] == "Bearer sk-test"
    assert body["model"] == "deepseek-flash" and body["temperature"] == 0
    assert body["thinking"] == {"type": "disabled"}


def test_rejected_deepseek_rewrite_goes_to_the_local_model(remote_harness, api):
    api.replies = [("Something else entirely, unrelated to the input.", "stop")]
    remote_harness.engine.replies = ["She goes home."]
    result = remote_harness.pipeline.refine("She go home.", "fix")
    assert result.text == "She goes home."
    assert result.model == "gemma3:4b"
    assert result.issues[0]["type"] == "drift_guard"
    assert "deepseek:deepseek-flash" in result.issues[0]["message"]


def test_failed_api_falls_back_and_is_skipped_until_retry(remote_harness, api, clock, cfg):
    api.down = True
    remote_harness.engine.replies = ["She goes home.", "She goes home now."]
    result = remote_harness.pipeline.refine("She go home.")
    assert result.model == "gemma3:4b"
    assert [i["type"] for i in result.issues] == ["deepseek_error"]
    assert len(remote_harness.engine.requests) == 2

    api.down = False
    api.replies = [("He goes home.", "stop")]
    remote_harness.engine.replies = ["He goes home."]
    assert remote_harness.pipeline.refine("He go home.", "fix").model == "gemma3:4b"

    clock.now = cfg.deepseek.retry_after_seconds
    assert remote_harness.pipeline.refine("He go home.", "fix").model == "deepseek:deepseek-flash"


def test_explicit_model_skips_deepseek(remote_harness, api):
    remote_harness.engine.replies = ["She goes home."]
    result = remote_harness.pipeline.refine("She go home.", "fix", model="qwen3:8b")
    assert result.model == "qwen3:8b"
    assert api.requests == []


def test_truncated_deepseek_answer_is_rejected(remote_harness, api):
    api.replies = [("She goes", "length")]
    remote_harness.engine.replies = ["She goes home."]
    result = remote_harness.pipeline.refine("She go home.", "fix")
    assert result.model == "gemma3:4b"
    assert result.issues[0]["type"] == "truncated_guard"


def test_health_reports_deepseek(cfg, remote_harness, api, harness):
    client = TestClient(create_app(cfg, remote_harness.pipeline), base_url="http://127.0.0.1:8765")
    assert client.get("/health").json()["deepseek"] == "ok"
    api.models = ["deepseek-v4-pro"]
    assert "not offered" in client.get("/health").json()["deepseek"]
    plain = TestClient(create_app(cfg, harness.pipeline), base_url="http://127.0.0.1:8765")
    assert plain.get("/health").json()["deepseek"].startswith("no key")


def test_key_comes_from_the_environment_then_the_file(cfg, monkeypatch, tmp_path):
    key_file = tmp_path / "api_key"
    key_file.write_text("sk-file\n")
    deepseek = cfg.deepseek.__class__(**{**cfg.deepseek.__dict__, "key_file": key_file})
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert load_key(deepseek) == "sk-file"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
    assert load_key(deepseek) == "sk-env"
    key_file.unlink()
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    assert load_key(deepseek) is None
