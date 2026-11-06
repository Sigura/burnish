import httpx
import pytest
from fastapi.testclient import TestClient

from burnish.models import ModelStore
from burnish.server import create_app


def test_real_services_fix_a_sentence(cfg):
    try:
        httpx.get(f"{cfg.languagetool.url}/v2/languages", timeout=1)
    except httpx.HTTPError:
        pytest.skip("LanguageTool is not running")
    store = ModelStore(cfg)
    tiers = (*cfg.selection.gpu_tiers, cfg.selection.cpu)
    if any(store.path(tier.model) is None for tier in tiers):
        pytest.skip("Default models are not downloaded")
    app = TestClient(create_app(cfg), base_url="http://127.0.0.1:8765")
    response = app.post("/refine", json={"text": "She go to the store with `my_var` yesterday.\n"})
    assert response.status_code == 200
    body = response.json()
    assert body["text"].endswith("\n")
    assert "`my_var`" in body["text"]
    assert "She go " not in body["text"]
