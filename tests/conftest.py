from pathlib import Path

import httpx
import pytest

from burnish.config import load_config
from burnish.engine import EngineError, Reply
from burnish.languagetool import LanguageToolClient
from burnish.pipeline import Pipeline
from burnish.selector import Selection


class StaticSelector:
    def __init__(self, selection):
        self.selection = selection
        self.invalidated = False

    def select(self, model=None):
        return Selection(model, 60, True) if model else self.selection

    def invalidate(self):
        self.invalidated = True


class FakeEngine:
    def __init__(self):
        self.replies = []
        self.truncated = False
        self.down = False
        self.requests = []

    def chat(self, selection, system, user, max_tokens):
        if self.down:
            raise EngineError("engine down")
        self.requests.append({"model": selection.model, "system": system, "user": user})
        return Reply(self.replies.pop(0), self.truncated)

    def supports_gpu(self):
        return True

    def is_pulled(self, model):
        return True


class Harness:
    def __init__(self, cfg):
        self.cfg = cfg
        self.engine = FakeEngine()
        self.lt_matches = []
        self.lt_requests = []
        self.lt_down = False
        self.selector = StaticSelector(Selection("gemma3:4b", 60, True))
        self.pipeline = Pipeline(
            cfg,
            self.engine,
            LanguageToolClient(cfg.languagetool, transport=httpx.MockTransport(self._lt)),
            self.selector,
        )

    def _lt(self, request):
        if self.lt_down:
            raise httpx.ConnectError("down")
        self.lt_requests.append(request.content.decode())
        return httpx.Response(200, json={"matches": self.lt_matches})


@pytest.fixture
def cfg():
    return load_config(Path("/nonexistent"))


@pytest.fixture
def harness(cfg):
    return Harness(cfg)


def lt_match(offset, length, replacement, category="TYPOS", rule="MORFOLOGIK_RULE_EN_US"):
    return {
        "offset": offset,
        "length": length,
        "message": "Possible error",
        "replacements": [{"value": replacement}] if replacement else [],
        "rule": {"id": rule, "category": {"id": category}},
    }
