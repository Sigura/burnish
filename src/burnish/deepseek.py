import os

import httpx

from burnish.config import DeepSeekConfig
from burnish.engine import Reply

HEALTH_TIMEOUT_SECONDS = 5


class DeepSeekError(RuntimeError):
    pass


def load_key(cfg: DeepSeekConfig) -> str | None:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key and cfg.key_file.is_file():
        key = cfg.key_file.read_text(encoding="utf-8").strip()
    return key or None


class DeepSeekClient:
    def __init__(self, cfg: DeepSeekConfig, key: str, transport: httpx.BaseTransport | None = None):
        self.cfg = cfg
        self.http = httpx.Client(
            base_url=cfg.url,
            timeout=cfg.timeout_seconds,
            transport=transport,
            headers={"Authorization": f"Bearer {key}"},
        )

    @property
    def name(self) -> str:
        return f"deepseek:{self.cfg.model}"

    def chat(self, system: str, user: str, max_tokens: int) -> Reply:
        body = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            # Thinking ignores temperature and spends tokens and seconds a copy edit doesn't need.
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            response = self.http.post("/chat/completions", json=body)
        except httpx.HTTPError as e:
            raise DeepSeekError(f"DeepSeek unreachable at {self.cfg.url}: {e}") from e
        if response.status_code != 200:
            raise DeepSeekError(f"DeepSeek returned {response.status_code}: {response.text[:300]}")
        try:
            choice = response.json()["choices"][0]
            content = choice["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise DeepSeekError(f"DeepSeek sent an unexpected answer: {response.text[:300]}") from e
        return Reply(content, choice.get("finish_reason") == "length")

    def check(self) -> str:
        """'ok', or why requests would fail. Listing models costs no tokens."""
        try:
            response = self.http.get("/models", timeout=HEALTH_TIMEOUT_SECONDS)
        except httpx.HTTPError as e:
            return f"unreachable at {self.cfg.url}: {e}"
        if response.status_code != 200:
            return f"returned {response.status_code}: {response.text[:200]}"
        try:
            offered = [m["id"] for m in response.json()["data"]]
        except (ValueError, KeyError, TypeError):
            return f"unexpected /models answer: {response.text[:200]}"
        if self.cfg.model not in offered:
            return f"model {self.cfg.model} is not offered; available: {', '.join(offered)}"
        return "ok"
