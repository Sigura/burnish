import re

import httpx

from burnish.config import LanguageToolConfig


class LanguageToolError(RuntimeError):
    pass


class LanguageToolClient:
    def __init__(self, cfg: LanguageToolConfig, transport: httpx.BaseTransport | None = None):
        self.cfg = cfg
        self.http = httpx.Client(base_url=cfg.url, timeout=cfg.timeout_seconds, transport=transport)

    def check(self, text: str) -> list[dict]:
        try:
            response = self.http.post("/v2/check", data={"language": self.cfg.language, "text": text})
        except httpx.HTTPError as e:
            raise LanguageToolError(f"LanguageTool unreachable at {self.cfg.url}: {e}") from e
        if response.status_code != 200:
            raise LanguageToolError(f"LanguageTool returned {response.status_code}: {response.text[:300]}")
        return response.json()["matches"]


def _sentence_start(text: str, offset: int) -> bool:
    before = text[:offset].rstrip(" \t")
    return not before or before[-1] in ".!?:\n"


# The spell checker flags product names and identifiers the LLM rightly left alone.
def _spelling_exempt(text: str, start: int, word: str, ignore_words: set[str]) -> bool:
    if word.lower() in ignore_words or any(c.isdigit() or c == "_" for c in word):
        return True
    if any(c.isupper() for c in word[1:]):
        return True
    return word[:1].isupper() and not _sentence_start(text, start)


def _contains_word(text: str, word: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text) is not None


def apply_fixes(
    text: str,
    matches: list[dict],
    protected: list[tuple[int, int]],
    cfg: LanguageToolConfig,
    llm_input: str | None = None,
) -> tuple[str, list[dict]]:
    ignore_words = {w.lower() for w in cfg.ignore_words}
    pieces: list[str] = []
    issues: list[dict] = []
    cursor = 0
    delta = 0
    for match in sorted(matches, key=lambda m: m["offset"]):
        start, end = match["offset"], match["offset"] + match["length"]
        if start < cursor or any(start < p_end and p_start < end for p_start, p_end in protected):
            continue
        word = text[start:end]
        category = match["rule"]["category"]["id"]
        replacements = [r["value"] for r in match["replacements"]]
        applied = bool(replacements) and category in cfg.apply_categories
        if applied and category == "TYPOS":
            # A word the LLM saw in its input and left alone is one it judged correct.
            kept_by_llm = llm_input is not None and _contains_word(llm_input, word)
            # Real typos rarely hit the first letter; suggestions that change it (Kubuntu -> Ubuntu,
            # lgtm -> GTM) are usually a different word.
            same_first_letter = word[:1].lower() == replacements[0][:1].lower()
            applied = same_first_letter and not kept_by_llm and not _spelling_exempt(text, start, word, ignore_words)
        issues.append(
            {
                "type": "languagetool",
                "rule": match["rule"]["id"],
                "category": category,
                "message": match["message"],
                "offset": start + delta,
                "length": len(replacements[0]) if applied else len(word),
                "replacements": replacements[:3],
                "applied": applied,
            }
        )
        if applied:
            pieces += [text[cursor:start], replacements[0]]
            cursor = end
            delta += len(replacements[0]) - len(word)
    pieces.append(text[cursor:])
    return "".join(pieces), issues
