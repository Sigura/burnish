import re
from dataclasses import dataclass

PLACEHOLDER = re.compile(r"\[\[P(\d+)\]\]")

_SEGMENT = r"[\w.+-]*[\w+-]"
_MARKER = r"[ \t]*(?:\#{1,6}|[-*+]|\d+[.)]|>)"
_SPANS = [
    r"(?ms:^[ \t]*(?P<fence>`{3,}|~{3,}).*?(?:^[ \t]*(?P=fence)[ \t]*$|\Z))",
    r"(?m:^(?: {4}|\t)[^\n]*$)",
    r"``[^\n]+?``",
    r"`[^`\n]+`",
    r"\[[^\]\n]*\]\([^)\s]*(?:\s+\"[^\"]*\")?\)",
    PLACEHOLDER.pattern,
    r"(?:https?://|www\.)[^\s<>\"'`)\]]*[^\s<>\"'`)\].,;:!?]",
    r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
    r"\b[A-Za-z]:\\[^\s]+",
    rf"(?<![\w/])(?:~|\.\.?)/{_SEGMENT}(?:/{_SEGMENT})*/?",
    rf"(?<![\w/.~])/{_SEGMENT}(?:/{_SEGMENT})*/?",
    rf"(?<![\w/.]){_SEGMENT}(?:/{_SEGMENT})+\.[A-Za-z0-9]+",
]
_MASKED = re.compile("|".join(_SPANS))
_PROTECTED = re.compile("|".join([*_SPANS[:4], rf"(?m:^{_MARKER}(?=[ \t]))", *_SPANS[4:]]))
# Models drop a placeholder at the start of a line, so line markers are cut off instead of masked.
_LINE_PREFIX = re.compile(rf"(?:{_MARKER}[ \t]+)+")


@dataclass(frozen=True)
class Masked:
    text: str
    spans: tuple[str, ...]
    prefixes: tuple[str, ...] = ()

    def lines_intact(self, candidate: str) -> bool:
        return not any(self.prefixes) or candidate.count("\n") + 1 == len(self.prefixes)

    def restore(self, masked_text: str) -> str:
        if any(self.prefixes):
            masked_text = "\n".join(p + line for p, line in zip(self.prefixes, masked_text.split("\n"), strict=True))
        return PLACEHOLDER.sub(lambda m: self.spans[int(m.group(1))], masked_text)

    def placeholders_intact(self, candidate: str) -> bool:
        return sorted(PLACEHOLDER.findall(candidate)) == sorted(PLACEHOLDER.findall(self.text))


def protect(text: str) -> Masked:
    spans: list[str] = []

    def mask(match: re.Match) -> str:
        spans.append(match.group(0))
        return f"[[P{len(spans) - 1}]]"

    prefixes: list[str] = []
    bodies: list[str] = []
    for line in _MASKED.sub(mask, text).split("\n"):
        prefix = m.group(0) if (m := _LINE_PREFIX.match(line)) else ""
        prefixes.append(prefix)
        bodies.append(line[len(prefix) :])
    return Masked("\n".join(bodies), tuple(spans), tuple(prefixes))


def protected_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _PROTECTED.finditer(text)]


def has_prose(masked_text: str) -> bool:
    return re.search(r"[^\W\d_]", PLACEHOLDER.sub("", masked_text)) is not None
