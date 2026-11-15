import functools
import re

NUM = r"0|[1-9]\d*"
IDENT = r"[0-9A-Za-z-]+"
FULL = re.compile(rf"^[v=]?({NUM})\.({NUM})\.({NUM})(?:-({IDENT}(?:\.{IDENT})*))?(?:\+({IDENT}(?:\.{IDENT})*))?$")
PART = re.compile(rf"^[v=]?(\*|[xX]|{NUM})(?:\.(\*|[xX]|{NUM})(?:\.(\*|[xX]|{NUM})(?:-({IDENT}(?:\.{IDENT})*))?(?:\+{IDENT}(?:\.{IDENT})*)?)?)?$")


@functools.total_ordering
class Version:
    def __init__(self, major, minor, patch, pre=()):
        self.major, self.minor, self.patch, self.pre = major, minor, patch, tuple(pre)

    def _key(self):
        pre = tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in self.pre)
        return (self.major, self.minor, self.patch, 0 if self.pre else 1, pre)

    def __eq__(self, other):
        return self._key() == other._key()

    def __lt__(self, other):
        return self._key() < other._key()

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        return f"{self.major}.{self.minor}.{self.patch}" + ("-" + ".".join(self.pre) if self.pre else "")


def parse(v):
    m = FULL.match(v.strip())
    if not m:
        raise ValueError(v)
    pre = m[4].split(".") if m[4] else []
    for p in pre:
        if p.isdigit() and len(p) > 1 and p[0] == "0":
            raise ValueError(v)
    return Version(int(m[1]), int(m[2]), int(m[3]), pre)


def _partial(s):
    m = PART.match(s)
    if not m:
        raise ValueError(s)
    parts = []
    for g in m.groups()[:3]:
        if g is None or g in "*xX":
            break
        parts.append(int(g))
    pre = m[4].split(".") if m[4] and len(parts) == 3 else []
    return parts, pre


def _lo(parts, pre=()):
    return Version(*(parts + [0] * (3 - len(parts))), pre)


def _next(parts):
    if len(parts) == 1:
        return Version(parts[0] + 1, 0, 0)
    return Version(parts[0], parts[1] + 1, 0)


def _comparator(tok):
    m = re.match(r"^(<=|>=|<|>|=)?(.*)$", tok)
    op, rest = m[1] or "", m[2]
    if rest[:1] in ("<", ">", "="):
        raise ValueError(tok)
    if rest.startswith(("~", "^")):
        if op:
            raise ValueError(tok)
        parts, pre = _partial(rest[1:])
        if not parts:
            return []
        lo = _lo(parts, pre)
        if rest[0] == "~":
            hi = _next(parts if len(parts) == 1 else parts[:2])
        elif parts[0] != 0 or len(parts) == 1:
            hi = Version(parts[0] + 1, 0, 0)
        elif parts[1] != 0 or len(parts) == 2:
            hi = Version(0, parts[1] + 1, 0)
        else:
            hi = Version(0, 0, parts[2] + 1)
        return [(">=", lo), ("<", hi)]
    parts, pre = _partial(rest)
    if len(parts) == 3:
        return [(op or "=", Version(*parts, pre))]
    if not parts:
        return [("<", Version(0, 0, 0))] if op in ("<", ">") else []
    return {
        "": [(">=", _lo(parts)), ("<", _next(parts))],
        "=": [(">=", _lo(parts)), ("<", _next(parts))],
        ">": [(">=", _next(parts))],
        ">=": [(">=", _lo(parts))],
        "<": [("<", _lo(parts))],
        "<=": [("<", _next(parts))],
    }[op]


def _parse_set(s):
    s = s.strip()
    m = re.match(r"^(\S+)\s+-\s+(\S+)$", s)
    if m:
        lp, lpre = _partial(m[1])
        hp, hpre = _partial(m[2])
        comps = [(">=", _lo(lp, lpre))] if lp else []
        if len(hp) == 3:
            comps.append(("<=", Version(*hp, hpre)))
        elif hp:
            comps.append(("<", _next(hp)))
        return comps
    toks = re.sub(r"(<=|>=|<|>|=)\s+", r"\1", s).split()
    if "-" in toks:
        raise ValueError(s)
    comps = []
    for t in toks:
        comps += _comparator(t)
    return comps


def _test(v, op, c):
    return {"=": v == c, "<": v < c, "<=": v <= c, ">": v > c, ">=": v >= c}[op]


def _sat_set(v, comps):
    if not all(_test(v, op, c) for op, c in comps):
        return False
    if v.pre:
        return any(c.pre and (c.major, c.minor, c.patch) == (v.major, v.minor, v.patch) for _, c in comps)
    return True


def _range(rng):
    return [_parse_set(s) for s in rng.split("||")]


def satisfies(version, rng):
    v = parse(version)
    return any(_sat_set(v, comps) for comps in _range(rng))


def max_satisfying(versions, rng):
    sets = _range(rng)
    best = None
    for s in versions:
        v = parse(s)
        if any(_sat_set(v, c) for c in sets) and (best is None or v > best[0]):
            best = (v, s)
    return best[1] if best else None
