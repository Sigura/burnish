import re

UNITS = {"d": 86400, "h": 3600, "m": 60, "s": 1}
PART = re.compile(r"(\d+)([dhms])", re.I)


def parse_duration(s):
    if not s or s != s.strip():
        raise ValueError(s)
    pos, total, last = 0, 0, -1
    order = "dhms"
    while pos < len(s):
        m = PART.match(s, pos)
        if not m:
            raise ValueError(s)
        u = m[2].lower()
        if order.index(u) <= last:
            raise ValueError(s)
        last = order.index(u)
        total += int(m[1]) * UNITS[u]
        pos = m.end()
        if pos < len(s) and s[pos] == " ":
            pos += 1
            if pos == len(s):
                raise ValueError(s)
    return total


def format_duration(seconds):
    if seconds < 0:
        raise ValueError(seconds)
    if seconds == 0:
        return "0s"
    out = []
    for u, n in UNITS.items():
        q, seconds = divmod(seconds, n)
        if q:
            out.append(f"{q}{u}")
    return "".join(out)
