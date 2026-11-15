import re
from collections import Counter
from datetime import datetime

LINE = re.compile(r"^(\S+) (DEBUG|INFO|WARN|ERROR) \[([^\]]+)\] (.*)$")


def summarize(lines):
    counts = {k: 0 for k in ("DEBUG", "INFO", "WARN", "ERROR")}
    errs, hours = Counter(), Counter()
    first = None
    invalid = 0
    for line in lines:
        if not line.strip():
            continue
        m = LINE.match(line)
        if m:
            try:
                ts = datetime.strptime(m[1], "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                m = None
        if not m:
            invalid += 1
            continue
        counts[m[2]] += 1
        hours[m[1][:13]] += 1
        if m[2] == "ERROR":
            errs[m[3]] += 1
            if first is None or ts < first[0]:
                first = (ts, m[4])
    busiest = min(hours, key=lambda h: (-hours[h], h)) if hours else None
    return {"counts": counts, "errors_by_component": dict(errs), "busiest_hour": busiest,
            "first_error": first[1] if first else None, "invalid": invalid}
