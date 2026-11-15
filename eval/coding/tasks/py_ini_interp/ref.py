import re

REF = re.compile(r"\$\$|\$\{([^}:]+)(?::([^}]+))?\}")


def _strip_comment(v, lineno):
    if v.startswith('"'):
        out, i = [], 1
        while i < len(v):
            if v.startswith('\\"', i):
                out.append('"')
                i += 2
            elif v[i] == '"':
                rest = v[i + 1:].strip()
                if rest and not rest.startswith(("#", ";")):
                    raise ValueError(f"line {lineno}: junk after quote")
                return "".join(out), True
            else:
                out.append(v[i])
                i += 1
        raise ValueError(f"line {lineno}: unterminated quote")
    m = re.search(r"\s[;#]", v)
    if m:
        v = v[:m.start()]
    return v.strip(), False


def load_ini(text):
    data = {"": {}}
    sec = ""
    last = None
    for n, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            last = None
            continue
        s = raw.strip()
        if s[0] in "#;":
            continue
        if raw[0] in " \t":
            if last is None:
                raise ValueError(f"line {n}: continuation without entry")
            data[sec][last] += "\n" + s
            continue
        if s.startswith("["):
            if not s.endswith("]"):
                raise ValueError(f"line {n}: unclosed section")
            sec = s[1:-1].strip()
            data.setdefault(sec, {})
            last = None
            continue
        m = re.search(r"[=:]", s)
        if not m:
            raise ValueError(f"line {n}: missing separator")
        key = s[:m.start()].strip().lower()
        val, _ = _strip_comment(s[m.end():].strip(), n)
        data[sec][key] = val
        last = key
    if not data[""]:
        del data[""]

    resolved = {}

    def get(section, key, stack):
        if (section, key) in resolved:
            return resolved[(section, key)]
        if (section, key) in stack:
            raise ValueError(f"reference cycle at {section}:{key}")
        if section not in data or key not in data[section]:
            raise ValueError(f"undefined reference {section}:{key}")
        stack = stack | {(section, key)}

        def sub(m):
            if m[0] == "$$":
                return "$"
            if m[2] is not None:
                return get(m[1], m[2].lower(), stack)
            return get(section, m[1].lower(), stack)

        v = REF.sub(sub, data[section][key])
        resolved[(section, key)] = v
        return v

    return {s: {k: get(s, k, frozenset()) for k in d} for s, d in data.items()}
