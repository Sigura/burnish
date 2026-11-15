import re


def flatten(obj, sep="."):
    if not isinstance(obj, dict):
        raise TypeError("top level must be a dict")
    out = {}

    def walk(v, path):
        if isinstance(v, dict) and v:
            for k, x in v.items():
                if not isinstance(k, str) or sep in k:
                    raise ValueError(f"bad key {k!r}")
                walk(x, path + [k])
        elif isinstance(v, list) and v:
            for i, x in enumerate(v):
                walk(x, path + [str(i)])
        else:
            out[sep.join(path)] = v

    for k, v in obj.items():
        if not isinstance(k, str) or sep in k:
            raise ValueError(f"bad key {k!r}")
        walk(v, [k])
    return out




def unflatten(flat, sep="."):
    root = {}
    for key, value in flat.items():
        parts = key.split(sep)
        node = root
        for p in parts[:-1]:
            nxt = node.get(p)
            if nxt is None:
                nxt = node[p] = {}
            elif not isinstance(nxt, dict):
                raise ValueError(f"conflict at {key}")
            node = nxt
        last = parts[-1]
        if last in node:
            raise ValueError(f"conflict at {key}")
        node[last] = ("leaf", value)

    def build(node):
        if isinstance(node, tuple):
            return node[1]
        keys = list(node)
        if keys and all(re.fullmatch(r"0|[1-9]\d*", k) for k in keys):
            idx = sorted(int(k) for k in keys)
            if idx != list(range(len(idx))):
                raise ValueError("sparse list")
            return [build(node[str(i)]) for i in idx]
        return {k: build(v) for k, v in node.items()}

    return build(root)
