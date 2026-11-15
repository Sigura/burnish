import copy


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        if k.endswith("+"):
            name = k[:-1]
            cur = out.get(name, [])
            if not isinstance(cur, list):
                raise TypeError(f"{name} is not a list")
            out[name] = cur + copy.deepcopy(list(v))
        elif v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def merge(base, *overrides):
    out = copy.deepcopy(base)
    for o in overrides:
        out = _merge(out, o)
    return out
