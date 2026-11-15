import dataclasses
import types
import typing


class LoadError(ValueError):
    def __init__(self, path, msg):
        super().__init__(f"{path}: {msg}")
        self.path = path


def _load(tp, data, path):
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if tp is type(None) or tp is None:
        if data is not None:
            raise LoadError(path, "expected null")
        return None
    if origin in (typing.Union, types.UnionType):
        rest = [a for a in args if a is not type(None)]
        if data is not None and len(rest) == 1:
            return _load(rest[0], data, path)
        for a in args:
            try:
                return _load(a, data, path)
            except LoadError:
                pass
        raise LoadError(path, f"no union member matched {data!r}")
    if origin is typing.Literal:
        if not any(data == a and type(data) is type(a) for a in args):
            raise LoadError(path, f"expected one of {args}")
        return data
    if tp is bool:
        if not isinstance(data, bool):
            raise LoadError(path, "expected bool")
        return data
    if tp is int:
        if isinstance(data, bool) or not isinstance(data, int):
            raise LoadError(path, "expected int")
        return data
    if tp is float:
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            raise LoadError(path, "expected float")
        return float(data)
    if tp is str:
        if not isinstance(data, str):
            raise LoadError(path, "expected str")
        return data
    if origin is list:
        if not isinstance(data, list):
            raise LoadError(path, "expected list")
        return [_load(args[0], v, f"{path}[{i}]") for i, v in enumerate(data)]
    if origin is tuple:
        if not isinstance(data, list):
            raise LoadError(path, "expected list")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_load(args[0], v, f"{path}[{i}]") for i, v in enumerate(data))
        if len(data) != len(args):
            raise LoadError(path, f"expected {len(args)} items")
        return tuple(_load(a, v, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, data)))
    if origin is dict:
        if not isinstance(data, dict):
            raise LoadError(path, "expected object")
        return {k: _load(args[1], v, f"{path}.{k}") for k, v in data.items()}
    if dataclasses.is_dataclass(tp):
        if not isinstance(data, dict):
            raise LoadError(path, "expected object")
        hints = typing.get_type_hints(tp)
        fields = {f.name: f for f in dataclasses.fields(tp)}
        for k in data:
            if k not in fields:
                raise LoadError(f"{path}.{k}", "unknown field")
        kwargs = {}
        for name, f in fields.items():
            if name in data:
                kwargs[name] = _load(hints[name], data[name], f"{path}.{name}")
            elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                raise LoadError(f"{path}.{name}", "missing field")
        return tp(**kwargs)
    raise TypeError(f"unsupported annotation {tp!r}")


def load(cls, data):
    return _load(cls, data, "$")
