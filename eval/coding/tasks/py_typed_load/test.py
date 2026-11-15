from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from solution import LoadError, load


@dataclass
class Port:
    number: int
    proto: Literal["tcp", "udp"] = "tcp"


@dataclass
class Server:
    host: str
    ports: list[Port] = field(default_factory=list)
    weight: float = 1.0
    tags: dict[str, str | int] = field(default_factory=dict)
    backup: Optional[Server] = None


@dataclass
class Config:
    name: str
    servers: list[Server]
    version: tuple[int, int, int] = (1, 0, 0)
    flags: tuple[bool, ...] = ()
    enabled: bool = True
    note: str | None = None


def err(cls, data, path):
    try:
        load(cls, data)
    except LoadError as e:
        assert isinstance(e, ValueError)
        assert e.path == path, (e.path, path)
        assert str(e).startswith(path + ": "), str(e)
        return
    raise AssertionError(f"no LoadError, expected at {path}")


GOOD = {
    "name": "web",
    "servers": [
        {"host": "a", "ports": [{"number": 80}, {"number": 53, "proto": "udp"}], "weight": 2},
        {"host": "b", "tags": {"zone": "eu", "rack": 4}, "backup": {"host": "c"}},
    ],
    "version": [2, 1, 0],
    "flags": [True, False],
}


def test_good():
    c = load(Config, GOOD)
    assert c == Config(
        name="web",
        servers=[
            Server("a", [Port(80), Port(53, "udp")], 2.0),
            Server("b", tags={"zone": "eu", "rack": 4}, backup=Server("c")),
        ],
        version=(2, 1, 0),
        flags=(True, False),
    )
    assert isinstance(c.servers[0].weight, float)
    assert isinstance(c.version, tuple) and isinstance(c.flags, tuple)
    assert isinstance(c.servers[1].backup, Server)


def test_defaults_not_shared():
    a = load(Config, {"name": "x", "servers": [{"host": "h"}]})
    b = load(Config, {"name": "y", "servers": [{"host": "h"}]})
    a.servers[0].ports.append(Port(1))
    assert b.servers[0].ports == []


def test_none_optional():
    c = load(Config, {"name": "x", "servers": [], "note": None})
    assert c.note is None


def test_missing_field():
    err(Config, {"name": "x", "servers": [{}, {"host": 1}]}, "$.servers[0].host")


def test_wrong_scalar():
    err(Config, {"name": "x", "servers": [{"host": "a"}, {"host": 1}]}, "$.servers[1].host")
    err(Config, {"name": "x", "servers": [{"host": "a", "ports": [{"number": 1}, {"number": "2"}]}]},
        "$.servers[0].ports[1].number")


def test_bool_not_int():
    err(Config, {"name": "x", "servers": [{"host": "a", "ports": [{"number": True}]}]}, "$.servers[0].ports[0].number")
    err(Config, {"name": "x", "servers": [{"host": "a", "weight": False}]}, "$.servers[0].weight")
    err(Config, {"name": "x", "servers": [], "enabled": 1}, "$.enabled")


def test_literal():
    err(Config, {"name": "x", "servers": [{"host": "a", "ports": [{"number": 1, "proto": "sctp"}]}]},
        "$.servers[0].ports[0].proto")


def test_union_and_dict_key_path():
    err(Config, {"name": "x", "servers": [{"host": "a", "tags": {"ok": 1, "bad": 1.5}}]}, "$.servers[0].tags.bad")


def test_union_order():
    @dataclass
    class U:
        v: int | str
        w: float | int

    u = load(U, {"v": "s", "w": 3})
    assert u.v == "s" and u.w == 3.0 and isinstance(u.w, float)


def test_unknown_key():
    err(Config, {"name": "x", "servers": [], "zzz": 1, "aaa": 2}, "$.zzz")


def test_tuple_fixed_length():
    err(Config, {"name": "x", "servers": [], "version": [1, 2]}, "$.version")
    err(Config, {"name": "x", "servers": [], "version": [1, 2, "3"]}, "$.version[2]")
    err(Config, {"name": "x", "servers": [], "flags": [True, 0]}, "$.flags[1]")


def test_container_types():
    err(Config, {"name": "x", "servers": {"host": "a"}}, "$.servers")
    err(Config, ["x"], "$")
    err(Config, {"name": "x", "servers": [{"host": "a", "backup": {"host": "b", "backup": {"host": 5}}}]},
        "$.servers[0].backup.backup.host")


from _runner import main
main(globals())
