import copy

from solution import merge

BASE = {"db": {"host": "localhost", "port": 5432, "opts": {"ssl": False}}, "plugins": ["a"], "debug": True}


def test_deep_merge():
    r = merge(BASE, {"db": {"port": 6543, "opts": {"ssl": True}}})
    assert r == {"db": {"host": "localhost", "port": 6543, "opts": {"ssl": True}}, "plugins": ["a"], "debug": True}


def test_replace_list_and_scalar_with_dict():
    r = merge(BASE, {"plugins": ["b"], "debug": {"level": 2}})
    assert r["plugins"] == ["b"] and r["debug"] == {"level": 2}


def test_delete():
    r = merge(BASE, {"debug": None, "db": {"opts": None}})
    assert "debug" not in r and r["db"] == {"host": "localhost", "port": 5432}


def test_append():
    r = merge(BASE, {"plugins+": ["b"]}, {"plugins+": ["c"], "extra+": [1]})
    assert r["plugins"] == ["a", "b", "c"] and r["extra"] == [1]
    assert "plugins+" not in r and "extra+" not in r


def test_append_type_error():
    try:
        merge(BASE, {"debug+": [1]})
    except TypeError:
        return
    raise AssertionError("expected TypeError")


def test_no_mutation_no_sharing():
    snap = copy.deepcopy(BASE)
    over = {"db": {"opts": {"x": [1]}}}
    r = merge(BASE, over)
    assert BASE == snap
    r["db"]["opts"]["x"].append(2)
    r["plugins"].append("z")
    assert over == {"db": {"opts": {"x": [1]}}} and BASE == snap


def test_no_overrides():
    r = merge(BASE)
    assert r == BASE and r is not BASE and r["db"] is not BASE["db"]


from _runner import main
main(globals())
