from solution import flatten, unflatten

NESTED = {"a": {"b": 1, "c": [10, {"d": None}, []]}, "e": {}, "f": "x"}
FLAT = {"a.b": 1, "a.c.0": 10, "a.c.1.d": None, "a.c.2": [], "e": {}, "f": "x"}


def raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return
    raise AssertionError(f"{fn.__name__}{args} did not raise {exc.__name__}")


def test_flatten():
    assert flatten(NESTED) == FLAT
    assert list(flatten(NESTED)) == list(FLAT)


def test_sep():
    assert flatten({"a": {"b": [1]}}, sep="/") == {"a/b/0": 1}


def test_unflatten():
    assert unflatten(FLAT) == NESTED
    assert unflatten({"x.1": "b", "x.0": "a"}) == {"x": ["a", "b"]}
    assert unflatten({"x.01": 1}) == {"x": {"01": 1}}


def test_roundtrip():
    obj = {"k": [[1, 2], [{"z": [3]}]], "m": {"n": {"o": True}}}
    assert unflatten(flatten(obj)) == obj


def test_errors():
    raises(TypeError, flatten, [1, 2])
    raises(ValueError, flatten, {"a.b": 1})
    raises(ValueError, unflatten, {"x.0": 1, "x.2": 2})
    raises(ValueError, unflatten, {"a": 1, "a.b": 2})
    raises(ValueError, unflatten, {"a.b": 2, "a": 1})


from _runner import main
main(globals())
