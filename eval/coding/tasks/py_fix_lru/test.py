from cache import ttl_lru


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def counting():
    calls = []

    def f(*args, **kwargs):
        calls.append((args, tuple(sorted(kwargs.items()))))
        return args[0] if args else None
    return f, calls


def test_falsy_values_cached():
    f, calls = counting()
    g = ttl_lru()(f)
    for v in (None, 0, "", None, 0, ""):
        g(v)
    assert len(calls) == 3, calls
    assert g.cache_info() == (3, 3, 3)


def test_kwargs_order():
    f, calls = counting()
    g = ttl_lru()(f)
    g(1, a=1, b=2)
    g(1, b=2, a=1)
    assert len(calls) == 1
    g(1, 2)
    g(1, b=2)
    assert len(calls) == 3


def test_lru_eviction_uses_recency():
    f, calls = counting()
    g = ttl_lru(maxsize=2)(f)
    g(1); g(2); g(1); g(3)
    assert g.cache_info()[2] == 2
    n = len(calls)
    g(1)
    assert len(calls) == n, "1 was used recently and must still be cached"
    g(2)
    assert len(calls) == n + 1, "2 was least recently used and must have been evicted"


def test_ttl():
    clk = Clock()
    f, calls = counting()
    g = ttl_lru(ttl=10, clock=clk)(f)
    g(1)
    clk.t = 9.99
    g(1)
    assert len(calls) == 1
    clk.t = 10.0
    g(1)
    assert len(calls) == 2
    clk.t = 19.0
    g(1)
    assert len(calls) == 2, "refreshed entry restarts its ttl"
    assert g.cache_info() == (2, 2, 1)


def test_exceptions_not_cached():
    n = 0

    @ttl_lru()
    def h(x):
        nonlocal n
        n += 1
        if n == 1:
            raise RuntimeError("boom")
        return x

    try:
        h(5)
    except RuntimeError:
        pass
    assert h(5) == 5 and h(5) == 5
    assert n == 2
    assert h.cache_info() == (1, 2, 1)


def test_clear_resets():
    f, calls = counting()
    g = ttl_lru()(f)
    g(1); g(1)
    g.cache_clear()
    assert g.cache_info() == (0, 0, 0)
    g(1)
    assert len(calls) == 2


def test_wraps():
    @ttl_lru()
    def named(x):
        "doc"
        return x
    assert named.__name__ == "named" and named.__doc__ == "doc"


from _runner import main
main(globals())
