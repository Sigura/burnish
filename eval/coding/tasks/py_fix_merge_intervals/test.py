import random

from intervals import IntervalSet


def brute(ops, lo, hi):
    cov = set()
    for op, s, e in ops:
        if op == "add":
            cov |= set(range(s, e))
        else:
            cov -= set(range(s, e))
    return cov


def runs(points):
    out = []
    for p in sorted(points):
        if out and out[-1][1] == p:
            out[-1][1] = p + 1
        else:
            out.append([p, p + 1])
    return [tuple(r) for r in out]


def test_touching_merge():
    s = IntervalSet([(1, 3), (3, 5)])
    assert list(s) == [(1, 5)]
    s.add(5, 7)
    assert list(s) == [(1, 7)]
    s.add(0, 1)
    assert list(s) == [(0, 7)]


def test_empty_ignored():
    s = IntervalSet([(1, 3)])
    s.add(5, 5)
    s.add(9, 2)
    assert list(s) == [(1, 3)]
    s.remove(2, 2)
    assert list(s) == [(1, 3)]


def test_contains():
    s = IntervalSet([(1, 3), (10, 20)])
    assert s.contains(1) and s.contains(2) and not s.contains(3)
    assert s.contains(10) and s.contains(19) and not s.contains(20) and not s.contains(0)


def test_free_clipped():
    s = IntervalSet([(0, 5), (8, 10), (20, 30)])
    assert s.free(3, 25) == [(5, 8), (10, 20)]
    assert s.free(-5, 2) == [(-5, 0)]
    assert s.free(11, 15) == [(11, 15)]
    assert s.free(21, 25) == []
    assert s.free(30, 30) == []


def test_random_against_bruteforce():
    rng = random.Random(7)
    for _ in range(300):
        s = IntervalSet()
        ops = []
        for _ in range(rng.randint(1, 12)):
            a, b = rng.randint(0, 40), rng.randint(0, 40)
            op = rng.choice(["add", "add", "remove"])
            getattr(s, op)(a, b)
            ops.append((op, a, b))
        cov = brute(ops, 0, 40)
        assert list(s) == runs(cov), (ops, list(s))
        assert len(s) == len(runs(cov))
        for x in range(-1, 42):
            assert s.contains(x) == (x in cov), (ops, x)
        lo, hi = sorted((rng.randint(-2, 42), rng.randint(-2, 42)))
        assert s.free(lo, hi) == runs(set(range(lo, hi)) - cov), (ops, lo, hi)


from _runner import main
main(globals())
