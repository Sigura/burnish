from solution import max_satisfying, parse, satisfies


def test_parse_and_order():
    order = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2", "1.0.0-beta.11",
             "1.0.0-rc.1", "1.0.0", "1.0.1", "1.1.0", "2.0.0"]
    vs = [parse(v) for v in order]
    for a, b in zip(vs, vs[1:]):
        assert a < b and b > a and not a == b
    assert sorted(reversed(vs)) == vs
    assert parse("v1.2.3+build.5") == parse("=1.2.3")
    assert parse("1.0.0-1") < parse("1.0.0-a")
    assert parse("1.0.0-a.b") > parse("1.0.0-a")


def test_parse_invalid():
    for bad in ["1.2", "01.2.3", "1.02.3", "1.2.3-01", "1.2.3-", "1.2.3+", "1.2.3-a..b", "a.b.c", "1.2.3.4", ""]:
        try:
            parse(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")
    parse("1.2.3-0a")


CASES = [
    ("1.2.3", "1.2.3", True), ("1.2.4", "1.2.3", False), ("1.2.3", "=v1.2.3", True),
    ("1.9.9", "1.x", True), ("2.0.0", "1", False), ("1.2.9", "1.2.*", True), ("1.3.0", "1.2.x", False),
    ("5.0.0", "*", True), ("5.0.0", "", True), ("5.0.0-a", "*", False),
    ("1.3.0", ">1.2", True), ("1.2.9", ">1.2", False), ("1.2.9", "<=1.2", True), ("1.3.0", "<=1.2", False),
    ("1.1.9", "<1.2", True), ("1.2.0", "<1.2", False), ("1.2.0", ">=1.2", True),
    ("1.2.9", "~1.2.3", True), ("1.3.0", "~1.2.3", False), ("1.2.2", "~1.2.3", False), ("1.9.0", "~1", True),
    ("1.9.9", "^1.2.3", True), ("2.0.0", "^1.2.3", False), ("0.2.9", "^0.2.3", True), ("0.3.0", "^0.2.3", False),
    ("0.0.3", "^0.0.3", True), ("0.0.4", "^0.0.3", False), ("1.9.0", "^1.2.x", True), ("0.0.9", "^0.0.x", True),
    ("0.1.0", "^0.0.x", False), ("0.9.0", "^0.x", True), ("1.0.0", "^0.x", False),
    ("2.3.4", "1.2.3 - 2.3.4", True), ("2.3.5", "1.2.3 - 2.3.4", False), ("2.3.9", "1.2.3 - 2.3", True),
    ("2.4.0", "1.2.3 - 2.3", False), ("1.2.0", "1.2 - 2", True), ("2.9.9", "1.2 - 2", True), ("3.0.0", "1.2 - 2", False),
    ("1.5.0", ">=1.2.3 <1.5.0 || >=2", False), ("2.1.0", ">=1.2.3 <1.5.0 || >=2", True),
    ("1.2.3-alpha.7", ">1.2.3-alpha.3", True), ("3.4.5-alpha.9", ">1.2.3-alpha.3", False),
    ("1.2.3-alpha.2", ">1.2.3-alpha.3", False), ("1.2.3-beta", "^1.2.3-alpha", True), ("1.2.4-beta", "^1.2.3-alpha", False),
    ("1.2.3-alpha.7", ">=1.0.0 <2.0.0", False), ("1.2.3-rc.1", "1.2.3-rc.1 || 9", True),
]


def test_satisfies_table():
    bad = [(v, r, exp) for v, r, exp in CASES if satisfies(v, r) != exp]
    assert not bad, bad


def test_max_satisfying():
    vs = ["1.2.3", "v1.2.4", "1.3.0-beta", "1.3.0", "2.0.0-rc.1", "0.9.0"]
    assert max_satisfying(vs, "~1.2") == "v1.2.4"
    assert max_satisfying(vs, "^1.0.0") == "1.3.0"
    assert max_satisfying(vs, ">=2.0.0-rc.0") == "2.0.0-rc.1"
    assert max_satisfying(vs, ">3") is None


def test_invalid_ranges():
    for bad in [">>1.2.3", "~>1", "1.2.3 -", "^", "1.2.3 - 2 - 3", "<a"]:
        try:
            satisfies("1.0.0", bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted range {bad!r}")


from _runner import main
main(globals())
