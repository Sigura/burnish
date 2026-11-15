from solution import evaluate as ev


def raises(exc, expr, env=None):
    try:
        ev(expr, env)
    except exc:
        return
    except Exception as e:
        raise AssertionError(f"{expr!r}: expected {exc.__name__}, got {type(e).__name__}")
    raise AssertionError(f"{expr!r}: expected {exc.__name__}")


def test_arith():
    assert ev("1 + 2 * 3") == 7
    assert ev("(1 + 2) * 3") == 9
    assert ev("10 - 4 - 3") == 3
    assert ev("7 % 3") == 1 and ev("-7 % 3") == 2
    assert ev("100 / 10 / 5") == 2


def test_division_types():
    r = ev("6 / 3")
    assert r == 2 and type(r) is int
    r = ev("7 / 2")
    assert r == 3.5 and type(r) is float
    r = ev("6.0 / 3")
    assert type(r) is float


def test_power_and_unary():
    assert ev("-2^2") == -4
    assert ev("2^-1") == 0.5
    assert ev("2^3^2") == 512
    assert ev("(-2)^2") == 4
    assert ev("--3") == 3
    assert ev("-+-3") == 3
    assert ev("!0 + !5") == 1


def test_decimals():
    assert ev(".5 + 2.") == 2.5
    assert ev("1.25*4") == 5.0


def test_compare_logic():
    assert ev("1 < 2 && 3 >= 3") == 1
    assert ev("1 == 2 || 2 != 2") == 0
    assert ev("0 || 7") == 1
    assert ev("1 + 1 == 2") == 1
    raises(SyntaxError, "1 < 2 < 3")


def test_short_circuit():
    assert ev("0 && 1/0") == 0
    assert ev("1 || x") == 1
    assert ev("1 ? 2 : 1/0") == 2
    assert ev("0 ? 1/0 : 3") == 3


def test_ternary_assoc():
    assert ev("0 ? 1 : 0 ? 2 : 3") == 3
    assert ev("1 ? 0 ? 5 : 6 : 7") == 6
    assert ev("1 + 1 ? 10 : 20") == 10


def test_vars_and_functions():
    env = {"x": 3, "y_2": 4.5}
    assert ev("x * y_2", env) == 13.5
    assert ev("max(1, x, 2) + min(x)", env) == 6
    assert ev("abs(-x) + abs(2 - 5)", env) == 6
    assert ev("max(min(1, 2), abs(-7)) ^ 2") == 49


def test_errors():
    raises(ZeroDivisionError, "1 / 0")
    raises(ZeroDivisionError, "5 % (2 - 2)")
    raises(NameError, "a + 1")
    raises(NameError, "foo(1)")
    for bad in ["", "1 +", "(1", "1)", "1 2", "abs()", "abs(1, 2)", "max()", "1 = 1", "1 & 1", "1 | 0",
                "? 1 : 2", "1 ? 2", "3..4", "min(1,)"]:
        raises(SyntaxError, bad)


from _runner import main
main(globals())
