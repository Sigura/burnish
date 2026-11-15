from datetime import datetime as D
from solution import next_fire


def raises(expr, after=D(2026, 1, 1)):
    try:
        next_fire(expr, after)
    except ValueError:
        return
    raise AssertionError(f"no ValueError for {expr!r}")


def test_every_minute():
    assert next_fire("* * * * *", D(2026, 3, 1, 10, 5, 30)) == D(2026, 3, 1, 10, 6)
    assert next_fire("* * * * *", D(2026, 3, 1, 10, 5)) == D(2026, 3, 1, 10, 6)


def test_rollover_year():
    assert next_fire("0 0 1 1 *", D(2026, 12, 31, 23, 59)) == D(2027, 1, 1)


def test_steps_and_lists():
    assert next_fire("*/15 9-17 * * *", D(2026, 5, 4, 17, 46)) == D(2026, 5, 5, 9, 0)
    assert next_fire("5,35 */6 * * *", D(2026, 5, 4, 6, 35)) == D(2026, 5, 4, 12, 5)
    assert next_fire("10/20 * * * *", D(2026, 5, 4, 6, 51)) == D(2026, 5, 4, 7, 10)
    assert next_fire("0 1-10/3 * * *", D(2026, 5, 4, 7, 0)) == D(2026, 5, 4, 10, 0)


def test_names():
    assert next_fire("0 12 * feb MON", D(2026, 1, 1)) == D(2026, 2, 2, 12)
    assert next_fire("0 0 * * mon-WED", D(2026, 5, 7)) == D(2026, 5, 11)
    assert next_fire("0 0 1 JUN-aug *", D(2026, 8, 2)) == D(2027, 6, 1)


def test_sunday_7():
    assert next_fire("30 8 * * 7", D(2026, 5, 4)) == D(2026, 5, 10, 8, 30)
    assert next_fire("30 8 * * 0", D(2026, 5, 4)) == D(2026, 5, 10, 8, 30)
    assert next_fire("0 0 * * 5-7", D(2026, 5, 4)) == D(2026, 5, 8)


def test_dom_dow_or():
    # 2026-05-04 is a Monday; the 13th or any Friday
    assert next_fire("0 0 13 * 5", D(2026, 5, 4)) == D(2026, 5, 8)
    assert next_fire("0 0 13 * 5", D(2026, 5, 9)) == D(2026, 5, 13)


def test_dom_restricted_dow_star():
    assert next_fire("0 0 31 * *", D(2026, 4, 1)) == D(2026, 5, 31)


def test_dow_step_restricts():
    # */2 on day-of-week is a restriction, so OR semantics with dom apply
    assert next_fire("0 0 1 * */2", D(2026, 5, 4, 1)) == D(2026, 5, 5)


def test_leap_day():
    assert next_fire("0 0 29 2 *", D(2026, 3, 1)) == D(2028, 2, 29)


def test_impossible():
    raises("0 0 30 2 *")


def test_errors():
    for e in ["* * * *", "60 * * * *", "* 24 * * *", "* * 0 * *", "* * * 13 *", "* * * * 8",
              "5-1 * * * *", "*/0 * * * *", "* * * FOO *", "a * * * *", "1,,2 * * * *"]:
        raises(e)


from _runner import main
main(globals())
