from solution import format_duration, parse_duration


def raises(s):
    try:
        parse_duration(s)
    except ValueError:
        return
    raise AssertionError(f"accepted {s!r}")


def test_parse():
    assert parse_duration("1h30m") == 5400
    assert parse_duration("2d 4h") == 187200
    assert parse_duration("90s") == 90
    assert parse_duration("0s") == 0
    assert parse_duration("90m") == 5400
    assert parse_duration("1D2H3M4S") == 93784


def test_parse_errors():
    for s in ["", "1x", "30m1h", "1h1h", "h", "1.5h", "1h  2m", " 1h", "1h-2m", "10"]:
        raises(s)


def test_format():
    assert format_duration(5400) == "1h30m"
    assert format_duration(0) == "0s"
    assert format_duration(86461) == "1d1m1s"
    assert format_duration(59) == "59s"
    assert format_duration(172800) == "2d"


def test_roundtrip():
    for n in [0, 1, 59, 60, 61, 3599, 3600, 86399, 86400, 1234567]:
        assert parse_duration(format_duration(n)) == n


def test_format_negative():
    try:
        format_duration(-1)
    except ValueError:
        return
    raise AssertionError("negative accepted")


from _runner import main
main(globals())
