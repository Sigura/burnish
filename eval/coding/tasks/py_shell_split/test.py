from solution import split_command as sp

ENV = {"HOME": "/home/u", "EMPTY": "", "MANY": "a  b\tc", "SP": " x ", "N": "1"}


def raises(line):
    try:
        sp(line, ENV)
    except ValueError:
        return
    raise AssertionError(f"no ValueError for {line!r}")


def test_basic():
    assert sp("ls  -la\t/tmp\n", ENV) == ["ls", "-la", "/tmp"]
    assert sp("", ENV) == []
    assert sp("   ", ENV) == []


def test_quotes():
    assert sp("echo 'a b' \"c d\"", ENV) == ["echo", "a b", "c d"]
    assert sp("a\"b c\"'d'", ENV) == ["ab cd"]
    assert sp("'' \"\" x", ENV) == ["", "", "x"]
    assert sp("'$HOME' \"$HOME\"", ENV) == ["$HOME", "/home/u"]


def test_backslashes():
    assert sp(r"a\ b c", ENV) == ["a b", "c"]
    assert sp(r'"a\"b" "\n" "\$x" "\\"', ENV) == ['a"b', r"\n", "$x", "\\"]
    assert sp(r"'a\b'", ENV) == [r"a\b"]
    assert sp("ab\\\ncd", ENV) == ["abcd"]
    assert sp('"ab\\\ncd"', ENV) == ["abcd"]
    assert sp(r"\$HOME", ENV) == ["$HOME"]


def test_expansion():
    assert sp("$HOME/x ${HOME}y $HOMEy", ENV) == ["/home/u/x", "/home/uy"]
    assert sp("${NOPE:-$HOME} ${EMPTY:-d} ${EMPTY-d} ${NOPE-d}", ENV) == ["/home/u", "d", "d"]
    assert sp("\"${EMPTY-d}\"", ENV) == [""]
    assert sp("${NOPE:-a b}", ENV) == ["a", "b"]
    assert sp("\"${NOPE:-a b}\"", ENV) == ["a b"]


def test_literal_dollar():
    assert sp("$ a$ $1x \"$\"", ENV) == ["$", "a$", "$1x", "$"]


def test_field_splitting():
    assert sp("x $MANY y", ENV) == ["x", "a", "b", "c", "y"]
    assert sp("x \"$MANY\" y", ENV) == ["x", "a  b\tc", "y"]
    assert sp("pre${SP}post", ENV) == ["pre", "x", "post"]
    assert sp("$EMPTY $NOPE z", ENV) == ["z"]
    assert sp("\"$EMPTY\" z", ENV) == ["", "z"]
    assert sp("a$EMPTY", ENV) == ["a"]
    assert sp("$SP", ENV) == ["x"]


def test_errors():
    for bad in ["'abc", "\"abc", "${HOME", "abc\\", "\"a\\\""]:
        raises(bad)


from _runner import main
main(globals())
