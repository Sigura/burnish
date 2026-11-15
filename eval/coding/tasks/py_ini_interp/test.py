from solution import load_ini

TEXT = r"""
top = 1
[paths]
Root = /srv/app   ; inline comment
logs: ${root}/logs
cache = ${root}/cache#notacomment
price = $$5
# full comment
  ; indented comment
[db]
host = db.local
url = postgres://${host}/${paths:root}
desc = "  spaced ; not comment  "  # comment
quote = "say \"hi\""
multi = first
  second line
	third
after = x

[paths]
extra = ${logs}/extra
"""


def err(text, needle):
    try:
        load_ini(text)
    except ValueError as e:
        assert needle in str(e), (str(e), needle)
        return
    raise AssertionError(f"no ValueError ({needle})")


def test_full():
    d = load_ini(TEXT)
    assert d[""] == {"top": "1"}
    assert d["paths"] == {"root": "/srv/app", "logs": "/srv/app/logs", "cache": "/srv/app/cache#notacomment",
                          "price": "$5", "extra": "/srv/app/logs/extra"}
    assert d["db"]["url"] == "postgres://db.local//srv/app"
    assert d["db"]["desc"] == "  spaced ; not comment  "
    assert d["db"]["quote"] == 'say "hi"'
    assert d["db"]["multi"] == "first\nsecond line\nthird"
    assert d["db"]["after"] == "x"


def test_last_wins_and_case():
    d = load_ini("[S]\nA=1\na=2\n[s]\na=3\n")
    assert d == {"S": {"a": "2"}, "s": {"a": "3"}}


def test_separators():
    d = load_ini("[x]\nurl = http://h:80/a=b\nk: v=w\n")
    assert d["x"] == {"url": "http://h:80/a=b", "k": "v=w"}


def test_empty_value():
    assert load_ini("[x]\nk =\n")["x"] == {"k": ""}


def test_errors():
    err("[x]\nnovalue\n", "line 2")
    err("[x\n", "line 1")
    err("  indented = 1\n", "line 1")
    err("[x]\nk = \"open\n", "line 2")
    err("[x]\na = ${b}\n", "")
    err("[x]\na = ${y:a}\n", "")
    err("[x]\na = ${b}\nb = ${c}\nc = ${a}\n", "cycle")
    err("[x]\na = ${a}\n", "cycle")


def test_blank_ends_continuation():
    err("[x]\na = 1\n\n  b\n", "line 4")


from _runner import main
main(globals())
