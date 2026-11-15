from solution import summarize

LOG = """2026-11-14T10:02:03Z INFO [api] started
2026-11-14T10:05:00Z ERROR [db] connection lost: timeout [retry 1]
2026-11-14T09:59:59Z ERROR [api] early failure
2026-11-14T10:30:00Z WARN [db] slow query

garbage line
2026-11-14T11:00:00Z ERROR [db] connection lost again
2026-11-14T11:01:00Z FATAL [db] unknown level
2026-11-14T11:02:00Z INFO api missing brackets
2026-13-14T11:02:00Z INFO [api] bad month
2026-11-14T09:00:00Z DEBUG [cache] warm
""".splitlines()


def test_counts():
    r = summarize(LOG)
    assert r["counts"] == {"DEBUG": 1, "INFO": 1, "WARN": 1, "ERROR": 3}


def test_errors_by_component():
    assert summarize(LOG)["errors_by_component"] == {"db": 2, "api": 1}


def test_busiest_hour():
    assert summarize(LOG)["busiest_hour"] == "2026-11-14T10"
    tie = ["2026-11-14T12:00:00Z INFO [a] x", "2026-11-14T08:00:00Z INFO [a] y"]
    assert summarize(tie)["busiest_hour"] == "2026-11-14T08"


def test_first_error():
    assert summarize(LOG)["first_error"] == "early failure"


def test_invalid():
    assert summarize(LOG)["invalid"] == 4


def test_empty():
    r = summarize([])
    assert r == {"counts": {"DEBUG": 0, "INFO": 0, "WARN": 0, "ERROR": 0}, "errors_by_component": {},
                 "busiest_hour": None, "first_error": None, "invalid": 0}


def test_generator_input():
    r = summarize(l for l in LOG)
    assert r["counts"]["ERROR"] == 3


from _runner import main
main(globals())
