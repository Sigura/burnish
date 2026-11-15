from paginate import paginate

ITEMS = list(range(1, 11))


def raises(*args):
    try:
        paginate(*args)
    except ValueError:
        return
    raise AssertionError(f"no ValueError for {args[1:]}")


def test_first_page():
    assert paginate(ITEMS, 1, 3) == {"items": [1, 2, 3], "page": 1, "total_pages": 4, "has_next": True, "has_prev": False}


def test_last_partial_page():
    assert paginate(ITEMS, 4, 3) == {"items": [10], "page": 4, "total_pages": 4, "has_next": False, "has_prev": True}


def test_exact_fit():
    r = paginate(ITEMS, 2, 5)
    assert r["items"] == [6, 7, 8, 9, 10] and r["total_pages"] == 2 and not r["has_next"] and r["has_prev"]


def test_beyond():
    r = paginate(ITEMS, 9, 3)
    assert r["items"] == [] and r["total_pages"] == 4 and not r["has_next"] and r["has_prev"]


def test_empty():
    assert paginate([], 1, 10) == {"items": [], "page": 1, "total_pages": 1, "has_next": False, "has_prev": False}


def test_invalid():
    raises(ITEMS, 0, 3)
    raises(ITEMS, 1, 0)
    raises(ITEMS, -1, 3)


from _runner import main
main(globals())
