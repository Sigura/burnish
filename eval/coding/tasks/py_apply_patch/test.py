import difflib
from solution import apply_patch


def mkdiff(a, b, n=3):
    lines = list(difflib.unified_diff(a.splitlines(True), b.splitlines(True), "a/f", "b/f", n=n))
    out = []
    for l in lines:
        if l.endswith("\n"):
            out.append(l)
        else:
            out.append(l + "\n\\ No newline at end of file\n")
    return "".join(out)


A = "".join(f"line {i}\n" for i in range(1, 31))


def roundtrip(a, b, n=3):
    assert apply_patch(a, mkdiff(a, b, n)) == b


def test_single_change():
    roundtrip(A, A.replace("line 10\n", "LINE TEN\n"))


def test_multiple_hunks_shift():
    b = A.replace("line 3\n", "x\ny\nz\n").replace("line 20\n", "").replace("line 28\n", "28a\n28b\n")
    roundtrip(A, b)
    roundtrip(A, b, n=1)
    roundtrip(A, b, n=0)


def test_insert_at_start_and_end():
    roundtrip(A, "new first\n" + A + "new last\n")
    roundtrip(A, "new first\n" + A + "new last\n", n=0)


def test_delete_everything():
    roundtrip("a\nb\n", "")
    roundtrip("", "a\nb\n")


def test_no_newline_at_eof():
    roundtrip("a\nb\nc", "a\nb\nc\n")
    roundtrip("a\nb\nc\n", "a\nb\nc")
    roundtrip("a\nb\nc", "a\nB\nc")
    roundtrip("a\nb\nc", "a\nb\nd")


def test_headers_and_section_text():
    diff = (
        "diff --git a/f b/f\nindex 123..456 100644\n--- a/f\n+++ b/f\n"
        "@@ -2,3 +2,3 @@ def foo():\n line 2\n-line 3\n+LINE 3\n line 4\n"
    )
    assert apply_patch(A, diff) == A.replace("line 3\n", "LINE 3\n")


def test_count_omitted():
    diff = "@@ -1 +1 @@\n-a\n+b\n"
    assert apply_patch("a\nc\n", diff) == "b\nc\n"


def test_context_mismatch():
    diff = mkdiff(A, A.replace("line 10\n", "X\n"))
    try:
        apply_patch(A.replace("line 9\n", "line nine\n"), diff)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on context mismatch")


def test_wrong_position():
    diff = "@@ -5,1 +5,1 @@\n-line 4\n+X\n"
    try:
        apply_patch(A, diff)
    except ValueError:
        pass
    else:
        raise AssertionError("must not fuzz to another line")


def test_bad_counts():
    diff = "@@ -1,3 +1,3 @@\n line 1\n-line 2\n+X\n"
    try:
        apply_patch(A, diff)
    except ValueError:
        pass
    else:
        raise AssertionError("header counts disagree with body")


def test_crlf_preserved():
    a = "a\r\nb\r\nc\r\n"
    b = "a\r\nB\r\nc\r\n"
    roundtrip(a, b)


from _runner import main
main(globals())
