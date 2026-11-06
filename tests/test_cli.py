import io

import pytest

from burnish import cli, client

RESULT = {"text": "She goes home.\n", "diff": "-a\n+b\n", "model": "m", "issues": []}


@pytest.fixture
def fake_refine(monkeypatch):
    calls = []

    def refine(url, text, mode, model):
        calls.append((text, mode, model))
        return RESULT

    monkeypatch.setattr(client, "refine", refine)
    return calls


@pytest.fixture
def failing_refine(monkeypatch):
    def refine(*args):
        raise client.RefineError("backend down")

    monkeypatch.setattr(client, "refine", refine)


def use_stdin(monkeypatch, text):
    monkeypatch.setattr("sys.stdin", io.StringIO(text))


def test_filter_writes_refined_text_in_all_mode_by_default(monkeypatch, capsys, fake_refine):
    use_stdin(monkeypatch, "She go home.\n")
    assert cli.main([]) == 0
    assert capsys.readouterr().out == "She goes home.\n"
    assert fake_refine == [("She go home.\n", "all", None)]


def test_flags_are_forwarded(monkeypatch, fake_refine):
    use_stdin(monkeypatch, "x")
    assert cli.main(["--mode", "polish", "--model", "phi4-mini"]) == 0
    assert fake_refine == [("x", "polish", "phi4-mini")]


def test_diff_flag(monkeypatch, capsys, fake_refine):
    use_stdin(monkeypatch, "x")
    assert cli.main(["--diff"]) == 0
    assert capsys.readouterr().out == "-a\n+b\n"


def test_failure_gives_input_back(monkeypatch, capsys, failing_refine):
    use_stdin(monkeypatch, "Keep me.\n")
    assert cli.main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == "Keep me.\n"
    assert "backend down" in captured.err


def test_file_argument_edits_in_place(tmp_path, fake_refine):
    path = tmp_path / "prompt.md"
    path.write_text("She go home.\n")
    assert cli.main([str(path)]) == 0
    assert path.read_text() == "She goes home.\n"


def test_file_is_untouched_on_failure(tmp_path, failing_refine):
    path = tmp_path / "prompt.md"
    path.write_text("Keep me.\n")
    assert cli.main([str(path)]) == 1
    assert path.read_text() == "Keep me.\n"
