import pytest

from burnish.protect import has_prose, protect


@pytest.mark.parametrize(
    "span",
    [
        "`git status`",
        "``a ` b``",
        "https://example.com/a_b?x=1",
        "www.example.com/x",
        "/etc/hosts",
        "~/.config/burnish/config.toml",
        "./run.sh",
        "src/app/main.py",
        "C:\\Users\\me\\file.txt",
        "me@example.com",
    ],
)
def test_span_is_masked_and_restored(span):
    text = f"Look at {span} now"
    masked = protect(text)
    assert masked.text == "Look at [[P0]] now"
    assert masked.restore(masked.text) == text


def test_trailing_punctuation_stays_outside_url_and_path():
    masked = protect("See https://a.io/x. Then open /etc/hosts.")
    assert masked.spans == ("https://a.io/x", "/etc/hosts")
    assert masked.text == "See [[P0]]. Then open [[P1]]."


@pytest.mark.parametrize("text", ["and/or", "he/she", "TCP/IP.", "1/2/2024"])
def test_ordinary_slashes_are_prose(text):
    assert protect(text).spans == ()


def test_fenced_block_is_one_span():
    block = "```python\nprint( 'teh' )\n```"
    assert protect(f"Intro text.\n\n{block}\n\nOutro text.").spans == (block,)


def test_unterminated_fence_runs_to_the_end():
    assert protect("Intro.\n```\ncode here").spans == ("```\ncode here",)


def test_line_markers_are_cut_off_and_restored():
    text = "# Title\n- item one\n1. item two\n> - nested quote\nplain line"
    masked = protect(text)
    assert masked.text == "Title\nitem one\nitem two\nnested quote\nplain line"
    assert masked.prefixes == ("# ", "- ", "1. ", "> - ", "")
    assert masked.restore("Title\nitem 1\nitem 2\nquote\nplain") == "# Title\n- item 1\n1. item 2\n> - quote\nplain"


def test_lines_intact_only_matters_with_line_markers():
    assert protect("- a\n- b").lines_intact("a\nb")
    assert not protect("- a\n- b").lines_intact("a b")
    assert protect("One line.\nTwo lines.").lines_intact("One line. Two lines.")


def test_markdown_link_is_one_span():
    masked = protect("Read the [setup guide](https://a.io/b) first")
    assert masked.spans == ("[setup guide](https://a.io/b)",)
    assert masked.text == "Read the [[P0]] first"


def test_literal_placeholder_in_input_survives():
    masked = protect("Keep [[P7]] here and `x` there")
    assert masked.restore(masked.text) == "Keep [[P7]] here and `x` there"


def test_placeholders_intact_detects_loss_and_duplicates():
    masked = protect("Use `a` and `b` here")
    assert masked.placeholders_intact("Use [[P1]] and [[P0]] here")
    assert not masked.placeholders_intact("Use [[P0]] here")
    assert not masked.placeholders_intact("Use [[P0]] [[P0]] and [[P1]] here")


def test_has_prose():
    assert not has_prose(protect("`ls -la`").text)
    assert not has_prose(protect("https://a.io/x /etc/hosts 123").text)
    assert has_prose(protect("Run `ls`").text)
