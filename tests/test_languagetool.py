from dataclasses import replace

from conftest import lt_match

from burnish.languagetool import apply_fixes


def fix(cfg, text, matches, protected=()):
    return apply_fixes(text, matches, list(protected), cfg.languagetool)


def test_applies_spelling_and_punctuation(cfg):
    matches = [lt_match(0, 3, "The"), lt_match(7, 1, ";", category="PUNCTUATION", rule="P")]
    text, issues = fix(cfg, "Teh cat, sat", matches)
    assert text == "The cat; sat"
    assert [i["applied"] for i in issues] == [True, True]


def test_grammar_is_reported_not_applied(cfg):
    text, issues = fix(cfg, "He go home", [lt_match(3, 2, "goes", category="GRAMMAR", rule="G")])
    assert text == "He go home"
    assert issues[0]["applied"] is False
    assert issues[0]["replacements"] == ["goes"]


def test_offsets_follow_earlier_replacements(cfg):
    matches = [lt_match(0, 3, "These"), lt_match(4, 2, "goes", category="GRAMMAR", rule="G")]
    text, issues = fix(cfg, "Teh go", matches)
    assert text == "These go"
    assert text[issues[1]["offset"] : issues[1]["offset"] + issues[1]["length"]] == "go"


def test_protected_spans_are_skipped(cfg):
    text, issues = fix(cfg, "Use `teh` now", [lt_match(5, 3, "the")], protected=[(4, 9)])
    assert text == "Use `teh` now"
    assert issues == []


def test_overlapping_match_is_skipped(cfg):
    matches = [
        lt_match(0, 4, "x", category="PUNCTUATION", rule="P"),
        lt_match(2, 4, "y", category="PUNCTUATION", rule="P"),
    ]
    text, issues = fix(cfg, "abcdef ok", matches)
    assert text == "xef ok"
    assert len(issues) == 1


def test_no_replacement_is_not_applied(cfg):
    text, issues = fix(cfg, "qqqq now", [lt_match(0, 4, None)])
    assert text == "qqqq now"
    assert issues[0]["applied"] is False


def test_spelling_leaves_names_and_identifiers_alone(cfg):
    for word in ("PyTorch", "CUDA", "gpt4", "my_var"):
        sentence = f"We use {word} here"
        text, issues = fix(cfg, sentence, [lt_match(7, len(word), "wrong")])
        assert text == sentence
        assert issues[0]["applied"] is False


def test_spelling_skips_midsentence_capitals_but_fixes_sentence_start(cfg):
    assert fix(cfg, "I like Kubuntu", [lt_match(7, 7, "Ubuntu")])[0] == "I like Kubuntu"
    assert fix(cfg, "Ok. Teh end", [lt_match(4, 3, "The")])[0] == "Ok. The end"


def test_spelling_skips_suggestions_that_change_the_first_letter(cfg):
    assert fix(cfg, "Kubuntu uses KDE", [lt_match(0, 7, "Ubuntu")])[0] == "Kubuntu uses KDE"
    assert fix(cfg, "PR lgtm, merge", [lt_match(3, 4, "GTM")])[0] == "PR lgtm, merge"
    assert fix(cfg, "i recieved it", [lt_match(0, 1, "I"), lt_match(2, 8, "received")])[0] == "I received it"


def test_ignore_words(cfg):
    custom = replace(cfg.languagetool, ignore_words=("kubectl",))
    text, _ = apply_fixes("run kubectl now", [lt_match(4, 7, "cubical")], [], custom)
    assert text == "run kubectl now"
