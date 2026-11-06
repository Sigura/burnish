import pytest
from conftest import lt_match

from burnish.pipeline import ServiceError, clean_output


def test_fix_returns_llm_rewrite(harness):
    harness.engine.replies = ["She goes home."]
    result = harness.pipeline.refine("She go home.", "fix")
    assert result.text == "She goes home."
    assert result.model == "gemma3:4b"
    assert "+She goes home." in result.diff
    assert result.issues == []


def test_default_mode_runs_fix_then_polish(harness):
    harness.engine.replies = ["She goes home.", "She goes home now."]
    result = harness.pipeline.refine("She go home.")
    assert result.text == "She goes home now."
    prompts = [r["system"] for r in harness.engine.requests]
    assert "Correct only" in prompts[0] and "Improve the fluency" in prompts[1]
    assert "She goes home." in harness.engine.requests[1]["user"]


def test_languagetool_fixes_typo_the_llm_introduced(harness):
    harness.engine.replies = ["Teh cat sat."]
    harness.lt_matches = [lt_match(0, 3, "The")]
    result = harness.pipeline.refine("The cat sat.", "fix")
    assert result.text == "The cat sat."
    assert result.issues[0]["applied"] is True


def test_languagetool_spelling_fix_skipped_when_llm_kept_the_word(harness):
    harness.engine.replies = ["Sanjay's PR lgtm, let's merge it."]
    harness.lt_matches = [lt_match(12, 4, "GTM")]
    result = harness.pipeline.refine("Sanjay's PR lgtm, let's merge it.", "fix")
    assert result.text == "Sanjay's PR lgtm, let's merge it."
    assert result.issues[0]["applied"] is False
    assert result.issues[0]["replacements"] == ["GTM"]


def test_languagetool_punctuation_fix_applies_when_llm_kept_the_text(harness):
    harness.engine.replies = ["Yes I agree."]
    harness.lt_matches = [lt_match(0, 3, "Yes,", category="PUNCTUATION", rule="P")]
    result = harness.pipeline.refine("Yes I agree.", "fix")
    assert result.text == "Yes, I agree."


def test_drift_falls_back_to_languagetool_only(harness):
    harness.engine.replies = ["Completely unrelated reply about other things altogether, friend."]
    harness.lt_matches = [lt_match(0, 3, "The")]
    result = harness.pipeline.refine("Teh cat sat.", "fix")
    assert result.text == "The cat sat."
    assert result.model == "languagetool"
    assert result.issues[0]["type"] == "drift_guard"


def test_polish_allows_a_larger_rewrite(harness):
    source = "The cat sat on the mat near the door."
    rewrite = "A cat rested on a mat by the front door."
    harness.engine.replies = [rewrite]
    assert harness.pipeline.refine(source, "polish").text == rewrite


def test_all_mode_keeps_fix_when_polish_drifts(harness):
    harness.engine.replies = ["She goes home.", "Totally different words appear in this long reply instead."]
    result = harness.pipeline.refine("She go home.", "all")
    assert result.text == "She goes home."
    assert result.model == "gemma3:4b"
    assert result.issues[0]["type"] == "drift_guard"


def test_protected_spans_are_hidden_from_llm_and_restored(harness):
    harness.engine.replies = ["Run [[P0]] now, please."]
    result = harness.pipeline.refine("Run `ls -la` now please.", "fix")
    assert "ls -la" not in harness.engine.requests[0]["user"]
    assert result.text == "Run `ls -la` now, please."


def test_dropped_placeholder_falls_back(harness):
    harness.engine.replies = ["Run it now please."]
    result = harness.pipeline.refine("Run `ls -la` now please.", "fix")
    assert result.text == "Run `ls -la` now please."
    assert result.model == "languagetool"
    assert result.issues[0]["type"] == "placeholder_guard"


def test_list_markers_are_hidden_from_llm_and_restored(harness):
    harness.engine.replies = ["He has the keys\nShe has the map"]
    result = harness.pipeline.refine("- He have the keys\n- She have the map", "fix")
    assert harness.engine.requests[0]["user"] == "<text>\nHe have the keys\nShe have the map\n</text>"
    assert result.text == "- He has the keys\n- She has the map"


def test_merged_list_lines_fall_back(harness):
    harness.engine.replies = ["He has the keys and she has the map."]
    result = harness.pipeline.refine("- He have the keys\n- She have the map", "fix")
    assert result.text == "- He have the keys\n- She have the map"
    assert result.issues[0]["type"] == "lines_guard"


def test_changed_bold_markers_fall_back(harness):
    harness.engine.replies = ["This is very important."]
    result = harness.pipeline.refine("This is **very** important.", "fix")
    assert result.text == "This is **very** important."
    assert result.issues[0]["type"] == "markdown_guard"


def test_truncated_reply_falls_back(harness):
    harness.engine.replies = ["She goes"]
    harness.engine.truncated = True
    result = harness.pipeline.refine("She go home.", "fix")
    assert result.text == "She go home."
    assert result.issues[0]["type"] == "truncated_guard"


def test_surrounding_whitespace_is_kept(harness):
    harness.engine.replies = ["She goes home."]
    harness.lt_matches = [lt_match(0, 3, "goes", category="GRAMMAR", rule="G")]
    result = harness.pipeline.refine("\n  She go home.\n", "fix")
    assert result.text == "\n  She goes home.\n"
    assert result.issues[0]["offset"] == 3


@pytest.mark.parametrize("text", ["", "   \n", "`ls -la`", "https://example.com/a"])
def test_nothing_to_edit_skips_services(harness, text):
    result = harness.pipeline.refine(text)
    assert result.text == text
    assert result.model == "none"
    assert harness.engine.requests == [] and harness.lt_requests == []


def test_model_override(harness):
    harness.engine.replies = ["She goes home."]
    result = harness.pipeline.refine("She go home.", "fix", model="phi4-mini")
    assert harness.engine.requests[0]["model"] == "phi4-mini"
    assert result.model == "phi4-mini"


def test_engine_failure_raises_and_drops_cached_selection(harness):
    harness.engine.down = True
    with pytest.raises(ServiceError):
        harness.pipeline.refine("She go home.")
    assert harness.selector.invalidated


def test_languagetool_failure_raises(harness):
    harness.engine.replies = ["She goes home."]
    harness.lt_down = True
    with pytest.raises(ServiceError):
        harness.pipeline.refine("She go home.", "fix")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('"She goes home."', "She goes home."),
        ("“She goes home.”", "She goes home."),
        ("<text>\nShe goes home.\n</text>", "She goes home."),
        ("<think>\n\n</think>\n\nShe goes home.", "She goes home."),
        ("Here is the corrected text:\n\nShe goes home.", "She goes home."),
        ("She goes home.\n\nNote: I changed the verb.", "She goes home."),
        ("```\nShe goes home.\n```", "She goes home."),
    ],
)
def test_clean_output(raw, expected):
    assert clean_output(raw, "She go home.") == expected


def test_clean_output_keeps_quotes_the_input_had():
    source = '"Go home," she said.'
    assert clean_output(source, source) == source
    assert clean_output('"Go home"', '"Go home"') == '"Go home"'


def test_clean_output_straightens_quotes_the_input_did_not_curl():
    assert clean_output("They’re “here”.", "Their 'here'.") == "They're \"here\"."
    assert clean_output("They’re “here”.", "Their ‘here’.") == "They’re \"here\"."


def test_placeholder_rule_only_in_prompt_when_text_has_placeholders(harness):
    harness.engine.replies = ["She goes home.", "Run [[P0]] now, please."]
    harness.pipeline.refine("She go home.", "fix")
    harness.pipeline.refine("Run `ls -la` now please.", "fix")
    plain, masked = (r["system"] for r in harness.engine.requests)
    assert "[[P0]]" not in plain
    assert "[[P0]]" in masked


def test_period_added_to_list_items_is_dropped(harness):
    harness.engine.replies = ["He has the keys.\nShe has the map.\nDone."]
    result = harness.pipeline.refine("- He have the keys\n- She have the map.\nDone", "fix")
    assert result.text == "- He has the keys\n- She has the map.\nDone."
