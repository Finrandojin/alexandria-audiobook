"""Standalone tests for the review-stage verbatim guarantee (app/review_script.py).

Audit finding: the review stage's LLM prompt used to instruct wholesale text
rewriting (attribution stripping, participle rephrasing, front-matter
merging) and its safety net (check_text_loss) tolerated up to 5% word loss.
That is a structural violation of the "audiobooks are verbatim" contract.

The fix makes text damage structurally impossible rather than merely
discouraged:
  - apply_positional_overlay(): "text" is ALWAYS taken from the original
    entry, never from the LLM's response. Only "speaker" (canonicalized)
    and "instruct" are taken from the LLM, matched positionally. A count
    mismatch between batch and corrected returns None -- the caller must
    treat that as a failed batch and keep the originals.
  - _join_narrator_texts()/merge_consecutive_narrators(): the narrator-merge
    join no longer unconditionally inserts a space -- it only does so when
    neither side already carries boundary whitespace, so verbatim entries
    (which carry their own boundary whitespace) stay byte-exact, while
    legacy .strip()'d scripts still merge readably.

This test exercises those two functions directly -- no live server, no LLM,
no subprocess.

Run directly:
    python app/test_review_verbatim.py
Exits 0 if all tests pass, non-zero otherwise.
"""
import io
import os
import sys
import traceback
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from review_script import (  # noqa: E402
    apply_positional_overlay,
    merge_consecutive_narrators,
    _join_narrator_texts,
    select_review_prompt,
    REVIEW_PROMPT_SCHEMA_MARKER,
    REVIEW_SYSTEM_PROMPT,
    REVIEW_USER_PROMPT,
)


results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, condition, detail))
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))


def test_overlay_preserves_text_applies_speaker_and_instruct():
    batch = [
        {"speaker": "narrator", "text": "  He lit a cigarette. ", "instruct": "Neutral, even narration."},
        {"speaker": "HOLMES", "text": "Quite so, he answered.", "instruct": "flat"},
    ]
    # The LLM tries to rewrite text (attribution-strip / rephrase) -- must
    # be discarded entirely. It also fixes a speaker and an instruct.
    corrected = [
        {"speaker": "NARRATOR", "text": "He lit a cigarette and threw himself into an armchair.", "instruct": "Neutral, even narration."},
        {"speaker": "holmes", "text": "Quite so.", "instruct": "Quietly analytical, dry amusement."},
    ]

    accepted = apply_positional_overlay(batch, corrected)

    check(
        "overlay: text is byte-identical to the ORIGINAL batch, not the LLM's",
        [e["text"] for e in accepted] == [b["text"] for b in batch],
        detail=repr([e["text"] for e in accepted]),
    )
    check(
        "overlay: speaker taken from correction, canonicalized",
        [e["speaker"] for e in accepted] == ["NARRATOR", "HOLMES"],
        detail=repr([e["speaker"] for e in accepted]),
    )
    check(
        "overlay: instruct taken from correction",
        accepted[1]["instruct"] == "Quietly analytical, dry amusement.",
        detail=repr(accepted[1]),
    )
    check(
        "overlay: original batch list/dicts not mutated",
        batch[0]["text"] == "  He lit a cigarette. " and batch[1]["speaker"] == "HOLMES",
        detail=repr(batch),
    )


def test_overlay_keeps_unspecified_fields_from_original():
    batch = [{"speaker": "elena", "text": "Hello.", "instruct": "warm"}]
    # LLM omits "instruct" entirely (only fixes speaker) -- original instruct
    # must survive.
    corrected = [{"speaker": "ELENA"}]
    accepted = apply_positional_overlay(batch, corrected)
    check(
        "overlay: instruct falls back to original when LLM omits it",
        accepted[0]["instruct"] == "warm",
        detail=repr(accepted[0]),
    )
    check(
        "overlay: text falls back to original regardless",
        accepted[0]["text"] == "Hello.",
    )


def test_overlay_empty_corrected_speaker_keeps_original():
    """N1 regression: a corrected speaker that is empty/None/blank, or that
    canonicalizes to empty (e.g. a stray parenthetical), must NOT blank out
    a good original label -- that would silently drop the entry from the
    voices roster and mute it at render time. The original (canonicalized)
    speaker must be kept in every such case."""
    cases = [
        ("empty string", ""),
        ("None", None),
        ("whitespace-only", "   "),
        ("canonicalizes to empty", "(shouting)"),
    ]
    for label, bad_speaker in cases:
        batch = [{"speaker": "mark", "text": "Stop right there.", "instruct": "Firm, commanding."}]
        corrected = [{"speaker": bad_speaker, "text": "irrelevant", "instruct": "Firm, commanding."}]
        accepted = apply_positional_overlay(batch, corrected)
        check(
            f"overlay: corrected speaker ({label}) -> original speaker retained (canonical)",
            accepted[0]["speaker"] == "MARK",
            detail=repr(accepted[0]),
        )


def test_overlay_absent_speaker_key_keeps_original_existing_behavior():
    """Existing behavior (unchanged by the N1 fix): an absent "speaker" key
    in the correction also falls back to the original speaker."""
    batch = [{"speaker": "elena", "text": "Hello.", "instruct": "warm"}]
    corrected = [{"instruct": "warm"}]  # no "speaker" key at all
    accepted = apply_positional_overlay(batch, corrected)
    check(
        "overlay: absent speaker key -> original speaker retained",
        accepted[0]["speaker"] == "ELENA",
        detail=repr(accepted[0]),
    )


def test_overlay_invalid_corrected_instruct_keeps_original():
    """N2 regression: a corrected instruct that is None, a list, a dict, or
    a whitespace-only string must be rejected -- only a non-empty string is
    accepted -- so malformed values never get written into
    annotated_script.json."""
    cases = [
        ("None", None),
        ("empty list", []),
        ("empty dict", {}),
        ("whitespace-only string", "  "),
    ]
    for label, bad_instruct in cases:
        batch = [{"speaker": "NARRATOR", "text": "It was raining.", "instruct": "Neutral, even narration."}]
        corrected = [{"speaker": "NARRATOR", "text": "irrelevant", "instruct": bad_instruct}]
        accepted = apply_positional_overlay(batch, corrected)
        check(
            f"overlay: corrected instruct ({label}) -> original instruct retained",
            accepted[0]["instruct"] == "Neutral, even narration.",
            detail=repr(accepted[0]),
        )


def test_overlay_valid_string_instruct_still_applied():
    """Sanity check that the N2 fix didn't overcorrect: a genuine non-empty
    string instruct from the LLM must still be applied."""
    batch = [{"speaker": "NARRATOR", "text": "It was raining.", "instruct": "Neutral, even narration."}]
    corrected = [{"speaker": "NARRATOR", "text": "irrelevant", "instruct": "Tense, clipped narration."}]
    accepted = apply_positional_overlay(batch, corrected)
    check(
        "overlay: valid string instruct is still applied",
        accepted[0]["instruct"] == "Tense, clipped narration.",
        detail=repr(accepted[0]),
    )


def test_overlay_count_mismatch_returns_none():
    batch = [
        {"speaker": "NARRATOR", "text": "One.", "instruct": "Neutral, even narration."},
        {"speaker": "NARRATOR", "text": "Two.", "instruct": "Neutral, even narration."},
    ]
    # LLM tries to split entry 1 into two -- explicitly forbidden now.
    corrected_too_many = [
        {"speaker": "NARRATOR", "text": "One.", "instruct": "Neutral, even narration."},
        {"speaker": "NARRATOR", "text": "One point five.", "instruct": "Neutral, even narration."},
        {"speaker": "NARRATOR", "text": "Two.", "instruct": "Neutral, even narration."},
    ]
    result = apply_positional_overlay(batch, corrected_too_many)
    check(
        "overlay: count mismatch (more entries) returns None",
        result is None,
        detail=repr(result),
    )

    corrected_too_few = [
        {"speaker": "NARRATOR", "text": "One and two.", "instruct": "Neutral, even narration."},
    ]
    result2 = apply_positional_overlay(batch, corrected_too_few)
    check(
        "overlay: count mismatch (fewer entries, merge attempt) returns None",
        result2 is None,
        detail=repr(result2),
    )


def test_overlay_malformed_correction_entry_falls_back():
    batch = [{"speaker": "NARRATOR", "text": "Text.", "instruct": "Neutral, even narration."}]
    corrected = ["not a dict"]
    accepted = apply_positional_overlay(batch, corrected)
    check(
        "overlay: non-dict correction entry falls back to original speaker/instruct",
        accepted[0]["speaker"] == "NARRATOR" and accepted[0]["instruct"] == "Neutral, even narration.",
        detail=repr(accepted),
    )
    check(
        "overlay: non-dict correction entry still gets original text",
        accepted[0]["text"] == "Text.",
    )


def test_join_narrator_texts_whitespace_aware():
    # Verbatim entries carrying their own boundary whitespace -> join with
    # "" so no byte is invented.
    check(
        "join: trailing space on left -> no extra space inserted",
        _join_narrator_texts("He left. ", "She stayed.") == "He left. She stayed.",
    )
    check(
        "join: leading space on right -> no extra space inserted",
        _join_narrator_texts("He left.", " She stayed.") == "He left. She stayed.",
    )
    check(
        "join: newline boundary preserved byte-exact",
        _join_narrator_texts("He left.\n", "She stayed.") == "He left.\nShe stayed.",
    )
    # Legacy stripped-text entries (neither side has boundary whitespace) ->
    # a space IS inserted so merged text still reads correctly.
    check(
        "join: neither side has boundary whitespace -> space inserted",
        _join_narrator_texts("He left.", "She stayed.") == "He left. She stayed.",
    )


def test_merge_consecutive_narrators_byte_exact_for_verbatim_entries():
    # Entries as they'd actually appear post-refactor: boundary whitespace
    # lives INSIDE the text field already (e.g. a trailing space carried
    # from the source), not stripped off.
    entries = [
        {"speaker": "NARRATOR", "text": "He lit a cigarette. ", "instruct": "Neutral, even narration."},
        {"speaker": "NARRATOR", "text": "Then he sat down.", "instruct": "Neutral, even narration."},
    ]
    merged, merges = merge_consecutive_narrators(entries, max_merged_length=800)
    check(
        "merge: verbatim entries join byte-exact (no injected space)",
        merged[0]["text"] == "He lit a cigarette. Then he sat down.",
        detail=repr(merged),
    )
    check("merge: one merge counted", merges == 1)


def test_merge_consecutive_narrators_legacy_stripped_entries_still_readable():
    # Legacy/older scripts whose entries were .strip()'d -- no boundary
    # whitespace on either side -- must still merge with a readable space.
    entries = [
        {"speaker": "NARRATOR", "text": "He lit a cigarette.", "instruct": "Neutral, even narration."},
        {"speaker": "NARRATOR", "text": "Then he sat down.", "instruct": "Neutral, even narration."},
    ]
    merged, merges = merge_consecutive_narrators(entries, max_merged_length=800)
    check(
        "merge: legacy stripped entries still get a readable space",
        merged[0]["text"] == "He lit a cigarette. Then he sat down.",
        detail=repr(merged),
    )
    check("merge: one merge counted (legacy case)", merges == 1)


def test_narrator_casing_canonicalized_by_overlay():
    """The review LLM might emit odd casing for NARRATOR; the overlay's
    canonicalize() call must normalize it so downstream `!= "NARRATOR"`
    comparisons (e.g. in merge_consecutive_narrators) stay reliable."""
    batch = [{"speaker": "Narrator", "text": "Once upon a time.", "instruct": "calm"}]
    corrected = [{"speaker": "NaRrAtOr", "text": "irrelevant, discarded", "instruct": "Neutral, even narration."}]
    accepted = apply_positional_overlay(batch, corrected)
    check(
        "overlay: odd-cased NARRATOR from the LLM is canonicalized",
        accepted[0]["speaker"] == "NARRATOR",
        detail=repr(accepted[0]),
    )
    check(
        "overlay: text still comes from the original, not the LLM",
        accepted[0]["text"] == "Once upon a time.",
    )


def test_select_review_prompt_stale_custom_falls_back_with_warning():
    """F8: a saved custom prompt from before the verbatim-review refactor
    (no REVIEW_PROMPT_SCHEMA_MARKER) must NOT be used as-is -- it still
    carries the retired "strip attribution tags / rewrite" methodology,
    which causes entry-count mismatches (failed batches) and degraded
    speaker/instruct fixes under the new positional-overlay contract. It
    must fall back to the built-in default and print a warning naming the
    offending config key."""
    stale_prompt = "You are a script reviewer. STRIP ATTRIBUTION TAGS FROM DIALOGUE. Rephrase as needed."
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = select_review_prompt(stale_prompt, REVIEW_SYSTEM_PROMPT, "review_system_prompt")
    output = buffer.getvalue()
    check(
        "select_review_prompt: stale custom (no marker) falls back to the default",
        result == REVIEW_SYSTEM_PROMPT,
        detail=repr(result[:80]),
    )
    check(
        "select_review_prompt: warning names the offending config key",
        "review_system_prompt" in output and "WARNING" in output,
        detail=repr(output),
    )


def test_select_review_prompt_marker_bearing_custom_used_as_is():
    """A custom prompt that DOES carry the marker is the operator's own,
    intentional, up-to-date customization -- it must be used byte-identically,
    not silently altered."""
    custom_prompt = "My own custom review prompt (prompt schema: verbatim-review-v1) with extra house rules."
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = select_review_prompt(custom_prompt, REVIEW_SYSTEM_PROMPT, "review_system_prompt")
    output = buffer.getvalue()
    check(
        "select_review_prompt: marker-bearing custom prompt used byte-identically",
        result == custom_prompt,
        detail=repr(result),
    )
    check(
        "select_review_prompt: no warning printed for a valid custom prompt",
        output == "",
        detail=repr(output),
    )


def test_select_review_prompt_absent_or_blank_uses_default_silently():
    """No saved custom prompt (None, missing) or a blank one -- both are the
    common/unconfigured case and must silently use the built-in default,
    with no warning noise."""
    for label, value in [("None", None), ("empty string", ""), ("whitespace-only", "   ")]:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            result = select_review_prompt(value, REVIEW_USER_PROMPT, "review_user_prompt")
        output = buffer.getvalue()
        check(
            f"select_review_prompt: absent/blank custom ({label}) -> default, silently",
            result == REVIEW_USER_PROMPT and output == "",
            detail=repr(output),
        )


def test_shipped_review_prompts_pass_their_own_guard():
    """The review_prompts.txt shipped with the repo must carry the marker in
    BOTH halves (system prompt and user prompt template), so a fresh
    install's own default prompts pass select_review_prompt's guard exactly
    like a valid custom prompt would."""
    check(
        "shipped prompts: REVIEW_SYSTEM_PROMPT carries the marker",
        REVIEW_PROMPT_SCHEMA_MARKER in REVIEW_SYSTEM_PROMPT,
        detail=repr(REVIEW_SYSTEM_PROMPT[:120]),
    )
    check(
        "shipped prompts: REVIEW_USER_PROMPT carries the marker",
        REVIEW_PROMPT_SCHEMA_MARKER in REVIEW_USER_PROMPT,
        detail=repr(REVIEW_USER_PROMPT[:200]),
    )
    # Also exercise the actual guard function end-to-end with the shipped
    # prompts, standing in for "prompts_config" being empty (the common
    # fresh-install case) and for a config that happens to echo the
    # shipped default back (still must pass, no warning).
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        sys_result = select_review_prompt(REVIEW_SYSTEM_PROMPT, REVIEW_SYSTEM_PROMPT, "review_system_prompt")
        usr_result = select_review_prompt(REVIEW_USER_PROMPT, REVIEW_USER_PROMPT, "review_user_prompt")
    output = buffer.getvalue()
    check(
        "shipped prompts: both pass select_review_prompt's guard with no warning",
        sys_result == REVIEW_SYSTEM_PROMPT and usr_result == REVIEW_USER_PROMPT and output == "",
        detail=repr(output),
    )


def main():
    tests = [
        test_overlay_preserves_text_applies_speaker_and_instruct,
        test_overlay_keeps_unspecified_fields_from_original,
        test_overlay_empty_corrected_speaker_keeps_original,
        test_overlay_absent_speaker_key_keeps_original_existing_behavior,
        test_overlay_invalid_corrected_instruct_keeps_original,
        test_overlay_valid_string_instruct_still_applied,
        test_overlay_count_mismatch_returns_none,
        test_overlay_malformed_correction_entry_falls_back,
        test_join_narrator_texts_whitespace_aware,
        test_merge_consecutive_narrators_byte_exact_for_verbatim_entries,
        test_merge_consecutive_narrators_legacy_stripped_entries_still_readable,
        test_narrator_casing_canonicalized_by_overlay,
        test_select_review_prompt_stale_custom_falls_back_with_warning,
        test_select_review_prompt_marker_bearing_custom_used_as_is,
        test_select_review_prompt_absent_or_blank_uses_default_silently,
        test_shipped_review_prompts_pass_their_own_guard,
    ]
    for t in tests:
        try:
            t()
        except Exception:
            check(t.__name__, False, detail=traceback.format_exc())

    failed = [name for name, ok, _ in results if not ok]
    print()
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED:")
        for name in failed:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
