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
import inspect
import io
import os
import sys
import traceback
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import review_script as review_script_module  # noqa: E402
from review_script import (  # noqa: E402
    apply_positional_overlay,
    merge_consecutive_narrators,
    _join_narrator_texts,
    select_review_prompt,
    REVIEW_PROMPT_SCHEMA_MARKER,
    REVIEW_SYSTEM_PROMPT,
    REVIEW_USER_PROMPT,
    normalize_text,
    check_text_loss,
    build_review_batches,
    _rendered_len,
    _truncate_context_entry,
    _maybe_print_truncation_hint,
    build_script_roster,
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


def test_shipped_review_prompt_states_the_attribution_tag_rule():
    """The reviewer must be told that an attribution tag stays narration.

    Measured cost of its absence: a full-book review made 58 NARRATOR ->
    character changes, 57 of them on entries containing no quotation mark at
    all -- "Dairine said.", "Ronan shook his head." and the like handed to
    character voices. Entries with no quotes voiced by a character nearly
    doubled (63 -> 119). The model was not malfunctioning; it was reading a
    tag that names a character as evidence the character speaks it, which is
    the one rule this fork exists to enforce and the only rule the review
    prompt did not state. default_prompts.txt carries it as its rule 2; both
    stages need it, so pin it here rather than relying on prose surviving an
    edit.
    """
    lowered = REVIEW_SYSTEM_PROMPT.lower()
    check(
        "review prompt: says attribution tags/action beats stay narration",
        "attribution tag" in lowered and "action beat" in lowered,
        detail=repr(REVIEW_SYSTEM_PROMPT[:200]),
    )
    check(
        "review prompt: distinguishes an audiobook from an audio drama",
        "audio drama" in lowered,
    )
    check(
        "review prompt: says naming a character is not the same as speaking",
        "he said" in lowered or "said." in lowered,
    )


def test_normalize_text_no_fusion_at_punctuation_quote_boundary():
    """F10: normalize_text() used to delete punctuation outright
    (`re.sub(r'[^\\w\\s]', '', text)`), so a quote/dash mark that directly
    abuts a word with no space of its own fused the two words together
    (`he said:--"Sicut` -> `saidsicut`). That fusion happened differently
    depending on whether the text was normalized as one long string or as
    separately-normalized entries split at a span/quote boundary, producing
    false-alarm coverage-ratio mismatches with zero actual text loss.
    Replacing punctuation with a space (then collapsing whitespace) fixes
    this: a punctuation mark always leaves a token separator behind."""
    check(
        'normalize_text: \'he said:--"Sicut quercus"\' does not fuse into "saidsicut"',
        normalize_text('he said:--"Sicut quercus"') == "he said sicut quercus",
        detail=repr(normalize_text('he said:--"Sicut quercus"')),
    )
    check(
        'normalize_text: \'ex-"spahi"\' does not fuse into "exspahi"',
        normalize_text('ex-"spahi"') == "ex spahi",
        detail=repr(normalize_text('ex-"spahi"')),
    )
    check(
        "normalize_text: apostrophe contractions now split into two tokens "
        "(don't -> 'don t', was 'dont' before -- fine as long as both sides "
        "of every comparison go through the same function)",
        normalize_text("don't") == "don t",
        detail=repr(normalize_text("don't")),
    )


def test_check_text_loss_whole_vs_entry_split_agree_on_fused_quote_fixture():
    """Regression for the false-alarm coverage-ratio bug: a source whose
    span/entry boundary falls exactly at a quote mark used to normalize
    differently depending on whether it went through normalize_text() as
    one long string vs. as separately-normalized entries whose words are
    then concatenated by check_text_loss. Both paths must now agree
    exactly, with zero actual characters gained or lost."""
    source = 'He said:--"Sicut quercus" and left, then said ex-"spahi" once more.'
    # Mimic a span/entry split landing exactly at the quote boundaries --
    # concatenating these pieces reproduces `source` exactly (byte-verbatim
    # by construction, same as the real pipeline).
    entry_pieces = [
        'He said:--',
        '"Sicut quercus" and left, then said ex-',
        '"spahi" once more.',
    ]
    check(
        "fixture: entry pieces concatenate back to the source exactly (sanity check)",
        "".join(entry_pieces) == source,
        detail=repr("".join(entry_pieces)),
    )

    entries = [{"text": p} for p in entry_pieces]
    passed, orig_joined, corr_joined, ratio = check_text_loss(
        [{"text": source}], entries, threshold=1.0, upper_bound=1.0
    )
    check(
        "check_text_loss: whole-source vs per-entry-split normalization now agree exactly (ratio == 1.0)",
        passed and ratio == 1.0 and orig_joined == corr_joined,
        detail=f"ratio={ratio!r} orig={orig_joined!r} corr={corr_joined!r}",
    )


# --- F12: dual-budget batching (count + rendered-JSON-size) --------------
#
# Production evidence: a fixed entry-count batch (25 entries) rendered to
# 26,001 chars (~6.5k tokens) when a batch happened to contain huge
# Gutenberg front-matter entries. The Ollama server was configured with a
# small serving context window and silently truncated the prompt instead
# of erroring -- the model never saw its instructions/entries and returned
# garbage (screenplay-format prose, or a 15-for-25 entry-count mismatch).
# The overlay kept annotated_script.json safe either way, but review
# silently became a no-op on exactly the batches with the longest entries.
# build_review_batches() bounds batches by BOTH count and rendered size so
# this can't happen regardless of the serving window in use.

def test_build_review_batches_splits_on_char_budget():
    entry = {"speaker": "NARRATOR", "text": "word " * 40, "instruct": "Neutral, even narration."}
    entries = [dict(entry) for _ in range(5)]

    size_2 = _rendered_len(entries[:2])
    size_3 = _rendered_len(entries[:3])
    check(
        "fixture sanity: rendered size strictly grows as entries are added",
        size_2 < size_3,
        detail=f"size_2={size_2} size_3={size_3}",
    )
    # Budget that fits exactly 2 of these identical entries but not a 3rd.
    budget = size_2

    batches = build_review_batches(entries, batch_size=100, char_budget=budget)

    check(
        "build_review_batches: no batch's rendered size exceeds the char budget "
        "(except an unavoidable singleton, not applicable here since entries are small)",
        all(_rendered_len(b) <= budget for b in batches),
        detail=repr([len(b) for b in batches]),
    )
    check(
        "build_review_batches: all entries preserved across batches, in order",
        [e for b in batches for e in b] == entries,
        detail=repr([len(b) for b in batches]),
    )
    check(
        "build_review_batches: char budget actually constrained batch sizes (more than 1 batch)",
        len(batches) > 1,
        detail=repr([len(b) for b in batches]),
    )


def test_build_review_batches_singleton_oversized_entry():
    huge_entry = {"speaker": "NARRATOR", "text": "Z" * 5000, "instruct": "Neutral, even narration."}
    small_entry = {"speaker": "NARRATOR", "text": "Short.", "instruct": "Neutral, even narration."}
    entries = [small_entry, huge_entry, small_entry]

    # Budget far too small for huge_entry alone -- it must still become its
    # own batch (never split, never silently dropped or merged away).
    tiny_budget = _rendered_len([small_entry]) + 50
    batches = build_review_batches(entries, batch_size=100, char_budget=tiny_budget)

    check(
        "build_review_batches: oversized entry becomes its own singleton batch (not split)",
        any(len(b) == 1 and b[0] is huge_entry for b in batches),
        detail=repr([len(b) for b in batches]),
    )
    check(
        "build_review_batches: all entries preserved, in order, none dropped",
        [e for b in batches for e in b] == entries,
        detail=repr(batches),
    )


def test_build_review_batches_count_budget_still_respected():
    """Even when the rendered size is well under the char budget, the
    entry-COUNT cap (review_batch_size) must still apply."""
    entries = [{"speaker": "NARRATOR", "text": "Hi.", "instruct": "Neutral, even narration."} for _ in range(10)]
    batches = build_review_batches(entries, batch_size=3, char_budget=1_000_000)
    check(
        "build_review_batches: count cap respected when char budget is not the constraint",
        [len(b) for b in batches] == [3, 3, 3, 1],
        detail=repr([len(b) for b in batches]),
    )


def test_build_review_batches_empty_and_single_entry():
    check(
        "build_review_batches: empty input -> empty list",
        build_review_batches([], 25, 12000) == [],
    )
    one = [{"speaker": "NARRATOR", "text": "Hi.", "instruct": "Neutral, even narration."}]
    check(
        "build_review_batches: single small entry -> one singleton batch",
        build_review_batches(one, 25, 12000) == [one],
    )


def test_truncate_context_entry_only_affects_context_copy():
    long_text = "X" * 500
    entry = {"speaker": "NARRATOR", "text": long_text, "instruct": "Neutral, even narration."}
    truncated = _truncate_context_entry(entry, max_text_chars=300)

    check(
        "_truncate_context_entry: long text truncated to 300 chars + ellipsis",
        truncated["text"] == ("X" * 300) + "...",
        detail=repr(truncated["text"][:40] + "..."),
    )
    check(
        "_truncate_context_entry: original entry dict is NOT mutated",
        entry["text"] == long_text,
        detail=repr(entry["text"][:40]),
    )
    check(
        "_truncate_context_entry: returns a different object than the original when truncating",
        truncated is not entry,
    )

    short_entry = {"speaker": "NARRATOR", "text": "Short.", "instruct": "Neutral, even narration."}
    unchanged = _truncate_context_entry(short_entry, max_text_chars=300)
    check(
        "_truncate_context_entry: text under the cap is left untouched",
        unchanged["text"] == "Short.",
    )


def test_overlay_target_batch_text_unaffected_by_context_truncation():
    """Extends the existing overlay tests: even for an entry long enough
    that _truncate_context_entry WOULD truncate it if used as neighbor
    context, apply_positional_overlay (which only ever sees TARGET BATCH
    entries, never context) must still produce byte-identical full-length
    text. These are two entirely separate code paths -- context truncation
    is applied in main() only to the +/-N neighbor windows passed via
    source_context, never to the batch passed to review_batch()/
    apply_positional_overlay -- this test pins that the overlay never
    routes through the context-truncation helper."""
    long_text = "Y" * 500
    batch = [{"speaker": "NARRATOR", "text": long_text, "instruct": "Neutral, even narration."}]
    corrected = [{"speaker": "NARRATOR", "text": "irrelevant, discarded", "instruct": "Tense, clipped narration."}]
    accepted = apply_positional_overlay(batch, corrected)
    check(
        "overlay: target batch text stays full-length/byte-identical (never context-truncated)",
        accepted[0]["text"] == long_text and len(accepted[0]["text"]) == 500,
        detail=f"len={len(accepted[0]['text'])}",
    )


def test_maybe_print_truncation_hint_fires_when_ratio_far_above_baseline():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        _maybe_print_truncation_hint(prompt_chars=32000, prompt_tokens=2050)  # ratio ~15.6, far above ~4
    output = buffer.getvalue()
    check(
        "truncation hint: fires when chars/tokens ratio is far above the ~4 baseline",
        "HINT" in output and "2050" in output and "32000" in output,
        detail=repr(output),
    )


def test_maybe_print_truncation_hint_silent_for_normal_ratio():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        _maybe_print_truncation_hint(prompt_chars=8000, prompt_tokens=2000)  # ratio 4.0, normal
    check(
        "truncation hint: silent for a normal chars/tokens ratio (no false positive)",
        buffer.getvalue() == "",
        detail=repr(buffer.getvalue()),
    )


def test_maybe_print_truncation_hint_silent_when_tokens_unknown():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        _maybe_print_truncation_hint(prompt_chars=32000, prompt_tokens=None)
        _maybe_print_truncation_hint(prompt_chars=32000, prompt_tokens=0)
    check(
        "truncation hint: silent when prompt_tokens is unknown/zero (no false positives, best-effort only)",
        buffer.getvalue() == "",
        detail=repr(buffer.getvalue()),
    )


def test_review_batch_char_budget_config_key_and_default():
    """F12: `generation.review_batch_char_budget` config key must be read
    with a default of 12000. Inspect main()'s actual source (rather than
    reimplementing the lookup separately) so this test tracks the real
    shipped default and config key name, not a copy that could drift."""
    source = inspect.getsource(review_script_module.main)
    check(
        'main(): reads generation_config.get("review_batch_char_budget", 12000)',
        'generation_config.get("review_batch_char_budget", 12000)' in source,
        detail=source[:600],
    )
    check(
        "main(): uses build_review_batches() (not raw fixed-count slicing) in both modes",
        source.count("build_review_batches(entries, batch_size, batch_char_budget)") == 2,
        detail=str(source.count("build_review_batches(entries, batch_size, batch_char_budget)")),
    )



# ---------------------------------------------------------------------------
# F15: roster-aware spelling resolution at the review seam
# ---------------------------------------------------------------------------

def _overlay_speaker(original, correction, roster=None):
    batch = [{"speaker": original, "text": "  Verbatim text. ", "instruct": "i"}]
    corrected = [{"speaker": correction, "instruct": "i"}]
    accepted = apply_positional_overlay(batch, corrected, roster=roster)
    return accepted[0]


def test_review_roster_snaps_drifted_spelling():
    roster = build_script_roster([
        {"speaker": "ABBE MARIGNAN"}, {"speaker": "NARRATOR"}])
    entry = _overlay_speaker("NARRATOR", "ABBEMARIGNAN", roster)
    check("review overlay snaps ABBEMARIGNAN onto ABBE MARIGNAN",
          entry["speaker"] == "ABBE MARIGNAN", detail=repr(entry["speaker"]))
    check("review overlay leaves text byte-identical",
          entry["text"] == "  Verbatim text. ", detail=repr(entry["text"]))


def test_review_roster_is_built_order_independently():
    # build_script_roster() sees the whole script, so which spelling it
    # establishes does not depend on the order the two variants appear in.
    forward = build_script_roster([{"speaker": "ABBE MARIGNAN"}, {"speaker": "ABBEMARIGNAN"}])
    backward = build_script_roster([{"speaker": "ABBEMARIGNAN"}, {"speaker": "ABBE MARIGNAN"}])
    check("review roster establishes the more-punctuated spelling either way",
          list(forward.values()) == list(backward.values()) == ["ABBE MARIGNAN"],
          detail=f"{forward!r} vs {backward!r}")


def test_review_overlay_conforms_to_the_established_spelling():
    # The overlay is read-only against the roster: a correction is snapped
    # onto whatever the SCRIPT established, so one review pass cannot emit
    # two spellings of one character into the same file.
    roster = build_script_roster([{"speaker": "ABBEMARIGNAN"}])
    entry = _overlay_speaker("NARRATOR", "ABBE MARIGNAN", roster)
    check("review overlay conforms a correction to the script's spelling",
          entry["speaker"] == "ABBEMARIGNAN", detail=repr(entry["speaker"]))


def test_review_roster_unifies_punctuation_drift():
    roster = build_script_roster([{"speaker": "O'BRIEN"}])
    entry = _overlay_speaker("NARRATOR", "OBRIEN", roster)
    check("review overlay snaps OBRIEN onto O'BRIEN",
          entry["speaker"] == "O'BRIEN", detail=repr(entry["speaker"]))


def test_review_roster_never_merges_similar_names():
    roster = build_script_roster([{"speaker": "JON"}, {"speaker": "ELLA"}])
    john = _overlay_speaker("NARRATOR", "JOHN", roster)
    bella = _overlay_speaker("NARRATOR", "BELLA", roster)
    check("JOHN stays distinct from JON in review", john["speaker"] == "JOHN",
          detail=repr(john["speaker"]))
    check("BELLA stays distinct from ELLA in review", bella["speaker"] == "BELLA",
          detail=repr(bella["speaker"]))


def test_review_roster_optional_and_fallback_intact():
    plain = _overlay_speaker("NARRATOR", "ABBEMARIGNAN")
    check("no roster -> plain canonicalization (back-compat)",
          plain["speaker"] == "ABBEMARIGNAN", detail=repr(plain["speaker"]))
    roster = build_script_roster([{"speaker": "ABBE MARIGNAN"}])
    fallback = _overlay_speaker("abbemarignan", "", roster)
    check("empty correction falls back to the roster-resolved original",
          fallback["speaker"] == "ABBE MARIGNAN", detail=repr(fallback["speaker"]))


def test_build_script_roster_shape():
    roster = build_script_roster([
        {"speaker": "ABBE MARIGNAN"}, {"speaker": "ABBEMARIGNAN"},
        {"speaker": "JON"}, {"speaker": ""}, "not a dict",
    ])
    check("roster index collapses the whitespace variant only",
          sorted(roster.values()) == ["ABBE MARIGNAN", "JON"], detail=repr(roster))



def main():
    tests = [
        test_review_roster_snaps_drifted_spelling,
        test_review_roster_is_built_order_independently,
        test_review_overlay_conforms_to_the_established_spelling,
        test_review_roster_unifies_punctuation_drift,
        test_review_roster_never_merges_similar_names,
        test_review_roster_optional_and_fallback_intact,
        test_build_script_roster_shape,
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
        test_shipped_review_prompt_states_the_attribution_tag_rule,
        test_normalize_text_no_fusion_at_punctuation_quote_boundary,
        test_check_text_loss_whole_vs_entry_split_agree_on_fused_quote_fixture,
        test_build_review_batches_splits_on_char_budget,
        test_build_review_batches_singleton_oversized_entry,
        test_build_review_batches_count_budget_still_respected,
        test_build_review_batches_empty_and_single_entry,
        test_truncate_context_entry_only_affects_context_copy,
        test_overlay_target_batch_text_unaffected_by_context_truncation,
        test_maybe_print_truncation_hint_fires_when_ratio_far_above_baseline,
        test_maybe_print_truncation_hint_silent_for_normal_ratio,
        test_maybe_print_truncation_hint_silent_when_tokens_unknown,
        test_review_batch_char_budget_config_key_and_default,
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
