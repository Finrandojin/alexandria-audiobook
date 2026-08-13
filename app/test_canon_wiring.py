"""Standalone tests for the speaker_canon wiring in app.py / review_script.py.

Covers:
  - build_voice_roster(): canonical, deduped, fragment-free roster
    construction from raw script entries; back-compat config lookup
    against non-canonical voice_config.json keys.
  - _canonical_roster_names() + GET /api/voices/alias_suggestions'
    underlying helpers: the on-demand, split-out alias-suggestion path.
  - review_script._canonicalize_speakers(): canonicalizes "speaker" fields
    only, leaving "text" byte-identical.

F13: suggest_aliases() is O(n^2) in roster size and was previously run
unconditionally inside build_voice_roster() on every GET /api/voices call,
even though the frontend never read the resulting "alias_suggestions"
field (measured on a real 589-speaker roster: ~52ms of an ~80ms request,
~160KB of unused response payload). build_voice_roster() no longer touches
suggest_aliases() at all -- it now returns a (roster_names, config_lookup)
2-tuple. Alias suggestions are computed only on demand, via a new
GET /api/voices/alias_suggestions endpoint backed by
_canonical_roster_names() + speaker_canon.suggest_aliases().

Run directly:
    python app/test_canon_wiring.py
Exits 0 if all tests pass, non-zero otherwise.
"""
import os
import sys
import time
import types
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# app.py pulls in `project` (-> tts.py -> numpy/torch/etc.) purely for
# unrelated TTS/project-management functionality that this test does not
# exercise. Stub it out before importing app.py, same pattern as
# app/test_epub_extract.py, to keep this test lightweight and standalone.
if 'project' not in sys.modules:
    _fake_project = types.ModuleType('project')

    class _FakeProjectManager:
        def __init__(self, *args, **kwargs):
            pass

        def load_chunks(self):
            return []

        def save_chunks(self, chunks):
            pass

        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                return None
            return _noop

    _fake_project.ProjectManager = _FakeProjectManager
    sys.modules['project'] = _fake_project

import app as app_module  # noqa: E402
from app import build_voice_roster, _canonical_roster_names, _compute_label_flags  # noqa: E402
from speaker_canon import suggest_aliases  # noqa: E402
from review_script import _canonicalize_speakers  # noqa: E402


results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, condition, detail))
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))


def test_roster_canonicalizes_and_dedupes():
    script_data = [
        {"speaker": "mark", "text": "a"},
        {"speaker": "MARK (shouting)", "text": "c"},
        {"speaker": "JON", "text": "d"},
        {"speaker": "JOHN", "text": "e"},
        {"speaker": "narrator", "text": "f"},
    ]
    roster_names, config_lookup = build_voice_roster(script_data, {})

    check(
        "roster: exactly 4 canonical names, MARK collapsed, JON/JOHN distinct",
        roster_names == sorted(["JON", "JOHN", "MARK", "NARRATOR"]),
        detail=repr(roster_names),
    )


def test_roster_handles_legacy_type_field_and_empty():
    script_data = [
        {"type": "elena", "text": "a"},
        {"speaker": "", "text": "b"},
        {"speaker": "   ", "text": "c"},
    ]
    roster_names, config_lookup = build_voice_roster(script_data, {})
    check(
        "roster: legacy 'type' field used, blanks dropped",
        roster_names == ["ELENA"],
        detail=repr(roster_names),
    )


def test_config_backcompat_resolution():
    # A config written under a raw, non-canonical spelling of the speaker
    # must still resolve for the canonical roster name. Note the canonical
    # name now KEEPS the gender-marking title ("Mr. Mark" -> "MISTER MARK"),
    # so Mr. Mark and a bare Mark are two roster entries, not one.
    script_data = [
        {"speaker": "Mr. Mark", "text": "a"},
        {"speaker": "mister mark (shouting)", "text": "b"},
    ]
    voice_config = {"Mr. Mark": {"engine": "clone", "ref": "mark.wav"}}
    roster_names, config_lookup = build_voice_roster(script_data, voice_config)

    check(
        "config back-compat: roster resolves to MISTER MARK",
        roster_names == ["MISTER MARK"],
        detail=repr(roster_names),
    )
    check(
        "config back-compat: non-canonical 'Mr. Mark' key resolves for MISTER MARK",
        config_lookup.get("MISTER MARK") == {"engine": "clone", "ref": "mark.wav"},
        detail=repr(config_lookup),
    )

    # Simulate what the handler does to derive persona_pending.
    persona_pending = "MISTER MARK" not in config_lookup
    check(
        "config back-compat: persona_pending is False when config resolves",
        persona_pending is False,
    )


def test_config_exact_key_wins_over_scan():
    # If BOTH an exact canonical key and a differently-cased key that would
    # canonicalize to the same name exist, the exact match must win.
    script_data = [{"speaker": "Mr. Mark", "text": "a"}]
    voice_config = {
        "MISTER MARK": {"engine": "exact"},
        "Mr. Mark": {"engine": "scanned"},
    }
    roster_names, config_lookup = build_voice_roster(script_data, voice_config)
    check(
        "config back-compat: exact canonical key wins over scanned fallback",
        config_lookup.get("MISTER MARK") == {"engine": "exact"},
        detail=repr(config_lookup),
    )


def test_roster_keeps_husband_and_wife_apart():
    # The defect: "Mr. Smith" and "Mrs. Smith" used to collapse into a single
    # roster entry, hence a single voice, with no way back.
    script_data = [
        {"speaker": "Mr. Smith", "text": "a"},
        {"speaker": "Mrs. Smith", "text": "b"},
    ]
    roster_names, _ = build_voice_roster(script_data, {})
    check(
        "roster: Mr. Smith and Mrs. Smith are two entries",
        roster_names == ["MISSUS SMITH", "MISTER SMITH"],
        detail=repr(roster_names),
    )


def test_config_metadata_keys_are_not_speakers():
    # voice_config.json carries a "_canon_version" stamp. Underscore-prefixed
    # keys are metadata and must never be scanned as a speaker name.
    script_data = [{"speaker": "mark", "text": "a"}]
    voice_config = {"_canon_version": 2, "MARK": {"engine": "exact"}}
    roster_names, config_lookup = build_voice_roster(script_data, voice_config)
    check(
        "config: '_canon_version' never resolves as a voice entry",
        roster_names == ["MARK"] and config_lookup == {"MARK": {"engine": "exact"}},
        detail=repr((roster_names, config_lookup)),
    )


def test_build_voice_roster_return_shape_is_two_tuple():
    """F13: build_voice_roster() no longer returns alias_suggestions --
    its return shape shrank from a 3-tuple to (roster_names,
    config_lookup)."""
    script_data = [{"speaker": "mark", "text": "a"}]
    result = build_voice_roster(script_data, {})
    check(
        "build_voice_roster: returns a 2-tuple (roster_names, config_lookup)",
        isinstance(result, tuple) and len(result) == 2,
        detail=repr(result),
    )


def test_build_voice_roster_does_not_compute_alias_suggestions():
    """F13: build_voice_roster() must never call suggest_aliases() -- that
    O(n^2) work moved off the GET /api/voices hot path entirely, onto the
    new on-demand GET /api/voices/alias_suggestions endpoint. Monkeypatch
    app's module-level suggest_aliases with a spy and confirm it is never
    invoked during build_voice_roster(), even on a roster large enough
    that a real call would be noticeable if it happened."""
    calls = []
    original = app_module.suggest_aliases

    def _spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    app_module.suggest_aliases = _spy
    try:
        script_data = [{"speaker": f"SPEAKER{i}", "text": "x"} for i in range(50)]
        build_voice_roster(script_data, {})
    finally:
        app_module.suggest_aliases = original

    check(
        "build_voice_roster: suggest_aliases() is never called (moved off the hot path)",
        calls == [],
        detail=repr(calls),
    )


def test_alias_suggestions_endpoint_helpers_match_old_inline_path():
    """The new on-demand endpoint's underlying helpers
    (_canonical_roster_names() + suggest_aliases()) must produce the exact
    same roster and suggestions the old inlined build_voice_roster() path
    did for a fixed fixture roster -- this is a pure code-motion refactor,
    not a behavior change to the suggestions themselves."""
    script_data = [
        {"speaker": "JON", "text": "a"},
        {"speaker": "JOHN", "text": "b"},
        {"speaker": "MARK", "text": "c"},
        {"speaker": "ELLA", "text": "d"},
        {"speaker": "BELLA", "text": "e"},
    ]
    endpoint_roster_names = _canonical_roster_names(script_data)
    endpoint_suggestions = suggest_aliases(endpoint_roster_names)

    # What build_voice_roster() computes its roster from -- must be the
    # exact same roster the on-demand endpoint's helper produces, since
    # they're both derived from the same script_data.
    roster_from_build_voice_roster, _config_lookup = build_voice_roster(script_data, {})

    check(
        "alias_suggestions endpoint: roster matches build_voice_roster's roster",
        endpoint_roster_names == roster_from_build_voice_roster,
        detail=repr((endpoint_roster_names, roster_from_build_voice_roster)),
    )
    per_voice = {
        name: [s for s in endpoint_suggestions if s["name"] == name]
        for name in endpoint_roster_names
    }
    check(
        "alias_suggestions endpoint: MARK gets an empty list (no similar name)",
        per_voice.get("MARK") == [],
        detail=repr(per_voice),
    )
    non_empty = sorted(name for name, sugs in per_voice.items() if sugs)
    check(
        "alias_suggestions endpoint: JON (of JON/JOHN) and ELLA (of ELLA/BELLA) "
        "carry suggestions (the shorter/alias side of each pair)",
        non_empty == ["ELLA", "JON"],
        detail=repr(per_voice),
    )


def test_alias_suggestions_perf_bound_600_names():
    """Perf regression guard: suggest_aliases() must stay well clear of an
    O(n^2) blowup on a roster as large as the biggest real one measured
    (589 speakers). 500ms is generous headroom -- enough to not flake on a
    slow CI machine -- while still catching a real algorithmic regression;
    a correctly-implemented O(n^2) string-similarity pass over 600 short
    names normally completes in well under 100ms."""
    roster = [f"CHARACTER{i:04d}" for i in range(600)]
    start = time.perf_counter()
    suggestions = suggest_aliases(roster)
    elapsed = time.perf_counter() - start
    check(
        f"suggest_aliases: 600-name roster completes in <500ms (actual: {elapsed * 1000:.1f}ms)",
        elapsed < 0.5,
        detail=f"elapsed={elapsed:.3f}s, suggestions_count={len(suggestions)}",
    )


def test_review_canonicalizes_speaker_not_text():
    entries = [
        {"speaker": "Narrator", "text": "Once upon a time.", "instruct": "calm"},
        {"speaker": "elena", "text": "  Hello there!  ", "instruct": "warm"},
        {"speaker": "Mr. Mark (shouting)", "text": "Stop!", "instruct": "angry"},
    ]
    result = _canonicalize_speakers(entries)

    check(
        "review: speakers canonicalized",
        [e["speaker"] for e in result] == ["NARRATOR", "ELENA", "MISTER MARK"],
        detail=repr([e["speaker"] for e in result]),
    )
    check(
        "review: text fields byte-identical to input",
        [e["text"] for e in result] == [e["text"] for e in entries],
        detail=repr([e["text"] for e in result]),
    )
    check(
        "review: instruct fields untouched",
        [e["instruct"] for e in result] == [e["instruct"] for e in entries],
    )
    check(
        "review: original entries list not mutated",
        entries[0]["speaker"] == "Narrator" and entries[2]["speaker"] == "Mr. Mark (shouting)",
        detail=repr(entries),
    )



# ---------------------------------------------------------------------------
# F15: roster-aware spelling resolution in the voices roster
# ---------------------------------------------------------------------------

def test_voices_roster_consolidates_whitespace_drift():
    script = [
        {"speaker": "ABBE MARIGNAN", "text": "a"},
        {"speaker": "NARRATOR", "text": "b"},
        {"speaker": "ABBEMARIGNAN", "text": "c"},
    ]
    names, _ = build_voice_roster(script, {})
    check("voices roster consolidates ABBEMARIGNAN onto first-seen ABBE MARIGNAN",
          names == ["ABBE MARIGNAN", "NARRATOR"], detail=repr(names))


def test_voices_roster_selection_is_order_independent():
    forward = _canonical_roster_names([
        {"speaker": "ABBE MARIGNAN", "text": "a"},
        {"speaker": "ABBEMARIGNAN", "text": "b"},
    ])
    backward = _canonical_roster_names([
        {"speaker": "ABBEMARIGNAN", "text": "a"},
        {"speaker": "ABBE MARIGNAN", "text": "b"},
    ])
    check("voices roster picks the more-punctuated spelling in either order",
          forward == backward == ["ABBE MARIGNAN"],
          detail=f"{forward!r} vs {backward!r}")


def test_voices_roster_unifies_punctuation_drift():
    names = _canonical_roster_names([
        {"speaker": "OBRIEN", "text": "a"},
        {"speaker": "O'BRIEN", "text": "b"},
        {"speaker": "OBRIAN", "text": "c"},
    ])
    check("O'BRIEN/OBRIEN unify; OBRIAN stays a separate character",
          names == ["O'BRIEN", "OBRIAN"], detail=repr(names))


def test_voices_roster_never_merges_similar_names():
    script = [{"speaker": n, "text": "x"} for n in ("JON", "JOHN", "ELLA", "BELLA")]
    names = _canonical_roster_names(script)
    check("JON/JOHN/ELLA/BELLA stay four distinct roster entries",
          names == ["BELLA", "ELLA", "JOHN", "JON"], detail=repr(names))


def test_voices_config_lookup_survives_whitespace_drift():
    script = [{"speaker": "ABBE MARIGNAN", "text": "a"}]
    names, lookup = build_voice_roster(script, {"ABBEMARIGNAN": {"voice": "v1"}})
    check("voice_config saved under a drifted spelling still resolves",
          lookup.get("ABBE MARIGNAN") == {"voice": "v1"}, detail=repr(lookup))



def test_label_flags_computation_never_mutates_entries_or_roster():
    """_compute_label_flags (backing GET /api/voices/label_flags) is
    advisory-only: running it must leave every entry's "speaker" value and
    the derived roster list byte-identical before and after, whether or not
    it flags anything."""
    import copy

    source_text = (
        "Alice walked into the room. \"Hello,\" said Alice. "
        "A stranger lurked nearby, saying nothing."
    )
    script_data = [
        {"speaker": "Alice", "text": "Hello,", "instruct": "warm"},
        # ZORBLAX never appears in source_text at all -> should be flagged.
        {"speaker": "Zorblax", "text": "A stranger lurked nearby", "instruct": "flat"},
    ]

    before_entries = copy.deepcopy(script_data)
    before_roster = _canonical_roster_names(script_data)

    flags = _compute_label_flags(script_data, source_text)

    after_entries = copy.deepcopy(script_data)
    after_roster = _canonical_roster_names(script_data)

    check("label_flags: script entries unchanged after computation",
          before_entries == after_entries, detail=repr(after_entries))
    check("label_flags: roster unchanged after computation",
          before_roster == after_roster, detail=repr((before_roster, after_roster)))
    check("label_flags: returns a flag record for each non-NARRATOR speaker",
          {f["name"] for f in flags} == {"ALICE", "ZORBLAX"}, detail=repr(flags))

    by_name = {f["name"]: f for f in flags}
    check("label_flags: ALICE is attested (appears near its own line)",
          by_name["ALICE"]["attested"] is True, detail=repr(by_name["ALICE"]))
    check("label_flags: ZORBLAX is flagged unattested (name never appears in source)",
          by_name["ZORBLAX"]["attested"] is False, detail=repr(by_name["ZORBLAX"]))


def test_generation_imports_the_repair_helpers_rather_than_reimplementing_them():
    """Speaker repair -- and the edit-distance predicate it rests on -- lives in
    speaker_canon, which is where the guards, the docstring and the tests are.
    A local reimplementation in generate_script would drift from all three, and
    is exactly how a bounded distance-1 check turns into an unbounded fuzzy
    match. Assert the wiring, and assert that no distance/similarity logic was
    copied into the caller."""
    import inspect
    import generate_script
    import speaker_canon

    check("generate_script imports repair_speaker from speaker_canon",
          generate_script.repair_speaker is speaker_canon.repair_speaker)
    check("generate_script imports source_word_index from speaker_canon",
          generate_script.source_word_index is speaker_canon.source_word_index)
    check("generate_script imports near_spellings from speaker_canon",
          generate_script.near_spellings is speaker_canon.near_spellings)

    source = inspect.getsource(generate_script)
    for banned in ("difflib", "def _is_distance_one",
                   "from rapidfuzz import fuzz", "ratio("):
        check(f"generate_script does not reimplement matching ({banned})",
              banned not in source)

    # generate_script may measure exact edit distance for ONE purpose: folding a
    # misspelled JSON schema key back onto the fixed four-word key vocabulary.
    # That vocabulary is this code's own, not the book's, so a near miss there
    # has one possible meaning -- unlike JON/JOHN. Keeping the measurement
    # confined to that function is what stops it from drifting onto names, so
    # pin the location rather than the mere absence of the word.
    key_recovery = inspect.getsource(generate_script._recover_label_keys)
    distance_lines = [line for line in source.splitlines()
                      if "Levenshtein" in line and not line.lstrip().startswith("#")]
    check("generate_script measures edit distance only for schema-key recovery",
          all(line in key_recovery or line.startswith("from rapidfuzz")
              for line in distance_lines),
          detail=repr([line for line in distance_lines
                       if line not in key_recovery
                       and not line.startswith("from rapidfuzz")]))
    check("schema-key recovery never targets the 'text' key",
          "text" not in generate_script._RECOVERABLE_LABEL_KEYS)
    check("speaker names are still matched only by speaker_canon",
          "Levenshtein" not in inspect.getsource(generate_script.resolve_span_labels))


def main():
    tests = [
        test_generation_imports_the_repair_helpers_rather_than_reimplementing_them,
        test_label_flags_computation_never_mutates_entries_or_roster,
        test_voices_roster_consolidates_whitespace_drift,
        test_voices_roster_selection_is_order_independent,
        test_voices_roster_unifies_punctuation_drift,
        test_voices_roster_never_merges_similar_names,
        test_voices_config_lookup_survives_whitespace_drift,
        test_roster_canonicalizes_and_dedupes,
        test_roster_handles_legacy_type_field_and_empty,
        test_config_backcompat_resolution,
        test_config_exact_key_wins_over_scan,
        test_roster_keeps_husband_and_wife_apart,
        test_config_metadata_keys_are_not_speakers,
        test_build_voice_roster_return_shape_is_two_tuple,
        test_build_voice_roster_does_not_compute_alias_suggestions,
        test_alias_suggestions_endpoint_helpers_match_old_inline_path,
        test_alias_suggestions_perf_bound_600_names,
        test_review_canonicalizes_speaker_not_text,
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
