"""Standalone tests for the speaker_canon wiring in app.py / review_script.py.

Covers:
  - app.build_voice_roster(): canonical, deduped, fragment-free roster
    construction from raw script entries; advisory alias_suggestions;
    back-compat config lookup against non-canonical voice_config.json keys.
  - review_script._canonicalize_speakers(): canonicalizes "speaker" fields
    only, leaving "text" byte-identical.

Run directly:
    python app/test_canon_wiring.py
Exits 0 if all tests pass, non-zero otherwise.
"""
import os
import sys
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

from app import build_voice_roster  # noqa: E402
from review_script import _canonicalize_speakers  # noqa: E402


results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, condition, detail))
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))


def test_roster_canonicalizes_and_dedupes():
    script_data = [
        {"speaker": "mark", "text": "a"},
        {"speaker": "Mr. Mark", "text": "b"},
        {"speaker": "MARK (shouting)", "text": "c"},
        {"speaker": "JON", "text": "d"},
        {"speaker": "JOHN", "text": "e"},
        {"speaker": "narrator", "text": "f"},
    ]
    roster_names, alias_suggestions, config_lookup = build_voice_roster(script_data, {})

    check(
        "roster: exactly 4 canonical names, MARK collapsed, JON/JOHN distinct",
        roster_names == sorted(["JON", "JOHN", "MARK", "NARRATOR"]),
        detail=repr(roster_names),
    )

    # JON/JOHN should produce a suggestion; MARK should not appear in any
    # suggestion (no other roster name is similar enough to it).
    names_in_suggestions = {s["name"] for s in alias_suggestions} | {
        s["alias_of"] for s in alias_suggestions
    }
    check(
        "roster: JON/JOHN produce an alias suggestion",
        {"JON", "JOHN"} <= names_in_suggestions,
        detail=repr(alias_suggestions),
    )
    check(
        "roster: MARK has no alias suggestion",
        "MARK" not in names_in_suggestions,
        detail=repr(alias_suggestions),
    )
    check(
        "roster: NARRATOR excluded from suggestions",
        "NARRATOR" not in names_in_suggestions,
        detail=repr(alias_suggestions),
    )


def test_roster_handles_legacy_type_field_and_empty():
    script_data = [
        {"type": "elena", "text": "a"},
        {"speaker": "", "text": "b"},
        {"speaker": "   ", "text": "c"},
    ]
    roster_names, alias_suggestions, config_lookup = build_voice_roster(script_data, {})
    check(
        "roster: legacy 'type' field used, blanks dropped",
        roster_names == ["ELENA"],
        detail=repr(roster_names),
    )


def test_config_backcompat_resolution():
    script_data = [
        {"speaker": "mark", "text": "a"},
        {"speaker": "MARK (shouting)", "text": "b"},
    ]
    voice_config = {"Mr. Mark": {"engine": "clone", "ref": "mark.wav"}}
    roster_names, alias_suggestions, config_lookup = build_voice_roster(script_data, voice_config)

    check(
        "config back-compat: roster resolves to MARK",
        roster_names == ["MARK"],
        detail=repr(roster_names),
    )
    check(
        "config back-compat: non-canonical 'Mr. Mark' key resolves for MARK",
        config_lookup.get("MARK") == {"engine": "clone", "ref": "mark.wav"},
        detail=repr(config_lookup),
    )

    # Simulate what the handler does to derive persona_pending.
    persona_pending = "MARK" not in config_lookup
    check(
        "config back-compat: persona_pending is False when config resolves",
        persona_pending is False,
    )


def test_config_exact_key_wins_over_scan():
    # If BOTH an exact canonical key and a differently-cased key that would
    # canonicalize to the same name exist, the exact match must win.
    script_data = [{"speaker": "mark", "text": "a"}]
    voice_config = {
        "MARK": {"engine": "exact"},
        "Mr. Mark": {"engine": "scanned"},
    }
    roster_names, alias_suggestions, config_lookup = build_voice_roster(script_data, voice_config)
    check(
        "config back-compat: exact canonical key wins over scanned fallback",
        config_lookup.get("MARK") == {"engine": "exact"},
        detail=repr(config_lookup),
    )


def test_alias_suggestions_attached_per_voice():
    """Mirrors what the get_voices handler does with alias_suggestions."""
    script_data = [
        {"speaker": "JON", "text": "a"},
        {"speaker": "JOHN", "text": "b"},
        {"speaker": "MARK", "text": "c"},
    ]
    roster_names, alias_suggestions, config_lookup = build_voice_roster(script_data, {})

    per_voice = {
        name: [s for s in alias_suggestions if s["name"] == name]
        for name in roster_names
    }
    check(
        "alias_suggestions: MARK gets an empty list",
        per_voice.get("MARK") == [],
        detail=repr(per_voice),
    )
    non_empty = [name for name, sugs in per_voice.items() if sugs]
    check(
        "alias_suggestions: exactly one of JON/JOHN carries the suggestion "
        "(the shorter/alias side)",
        non_empty == ["JON"],
        detail=repr(per_voice),
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
        [e["speaker"] for e in result] == ["NARRATOR", "ELENA", "MARK"],
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


def main():
    tests = [
        test_roster_canonicalizes_and_dedupes,
        test_roster_handles_legacy_type_field_and_empty,
        test_config_backcompat_resolution,
        test_config_exact_key_wins_over_scan,
        test_alias_suggestions_attached_per_voice,
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
