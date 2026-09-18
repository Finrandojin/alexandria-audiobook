"""Standalone tests for tts_normalizer.py.

Run directly: `python app/test_tts_normalization.py` (from anywhere -- it
fixes up sys.path itself). Exits 0 on success, 1 on any failure.

Imports ONLY tts_normalizer -- never tts.py. tts.py pulls in numpy/
soundfile/pydub at module level, which may not be installed in this
interpreter; this test module deliberately avoids that so it can run on
any machine.

tts_normalizer selects a backend lazily: nemo_text_processing first, else
wetext, else none (passthrough + one warning). The backend-selection and
fidelity tests below use a skip-with-notice pattern where the choice of
backend matters (they assert the wetext path when nemo is genuinely
absent -- the real situation on this Windows box -- and print a [SKIP]
notice instead of failing if nemo happens to be importable, e.g. on
Linux/Colab CI with both installed). The both-backends-absent test
poisons `sys.modules` to force that path deterministically regardless of
what's actually installed. Pronunciation-dictionary tests force the
backend to "none" (a white-box module-state override) so they exercise
only the dictionary logic, independent of whichever backend is present.
"""

import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tts_normalizer  # noqa: E402
from tts_normalizer import TTSNormalizer  # noqa: E402


FAILURES = []

FIXED_SENTENCE = "Dr. Halloway lived on Elm St."

BATTERY = [
    FIXED_SENTENCE,
    "“Well,” she said, — pausing — “I didn't expect that.”",
    "It was... complicated. St. Peter's Basilica loomed ahead.",
    "",
    "Plain ASCII sentence with no abbreviations at all.",
]

_UNSET = object()


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def reset_backend_state():
    """White-box reset of the sticky module-level backend cache/warning
    flags, so each test that cares about backend-selection behavior can
    control it independently of what earlier tests did."""
    tts_normalizer._BACKEND_NORMALIZER = None
    tts_normalizer._BACKEND_NAME = None
    tts_normalizer._BACKEND_ATTEMPTED = False
    tts_normalizer._WARNING_PRINTED = False


def force_backend_none():
    """Pin the module-level backend state to 'none' without touching
    imports at all, so pronunciation-dict tests can isolate dictionary
    behavior regardless of which backend(s) are actually installed."""
    tts_normalizer._BACKEND_ATTEMPTED = True
    tts_normalizer._BACKEND_NORMALIZER = None
    tts_normalizer._BACKEND_NAME = "none"


def test_flag_off_is_noop():
    normalizer = TTSNormalizer(enabled=False, pronunciation_dict_path=None)
    for s in BATTERY:
        result = normalizer.normalize(s)
        check(
            f"flag-off no-op: {s[:30]!r}",
            result is s,
            f"expected identical object, got {result!r}",
        )


def test_flag_off_ignores_pronunciation_dict():
    with tempfile.TemporaryDirectory() as d:
        dict_path = os.path.join(d, "pronunciation_dict.json")
        with open(dict_path, "w", encoding="utf-8") as f:
            json.dump({"Marlborough Dr.": "Marlborough Drive"}, f)

        normalizer = TTSNormalizer(enabled=False, pronunciation_dict_path=dict_path)
        text = "She lived on Marlborough Dr. for years."
        result = normalizer.normalize(text)
        check("flag-off ignores pronunciation dict (byte-identical passthrough)", result is text)


def test_flag_off_passthrough_even_with_wetext_installed():
    # wetext is installed in this environment (see app/requirements.txt),
    # but flag-off must still be a pure no-op regardless.
    normalizer = TTSNormalizer(enabled=False, pronunciation_dict_path=None)
    result = normalizer.normalize(FIXED_SENTENCE)
    check(
        "flag-off byte-identical passthrough even with wetext installed",
        result is FIXED_SENTENCE,
        f"got {result!r}",
    )


def test_backend_selection_and_fidelity_with_wetext():
    reset_backend_state()
    buf = io.StringIO()
    normalizer = TTSNormalizer(enabled=True, pronunciation_dict_path=None)
    with contextlib.redirect_stdout(buf):
        result = normalizer.normalize(FIXED_SENTENCE)
    output = buf.getvalue()

    if tts_normalizer._BACKEND_NAME == "nemo":
        # nemo_text_processing IS importable in this environment (e.g. a
        # Linux/Colab box with it installed) and takes priority over
        # wetext -- skip with notice rather than failing; the
        # nemo-selected path is verified empirically via
        # tools/verify_tts_normalization.py there, not by this assertion.
        print("[SKIP] nemo_text_processing is importable in this environment and "
              "takes priority; the wetext-selection path was not exercised here.")
        return

    check(
        "backend selection picks wetext when nemo is absent",
        tts_normalizer._BACKEND_NAME == "wetext",
        f"got backend={tts_normalizer._BACKEND_NAME!r}",
    )
    check("backend log line names wetext", "backend: wetext" in output, f"log was {output!r}")
    check("wetext expands 'Dr.' -> 'doctor'", "doctor" in result, f"got {result!r}")
    check("wetext fidelity: 'Elm St.' left untouched", "Elm St." in result, f"got {result!r}")
    check("wetext fidelity: does not over-expand (no 'Saint')", "Saint" not in result, f"got {result!r}")


def test_both_backends_absent_warns_once_and_passthrough():
    reset_backend_state()
    # Poison sys.modules so both nemo_text_processing and wetext imports
    # fail, regardless of what's actually installed in this environment.
    saved = {}
    for name in ("nemo_text_processing", "wetext"):
        saved[name] = sys.modules.get(name, _UNSET)
        sys.modules[name] = None
    try:
        normalizer = TTSNormalizer(enabled=True, pronunciation_dict_path=None)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r1 = normalizer.normalize(FIXED_SENTENCE)
            r2 = normalizer.normalize(FIXED_SENTENCE)
            r3 = normalizer.normalize("Another sentence entirely.")
        output = buf.getvalue()

        check("both-absent: backend resolves to 'none'", tts_normalizer._BACKEND_NAME == "none")
        check("both-absent: byte-identical passthrough [1]", r1 == FIXED_SENTENCE, f"got {r1!r}")
        check("both-absent: byte-identical passthrough [2]", r2 == FIXED_SENTENCE, f"got {r2!r}")
        check("both-absent: byte-identical passthrough [3]", r3 == "Another sentence entirely.", f"got {r3!r}")

        warning_count = output.count("WARNING: no text-normalization backend is available")
        check(
            "both-absent: warning printed exactly once across 3 calls",
            warning_count == 1,
            f"got {warning_count}, output={output!r}",
        )
    finally:
        for name, val in saved.items():
            if val is _UNSET:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = val
        reset_backend_state()


def test_pronunciation_dict_applied_when_flag_on():
    with tempfile.TemporaryDirectory() as d:
        dict_path = os.path.join(d, "pronunciation_dict.json")
        with open(dict_path, "w", encoding="utf-8") as f:
            json.dump({"Marlborough Dr.": "Marlborough Drive"}, f)

        reset_backend_state()
        # Pin the backend to "none" so this test isolates the
        # pronunciation-dictionary behavior specifically, independent of
        # whichever backend(s) are actually installed.
        force_backend_none()

        normalizer = TTSNormalizer(enabled=True, pronunciation_dict_path=dict_path)
        text = "She lived on Marlborough Dr. for years."
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = normalizer.normalize(text)
        check(
            "pronunciation dict substitution applied (flag on)",
            result == "She lived on Marlborough Drive for years.",
            f"got {result!r}",
        )
        check(
            "pronunciation dict substitution logged",
            "Marlborough Dr." in buf.getvalue() and "Marlborough Drive" in buf.getvalue(),
            f"log output was: {buf.getvalue()!r}",
        )

        reset_backend_state()


def test_pronunciation_dict_not_applied_when_flag_off():
    with tempfile.TemporaryDirectory() as d:
        dict_path = os.path.join(d, "pronunciation_dict.json")
        with open(dict_path, "w", encoding="utf-8") as f:
            json.dump({"Marlborough Dr.": "Marlborough Drive"}, f)

        normalizer = TTSNormalizer(enabled=False, pronunciation_dict_path=dict_path)
        text = "She lived on Marlborough Dr. for years."
        result = normalizer.normalize(text)
        check("pronunciation dict NOT applied (flag off, byte-identical)", result is text)


def test_pronunciation_dict_longest_match_first():
    with tempfile.TemporaryDirectory() as d:
        dict_path = os.path.join(d, "pronunciation_dict.json")
        with open(dict_path, "w", encoding="utf-8") as f:
            json.dump({"Dr.": "Doctor", "Marlborough Dr.": "Marlborough Drive"}, f)

        reset_backend_state()
        force_backend_none()

        normalizer = TTSNormalizer(enabled=True, pronunciation_dict_path=dict_path)
        result = normalizer.normalize("Marlborough Dr.")
        check(
            "longest-match-first: 'Marlborough Dr.' -> 'Marlborough Drive' (not 'Marlborough Doctor')",
            result == "Marlborough Drive",
            f"got {result!r}",
        )

        reset_backend_state()


def test_missing_dict_file_no_error():
    missing_path = os.path.join(tempfile.gettempdir(), "definitely_does_not_exist_pronunciation_dict.json")
    if os.path.exists(missing_path):
        os.remove(missing_path)
    try:
        normalizer = TTSNormalizer(enabled=True, pronunciation_dict_path=missing_path)
        check("missing dict file -> empty dict, no error", normalizer._pronunciation_dict == {})
    except Exception as exc:
        check("missing dict file -> empty dict, no error", False, f"raised {exc!r}")


def test_invalid_json_dict_warns_once_and_empty():
    with tempfile.TemporaryDirectory() as d:
        dict_path = os.path.join(d, "pronunciation_dict.json")
        with open(dict_path, "w", encoding="utf-8") as f:
            f.write("{not valid json,,,")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            normalizer = TTSNormalizer(enabled=True, pronunciation_dict_path=dict_path)
        output = buf.getvalue()
        check("invalid JSON dict -> empty dict", normalizer._pronunciation_dict == {})
        check(
            "invalid JSON dict -> exactly one warning",
            output.count("WARNING: could not load pronunciation dict") == 1,
            f"got output: {output!r}",
        )


def test_invalid_dict_shape_warns():
    with tempfile.TemporaryDirectory() as d:
        dict_path = os.path.join(d, "pronunciation_dict.json")
        with open(dict_path, "w", encoding="utf-8") as f:
            json.dump(["not", "a", "dict"], f)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            normalizer = TTSNormalizer(enabled=True, pronunciation_dict_path=dict_path)
        check("non-dict-shaped JSON -> empty dict", normalizer._pronunciation_dict == {})
        check("non-dict-shaped JSON -> warning printed", "WARNING" in buf.getvalue())


def test_pronunciation_dict_respects_word_boundaries():
    """A dict entry must not fire inside a longer word.

    This is the user-facing counterpart of the rule CLAUDE.md states for the
    source pipeline: blindly expanding "Dr."/"St." breaks fidelity. Before the
    boundary rule, {"Dr": "Doctor"} rewrote *Drake* to *Doctorake*.
    """
    from tts_normalizer import _apply_pronunciation_dict as apply_dict

    check(
        "a key does not fire inside a longer word",
        apply_dict("Drake walked on", {"Dr": "Doctor"}) == "Drake walked on",
        detail=apply_dict("Drake walked on", {"Dr": "Doctor"}),
    )
    check(
        "the same key still fires as a standalone word",
        apply_dict("Elm Dr today", {"Dr": "Doctor"}) == "Elm Doctor today",
        detail=apply_dict("Elm Dr today", {"Dr": "Doctor"}),
    )
    check(
        "a key ending in punctuation keeps working",
        apply_dict("Dr. Who", {"Dr.": "Doctor"}) == "Doctor Who",
        detail=apply_dict("Dr. Who", {"Dr.": "Doctor"}),
    )
    # A letter next to a digit is a real boundary, so unit entries still work.
    check(
        "a letter key still fires against a digit neighbour",
        apply_dict("5km away", {"km": "kilometers"}) == "5kilometers away",
        detail=apply_dict("5km away", {"km": "kilometers"}),
    )
    # Uncased, unsegmented scripts have no word boundaries to respect, so
    # substring replacement remains the correct behaviour there.
    check(
        "uncased script keeps substring semantics",
        apply_dict("林考言道", {"林": "Lin"}) == "Lin考言道",
        detail=apply_dict("林考言道", {"林": "Lin"}),
    )
    # Nothing is replaced when every occurrence is embedded.
    check(
        "an entry matching nothing leaves text byte-identical",
        apply_dict("Andrew and Drake", {"Dr": "Doctor"}) == "Andrew and Drake",
        detail=apply_dict("Andrew and Drake", {"Dr": "Doctor"}),
    )


def main():
    tests = [
        test_flag_off_is_noop,
        test_flag_off_ignores_pronunciation_dict,
        test_flag_off_passthrough_even_with_wetext_installed,
        test_backend_selection_and_fidelity_with_wetext,
        test_both_backends_absent_warns_once_and_passthrough,
        test_pronunciation_dict_applied_when_flag_on,
        test_pronunciation_dict_not_applied_when_flag_off,
        test_pronunciation_dict_longest_match_first,
        test_pronunciation_dict_respects_word_boundaries,
        test_missing_dict_file_no_error,
        test_invalid_json_dict_warns_once_and_empty,
        test_invalid_dict_shape_warns,
    ]
    for t in tests:
        print(f"\n--- {t.__name__} ---")
        t()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print(f"All {len(tests)} test groups passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
