"""Standalone tests for the F3 integration fixes:

  - tts.resolve_voice: canonical-aware voice_config lookup (audit #4),
    shared between project.py's _resolve_alias and tts.py's own lookups.
  - tts._prepare_batch_chunks: exactly-once text normalization + speaker
    resolution for generate_batch's entry point (audit #5), built as a
    read-only pure helper (audit re-finding N3) -- it must never mutate
    the caller's chunk dicts in place.
  - project.group_into_chunks: whitespace-aware join (audit #9-partial).
  - app._process_completion_message: exit-code-3 message selection for
    the "script" task family (audit #7 UX).

`tts.py` / `project.py` import numpy, soundfile, and pydub for their real
TTS/audio-processing work, none of which this test exercises (it only
calls resolve_voice(), _prepare_batch_chunks(), group_into_chunks(), and
_process_completion_message -- all pure functions with no audio/tensor
dependencies). Those heavy, possibly-uninstalled third-party packages are
stubbed in sys.modules before import, following the same pattern used by
test_epub_extract.py for the (unrelated) `project` stub there.

Run directly:
    python app/test_integration_fixes.py
Exits 0 if all tests pass, non-zero otherwise.
"""
import os
import sys
import types
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Stub heavy/possibly-uninstalled third-party deps that tts.py / project.py
# import at module level purely for real audio/model work this test never
# calls into. `speaker_canon` (rapidfuzz) is a real, light dependency and is
# NOT stubbed -- resolve_voice's canonicalization must be genuine.
# ---------------------------------------------------------------------------
for _mod_name in ("numpy", "soundfile"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)

if "pydub" not in sys.modules:
    _fake_pydub = types.ModuleType("pydub")

    class _FakeAudioSegment:
        """Placeholder; tts.py/project.py only reference AudioSegment inside
        function bodies this test never calls."""
        pass

    _fake_pydub.AudioSegment = _FakeAudioSegment
    sys.modules["pydub"] = _fake_pydub

if "huggingface_hub" not in sys.modules:
    sys.modules["huggingface_hub"] = types.ModuleType("huggingface_hub")

import tts  # noqa: E402
import project  # noqa: E402

resolve_voice = tts.resolve_voice
group_into_chunks = project.group_into_chunks


# ---------------------------------------------------------------------------
# Stub 'project' for app.py's import, so importing app.py (for
# _process_completion_message) doesn't need a real ProjectManager or pull in
# anything else transitively. Mirrors test_epub_extract.py's approach.
# Must happen with a *separate* module identity from the real `project`
# module we just imported above -- app.py does `from project import
# ProjectManager`, so we only need that one name to resolve.
# ---------------------------------------------------------------------------
_real_project_module = sys.modules.get("project")


class _FakeProjectManager:
    def __init__(self, *args, **kwargs):
        pass

    def load_chunks(self):
        return []

    def save_chunks(self, chunks):
        pass

    def __getattr__(self, name):
        # app.py's module-level startup code may touch other attributes;
        # turn any of those into harmless no-op callables.
        def _noop(*args, **kwargs):
            return None
        return _noop


# Temporarily swap in a fake ProjectManager just for app.py's `from project
# import ProjectManager` line, then restore the real module so later tests
# in this file (if any) still see the genuine `project` module.
_fake_project_for_app = types.ModuleType("project")
_fake_project_for_app.ProjectManager = _FakeProjectManager
sys.modules["project"] = _fake_project_for_app
try:
    import app as alexandria_app  # noqa: E402
finally:
    if _real_project_module is not None:
        sys.modules["project"] = _real_project_module

_process_completion_message = alexandria_app._process_completion_message


# ---------------------------------------------------------------------------
# Test harness (no external test framework dependency)
# ---------------------------------------------------------------------------
_failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS: {name}")
    else:
        msg = f"  FAIL: {name}" + (f" -- {detail}" if detail else "")
        print(msg)
        _failures.append(name)


def check_eq(name, actual, expected):
    check(name, actual == expected, f"expected {expected!r}, got {actual!r}")


# ---------------------------------------------------------------------------
# resolve_voice tests
# ---------------------------------------------------------------------------
def test_resolve_voice_raw_hit():
    voice_config = {"MARK": {"type": "custom", "voice": "Ryan"}}
    check_eq("raw exact key hit", resolve_voice(voice_config, "MARK"), "MARK")


def test_resolve_voice_canonical_hit_narrator():
    # Single-speaker scripts historically wrote raw "Narrator" (mixed case)
    # into chunks.json; voice_config is now keyed by canonical "NARRATOR".
    voice_config = {"NARRATOR": {"type": "custom", "voice": "Ryan"}}
    check_eq(
        "canonical hit: raw 'Narrator' resolves config under 'NARRATOR'",
        resolve_voice(voice_config, "Narrator"),
        "NARRATOR",
    )
    check_eq(
        "canonical hit: raw 'narrator' (lowercase) also resolves",
        resolve_voice(voice_config, "narrator"),
        "NARRATOR",
    )


def test_resolve_voice_canonicalized_key_scan():
    # Legacy voice_config keyed by a raw, non-canonical label ("Mr. Mark"),
    # while the script/UI now uses the canonical speaker "MARK".
    voice_config = {"Mr. Mark": {"type": "custom", "voice": "Ryan"}}
    check_eq(
        "canonicalized-key scan: canonical speaker 'MARK' resolves 'Mr. Mark' config",
        resolve_voice(voice_config, "MARK"),
        "Mr. Mark",
    )
    # And the reverse direction: config already keyed canonically, raw
    # honorific-prefixed speaker label should still resolve to it.
    voice_config2 = {"MARK": {"type": "custom", "voice": "Ryan"}}
    check_eq(
        "canonicalized-key scan (reverse): raw 'Mr. Mark' resolves 'MARK' config",
        resolve_voice(voice_config2, "Mr. Mark"),
        "MARK",
    )


def test_resolve_voice_miss_returns_none():
    voice_config = {"NARRATOR": {"type": "custom"}}
    check("miss returns None", resolve_voice(voice_config, "ELLA") is None)
    check("empty speaker returns None", resolve_voice(voice_config, "") is None)
    check("empty voice_config returns None", resolve_voice({}, "NARRATOR") is None)


def test_resolve_voice_alias_chain():
    voice_config = {
        "JON": {"alias_of": "JOHN"},
        "JOHN": {"type": "custom", "voice": "Ryan"},
    }
    check_eq("alias chain follows alias_of", resolve_voice(voice_config, "JON"), "JOHN")
    # Alias chain should also be canonical-aware at the raw-speaker hop.
    check_eq(
        "alias chain resolves from a raw (non-canonical) speaker label",
        resolve_voice(voice_config, "Jon"),
        "JOHN",
    )


def test_resolve_voice_alias_cycle_safe():
    voice_config = {
        "A": {"alias_of": "B"},
        "B": {"alias_of": "A"},
    }
    # Must not infinite-loop; some deterministic name is returned.
    result = resolve_voice(voice_config, "A")
    check("alias cycle terminates", result in ("A", "B"))


# ---------------------------------------------------------------------------
# _prepare_batch_chunks tests (audit re-finding N3: generate_batch's entry
# loop must never mutate the caller's chunk dicts in place -- a future
# caller passing persisted load_chunks() dicts would otherwise get
# normalized text + rewritten speakers silently written back into
# chunks.json, causing double-normalization on the next run).
# ---------------------------------------------------------------------------
import copy  # noqa: E402


class _RecordingNormalizer:
    """Minimal stand-in for TTSNormalizer: uppercases text so a change is
    visually obvious, and records every string it was asked to normalize."""

    def __init__(self):
        self.calls = []

    def normalize(self, text):
        self.calls.append(text)
        return text.upper()


class _IdentityNormalizer:
    """Stand-in for a disabled TTSNormalizer: returns the exact same string
    object, unchanged -- this is the real TTSNormalizer's flag-off contract."""

    def normalize(self, text):
        return text


def test_prepare_batch_chunks_does_not_mutate_input():
    voice_config = {"NARRATOR": {"type": "custom", "voice": "Ryan"}}
    original = [
        {"index": 0, "speaker": "Narrator", "text": "hello there", "instruct": "neutral"},
        {"index": 1, "speaker": "Narrator", "text": "second line", "instruct": ""},
    ]
    snapshot = copy.deepcopy(original)

    _ = tts._prepare_batch_chunks(original, voice_config, _RecordingNormalizer())

    check_eq("input chunk list unchanged after call (deep-equality)", original, snapshot)


def test_prepare_batch_chunks_output_normalized_and_resolved():
    voice_config = {"NARRATOR": {"type": "custom", "voice": "Ryan"}}
    original = [
        {"index": 0, "speaker": "Narrator", "text": "hello there", "instruct": "neutral"},
    ]

    prepared = tts._prepare_batch_chunks(original, voice_config, _RecordingNormalizer())

    check_eq("prepared list has one entry", len(prepared), 1)
    check_eq("prepared text is normalized", prepared[0]["text"], "HELLO THERE")
    check_eq(
        "prepared speaker resolved to canonical voice_config key",
        prepared[0]["speaker"],
        "NARRATOR",
    )
    check_eq("prepared entry keeps unrelated fields (index)", prepared[0]["index"], 0)
    check_eq("prepared entry keeps instruct untouched", prepared[0]["instruct"], "neutral")
    check(
        "prepared dict is a distinct object from the input dict",
        prepared[0] is not original[0],
    )


def test_prepare_batch_chunks_instruct_never_normalized():
    voice_config = {"NARRATOR": {"type": "custom"}}
    original = [{"index": 0, "speaker": "NARRATOR", "text": "hi", "instruct": "hi"}]
    normalizer = _RecordingNormalizer()

    prepared = tts._prepare_batch_chunks(original, voice_config, normalizer)

    check_eq("only 'text' passed to normalizer, never 'instruct'", normalizer.calls, ["hi"])
    check_eq("instruct field left exactly as-is", prepared[0]["instruct"], "hi")


def test_prepare_batch_chunks_flag_off_is_byte_identical():
    voice_config = {"NARRATOR": {"type": "custom"}}
    original = [{"index": 0, "speaker": "NARRATOR", "text": "St. Peter's arrived.", "instruct": ""}]

    prepared = tts._prepare_batch_chunks(original, voice_config, _IdentityNormalizer())

    check_eq(
        "flag-off (identity normalizer): text is byte-identical",
        prepared[0]["text"],
        original[0]["text"],
    )
    check(
        "flag-off (identity normalizer): text is the SAME string object (no copy)",
        prepared[0]["text"] is original[0]["text"],
    )


def test_prepare_batch_chunks_unresolvable_speaker_kept_raw():
    # No canonical-aware match anywhere -- speaker is left as the original
    # raw value (so the caller's later raw .get(speaker) lookup, and the
    # printed warning, both still make sense) rather than becoming None.
    voice_config = {"NARRATOR": {"type": "custom"}}
    original = [{"index": 0, "speaker": "SOMEONE_ELSE", "text": "hi", "instruct": ""}]

    prepared = tts._prepare_batch_chunks(original, voice_config, _IdentityNormalizer())

    check_eq(
        "unresolvable speaker is left unchanged, not set to None",
        prepared[0]["speaker"],
        "SOMEONE_ELSE",
    )


# ---------------------------------------------------------------------------
# group_into_chunks whitespace-aware join tests
# ---------------------------------------------------------------------------
def test_join_verbatim_no_injected_space():
    # Both entries already carry their own boundary whitespace (trailing
    # space on the first, none needed on the second) -- verbatim join must
    # NOT inject an extra character that was never in the source.
    entries = [
        {"speaker": "NARRATOR", "text": "Hello there. ", "instruct": ""},
        {"speaker": "NARRATOR", "text": "The next sentence.", "instruct": ""},
    ]
    chunks = group_into_chunks(entries)
    check_eq("verbatim join count", len(chunks), 1)
    check_eq(
        "verbatim join produces no injected space",
        chunks[0]["text"],
        "Hello there. The next sentence.",
    )


def test_join_verbatim_leading_space_on_second():
    entries = [
        {"speaker": "NARRATOR", "text": "Hello there.", "instruct": ""},
        {"speaker": "NARRATOR", "text": " The next sentence.", "instruct": ""},
    ]
    chunks = group_into_chunks(entries)
    check_eq(
        "verbatim join: leading space on second entry is respected, not doubled",
        chunks[0]["text"],
        "Hello there. The next sentence.",
    )


def test_join_legacy_stripped_entries_get_readable_join():
    # Legacy/stripped entries with no boundary whitespace on either side
    # still get a readable space injected (old behavior preserved).
    entries = [
        {"speaker": "NARRATOR", "text": "Hello there.", "instruct": ""},
        {"speaker": "NARRATOR", "text": "The next sentence.", "instruct": ""},
    ]
    chunks = group_into_chunks(entries)
    check_eq(
        "legacy stripped join still inserts one readable space",
        chunks[0]["text"],
        "Hello there. The next sentence.",
    )


def test_join_newline_boundary_not_doubled():
    entries = [
        {"speaker": "NARRATOR", "text": "Hello there.\n", "instruct": ""},
        {"speaker": "NARRATOR", "text": "The next sentence.", "instruct": ""},
    ]
    chunks = group_into_chunks(entries)
    check_eq(
        "newline boundary counts as whitespace; no extra space injected",
        chunks[0]["text"],
        "Hello there.\nThe next sentence.",
    )


# ---------------------------------------------------------------------------
# app._process_completion_message tests
# ---------------------------------------------------------------------------
def test_process_completion_message_success():
    check_eq(
        "success message unchanged",
        _process_completion_message("script", 0),
        "Task script completed successfully.",
    )


def test_process_completion_message_exit3_script_task():
    msg = _process_completion_message("script", 3)
    check("exit-3 script message mentions degradations, not 'failed'", "degradations" in msg and "failed" not in msg)
    check("exit-3 script message mentions NARRATOR fallback", "NARRATOR" in msg)


def test_process_completion_message_other_nonzero_unchanged():
    check_eq(
        "generic nonzero code still says failed",
        _process_completion_message("script", 1),
        "Task script failed with return code 1.",
    )
    check_eq(
        "exit code 3 for an unrelated task family is still a generic failure",
        _process_completion_message("audio", 3),
        "Task audio failed with return code 3.",
    )


# ---------------------------------------------------------------------------
# Normalization flag-off passthrough (structural check, not a live import of
# TTSEngine which needs a real torch/model stack). This asserts the actual
# TTSNormalizer contract used by generate_batch/generate_voice: disabled ->
# returns the exact same string object, unchanged.
# ---------------------------------------------------------------------------
def test_normalizer_disabled_is_identity_passthrough():
    from tts_normalizer import TTSNormalizer

    normalizer = TTSNormalizer(enabled=False)
    text = "St. Peter's arrived at 3pm."
    result = normalizer.normalize(text)
    check("disabled normalizer returns same object (no copy/mutation)", result is text)


def run():
    tests = [
        test_resolve_voice_raw_hit,
        test_resolve_voice_canonical_hit_narrator,
        test_resolve_voice_canonicalized_key_scan,
        test_resolve_voice_miss_returns_none,
        test_resolve_voice_alias_chain,
        test_resolve_voice_alias_cycle_safe,
        test_prepare_batch_chunks_does_not_mutate_input,
        test_prepare_batch_chunks_output_normalized_and_resolved,
        test_prepare_batch_chunks_instruct_never_normalized,
        test_prepare_batch_chunks_flag_off_is_byte_identical,
        test_prepare_batch_chunks_unresolvable_speaker_kept_raw,
        test_join_verbatim_no_injected_space,
        test_join_verbatim_leading_space_on_second,
        test_join_legacy_stripped_entries_get_readable_join,
        test_join_newline_boundary_not_doubled,
        test_process_completion_message_success,
        test_process_completion_message_exit3_script_task,
        test_process_completion_message_other_nonzero_unchanged,
        test_normalizer_disabled_is_identity_passthrough,
    ]

    for t in tests:
        print(f"{t.__name__}:")
        try:
            t()
        except Exception:
            print(f"  ERROR: {t.__name__} raised an exception:")
            traceback.print_exc()
            _failures.append(t.__name__)

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): {_failures}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
