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
import contextlib
import io
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
    # while the script/UI uses the canonical speaker "MISTER MARK".
    voice_config = {"Mr. Mark": {"type": "custom", "voice": "Ryan"}}
    check_eq(
        "canonicalized-key scan: canonical speaker 'MISTER MARK' resolves 'Mr. Mark' config",
        resolve_voice(voice_config, "MISTER MARK"),
        "Mr. Mark",
    )
    # And the reverse direction: config already keyed canonically, raw
    # title-prefixed speaker label should still resolve to it.
    voice_config2 = {"MISTER MARK": {"type": "custom", "voice": "Ryan"}}
    check_eq(
        "canonicalized-key scan (reverse): raw 'Mr. Mark' resolves 'MISTER MARK' config",
        resolve_voice(voice_config2, "Mr. Mark"),
        "MISTER MARK",
    )


# ---------------------------------------------------------------------------
# resolve_voice strategy 4: the gendered-title migration shim.
#
# canonicalize() used to DROP Mr/Mrs/Mme/..., so a voice_config.json written
# before the change keys "Mr. Mark" as "MARK". It now preserves them, so the
# speaker arriving from the script is "MISTER MARK". Without a shim, every
# line of every pre-existing project would miss its voice.
# ---------------------------------------------------------------------------
def test_resolve_voice_legacy_bare_config_voices_gendered_speaker():
    # Direction 1, unguarded: a legacy config cannot contain a gendered rival,
    # and refusing here would silently drop the character's lines.
    voice_config = {"MARK": {"type": "custom", "voice": "Ryan"}}
    check_eq(
        "migration: legacy 'MARK' config voices 'Mr. Mark'",
        resolve_voice(voice_config, "Mr. Mark"),
        "MARK",
    )
    check_eq(
        "migration: legacy 'MARK' config voices canonical 'MISTER MARK'",
        resolve_voice(voice_config, "MISTER MARK"),
        "MARK",
    )


def test_resolve_voice_migration_matrix_legacy_bare_config():
    # One legacy key, three speaker spellings, all must render.
    voice_config = {"SMITH": {"type": "custom", "voice": "Ryan"}}
    for speaker in ("MISTER SMITH", "MISSUS SMITH", "SMITH"):
        check_eq(
            f"migration matrix: legacy {{SMITH}} config voices '{speaker}'",
            resolve_voice(voice_config, speaker),
            "SMITH",
        )


def test_resolve_voice_gendered_rival_does_not_unvoice_the_other():
    # The regression that matters most: assigning a voice to MRS SMITH must
    # not steal, or silence, MISTER SMITH's legacy bare entry.
    voice_config = {"SMITH": {"voice": "a"}, "MRS SMITH": {"voice": "b"}}
    check_eq(
        "migration: 'MISTER SMITH' still resolves to legacy SMITH beside a MRS SMITH key",
        resolve_voice(voice_config, "MISTER SMITH"),
        "SMITH",
    )
    check_eq(
        "migration: 'MRS SMITH' resolves to its own key",
        resolve_voice(voice_config, "Mrs. Smith"),
        "MRS SMITH",
    )


def test_resolve_voice_bare_speaker_finds_a_single_gendered_key():
    # Direction 2, unambiguous: only one gendered key could be meant.
    voice_config = {"MRS SMITH": {"voice": "b"}}
    check_eq(
        "migration: bare 'SMITH' resolves the single gendered key",
        resolve_voice(voice_config, "SMITH"),
        "MRS SMITH",
    )


def test_resolve_voice_bare_speaker_ambiguous_between_gendered_keys():
    # Direction 2, guarded: two gendered rivals and no evidence which one an
    # unqualified "SMITH" meant. Picking by insertion order would give a
    # character the wrong person's voice, so refuse -- in BOTH orders.
    forward = {"MR SMITH": {"voice": "a"}, "MRS SMITH": {"voice": "b"}}
    backward = {"MRS SMITH": {"voice": "b"}, "MR SMITH": {"voice": "a"}}
    check("migration: ambiguous bare 'SMITH' returns None (MR first)",
          resolve_voice(forward, "SMITH") is None)
    check("migration: ambiguous bare 'SMITH' returns None (MRS first)",
          resolve_voice(backward, "SMITH") is None)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        resolve_voice(forward, "SMITH")
    printed = buffer.getvalue()
    check("migration: the ambiguous miss warns loudly and names the speaker",
          "SMITH" in printed and "ambiguous" in printed.lower(),
          f"stdout was {printed!r}")


def test_resolve_voice_ignores_metadata_keys():
    # "_canon_version" is file metadata, never a speaker.
    voice_config = {"_canon_version": 2, "MARK": {"voice": "a"}}
    check_eq("metadata: '_canon_version' is skipped by the key scan",
             resolve_voice(voice_config, "MARK"), "MARK")
    check("metadata: '_canon_version' is never returned as a resolved key",
          resolve_voice(voice_config, "_canon_version") is None)


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
# Over-long entry splitting (nothing between the script and the TTS call
# bounded entry length before this; group_into_chunks capped MERGING only).
#
# Properties, not fixtures: every assertion below holds for any text, and the
# inputs are generated rather than quoted from a book.
# ---------------------------------------------------------------------------
def test_split_long_text_join_is_byte_exact():
    # THE fidelity property: choosing cut points must never change the text.
    cap = 50
    for text in (
        "word " * 200,                      # whitespace everywhere
        "Sentence one. Sentence two! " * 40,  # sentence enders
        "para\n\nbreak\n\n" * 40,           # paragraph breaks
        "x" * 500,                          # no boundary at all
        "林" * 300,                     # unsegmented script, no spaces
        "a",                                # shorter than the cap
        "",                                 # empty
    ):
        pieces = project.split_long_text(text, cap)
        check_eq(f"split join byte-exact for {text[:12]!r}", "".join(pieces), text)
        check(
            f"every piece within cap for {text[:12]!r}",
            all(len(p) <= cap for p in pieces),
        )


def test_split_long_text_hard_cuts_unsegmented_script():
    # Chinese/Japanese/Thai have no whitespace to cut at. A hard cut is the
    # only option and must still bound the piece length rather than giving up
    # and returning the whole string.
    text = "林" * 260
    pieces = project.split_long_text(text, 100)
    check_eq("unsegmented text is split into bounded pieces", len(pieces), 3)
    check_eq("unsegmented split is lossless", "".join(pieces), text)


def test_split_long_text_prefers_natural_boundaries():
    # A paragraph break inside the window wins over a mid-word cut.
    text = "alpha beta\n\n" + ("gamma " * 40)
    pieces = project.split_long_text(text, 30)
    check("paragraph break is preferred as a cut point", pieces[0].endswith("\n\n"))

    # With no line breaks, a sentence end beats an arbitrary space.
    text = "One two three. Four five six seven eight nine ten eleven."
    pieces = project.split_long_text(text, 20)
    check("sentence end is preferred over a mid-clause space",
          pieces[0].startswith("One two three."))


def test_group_into_chunks_bounds_a_single_oversize_entry():
    # Before the split, ONE entry longer than the cap passed through whole.
    long_text = "This is a sentence. " * 60  # 1200 chars, one entry
    entries = [{"speaker": "NARRATOR", "text": long_text, "instruct": "x"}]
    chunks = group_into_chunks(entries)
    check("an oversize entry is split into several chunks", len(chunks) > 1)
    check(
        "no chunk exceeds MAX_CHUNK_CHARS after grouping",
        all(len(c["text"]) <= project.MAX_CHUNK_CHARS for c in chunks),
    )
    check_eq(
        "splitting an oversize entry preserves its text exactly",
        "".join(c["text"] for c in chunks),
        long_text,
    )


def test_split_pieces_do_not_introduce_pauses():
    # Same-speaker chunks get SAME_SPEAKER_PAUSE_MS between them, which would
    # insert silence mid-sentence. Only the LAST piece keeps the original
    # pause; the rest are pinned to 0 ("no gap" to combine_audio_with_pauses).
    entries = [{
        "speaker": "NARRATOR",
        "text": "Filler sentence here. " * 60,
        "instruct": "x",
        "pause_after": 700,
    }]
    chunks = group_into_chunks(entries)
    check("test needs a split to be meaningful", len(chunks) > 1)
    check_eq(
        "all but the last piece suppress the pause",
        [c.get("pause_after") for c in chunks[:-1]],
        [0] * (len(chunks) - 1),
    )
    check_eq("the last piece keeps the original pause",
             chunks[-1].get("pause_after"), 700)


def test_split_preserves_speaker_and_instruct():
    entries = [{"speaker": "ALICE", "text": "Talking. " * 100, "instruct": "warm"}]
    chunks = group_into_chunks(entries)
    check("test needs a split to be meaningful", len(chunks) > 1)
    check("every piece keeps the speaker, so the voice does not change mid-line",
          all(c["speaker"] == "ALICE" for c in chunks))
    check("every piece keeps the instruct",
          all(c["instruct"] == "warm" for c in chunks))


def test_grouping_is_unchanged_for_normal_entries():
    # The split must be a no-op on anything already within the cap, so this
    # change cannot perturb the overwhelming majority of chunks.
    entries = [
        {"speaker": "NARRATOR", "text": "Short one. ", "instruct": ""},
        {"speaker": "ALICE", "text": "A line of dialogue.", "instruct": "warm"},
        {"speaker": "NARRATOR", "text": " And narration after it.", "instruct": ""},
    ]
    chunks = group_into_chunks(entries)
    check_eq("no-op on within-cap entries: chunk count", len(chunks), 3)
    check("no-op on within-cap entries: no pause keys invented",
          all("pause_after" not in c for c in chunks))


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
        test_resolve_voice_legacy_bare_config_voices_gendered_speaker,
        test_resolve_voice_migration_matrix_legacy_bare_config,
        test_resolve_voice_gendered_rival_does_not_unvoice_the_other,
        test_resolve_voice_bare_speaker_finds_a_single_gendered_key,
        test_resolve_voice_bare_speaker_ambiguous_between_gendered_keys,
        test_resolve_voice_ignores_metadata_keys,
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
        test_split_long_text_join_is_byte_exact,
        test_split_long_text_hard_cuts_unsegmented_script,
        test_split_long_text_prefers_natural_boundaries,
        test_group_into_chunks_bounds_a_single_oversize_entry,
        test_split_pieces_do_not_introduce_pauses,
        test_split_preserves_speaker_and_instruct,
        test_grouping_is_unchanged_for_normal_entries,
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
