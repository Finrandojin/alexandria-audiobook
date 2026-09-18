"""Optional TTS-boundary text normalization for Alexandria.

FIDELITY CONTRACT (do not violate):
  - `annotated_script.json` (and every other script file) stays byte-verbatim
    on disk. Nothing in this module reads or writes script files -- it only
    ever transforms an in-memory `text` string at the moment it is about to
    be handed to the TTS engine (see `TTSEngine.generate_voice` in tts.py).
    The source -> annotated_script.json word-coverage invariant (1.0) is
    unaffected because this module never touches that pipeline.
  - `instruct_text` must NEVER be passed through `normalize()`. The TTS
    instruct/delivery-direction string is not narration text and must reach
    the model unchanged. Callers are responsible for only ever calling
    `normalize()` on the narration `text`, never on `instruct_text`.
  - With the feature disabled (the default), `normalize()` is a pure no-op:
    it returns the exact same string object passed in, with zero side
    effects (no imports triggered, no file reads, nothing printed).

WHY THIS DEFAULTS OFF:
  Alexandria's audiobooks are verbatim -- the spoken word must match the
  source text exactly. The original Alexandria pipeline performs zero text
  normalization, and Qwen3-TTS may already handle abbreviations reasonably
  well on its own. Both normalization backends below lowercase some of
  their output (inaudible, but it breaks byte-verbatim comparisons) and
  are non-trivial dependencies. Whether enabling normalization is a net
  win is an empirical question to be answered per-project with
  `tools/verify_tts_normalization.py`, not assumed here.

BACKEND CHAIN (selected lazily, cached for the life of the process):
  1. `nemo_text_processing` -- tried first when the flag is on. Highest
     fidelity: context-aware (e.g. it turns "St. Peter's" into "Saint
     Peter's" but correctly leaves "Elm St." alone). Its `pynini`
     dependency ships source-only for Windows (no prebuilt wheel; it
     requires the OpenFst C++ toolchain to build), so it is normally only
     available on Linux/Colab. Install with `pip install
     nemo_text_processing`.
  2. `wetext` -- tried if nemo is unavailable. A pynini-free FST
     normalizer with prebuilt wheels (including Windows), so it is the
     practical option on a Windows dev box. It expands things like
     "Dr." -> "doctor", money, times, and dates, and -- like nemo --
     correctly leaves "Elm St." untouched (it does not catch every case
     nemo does, e.g. it does not expand "St. Peter's" to "Saint Peter's",
     but it never breaks fidelity by over-expanding). Install with
     `pip install wetext`.
  3. Neither available -- normalization for this step is skipped, a
     single loud warning is printed (once per process), and text passes
     through unchanged for this step.
  In all three cases, the per-book pronunciation dictionary (below) is
  applied afterward, since it exists precisely to correct whichever
  backend's mistakes (or to do the only normalization available, when
  neither backend is present).

PRONUNCIATION DICTIONARY:
  A per-book, user-maintained, optional JSON file (`pronunciation_dict.json`
  at the repo root, alongside `annotated_script.json`) of flat
  `{"from": "to"}` literal string replacements. This exists to let a user
  correct mistakes the backend makes -- e.g. it might turn "Marlborough
  Dr." into "Marlborough doctor" when it should be "Marlborough Drive";
  the user adds `{"Marlborough Dr.": "Marlborough Drive"}` to fix just
  that case. This is NOT a shipped abbreviation-expansion table (that
  would break fidelity for the general case, e.g. blindly expanding "Elm
  St." to "Elm Saint") -- it is manually curated, per-book data, applied
  only when normalization is enabled.
"""

import json
import os

# Sticky, process-wide state so we select a backend, compile its (slow)
# normalizer graph, and print any "backend unavailable" warning at most
# once per process, no matter how many TTSNormalizer instances are created
# or how many times normalize() is called.
_BACKEND_NORMALIZER = None   # cached backend instance (nemo Normalizer or wetext Normalizer)
_BACKEND_NAME = None         # "nemo" | "wetext" | "none", once selection has run
_BACKEND_ATTEMPTED = False   # sticky: selection has been attempted (success or not)
_WARNING_PRINTED = False     # "no backend available" warning, printed at most once


def _warn_no_backend_available(nemo_exc, wetext_exc):
    global _WARNING_PRINTED
    if _WARNING_PRINTED:
        return
    _WARNING_PRINTED = True
    print(
        "[tts-normalize] " + "=" * 68 + "\n"
        "[tts-normalize] WARNING: no text-normalization backend is available in\n"
        "[tts-normalize] this Python environment -- normalization is enabled in\n"
        "[tts-normalize] config but will be SKIPPED for every TTS call in this\n"
        "[tts-normalize] process (text is passed through as-is for this step; the\n"
        "[tts-normalize] pronunciation dictionary, if any, still applies).\n"
        f"[tts-normalize]   nemo_text_processing import error: {nemo_exc}\n"
        f"[tts-normalize]   wetext import error: {wetext_exc}\n"
        "[tts-normalize] Install one of:\n"
        "[tts-normalize]     pip install nemo_text_processing   (Linux/Colab; highest fidelity)\n"
        "[tts-normalize]     pip install wetext                 (Windows-friendly, prebuilt wheels)\n"
        "[tts-normalize] " + "=" * 68
    )


def _select_backend():
    """Lazily select and cache a normalization backend: nemo, else wetext,
    else none. Attempted at most once per process; logs exactly one
    `[tts-normalize] backend: ...` line for that attempt.

    Returns (normalizer_instance_or_None, backend_name).
    """
    global _BACKEND_NORMALIZER, _BACKEND_NAME, _BACKEND_ATTEMPTED

    if _BACKEND_ATTEMPTED:
        return _BACKEND_NORMALIZER, _BACKEND_NAME
    _BACKEND_ATTEMPTED = True

    nemo_exc = None
    wetext_exc = None

    try:
        from nemo_text_processing.text_normalization.normalize import Normalizer as _NemoNormalizer
        _BACKEND_NORMALIZER = _NemoNormalizer(input_case="cased", lang="en")
        _BACKEND_NAME = "nemo"
    except Exception as exc:  # ImportError, missing pynini, graph build failure, etc.
        nemo_exc = exc
        try:
            from wetext import Normalizer as _WetextNormalizer
            _BACKEND_NORMALIZER = _WetextNormalizer(lang="en", operator="tn")
            _BACKEND_NAME = "wetext"
        except Exception as exc2:
            wetext_exc = exc2
            _BACKEND_NORMALIZER = None
            _BACKEND_NAME = "none"
            _warn_no_backend_available(nemo_exc, wetext_exc)

    print(f"[tts-normalize] backend: {_BACKEND_NAME}")
    return _BACKEND_NORMALIZER, _BACKEND_NAME


def _apply_backend_normalization(text):
    normalizer, backend_name = _select_backend()
    if normalizer is None:
        return text

    if backend_name == "nemo":
        normalized = normalizer.normalize(text, verbose=False)
    else:  # "wetext"
        normalized = normalizer.normalize(text)

    if normalized != text:
        print(f"[tts-normalize] {backend_name}: {text!r} -> {normalized!r}")
    return normalized


def load_pronunciation_dict(path):
    """Load the flat {"from": "to"} pronunciation dictionary.

    Missing file -> {} silently (this is the common, expected case; not
    every book needs one). Invalid JSON or wrong shape -> {} with exactly
    one printed warning.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("pronunciation dict JSON must be a flat object of string -> string")
        for key, value in data.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("pronunciation dict entries must all be strings")
        return data
    except Exception as exc:
        print(f"[tts-normalize] WARNING: could not load pronunciation dict at {path!r}: {exc}. Using empty dict.")
        return {}


def _is_cased_letter(ch):
    """True for a letter belonging to a script that distinguishes upper and
    lower case (Latin, Greek, Cyrillic, Armenian, ...). Derived from the
    Unicode properties of the character itself -- no script list, no language
    table.

    Used to decide whether word-boundary logic is meaningful around a match.
    Cased scripts in this pipeline are the space-segmented ones where "Dr"
    must not match inside "Drake"; uncased scripts include the unsegmented
    ones (Chinese, Japanese) where a substring match is the CORRECT behaviour
    because there are no word boundaries to respect.
    """
    return ch.isalpha() and ch.lower() != ch.upper()


def _match_respects_word_boundary(text, start, key):
    """True when the occurrence of `key` at `text[start:]` is not embedded in
    a larger word.

    A boundary is only required on an edge where the KEY itself ends in a
    cased letter or a digit: an entry like "Dr." ends in punctuation, so the
    period is already the boundary, while a bare "Dr" must not fire inside
    "Drake". An edge whose key character is an uncased letter imposes no
    requirement, so unsegmented scripts keep substring semantics.

    Only a SAME-CLASS neighbour blocks -- letter beside letter, digit beside
    digit. A letter adjacent to a digit is a real boundary, which is what
    keeps the common unit entry working: "km" must still fire in "5km", while
    "Dr" must still not fire in "Drake".
    """
    end = start + len(key)

    def blocks(edge_char, neighbour):
        if _is_cased_letter(edge_char):
            return _is_cased_letter(neighbour)
        if edge_char.isdigit():
            return neighbour.isdigit()
        return False

    if start > 0 and blocks(key[0], text[start - 1]):
        return False
    if end < len(text) and blocks(key[-1], text[end]):
        return False
    return True


def _replace_on_boundaries(text, key, value):
    """Replace every word-boundary-respecting occurrence of `key`. Returns the
    new text. Scans left to right and never re-examines inserted text, so a
    value containing its own key cannot loop.
    """
    out = []
    index = 0
    while True:
        found = text.find(key, index)
        if found == -1:
            out.append(text[index:])
            return "".join(out)
        if _match_respects_word_boundary(text, found, key):
            out.append(text[index:found])
            out.append(value)
            index = found + len(key)
        else:
            out.append(text[index:found + len(key)])
            index = found + len(key)


def _apply_pronunciation_dict(text, mapping):
    """Apply the user's per-book pronunciation dictionary.

    WORD-BOUNDARY AWARE. This was a raw ``str.replace``, which made the
    user-facing pronunciation_dict.json a way to reintroduce exactly the
    damage CLAUDE.md bans in the source pipeline: an entry of
    ``{"Dr": "Doctor"}`` rewrote *Drake* to *Doctorake*, and *Elm St.* to
    *Elm Saint.* if the user added "St". Longest-match-first ordering only
    protects against a shorter key clobbering a longer phrase; it does
    nothing about a key matching inside an unrelated word.

    See _match_respects_word_boundary for the exact rule and why it is
    derived from Unicode casedness rather than a script list.
    """
    if not mapping:
        return text
    # Longest-match-first: "Marlborough Dr." must win over a shorter "Dr."
    # entry so we don't clobber part of a longer phrase first.
    for key in sorted(mapping.keys(), key=len, reverse=True):
        if key and key in text:
            value = mapping[key]
            new_text = _replace_on_boundaries(text, key, value)
            if new_text != text:
                print(f"[tts-normalize] dict: {key!r} -> {value!r}")
            text = new_text
    return text


class TTSNormalizer:
    """Optional text normalization applied only at the TTS call boundary.

    Construction never fails and never raises: a missing/invalid
    pronunciation dictionary degrades to an empty dict (with a warning for
    the invalid case). The normalization backend (nemo, then wetext, then
    none) is selected lazily on first use of normalize(), not at
    construction time.

    IMPORTANT: only ever call `normalize()` on narration `text`. Never call
    it on `instruct_text` -- see module docstring.
    """

    def __init__(self, enabled: bool, pronunciation_dict_path: str = None):
        self.enabled = bool(enabled)
        self._dict_path = pronunciation_dict_path
        # The pronunciation dict is cheap, independent, user-owned data. We
        # only bother loading it when the feature is enabled -- when
        # disabled, normalize() never even glances at it.
        self._pronunciation_dict = (
            load_pronunciation_dict(pronunciation_dict_path) if self.enabled else {}
        )

    def normalize(self, text: str) -> str:
        """Return `text`, optionally normalized.

        When disabled: returns `text` unchanged (same object, zero side
        effects) -- this is the default and is what keeps the TTS input
        byte-identical to the source script text.

        When enabled: applies the selected backend's text normalization
        (nemo, else wetext, else a no-op with a warning) followed by the
        pronunciation dictionary (longest-match-first, literal,
        case-sensitive substitution).
        """
        if not self.enabled:
            return text

        normalized = _apply_backend_normalization(text)
        normalized = _apply_pronunciation_dict(normalized, self._pronunciation_dict)
        return normalized
