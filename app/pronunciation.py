"""A per-book pronunciation lexicon for names the TTS cannot guess.

46% of the active book's lines contain a Latinized Japanese proper noun -
`Subaru` alone in 811 of 2,609 - and an English-trained model has no way to
know that it is SU-ba-ru rather than soo-BAR-oo. Observed errors already
include `rom and` heard as `roman` and `rezero` as `risero`.

WHY A LEXICON RATHER THAN BETTER PROMPTING. The pronunciation of a name is a
fact about the book, not about the model or the sentence. It belongs in data
that a person can correct once and have applied everywhere, which is the same
reasoning behind `character_aliases.json` and `voice_config.json`.

ORTHOGRAPHIC, NOT PHONETIC. Qwen3-TTS takes text; there is no phoneme input to
address. So an entry is a RESPELLING - "Subaru" -> "Soo-bah-roo" - which the
model reads as ordinary text. That is a real limitation: respelling is a blunt
instrument compared with IPA, and what respelling actually helps is an
empirical question per name.

WHICH IS WHY THIS SHIPS EMPTY. The mechanism is here; the entries are not
guessed. `proper_noun_pronunciation.py` provides the harness to test candidate
respellings against real lines, and an entry should be added because it was
measured to help, not because it looked right.

A TENSION WORTH STATING BEFORE ANYONE MEASURES THIS. WER is the wrong gate for
a lexicon. Validation scores the transcript against the ORIGINAL text, which is
correct - "Subaru" is what the book says. But if a respelling makes the model
say the name properly and the ASR then transcribes it as "Soo bah roo", WER
gets WORSE while the audio gets BETTER. A lexicon is judged by listening, or by
an ASR comparison that normalises the name on both sides. Optimising it against
raw WER would select for names that are easy to transcribe rather than right.
"""
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(REPO, "pronunciation.json")

_cache = {"path": None, "mtime": None, "entries": {}, "pattern": None}


def _compile(entries):
    """One regex over all keys, longest first.

    Longest-first matters: with both "Natsuki" and "Natsuki Subaru" present,
    alternation would otherwise match the shorter key inside the longer name
    and leave a half-substituted phrase.
    """
    keys = sorted((k for k in entries if k), key=len, reverse=True)
    if not keys:
        return None
    return re.compile(r"(?<![\w'])(" + "|".join(re.escape(k) for k in keys)
                      + r")(?![\w'])")


def load_lexicon(path=None, force=False):
    """-> {name: respelling}. Cached on mtime so an edit takes effect."""
    path = path or DEFAULT_PATH
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        _cache.update({"path": path, "mtime": None, "entries": {},
                       "pattern": None})
        return {}
    if not force and _cache["path"] == path and _cache["mtime"] == mtime:
        return _cache["entries"]
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (ValueError, OSError):
        # A malformed lexicon must not stop a book from generating. It
        # degrades to "no substitutions", which is the previous behaviour.
        _cache.update({"path": path, "mtime": mtime, "entries": {},
                       "pattern": None})
        return {}
    source = raw.get("names") if isinstance(raw.get("names"), dict) else raw
    entries = {str(k): str(v) for k, v in source.items()
               if isinstance(k, str) and isinstance(v, str)
               and k.strip() and v.strip()}
    _cache.update({"path": path, "mtime": mtime, "entries": entries,
                   "pattern": _compile(entries)})
    return entries


def apply_pronunciation(text, path=None):
    """-> (text, [{name, spoken}]) with every applied substitution recorded.

    Returns the substitutions rather than mutating anything, so a caller can
    put them in an artifact. A silent respelling would be untraceable in the
    audio - the listener hears one thing and the script says another.
    """
    if not text:
        return text, []
    load_lexicon(path)
    pattern, entries = _cache["pattern"], _cache["entries"]
    if not pattern:
        return text, []
    applied = []

    def swap(match):
        name = match.group(1)
        spoken = entries[name]
        applied.append({"name": name, "spoken": spoken})
        return spoken

    return pattern.sub(swap, text), applied


def lexicon_names(path=None):
    """The names the lexicon knows, for reporting and tests."""
    return sorted(load_lexicon(path))
