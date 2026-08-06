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


# ── keeping the name list honest ─────────────────────────────────────────

def character_forms(script_path=None, aliases_path=None, voice_config_path=None):
    """Every written form of every character that ACTUALLY OCCURS in the text.

    WHY DERIVED RATHER THAN LISTED. The first version of this file carried a
    hardcoded list of thirteen names typed from memory. It included Ram and
    Rem, who do not appear in this book at all, and it would silently have no
    entry for a character it forgot or for any other book. The roster already
    exists in character_aliases.json and voice_config.json; a second list
    beside them is the parallel-maintenance problem this repo has been bitten
    by before.

    ALIASES ARE USED FOR DISCOVERY, NEVER FOR SUBSTITUTION. The alias map
    records IDENTITY - 'BETTY' -> 'BEATRICE' is one character - while a
    lexicon records SOUND, and those come apart. Speaking "Betty" as
    "Beatrice" would put a word in the audio that is not in the book, which is
    worse than mispronouncing it. So every written form gets its own entry and
    the canonical name confers nothing on its variants.

    CASE IS LOAD-BEARING. `Felt` is a character here and `felt` is a verb:
    242 and 65 occurrences in the same book. Matching case-insensitively would
    rewrite the verb. Only forms that occur with their own capitalisation are
    returned.
    """
    import json as _json
    root = REPO
    script_path = script_path or os.path.join(root, "chunks.json")
    aliases_path = aliases_path or os.path.join(root, "character_aliases.json")
    voice_config_path = voice_config_path or os.path.join(root,
                                                          "voice_config.json")

    def _read(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return _json.load(fh)
        except (OSError, ValueError):
            return None

    candidates = set()
    aliases = _read(aliases_path)
    if isinstance(aliases, dict):
        for variant, canonical in aliases.items():
            candidates.add(str(variant))
            candidates.add(str(canonical))
    vc = _read(voice_config_path)
    if isinstance(vc, dict):
        chars = vc.get("characters") if isinstance(vc.get("characters"), dict) else vc
        candidates.update(str(k) for k in chars)

    chunks = _read(script_path)
    if not isinstance(chunks, list):
        return {}
    text = "\n".join(str(c.get("text") or "") for c in chunks
                      if isinstance(c, dict))
    if isinstance(chunks, list):
        candidates.update(str(c.get("speaker")) for c in chunks
                          if isinstance(c, dict) and c.get("speaker"))

    # A speaker label is usually upper case while prose uses natural case, so
    # try the label AND its title-cased form - but count each spelling
    # separately, because that is what the substitution will match.
    forms = {}
    for name in candidates:
        name = name.strip()
        if len(name) < 2:
            continue
        for form in {name, name.title()}:
            n = len(re.findall(r"(?<![\w'])" + re.escape(form) + r"(?![\w'])",
                               text))
            if not n:
                continue
            # A form whose LOWERCASE also occurs is a name that collides with
            # an ordinary word: Felt/felt (242/65), Man/man (11/95), Cat/cat
            # (1/22). Case-sensitive matching already protects the verb, but
            # an editor filling this file in deserves to be warned before
            # giving "Man" a respelling.
            lower = len(re.findall(
                r"(?<![\w'])" + re.escape(form.lower()) + r"(?![\w'])", text))
            forms[form] = {"occurrences": n,
                           "collides_with_common_word": bool(lower and
                                                             form != form.lower()),
                           "lowercase_occurrences": lower}
    return dict(sorted(forms.items(),
                       key=lambda kv: -kv[1]["occurrences"]))
