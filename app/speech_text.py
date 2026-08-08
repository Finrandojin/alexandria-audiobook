"""Conservative speech preparation and non-prose risk classification."""
import re
import unicodedata


SPEECH_BREAKS = "•·▪◦‣∙■□◆●▲─━―*_~"
SPEECH_WORDS = {
    "©": "copyright", "®": "registered trademark", "™": "trademark",
    "&": "and", "@": "at", "%": "percent", "°": "degrees",
    "№": "number", "§": "section", "†": "", "‡": "",
}

# Symbols with a genuine spoken form. Kept apart from SPEECH_WORDS because
# these are swept up by the catch-all below if they are not named here, and
# the distinction matters: a named symbol is spoken, an unnamed one is
# dropped. Goal 5.1.
VERBALIZED_SYMBOLS = {
    "∞": "infinity", "→": "to", "←": "from", "≈": "approximately",
    "≠": "not equal to", "≤": "at most", "≥": "at least",
    "±": "plus or minus", "×": "times", "÷": "divided by",
    "√": "the square root of", "∑": "the sum of", "µ": "micro",
}

# Unicode categories with no spoken form at all: symbols, private use,
# unassigned and surrogates. Currency (Sc) is deliberately excluded - "$" and
# "£" are speakable and belong in a table, not in a silent drop.
#
# WHY A CATCH-ALL AND NOT A LIST. The named tables above can only cover
# symbols someone thought of. An audiobook meets whatever the source file
# contains, and the failure is silent: the engine receives a character with no
# pronunciation and its behaviour is undefined - it may skip it, or emit noise,
# or mispronounce the surrounding words. Dropping the unknown is the
# conservative choice, because a dropped symbol is a symbol the listener was
# never going to hear correctly anyway.
_UNSPEAKABLE_CATEGORIES = frozenset({"So", "Sm", "Sk", "Co", "Cn", "Cs"})
REPLACEMENT_CHARACTER = "�"

_BREAK_RE = re.compile(f"\\s*[{re.escape(SPEECH_BREAKS)}]+\\s*")
_SPACE_RE = re.compile(r"[ \t]{2,}")
_ORPHAN_PUNCT_RE = re.compile(r"(?:\.\s*){2,}")
_DUPE_WORD_RE = re.compile(r"\b(\w+)(\s+\1)+\b", re.IGNORECASE)
_URL_RE = re.compile(
    r"(?:https?://|https?//|www\.)\S+|\b[a-z0-9-]+(?:\.[a-z0-9-]+)+"
    r"(?:/[a-z0-9._~:/?#\[\]@!$&'()*+,;=%-]*)?",
    re.IGNORECASE)
_LABELLED_IDENTIFIER_RE = re.compile(
    r"\b(?:isbn(?:-1[03])?|issn|lccn|lcc|ddc|doi|sku|catalog(?:ue)?|serial|model|account|"
    r"reference|ref|id)\s*(?:no\.?|number|#|:)?\s*[a-z0-9][a-z0-9._:/-]*",
    re.IGNORECASE)
_ISBN_RE = re.compile(r"\b97[89](?:-?\d){10}\b")
_CONTENTS_RE = re.compile(r"^\s*(?:contents|navigation|table of contents)\b",
                          re.IGNORECASE)
_COMPACT_IDENTIFIER_RE = re.compile(
    r"\b(?=[a-z0-9._:/-]{5,}\b)(?=[a-z0-9._:/-]*[a-z])"
    r"(?=[a-z0-9._:/-]*\d)[a-z0-9]+(?:[._:/-][a-z0-9]+)+\b",
    re.IGNORECASE)


def get_speech_risks(text):
    """Return evidence-backed categories that merit review, without mutation."""
    value = str(text or "")
    risks = []
    if _URL_RE.search(value):
        risks.append("url")
    if (_LABELLED_IDENTIFIER_RE.search(value) or _COMPACT_IDENTIFIER_RE.search(value)
            or _ISBN_RE.search(value)):
        risks.append("identifier")
    break_count = sum(value.count(mark) for mark in SPEECH_BREAKS)
    if (break_count >= 2 or (value.count("|") >= 2 and "\n" in value)
            or (_CONTENTS_RE.search(value) and value.count(".") >= 3)):
        risks.append("list_or_table")
    return risks


def verbalize_symbols(text):
    """-> (text, transformations). Named symbols spoken, unknown ones dropped.

    Runs after SPEECH_WORDS and the structural-break pass so anything those
    already handle keeps its existing behaviour - `■` stays a sentence break
    rather than becoming a silent drop.
    """
    spoken, dropped, out = [], [], []
    for ch in text:
        word = VERBALIZED_SYMBOLS.get(ch)
        if word:
            out.append(f" {word} ")
            spoken.append(ch)
        elif ch == REPLACEMENT_CHARACTER or (
                not ch.isspace()
                and unicodedata.category(ch) in _UNSPEAKABLE_CATEGORIES):
            out.append(" ")
            dropped.append(ch)
        else:
            out.append(ch)
    transformations = []
    if spoken:
        transformations.append({"type": "verbalized_symbol",
                                "symbols": sorted(set(spoken))})
    if dropped:
        # Recorded, not silent: a character removed without a trace is
        # indistinguishable from one that was never in the source, and this
        # list is the evidence for goal 5.1's count.
        transformations.append({"type": "dropped_unspeakable",
                                "symbols": sorted(set(dropped)),
                                "count": len(dropped)})
    return "".join(out), transformations


def get_speech_normalization(text):
    """Return normalized text and every applied transformation as new data."""
    if not text:
        return {"text": text, "changed": False, "transformations": [],
                "risk_categories": get_speech_risks(text)}
    original = str(text)
    normalized = original
    transformations = []
    for symbol, word in SPEECH_WORDS.items():
        if symbol in normalized:
            normalized = normalized.replace(symbol, f" {word} " if word else " ")
            transformations.append({"type": "spoken_symbol" if word else "dropped_reference_mark",
                                    "symbol": symbol, "replacement": word})
    if _BREAK_RE.search(normalized):
        normalized = _BREAK_RE.sub(". ", normalized)
        transformations.append({"type": "structural_break"})
    normalized, symbol_transformations = verbalize_symbols(normalized)
    transformations.extend(symbol_transformations)
    normalized = _ORPHAN_PUNCT_RE.sub(". ", normalized)
    normalized = _SPACE_RE.sub(" ", normalized)
    deduplicated = _DUPE_WORD_RE.sub(r"\1", normalized)
    if deduplicated != normalized:
        transformations.append({"type": "duplicate_spoken_word"})
    normalized = deduplicated
    # Proper-noun respellings, applied LAST so a lexicon entry is never
    # mangled by symbol or break handling. Recorded as a transformation like
    # everything else: a silent respelling would be untraceable, with the
    # listener hearing one thing and the script saying another.
    try:
        from pronunciation import apply_pronunciation
        spoken, applied = apply_pronunciation(normalized)
        if applied:
            normalized = spoken
            transformations.append({"type": "pronunciation_lexicon",
                                    "substitutions": applied})
    except Exception:                                   # noqa: BLE001
        # A broken lexicon must never stop a book generating.
        pass
    stripped = normalized.strip(" .\t\n")
    normalized = stripped + "." if stripped else ""
    return {"text": normalized, "changed": normalized != original,
            "transformations": transformations,
            "risk_categories": get_speech_risks(original)}


def normalize_for_speech(text):
    """Return only the prepared text for existing TTS callers."""
    return get_speech_normalization(text)["text"]
