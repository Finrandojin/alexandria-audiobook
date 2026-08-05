"""Conservative speech preparation and non-prose risk classification."""
import re


SPEECH_BREAKS = "•·▪◦‣∙■□◆●▲─━―*_~"
SPEECH_WORDS = {
    "©": "copyright", "®": "registered trademark", "™": "trademark",
    "&": "and", "@": "at", "%": "percent", "°": "degrees",
    "№": "number", "§": "section", "†": "", "‡": "",
}
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
    normalized = _ORPHAN_PUNCT_RE.sub(". ", normalized)
    normalized = _SPACE_RE.sub(" ", normalized)
    deduplicated = _DUPE_WORD_RE.sub(r"\1", normalized)
    if deduplicated != normalized:
        transformations.append({"type": "duplicate_spoken_word"})
    normalized = deduplicated
    stripped = normalized.strip(" .\t\n")
    normalized = stripped + "." if stripped else ""
    return {"text": normalized, "changed": normalized != original,
            "transformations": transformations,
            "risk_categories": get_speech_risks(original)}


def normalize_for_speech(text):
    """Return only the prepared text for existing TTS callers."""
    return get_speech_normalization(text)["text"]
