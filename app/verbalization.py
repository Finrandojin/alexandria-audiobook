"""Decide what the narrator voices when the source contains unspeakable marks.

The script keeps the source's characters for fidelity; this module decides what
actually reaches TTS. Split by class because the classes need different
handling: a scene break is a pause, an arrow is a word, a music note is a
delivery cue, and a pictographic kana needs a human.

Sizing comes from a scan of 4,195 novels: scene breaks are 105,897 occurrences
against 17,115 for verbalization, and U+2500 alone outweighs every other symbol
combined. See docs/superpowers/specs/2026-07-25-unspeakable-passthrough-design.md.
"""

import unicodedata

# Scene / section breaks: silence, never speech. Ordered by measured frequency.
SCENE_BREAK_CHARS = frozenset("─◇◆■□○━█△●▪▫✧❄")

# Spoken renderings. Deliberately small and explicit rather than a
# category-wide rule, so every substitution is auditable.
VERBALIZE = {
    "∞": "infinity",
    "←": "left arrow", "→": "right arrow", "↑": "up arrow", "↓": "down arrow",
    "°": "degrees", "©": "copyright", "×": "times", "÷": "divided by",
    "±": "plus or minus", "≠": "not equal to", "≈": "approximately",
    "★": "star", "☆": "star", "♥": "heart",
}

# Delivery cues, NOT words. A music note is deliberately absent from VERBALIZE:
# it usually brackets sung dialogue, and "music note la la la music note" is
# worse than leaving it in place.
ELONGATION_CHARS = frozenset("~～")
MUSIC_CHARS = frozenset("♪♫")

# Appended to an entry's existing instruct when a cue is removed from the text,
# so the signal survives instead of being silently deleted.
ELONGATION_HINT = "Drawn-out, elongated delivery."
SUNG_HINT = "Sung rather than spoken."

_SYMBOL_CATEGORIES = frozenset({"So", "Sm", "Sk"})
_KANA_START, _KANA_END = "぀", "ヿ"


def is_kana(char):
    return _KANA_START <= char <= _KANA_END


def is_pictographic_kana(char, neighbours):
    """Whether a kana is used as a picture rather than as language.

    Detects the manga convention where a kana describes a shape, as in
    "her mouth へ". A lone kana with no kana neighbours is a drawing; kana
    among kana is Japanese text.
    """
    if not is_kana(char):
        return False
    return not any(is_kana(c) for c in neighbours)


def classify(char):
    """Return scene_break, verbalize, elongation, music, review, or speakable."""
    if char in SCENE_BREAK_CHARS:
        return "scene_break"
    if char in VERBALIZE:
        return "verbalize"
    if char in ELONGATION_CHARS:
        return "elongation"
    if char in MUSIC_CHARS:
        return "music"
    if unicodedata.category(char) in _SYMBOL_CATEGORIES:
        return "review"
    return "speakable"


def extract_delivery_cues(text):
    """Strip delivery cues from text, returning (text, instruct_hints).

    Bracketing music notes mark sung dialogue and become a hint. A lone
    mid-sentence note has no safe interpretation, so it is left in place and
    reported for review rather than guessed at.
    """
    hints = []
    stripped = text.strip()
    if (len(stripped) > 1 and stripped[0] in MUSIC_CHARS
            and stripped[-1] in MUSIC_CHARS):
        text = stripped[1:-1].strip()
        hints.append(SUNG_HINT)
    if any(char in ELONGATION_CHARS for char in text):
        text = "".join(c for c in text if c not in ELONGATION_CHARS)
        hints.append(ELONGATION_HINT)
    return text, hints
