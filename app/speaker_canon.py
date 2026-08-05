"""Speaker-name canonicalization for the Alexandria audiobook pipeline.

This module is standalone and dependency-light (stdlib + rapidfuzz only).
It must NOT import anything from app.py / review_script.py / generate_script.py —
those modules will import THIS module in a later integration stage.

Two tiers:

  Tier 1 (canonicalize): fully automatic, deterministic normalization of a
  single raw speaker label into a canonical UPPERCASE form. Safe to apply
  everywhere a speaker label is produced or compared.

  Tier 2 (suggest_aliases): advisory-only fuzzy matching across an already-
  canonicalized roster. It NEVER merges or mutates the roster; it only
  returns suggestion records for a human (via the Voices UI) to accept or
  reject. JON/JOHN and ELLA/BELLA are deliberately treated as *distinct*
  people in the roster even though they are similar strings -- this module
  will surface a suggestion for a human to review, but will not merge them.

IMPORTANT: canonicalization applies to speaker LABELS only. Audiobook body
text must remain byte-for-byte verbatim; nothing in this module should ever
be applied to spoken/narrated text.
"""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Tier 1: canonicalize()
# ---------------------------------------------------------------------------

# Honorifics stripped from the front of a name, longest-first so multi-word
# honorifics (e.g. "Professor") aren't shadowed by a shorter prefix match.
# Matched case-insensitively, with an optional trailing period, followed by
# whitespace (so "Drake" is never mistaken for "Dr" + "ake").
_HONORIFICS = [
    "professor", "captain", "prof", "capt", "mrs", "miss", "lady", "lord",
    "dame", "col", "rev", "sgt", "mr", "ms", "mx", "dr", "sir", "lt", "fr",
    "st",
]

_HONORIFIC_RE = re.compile(
    r"^(?:" + "|".join(_HONORIFICS) + r")\.?\s+",
    re.IGNORECASE,
)

# A parenthetical (and any surrounding whitespace) anywhere in the string,
# e.g. "MARK (shouting)" -> "MARK", "MARK (angrily) enters" -> "MARK enters".
_PARENS_RE = re.compile(r"\s*\([^)]*\)\s*")

# Characters allowed to remain in a canonical name: letters, digits,
# whitespace, apostrophes (O'Brien), and hyphens (Jean-Luc). Everything else
# is considered "stray punctuation" and stripped. This intentionally runs
# AFTER accent normalization (NFKD + combining-mark strip), so accented
# letters have already been folded down to plain ASCII letters by this point.
_ALLOWED_CHARS_RE = re.compile(r"[^A-Za-z0-9'\-\s]")

# Collapse any run of whitespace to a single space.
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_accents(text: str) -> str:
    """Fold accented characters down to their base ASCII form.

    e.g. 'José' -> 'Jose', 'François' -> 'Francois'.
    """
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def canonicalize(raw: str) -> str:
    """Canonicalize a raw speaker label into its canonical UPPERCASE form.

    Steps (in order):
      1. Bail out to "" for empty/whitespace-only input.
      2. Strip accents (NFKD decompose + drop combining marks).
      3. Remove parenthetical asides, e.g. "MARK (shouting)" -> "MARK".
      4. Strip stray punctuation, keeping apostrophes and hyphens (so
         "O'Brien" and "Jean-Luc" survive intact).
      5. Collapse internal whitespace runs and strip leading/trailing space.
      6. Strip a single leading honorific (Mr, Mrs, Dr, Professor, ...) --
         but never let this reduce a non-empty input to emptiness; if
         stripping the honorific would leave nothing, keep the honorific
         itself as the name (e.g. "Dr." alone -> "DR").
      7. Uppercase the result.
      8. Map any casing of "narrator" to exactly "NARRATOR".

    Canonicalization is idempotent: canonicalize(canonicalize(x)) ==
    canonicalize(x).
    """
    if raw is None:
        return ""

    text = raw.strip()
    if not text:
        return ""

    text = _strip_accents(text)
    text = _PARENS_RE.sub(" ", text)
    text = _ALLOWED_CHARS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    if not text:
        return ""

    honorific_match = _HONORIFIC_RE.match(text)
    if honorific_match:
        remainder = text[honorific_match.end():].strip()
        if remainder:
            text = remainder
        # else: stripping would empty the name out -- keep original text
        # (the honorific itself becomes the canonical name).

    text = _WHITESPACE_RE.sub(" ", text).strip()
    canonical = text.upper()

    if canonical == "NARRATOR":
        return "NARRATOR"

    return canonical


# ---------------------------------------------------------------------------
# Tier 2: suggest_aliases()
# ---------------------------------------------------------------------------

# Similarity metric: rapidfuzz.fuzz.ratio, which is an Indel/Levenshtein-based
# normalized similarity in the range [0, 100]. Chosen (over Jaro-Winkler)
# because it penalizes inserted/deleted characters roughly in proportion to
# name length, which lines up well with short-name confusions like
# JON/JOHN (single inserted "H") and ELLA/BELLA (single inserted "B") while
# still keeping unrelated names like MARCUS/ELENA far apart.
#
# Threshold: 75 (out of 100). Chosen empirically:
#   - JON vs JOHN     -> ratio ~85.7  (>= 75, suggestion emitted)
#   - ELLA vs BELLA   -> ratio ~88.9  (>= 75, suggestion emitted)
#   - MARCUS vs ELENA -> ratio ~18.2  (< 75, no suggestion)
#
# Direction rule: the SHORTER canonical name is suggested as an alias of the
# LONGER canonical name (ties broken alphabetically, shorter/earlier name
# first). Rationale: honorific/typo/nickname variants are usually shorter
# than or equal in length to the "fuller" form (e.g. JON -> JOHN), so the
# longer string is treated as the more complete / likely-canonical target.
# This is a heuristic default, not a claim about which name is "more
# frequent" in the manuscript -- callers with real frequency counts from the
# roster may want to override this ordering.
_SIMILARITY_THRESHOLD = 75.0


def suggest_aliases(roster: list) -> list:
    """Suggest advisory alias pairs within an already-canonicalized roster.

    This function is READ-ONLY: it never mutates `roster` and never merges
    or removes any entry. It only returns a list of suggestion dicts for a
    human reviewer (e.g. the Voices UI) to accept or reject. Names deemed
    similar (e.g. JON/JOHN, ELLA/BELLA) remain fully distinct roster entries
    -- this function does not, and must not, auto-merge them.

    Args:
        roster: list of speaker name strings. Names are expected to already
            be canonical (e.g. produced by `canonicalize`), but this function
            re-canonicalizes defensively and de-duplicates before comparing,
            so raw/mixed-case input degrades gracefully rather than raising.

    Returns:
        A list of dicts, each shaped like:
            {"name": <shorter name>, "alias_of": <longer name>, "score": <float 0-1>}
        sorted by descending score. "score" is normalized to the 0.0-1.0
        range (rapidfuzz's 0-100 ratio divided by 100), per the example in
        the design brief. NARRATOR is always excluded, from either side of
        a pair.
    """
    if not roster:
        return []

    # Defensive re-canonicalization + de-duplication, preserving first-seen
    # order. This does NOT mutate the caller's list/object.
    seen = set()
    names = []
    for raw_name in roster:
        name = canonicalize(raw_name)
        if not name or name == "NARRATOR":
            continue
        if name in seen:
            continue
        seen.add(name)
        names.append(name)

    suggestions = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            score = fuzz.ratio(a, b)
            if score < _SIMILARITY_THRESHOLD:
                continue

            # Direction: shorter name is the alias, longer name is the
            # target. Ties broken alphabetically for determinism.
            if len(a) < len(b) or (len(a) == len(b) and a < b):
                shorter, longer = a, b
            else:
                shorter, longer = b, a

            suggestions.append({
                "name": shorter,
                "alias_of": longer,
                "score": round(score / 100.0, 4),
            })

    suggestions.sort(key=lambda s: s["score"], reverse=True)
    return suggestions
