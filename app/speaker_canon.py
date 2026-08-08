"""Speaker-name canonicalization for the Alexandria audiobook pipeline.

This module is standalone and dependency-light (stdlib + rapidfuzz only).
It must NOT import anything from app.py / review_script.py / generate_script.py —
those modules will import THIS module in a later integration stage.

Two tiers:

  Tier 1 (canonicalize): fully automatic, deterministic normalization of a
  single raw speaker label into a canonical UPPERCASE form. Safe to apply
  everywhere a speaker label is produced or compared.

  Tier 1b (roster_key / remember_in_roster / resolve_against_roster):
  roster-AWARE spelling resolution, kept strictly separate from canonicalize()
  so the latter stays pure. It unifies two spellings ONLY when they are equal
  after removing every boundary mark -- whitespace, hyphens, apostrophes --
  ("ABBEMARIGNAN" -> "ABBE MARIGNAN", "OBRIEN" -> "O'BRIEN"), keeping the
  more-punctuated form. Exact key equality, never fuzzy similarity, so names
  that merely look alike are never merged.

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
#
# INCLUSION CRITERION: strip a word only when it is a TITLE whose abbreviation
# varies between mentions of the same person -- "Dr." / "Doctor", "Capt." /
# "Captain" -- which is exactly the drift that fragments a roster. Never strip
# a word that can be part of the name itself.
#
# "ST" was removed under that criterion: it is a name constituent, not a title
# (St. Clair, St. John, St. Laurent), and saint-derived surnames are common in
# the French literature this pipeline processes. While it was listed,
# "ST JOHN RIVERS" canonicalized to "JOHN RIVERS" -- a different character.
_HONORIFICS = [
    "professor", "captain", "prof", "capt", "mrs", "miss", "lady", "lord",
    "dame", "col", "rev", "sgt", "mr", "ms", "mx", "dr", "sir", "lt", "fr",
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

# Same as _ALLOWED_CHARS_RE, but additionally keeps straight/curly quote
# characters alive through the first punctuation pass, so a name like
# '"BLACK SCOUT"' still has both of its wrapping quote characters present
# when _strip_wrapping_quotes() runs. Anything it lets through that isn't
# consumed as part of a matched wrapping pair is cleaned up afterward by a
# second pass through the strict _ALLOWED_CHARS_RE above.
_ALLOWED_CHARS_WITH_QUOTES_RE = re.compile(
    "[^A-Za-z0-9'\\-\\s\"‘’“”]"
)

# Collapse any run of whitespace to a single space.
_WHITESPACE_RE = re.compile(r"\s+")

# Wrapping quote pairs recognized by _strip_wrapping_quotes(): straight
# single quote/apostrophe, straight double quote, and the curly ("smart")
# single and double quote pairs. The straight apostrophe intentionally maps
# to itself (') since ASCII text uses the same glyph for both open and
# close; the curly variants use their proper distinct open/close characters.
_QUOTE_PAIRS = {
    "'": "'",
    '"': '"',
    "‘": "’",  # ‘ ... ’
    "“": "”",  # “ ... ”
}


def _strip_accents(text: str) -> str:
    """Fold accented characters down to their base ASCII form.

    e.g. 'José' -> 'Jose', 'François' -> 'Francois'.
    """
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _strip_wrapping_quotes(text: str) -> str:
    """Iteratively strip MATCHED wrapping quote pairs from `text`.

    Handles the case where an LLM emits a speaker label wrapped in quotes,
    e.g. 'MOTHER OF MONSTERS' or "BLACK SCOUT" or nested combinations like
    '"BLACK SCOUT"' or curly ‘Mother of Monsters’ / “Black Scout”. Each pass
    strips one matched outer pair (straight/curly single or double quote,
    per _QUOTE_PAIRS) and re-trims whitespace, so nested wraps ("''X''")
    unwind one layer at a time.

    Deliberately conservative: only strips when the FIRST and LAST character
    of the (whitespace-trimmed) string form a matching open/close pair AND
    the string is longer than 2 characters (so a bare "'" or a 2-char
    wrapper isn't hollowed out to nothing). This means an UNMATCHED leading
    or trailing apostrophe -- possessive "JONES'" or elision "'TIS" -- is
    left untouched, since in both cases only one end is a quote character
    and the other is a letter, so first/last never form a pair.

    Bounded by len(text) iterations, which is far more than any realistic
    nesting depth, to guarantee termination on pathological input.
    """
    for _ in range(len(text)):
        stripped = text.strip()
        if len(stripped) <= 2:
            break
        first, last = stripped[0], stripped[-1]
        if _QUOTE_PAIRS.get(first) == last:
            candidate = stripped[1:-1].strip()
            if not candidate:
                break
            text = candidate
        else:
            break
    return text.strip()


def canonicalize(raw: str) -> str:
    """Canonicalize a raw speaker label into its canonical UPPERCASE form.

    Steps (in order):
      1. Bail out to "" for empty/whitespace-only input.
      2. Strip accents (NFKD decompose + drop combining marks).
      3. Remove parenthetical asides, e.g. "MARK (shouting)" -> "MARK".
      4. Strip stray punctuation EXCEPT apostrophes/hyphens/quote marks, so
         "O'Brien" and "Jean-Luc" survive intact and quote characters are
         still around for the wrapping-quote check in the next step.
      5. Collapse internal whitespace runs and strip leading/trailing space.
      6. Strip MATCHED wrapping quote pairs iteratively (straight or curly,
         single or double), e.g. "'Mother of Monsters'" -> "Mother of
         Monsters", '"BLACK SCOUT"' -> "BLACK SCOUT". Unmatched leading/
         trailing apostrophes (possessive "JONES'", elision "'TIS") are left
         alone -- see _strip_wrapping_quotes for the exact rule.
      7. Strip any remaining stray punctuation (incl. leftover unmatched
         quote characters), keeping apostrophes and hyphens, then re-collapse
         whitespace.
      8. Strip a single leading honorific (Mr, Mrs, Dr, Professor, ...) --
         but never let this reduce a non-empty input to emptiness; if
         stripping the honorific would leave nothing, keep the honorific
         itself as the name (e.g. "Dr." alone -> "DR").
      9. Uppercase the result.
      10. Map any casing of "narrator" to exactly "NARRATOR".

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
    text = _ALLOWED_CHARS_WITH_QUOTES_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    if not text:
        return ""

    text = _strip_wrapping_quotes(text)
    text = _ALLOWED_CHARS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    if not text:
        return ""

    # Strip honorifics to a FIXPOINT. A single pass is not idempotent, and the
    # damage was silent: "Mr. St. Clair" canonicalized to "ST CLAIR", which
    # canonicalized AGAIN (as it does -- generate_script.py canonicalizes for
    # the entry and remember_in_roster canonicalizes for the roster) to
    # "CLAIR". The script said one name, the roster and voice_config another,
    # and the character was split across two voices. Stacked honorifics are
    # real: "Sir Lt. Col. Blimp", "Mrs. Dr. Watson".
    while True:
        honorific_match = _HONORIFIC_RE.match(text)
        if not honorific_match:
            break
        remainder = text[honorific_match.end():].strip()
        if not remainder:
            # Stripping would empty the name out -- keep what we have (the
            # honorific itself becomes the canonical name, so a bare "Dr."
            # stays "DR" rather than vanishing).
            break
        text = remainder

    text = _WHITESPACE_RE.sub(" ", text).strip()
    canonical = text.upper()

    if canonical == "NARRATOR":
        return "NARRATOR"

    return canonical


# ---------------------------------------------------------------------------
# Tier 1b: roster-aware spelling resolution
#
# canonicalize() is, and must remain, a PURE function of one string: it is
# roster-free and idempotent, and suggest_aliases() depends on that for
# comparison. Roster awareness therefore lives here, in separate functions
# that take the roster explicitly.
#
# Why this exists: an LLM was observed emitting "ABBEMARIGNAN" one chunk after
# correctly emitting "ABBE MARIGNAN", with the correct spelling right there in
# its context. Ordinary spelling drift. canonicalize() has no space-deleting
# path (correctly -- it must never invent or delete word boundaries), so both
# forms otherwise survive as two roster entries, two voice assignments, and two
# voices for one character.
#
# CONTRACT -- never auto-merge similar names. JON/JOHN and ELLA/BELLA are
# different people and must stay distinct roster entries. The ONLY unification
# permitted is EXACT equality on a derived key: no threshold, no similarity
# score, nothing fuzzy. The key is the canonical form with every
# non-alphanumeric character removed, so two spellings unify iff they have
# identical letters and digits in identical order and differ ONLY in their
# boundary marks -- spaces, hyphens and apostrophes:
#
#     ABBE MARIGNAN / ABBEMARIGNAN      -> ABBEMARIGNAN      (unified)
#     O'BRIEN       / OBRIEN            -> OBRIEN            (unified)
#     JEAN-LUC      / JEAN LUC          -> JEANLUC           (unified)
#     JON / JOHN, ELLA / BELLA          -> different keys    (NOT unified)
#
# One key rather than a whitespace-tier-then-punctuation-tier lookup: the
# two-tier version buys nothing here (a narrower whitespace-only match is
# always also a punctuation match, so precedence never actually differs on
# real data) and costs a second lookup plus a precedence rule to explain.
#
# MEASURED on the real 578-label production roster: the whitespace-only key
# produced 2 collision families, and this wider key produces exactly the same
# 2 -- both of them the known ABBE MARIGNAN pair ("ABBE MARIGNAN" and
# "ABBE MARIGNAN'S NIECE"). Adding punctuation to the key introduced ZERO new
# collisions on real data. test_real_roster_key_introduces_no_new_collisions
# in test_speaker_canon.py pins that property so the key cannot silently widen.
#
# The residual risk this DOES carry: names where a boundary mark is a real
# morpheme boundary rather than a typo -- transliterated Korean/Chinese/
# Japanese short names such as LI NA vs LINA or O KIN vs OKIN would be unified
# even if they are two people. Judged acceptable because the Voices UI's alias
# mechanism is the escape hatch, though an imperfect one: aliasing can merge
# two roster entries, it cannot split one that was wrongly merged here.
#
# SELECTION RULE -- MOST BOUNDARY MARKS WINS; ties go to the incumbent (the
# first spelling seen). Two spellings sharing a key have identical letters and
# digits, so the LONGER string is exactly the one carrying more boundary marks
# -- that is the whole rule, and it is why the comparison below is a length
# comparison. "ABBE MARIGNAN" beats "ABBEMARIGNAN", "O'BRIEN" beats "OBRIEN",
# regardless of arrival order. Rationale: LLMs drop boundary marks far more
# often than they insert them, so the more-punctuated form is the more likely
# original. It picks correctly for both observed collision families.
#
# Equal-length variants are genuine ties -- "JEAN-LUC" vs "JEAN LUC",
# "MARY-ANNE" vs "MARY ANNE" -- where neither form is more likely correct than
# the other. The incumbent keeps the slot: deterministic, and the outcome
# depends only on which spelling the roster met first.
#
# Critically, this rule is ORDER-INDEPENDENT wherever the lengths differ.
# First-seen-wins was not: it made the canonical spelling depend on arrival
# order, so one malformed first sighting would become canonical for the rest
# of the book AND be fed back into the prompt's roster block, and generation
# (chunk order) and review (entry order) could genuinely disagree about who
# was "first".
# ---------------------------------------------------------------------------

# Everything that is not a letter or a digit. Applied to an ALREADY canonical
# name, where the only survivors are spaces, hyphens and apostrophes.
_KEY_STRIP_RE = re.compile(r"[^A-Z0-9]")


def roster_key(name: str) -> str:
    """Derived comparison key for a speaker name: the canonical form with all
    boundary marks (whitespace, hyphens, apostrophes) removed.

    Lossy, and used ONLY for exact-equality lookup -- never displayed, never
    stored as a speaker label. Two names share a key iff they differ solely in
    where their boundary marks fall.
    """
    return _KEY_STRIP_RE.sub("", canonicalize(name))


def remember_in_roster(index: dict, name: str) -> str:
    """Record ``name`` in a roster index and return the winning spelling.

    ``index`` is a plain ``{roster_key: established_spelling}`` dict; callers
    create it as ``{}`` and feed it back on every label, so each lookup is
    O(1). It is mutated in place.

    The winner is the spelling carrying the most boundary marks, which -- since
    colliding spellings share every letter and digit -- is simply the longer
    string; equal-length ties keep the incumbent. See the section comment above
    for why this rule, and not first-seen-wins, is the safe one. Returns the
    winning spelling, or "" for a name that canonicalizes to nothing.
    """
    canonical = canonicalize(name)
    if not canonical:
        return ""

    key = _KEY_STRIP_RE.sub("", canonical)
    established = index.get(key)
    if established is None:
        index[key] = canonical
        return canonical

    if len(canonical) > len(established):
        index[key] = canonical
        return canonical
    return established


def resolve_against_roster(raw: str, index: dict) -> str:
    """Canonicalize ``raw`` and snap it onto the established spelling that
    differs from it only in its boundary marks, if the roster has one.

    Read-only counterpart to ``remember_in_roster``: ``index`` is never
    mutated, and a name the roster has not seen simply passes through as
    ``canonicalize(raw)``. Idempotent.
    """
    canonical = canonicalize(raw)
    if not canonical:
        return ""
    return (index or {}).get(_KEY_STRIP_RE.sub("", canonical)) or canonical


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
