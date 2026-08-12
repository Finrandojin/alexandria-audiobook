"""Speaker-name canonicalization for the Alexandria audiobook pipeline.

This module is standalone and dependency-light (stdlib + rapidfuzz only).
It must NOT import anything from app.py / review_script.py / generate_script.py —
those modules will import THIS module in a later integration stage.

Two tiers:

  Tier 1 (canonicalize): fully automatic, deterministic normalization of a
  single raw speaker label into a canonical UPPERCASE form. Safe to apply
  everywhere a speaker label is produced or compared. Rank titles are
  dropped ("Dr. Millman" -> "MILLMAN"); gender-marking titles are PRESERVED
  and normalized ("Mr. Smith" -> "MISTER SMITH", "Mme Tellier" -> "MADAME
  TELLIER"), because they are often the only thing telling two characters
  apart -- see _RANK_TITLES / _GENDERED_TITLES.

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

  Tier 3 (attest_label / attest_speaker): advisory local-window attestation
  -- flags labels whose core name tokens don't appear near the label's own
  entries in the source, catching drift/invention the LLM produces (e.g.
  transposed letters, invented names). READ-ONLY / advisory: never mutates
  anything. attest_label() returns the boolean detail record the Voices UI
  badge consumes; attest_speaker() returns the three-way ATTESTED /
  UNATTESTED / UNVERIFIABLE verdict a would-be *gate* needs, because a
  caller that rejects labels must distinguish "refuted" from "our check does
  not apply here".

EVERY CHARACTER TEST IN THIS MODULE IS UNICODE, NOT ASCII, AND MUST STAY THAT
WAY. The pipeline processes non-English books (French honorifics are listed
below; span_tokenizer.PAIRED_QUOTES carries German, Russian and CJK quote
conventions), and an ASCII character class silently DELETES non-Latin text
rather than failing. Three ASCII classes lived here, each failing differently,
and all three are now category predicates (_is_name_char and the helpers built
on it -- _strip_disallowed, _strip_boundary_marks, _scan_tokens):

  * canonicalize()'s punctuation strip erased non-Latin names to "", and
    resolve_span_labels only accepts a truthy canonical form -- so EVERY
    dialogue line in a Cyrillic/Hebrew/Arabic/Greek/CJK book was narrated and
    the book got no character voices at all.
  * the roster key dropped non-Latin letters, so every such name would have
    keyed to "" -- one roster entry and one voice for an entire cast,
    irreversibly. Widening the canonical form WITHOUT widening the key turns
    "no voices" into "one voice for everybody", which is strictly worse; see
    the note at _strip_boundary_marks.
  * the attestation tokenizer extracted zero core tokens, so the Voices UI's
    attestation badge was wrong for every speaker in every non-Latin book.

Predicates, not regex classes: `re` cannot express "any Unicode mark" without
the third-party `regex` module, and the spacing combining marks that carry
vowels in Indic scripts are category Mc, which `\w` does not match -- "सीता"
came out as two bare consonants under a `\w`-based class.

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

# Leading titles are handled in TWO classes, because they carry two very
# different amounts of information about WHO is speaking.
#
# INCLUSION CRITERION (both classes): treat a word as a title only when its
# spelling varies between mentions of the same person -- "Dr." / "Doctor",
# "Capt." / "Captain", "Mme" / "Madame" -- which is exactly the drift that
# fragments a roster. Never list a word that can be part of the name itself.
#
# "ST" is deliberately absent under that criterion: it is a name constituent,
# not a title (St. Clair, St. John, St. Laurent), and saint-derived surnames
# are common in the French literature this pipeline processes. While it was
# listed, "ST JOHN RIVERS" canonicalized to "JOHN RIVERS" -- a different
# character.
#
# CLASS 1 -- RANK titles are DROPPED. Rank says nothing about identity that the
# surname doesn't already say, and the same person is freely "Dr. Millman",
# "Doctor Millman" and "Millman" within one book. Dropping them merges those
# three mentions into one voice, which is the desired outcome.
_RANK_TITLES = [
    "professor", "captain", "prof", "capt", "col", "rev", "sgt", "dr", "lt",
]

# CLASS 2 -- GENDER-MARKING titles are PRESERVED, normalized to one spelled-out
# form per title. Dropping them was a silent data-loss bug: "Mr. Smith" and
# "Mrs. Smith" both canonicalized to "SMITH", so a husband and wife were merged
# into a single roster entry and a single voice, at annotation time and
# irreversibly -- the Voices UI's alias mechanism can merge two entries but can
# never split one. The title is the ONLY thing distinguishing them, so it has
# to survive canonicalization.
#
# Normalization is language-PRESERVING: "Mme Tellier" and "Madame Tellier"
# unify on MADAME (they are one character), while "Monsieur Dufour" and
# "Madame Dufour" stay apart. Folding French onto English (MME -> MISSUS) would
# be a translation of the author's text into the roster, and the abbreviated
# and spelled-out French forms drift against each other in real books -- on the
# 578-label production roster, MADAME/MME split five characters in two.
#
# Judgment calls, recorded so they are not silently re-litigated:
#   * SIR / LADY / LORD / DAME / MX are gendered, not rank, and are preserved.
#     "Sir John" and "Lady John" are two people.
#   * FR moved here rather than being dropped. It was flagged as a
#     name-constituent risk of the same class as ST (a surname beginning
#     "Fr ..."); preserving it contains that risk -- the worst case is now a
#     harmless "FR" prefix on one roster entry, not a silently different
#     character. FR is NOT folded onto FATHER: "Father Milon" reads as part of
#     the name in this corpus and has never been in the title list.
#   * M -> MONSIEUR only as a standalone leading token (same position and
#     tokenization as every other title), so "M. Marambot" unifies with
#     "Monsieur Marambot". A single-letter initial in that position -- "M.
#     Night" -- is therefore read as a title. Accepted: consistency with the
#     other titles beats a special case, and the failure is a visible wrong
#     prefix rather than two characters merged.
#   * The normalized forms are themselves recognized titles mapping to
#     themselves (MISTER -> MISTER), which is what keeps canonicalize()
#     idempotent once its own output is fed back in.
_GENDERED_TITLES = {
    "mr": "MISTER", "mister": "MISTER",
    "mrs": "MISSUS", "missus": "MISSUS",
    "ms": "MS",
    "miss": "MISS",
    "mx": "MX",
    "m": "MONSIEUR", "monsieur": "MONSIEUR",
    "mme": "MADAME", "madame": "MADAME",
    "mlle": "MADEMOISELLE", "mademoiselle": "MADEMOISELLE",
    "sir": "SIR",
    "lady": "LADY",
    "lord": "LORD",
    "dame": "DAME",
    "fr": "FR",
}

# The normalized gender-marking prefixes, exported for consumers that need to
# reason about "same surname, different title" -- notably tts.resolve_voice's
# migration shim for voice_config.json files written before titles were
# preserved. Consumers MUST import this rather than re-listing the titles.
GENDERED_TITLES = frozenset(_GENDERED_TITLES.values())

# Every leading token treated as a title, in lowercase.
_ALL_TITLES = frozenset(_RANK_TITLES) | frozenset(_GENDERED_TITLES)

# A parenthetical (and any surrounding whitespace) anywhere in the string,
# e.g. "MARK (shouting)" -> "MARK", "MARK (angrily) enters" -> "MARK enters".
_PARENS_RE = re.compile(r"\s*\([^)]*\)\s*")

# Characters allowed to remain in a canonical name: letters, digits,
# whitespace, apostrophes (O'Brien), and hyphens (Jean-Luc). Everything else
# is considered "stray punctuation" and stripped. This intentionally runs
# AFTER accent normalization (NFKD + combining-mark strip), so accented
# letters have already been folded down to their base letters by this point.
#
# UNICODE, not ASCII. This class was `[^A-Za-z0-9'\-\s]` and therefore treated
# every non-Latin letter as stray punctuation, which was a total-loss bug for
# any book not written in a Latin script: "Ирина" canonicalized to "" (every
# character stripped, then the empty-string bail-out at the top of
# canonicalize), and resolve_span_labels only accepts a speaker when
# `canonical` is truthy -- so EVERY dialogue line in a Cyrillic, Hebrew,
# Arabic, Greek or CJK book fell back to NARRATOR and the book got no
# character voices at all. `[^\W_]` is "any word character except
# underscore", i.e. letters and digits in ANY script; it encodes
# word-character-ness, not knowledge of any particular language.
# Implemented as a CATEGORY PREDICATE, not a regex character class, because a
# class cannot express "any Unicode mark" without the third-party `regex`
# module. `\w` matches str.isalnum() characters, and the spacing combining
# marks that carry vowels in Indic scripts are category Mc, which is NOT
# alphanumeric: "सीता" lost its vowel signs under a `\w`-based class and came
# out as two bare consonants. Marks are part of the letter, so they are kept.
_QUOTE_CHARS = "\"'‘’“”"
_NAME_PUNCT = "'‘’"


def _is_name_char(ch, allow_quotes=False):
    """True for a character that may appear inside a canonical name: any
    alphanumeric in any script, any Unicode mark (Mn/Mc/Me -- Indic vowel
    signs, Hebrew points, combining diacritics), and the apostrophe glyphs.
    Language-agnostic by construction: it asks the Unicode database what kind
    of character this is, and knows nothing about any particular script.
    """
    if ch.isalnum() or ch in _NAME_PUNCT:
        return True
    if allow_quotes and ch in _QUOTE_CHARS:
        return True
    return unicodedata.category(ch).startswith("M")


def _strip_disallowed(text, allow_quotes=False):
    """Replace every character that cannot appear in a canonical name with a
    space, keeping hyphens and whitespace. Replacement (not deletion) matches
    the previous regex behaviour, so "MARK(shouting)" still separates into
    words rather than fusing.
    """
    return "".join(
        ch if (ch in "-" or ch.isspace() or _is_name_char(ch, allow_quotes)) else " "
        for ch in text
    )

# _strip_disallowed(..., allow_quotes=True) additionally keeps straight/curly
# quote characters alive through the first punctuation pass, so a name like
# '"BLACK SCOUT"' still has both of its wrapping quote characters present when
# _strip_wrapping_quotes() runs. Anything it lets through that isn't consumed
# as part of a matched wrapping pair is cleaned up by the second, strict pass.

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
      8. Resolve the run of leading titles (see _RANK_TITLES /
         _GENDERED_TITLES): rank titles are dropped ("Dr. Millman" ->
         "MILLMAN"), while the FIRST gender-marking title becomes a
         normalized preserved prefix ("Mr. Smith" -> "MISTER SMITH",
         "Mme Bovary" -> "MADAME BOVARY"). Stacked titles are all consumed,
         so "Mrs. Dr. Watson" -> "MISSUS WATSON" and "Sir Lt. Col. Blimp"
         -> "SIR BLIMP". Never lets this reduce a non-empty input to
         emptiness: a label that is nothing but titles keeps its last title
         as the name ("Dr." -> "DR", "Mrs." -> "MRS", "Mr. Mrs." -> "MRS").
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
    text = _strip_disallowed(text, allow_quotes=True)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    if not text:
        return ""

    text = _strip_wrapping_quotes(text)
    text = _strip_disallowed(text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    if not text:
        return ""

    # Consume the WHOLE run of leading titles in one pass, not one title per
    # pass. Stacked titles are real ("Sir Lt. Col. Blimp", "Mrs. Dr. Watson"),
    # and a single-title pass was not idempotent -- the damage was silent:
    # "Mr. St. Clair" canonicalized to "ST CLAIR", which canonicalized AGAIN
    # (as it does -- generate_script.py canonicalizes for the entry and
    # remember_in_roster canonicalizes for the roster) to "CLAIR". The script
    # said one name, the roster and voice_config another, and the character was
    # split across two voices.
    #
    # Tokenizing on whitespace is safe here: step 4/7 already replaced every
    # period with a space, so "Dr. Smith" is "Dr Smith" by now and no title can
    # be matched inside a word ("Drake" is one token, and "drake" is not a
    # title).
    tokens = text.split(" ")
    prefix = None
    index = 0
    while index < len(tokens) and tokens[index].lower() in _ALL_TITLES:
        if prefix is None:
            prefix = _GENDERED_TITLES.get(tokens[index].lower())
        index += 1

    if index >= len(tokens):
        # The label is nothing but titles. Consuming them all would empty the
        # name out, so the LAST title becomes the name itself and no prefix is
        # emitted: a bare "Dr." stays "DR" rather than vanishing, and
        # "Mr. Mrs." stays "MRS" rather than becoming "MISTER MRS".
        tokens = tokens[-1:]
    else:
        tokens = ([prefix] if prefix else []) + tokens[index:]

    text = _WHITESPACE_RE.sub(" ", " ".join(tokens)).strip()
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
#
# UNICODE, and this one is load-bearing for safety, not just coverage. While
# this class was `[^A-Z0-9]` it stripped every non-Latin letter too, so once
# canonicalize() (correctly) began preserving them, roster_key("ИРИНА") and
# roster_key("ПЕТР") would both have been "" -- an empty key that EVERY
# non-Latin name collides on. remember_in_roster would then have merged the
# entire cast of a Russian novel into one speaker with one voice, and a wrong
# merge is unsplittable (see the SELECTION RULE note above). Widening
# canonicalize() without widening this key would have turned "no character
# voices" into "one character voice for everybody", which is strictly worse.
# Implemented with the same category predicate as canonicalize() (see
# _is_name_char) so the two agree on what a letter is in every script,
# including Indic vowel signs. Keeps alphanumerics and marks; removes exactly
# the boundary marks the key is meant to ignore -- spaces, hyphens,
# apostrophes.


def _strip_boundary_marks(canonical):
    """Derive the lossy comparison key: keep letters/digits/marks of any
    script, drop every boundary mark. Pure.
    """
    return "".join(
        ch for ch in canonical
        if ch not in _NAME_PUNCT and (ch.isalnum() or unicodedata.category(ch).startswith("M"))
    )


def roster_key(name: str) -> str:
    """Derived comparison key for a speaker name: the canonical form with all
    boundary marks (whitespace, hyphens, apostrophes) removed.

    Lossy, and used ONLY for exact-equality lookup -- never displayed, never
    stored as a speaker label. Two names share a key iff they differ solely in
    where their boundary marks fall.
    """
    return _strip_boundary_marks(canonicalize(name))


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

    key = _strip_boundary_marks(canonical)
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
    return (index or {}).get(_strip_boundary_marks(canonical)) or canonical


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


# ---------------------------------------------------------------------------
# Tier 3: attest_label()
#
# Purely advisory, local-window text attestation. The LLM occasionally
# invents or drifts a speaker name (transposed letters, an outright
# hallucinated name) that is NOT the byte-verbatim text of the book -- since
# labels are metadata, not book text, nothing else in the pipeline checks
# them against the source. This tier gives a human reviewer a signal:
# "this label's name doesn't appear anywhere near its own lines" -- without
# ever touching a speaker value, roster entry, or voice_config.
#
# Deliberately NOT a general search across the whole book: a name appearing
# ANYWHERE in a 300-page novel is a near-useless signal (most names appear
# many times, far from any particular character's own lines). Attestation is
# scoped to windows drawn from the label's OWN entries, which is exactly the
# text this label is describing.
# ---------------------------------------------------------------------------

# A small, deliberately conservative list of generic connector words that
# never carry identifying information on their own and therefore should not
# be required to appear near a label's lines. Extensible -- add another
# lowercase connector here if a real roster surfaces one, but keep the list
# short: anything that could plausibly BE a surname (e.g. "DE" as part of
# "DE LA CRUZ" combos is still excluded on purpose, mirroring how titles are
# handled) should stay out unless it is purely a function word.
_CORE_TOKEN_STOPWORDS = frozenset({
    "THE", "A", "AN", "OF", "AND", "OR", "DE", "DU", "LA", "LE", "VON", "VAN",
})

# Uppercase forms of every recognized title token (both classes), derived
# from the existing module constants rather than re-listed, so this stays in
# sync with _RANK_TITLES / _GENDERED_TITLES automatically.
_CORE_TOKEN_TITLES = frozenset(t.upper() for t in _ALL_TITLES) | GENDERED_TITLES

# A bare word-boundary token: letters/digits/apostrophes (straight or curly)
# and internal hyphens, used both to pull core tokens out of a label and to
# scan a source window for whole-word occurrences of those tokens. Hyphens
# are included so a hyphenated label token ("ARCH-VOTARY") matches a
# hyphenated source occurrence ("Arch-votary") as a single unit rather than
# splitting into two words that individually pass but the label as a whole
# never does.
#
# UNICODE, for the same reason as canonicalize()'s character handling: as
# `[A-Za-z0-9'‘’]+` this extracted ZERO tokens from a Hebrew, Arabic, Greek or
# CJK name (measured), and a label with no core tokens is reported unattested
# -- so the Voices UI's attestation badge was wrong for every speaker in every
# non-Latin book. Scanners rather than regexes, for the same reason as
# _strip_disallowed: a character class cannot say "any Unicode mark", and
# Indic vowel signs (category Mc) are part of the letter.


def _scan_tokens(text, join_hyphens):
    """Split `text` into name tokens: maximal runs of name characters (see
    _is_name_char), optionally joining runs separated by a single hyphen so a
    hyphenated name is one token. Pure; returns a list in source order.
    """
    tokens = []
    current = []
    index = 0
    length = len(text)
    while index < length:
        ch = text[index]
        if _is_name_char(ch):
            current.append(ch)
        elif (
            join_hyphens and ch == "-" and current
            and index + 1 < length and _is_name_char(text[index + 1])
        ):
            current.append("-")
        elif current:
            tokens.append("".join(current))
            current = []
        index += 1
    if current:
        tokens.append("".join(current))
    return tokens

# Curly ("smart") apostrophe variants, folded to the straight ASCII
# apostrophe before comparison. The LLM and the book's own prose disagree on
# which glyph they use for the same character's name -- "SKER'RET" (label,
# straight) vs "Sker'ret" (source, curly U+2019) -- and that disagreement
# carries no identifying information, so it must not affect attestation.
_CURLY_APOSTROPHE_RE = re.compile("[‘’]")


def _fold_word(word: str) -> str:
    """Fold a single word for attestation comparison: accent-strip, curly
    apostrophes to straight, uppercase, then drop one trailing possessive
    "'S" (so "Dairine's" matches core token "DAIRINE"). Does NOT strip other
    apostrophes, so "O'BRIEN" stays "O'BRIEN" on both sides of the
    comparison -- only the glyph is normalized, not the character's presence.
    """
    folded = _strip_accents(word)
    folded = _CURLY_APOSTROPHE_RE.sub("'", folded).upper()
    if folded.endswith("'S") and len(folded) > 2:
        folded = folded[:-2]
    return folded


def _window_words(window: str) -> set:
    """Extract the set of folded whole-word tokens present in a source
    window, for attestation lookup.

    Includes hyphenated tokens as single folded units (via _scan_tokens) so a
    hyphenated label token matches a hyphenated source occurrence. Also adds
    hyphen-joined bigrams of adjacent plain words (e.g. source "Arch votary"
    contributes "ARCH-VOTARY") so a hyphenated label token still matches when
    the source spells the same name with a space instead of a hyphen. This is
    exact whole-word/whole-token matching, not fuzzy: a name must appear as
    an unbroken hyphenated form, or as two adjacent plain words joined the
    same way the label joins them.
    """
    words = set()
    for match in _scan_tokens(window, join_hyphens=True):
        folded = _fold_word(match)
        if folded:
            words.add(folded)

    simple = [_fold_word(w) for w in _scan_tokens(window, join_hyphens=False)]
    for first, second in zip(simple, simple[1:]):
        if first and second:
            words.add(f"{first}-{second}")

    return words


def _core_tokens(label: str) -> list:
    """Extract the "core" identifying tokens from a canonical speaker label.

    A core token is any whitespace-separated token of the label that is
    NOT a generic stopword (_CORE_TOKEN_STOPWORDS) and NOT a recognized
    title token (_CORE_TOKEN_TITLES, derived from the Tier 1 title lists).
    Tokens are accent-folded and possessive-stripped the same way source
    words are, so "MISTER SMITH" -> ["SMITH"] and a label containing an
    accented or possessive-looking token still compares consistently.

    Pure function of `label`; does no I/O and touches no roster state.
    """
    if not label:
        return []
    tokens = []
    for raw_token in label.split():
        folded = _fold_word(raw_token)
        if not folded:
            continue
        if folded in _CORE_TOKEN_STOPWORDS or folded in _CORE_TOKEN_TITLES:
            continue
        tokens.append(folded)
    return tokens


def attest_label(label: str, windows: list) -> dict:
    """Advisory, read-only check that a label's core name tokens appear as
    whole words somewhere in its own local source `windows`.

    Args:
        label: a (canonical) speaker label, e.g. "MISTER SMITH".
        windows: a list of source-text substrings (strings) drawn from
            around this label's own entries in the book. Not mutated.

    Returns:
        {"attested": bool, "missing_tokens": [str, ...], "core_tokens": [str, ...]}

        A label is attested=True iff EVERY core token extracted from it
        appears (accent-folded, case-insensitive, possessive-stripped) as a
        whole word in at least one window.

        If the label has ZERO core tokens (e.g. it is only a title, or only
        stopwords -- "MISTER", "NARRATOR"-like edge cases), this is treated
        conservatively as attested=False with missing_tokens=[] and a note
        that this is a trivial/unknown case: there is nothing to check, so
        there is nothing to positively confirm either. Callers that want to
        suppress trivial cases from a UI can do so by checking
        `core_tokens == []`.

    Pure function of (label, windows): no file I/O, no globals mutated, no
    roster access. `windows` is read-only -- neither the list nor its
    string elements are modified.
    """
    core_tokens = _core_tokens(label)

    if not core_tokens:
        return {
            "attested": False,
            "missing_tokens": [],
            "core_tokens": [],
            "note": "trivial_or_unknown_label",
        }

    # Build the set of whole words present across all windows, folded the
    # same way as the core tokens, once, rather than re-scanning per token.
    words_seen = set()
    for window in windows or []:
        if not window:
            continue
        words_seen |= _window_words(window)

    missing = [token for token in core_tokens if token not in words_seen]

    return {
        "attested": len(missing) == 0,
        "missing_tokens": missing,
        "core_tokens": core_tokens,
    }


# Verdicts returned by attest_speaker(). Exported as constants so callers
# compare against these rather than re-spelling the strings.
ATTESTED = "attested"
UNATTESTED = "unattested"
UNVERIFIABLE = "unverifiable"


def attest_speaker(label, windows):
    """Three-way attestation verdict for a speaker label: ATTESTED,
    UNATTESTED, or UNVERIFIABLE.

    Wraps attest_label() rather than replacing it: attest_label's boolean
    contract is depended on by GET /api/voices/label_flags and by
    test_speaker_canon.py, and stays exactly as it was. This function adds
    the third outcome that a *gate* needs and a badge does not.

    Why a third verdict exists. A boolean forces every label that cannot be
    positively confirmed into the same bucket as a label that was positively
    refuted, and those two must not share a fate: refuted means the LLM
    invented or misspelled a name, while merely-unconfirmed means our own
    check does not apply to this text. Any caller that rejects labels must
    reject only the first kind, or it silently destroys books written in
    scripts this module cannot tokenize.

    Verdicts:
      ATTESTED      every core token appears as a whole word in some window.
      UNVERIFIABLE  the label yields no core tokens at all (it is only
                    titles/stopwords), or every otherwise-missing token IS
                    present in a window but not at a word boundary. The
                    second case is what unsegmented scripts look like:
                    Chinese, Japanese and Thai do not delimit words with
                    spaces, so a correct name is a substring of a longer run
                    and no amount of whole-word matching will confirm it.
                    It is ALSO what a too-short Latin token looks like
                    ("AL" inside "ALICE"), which is exactly why this is
                    UNVERIFIABLE and not ATTESTED -- it is the honest
                    "cannot tell" answer, and callers must treat it as
                    accept-and-count, never as grounds for rejection.
      UNATTESTED    at least one core token appears nowhere in the windows,
                    in any form. This is the only verdict that constitutes
                    positive evidence the label is wrong.

    Deliberately NOT language detection. No script table, no language
    parameter, no per-language branch: the substring probe is a property of
    the text in front of it, so a book in an unsegmented script degrades to
    UNVERIFIABLE by the same rule that any other ambiguous match does.

    Args:
        label: a (canonical) speaker label, e.g. "MISTER SMITH".
        windows: list of source-text substrings drawn from around this
            label's own entries. Not mutated.

    Returns:
        One of the ATTESTED / UNATTESTED / UNVERIFIABLE string constants.

    Pure function of (label, windows): no I/O, no globals mutated, no roster
    access, and neither argument is modified. Idempotent -- repeated calls on
    equal inputs return equal verdicts.
    """
    result = attest_label(label, windows)

    if not result["core_tokens"]:
        return UNVERIFIABLE
    if result["attested"]:
        return ATTESTED

    # Fold each window once, the same way _fold_word folds a token, so the
    # substring probe compares like with like (accents stripped, curly
    # apostrophes straightened, uppercased) instead of comparing a folded
    # token against raw source bytes.
    folded_windows = [
        _CURLY_APOSTROPHE_RE.sub("'", _strip_accents(window)).upper()
        for window in (windows or []) if window
    ]

    for token in result["missing_tokens"]:
        if not any(token in window for window in folded_windows):
            return UNATTESTED

    return UNVERIFIABLE
