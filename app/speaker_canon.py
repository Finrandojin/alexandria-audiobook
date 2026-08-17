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
  transposed letters, invented names). A MULTI-token label must occur as a
  PHRASE, not merely as tokens scattered through the window, or a name
  recombined from two real characters' name parts attests. READ-ONLY /
  advisory: never mutates
  anything. attest_label() returns the boolean detail record the Voices UI
  badge consumes; attest_speaker() returns the three-way ATTESTED /
  UNATTESTED / UNVERIFIABLE verdict a would-be *gate* needs, because a
  caller that rejects labels must distinguish "refuted" from "our check does
  not apply here".

  Tier 3b (repair_speaker): the ONE place in this module that rewrites a
  speaker name. It fires only on a label Tier 3 already ruled UNATTESTED, and
  only when the book's own text refutes that spelling outright and points at
  exactly one established alternative. It is a policy argument, NOT a proof of
  immunity to the fuzzy-merge ban -- read its docstring, including its list of
  accepted limitations, before touching it. Roster-aware, and therefore takes
  the roster as an explicit argument, exactly like Tier 1b.

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


def _split_possessive(word: str):
    """Fold a single word and split off one trailing possessive "'S".

    Returns ``(base, suffix)`` where ``suffix`` is "'S" or "". The suffix is
    returned rather than discarded so a caller that REBUILDS a label (see
    repair_speaker) can put it back: "KITT'S" must repair to "KITT'S", not to
    a bare "KIT".
    """
    folded = _strip_accents(word)
    folded = _CURLY_APOSTROPHE_RE.sub("'", folded).upper()
    if folded.endswith("'S") and len(folded) > 2:
        return folded[:-2], "'S"
    return folded, ""


def _fold_word(word: str) -> str:
    """Fold a single word for attestation comparison: accent-strip, curly
    apostrophes to straight, uppercase, then drop one trailing possessive
    "'S" (so "Dairine's" matches core token "DAIRINE"). Does NOT strip other
    apostrophes, so "O'BRIEN" stays "O'BRIEN" on both sides of the
    comparison -- only the glyph is normalized, not the character's presence.
    """
    return _split_possessive(word)[0]


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


def _window_token_sequence(window: str) -> list:
    """The ORDERED list of folded word tokens in a source window.

    Companion to _window_words (which is an unordered set): adjacency cannot be
    decided from a set, and a multi-token label is only attested when its parts
    occur next to each other -- see _phrase_present.

    Hyphens are split, not joined, on BOTH sides of the comparison (here and in
    _phrase_present's label side), so "ARCH-VOTARY" and a source "Arch votary"
    or "Arch-votary" all reduce to the same two-element run. Folding is
    _fold_word, so case, accents, curly apostrophes and one trailing possessive
    are already normalized -- which is what makes "X'S DAD" match a source
    "X's dad" without any possessive-specific rule here.
    """
    sequence = []
    for raw_token in _scan_tokens(window, join_hyphens=False):
        folded = _fold_word(raw_token)
        if folded:
            sequence.extend(part for part in folded.split("-") if part)
    return sequence


# Source words allowed to sit BETWEEN two core tokens without breaking the
# phrase: the words _core_tokens drops from the label, MINUS the conjunctions,
# so a label and the source phrase it came from stay comparable ("MOTHER OF
# MONSTERS" matches "Mother of Monsters"; a leading article or title either
# side carries or omits is invisible to both).
#
# CONJUNCTIONS ARE EXCLUDED ON PURPOSE. "and"/"or" is precisely the shape that
# joins two DIFFERENT entities -- "the TV and the DVD", "Kit and Nita" -- so
# skipping them would re-admit the recombined label this rule exists to refuse.
# MEASURED: keeping them let 2 of the 5 fabricated labels back through.
_PHRASE_SKIPPABLE = (
    (_CORE_TOKEN_STOPWORDS - {"AND", "OR"}) | _CORE_TOKEN_TITLES
)


def _phrase_present(core_tokens: list, windows: list) -> bool:
    """True iff ``core_tokens`` occur ADJACENT and IN ORDER in some window.

    "Adjacent" tolerates only _PHRASE_SKIPPABLE words in between. Nothing fuzzy:
    every token must match exactly after folding.
    """
    # ponytail: adjacency is measured in WORDS, so punctuation and even a
    # sentence break between two tokens is invisible ("...the TV. The news...")
    # -- it only ever makes the gate more permissive, never less. Track
    # sentence boundaries here if a real book is measured slipping through.
    wanted = []
    for token in core_tokens:
        wanted.extend(part for part in token.split("-") if part)
    if not wanted:
        return False

    for window in windows or []:
        if not window:
            continue
        sequence = _window_token_sequence(window)
        for start in range(len(sequence)):
            index = start
            matched = 0
            while index < len(sequence) and matched < len(wanted):
                if sequence[index] == wanted[matched]:
                    matched += 1
                elif matched == 0 or sequence[index] not in _PHRASE_SKIPPABLE:
                    break
                index += 1
            if matched == len(wanted):
                return True
    return False


def _is_possessive_composition(label: str) -> bool:
    """True when the label carries a possessive token -- "X'S DAD", "X'S
    FATHER": a RELATION described in terms of a named person, not a name.

    Such labels are EXEMPT from the phrase requirement, and the exemption is
    load-bearing rather than cosmetic. MEASURED: on one 8,081-entry artifact
    "NITA'S DAD" carries 50 correctly-attributed entries, and the source spells
    that phrase just 5 times in 790,543 characters -- the character is
    overwhelmingly referred to by pronoun near his own speech, so demanding the
    phrase inside his own attestation window refuses 50 good entries to catch
    none. (A label built on a name the book names OFTEN, "ROSHAUN'S FATHER"
    with 25 occurrences, happens to survive the phrase rule -- which is exactly
    the point: without the exemption the verdict depends on how chatty the
    narrator is, not on whether the label is real.)

    Safe against the recombination the phrase rule exists to catch: an invented
    name is assembled from two NAME parts ("<first> <surname-of-someone-else>")
    and carries no possessive. The residual cost is stated in attest_label.

    A property of how labels are composed in English, not of any book.
    """
    return any(_split_possessive(token)[1] for token in (label or "").split())


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

        A SINGLE-core-token label is attested=True iff that token appears
        (accent-folded, case-insensitive, possessive-stripped) as a whole word
        in at least one window.

        A MULTI-core-token label must additionally occur as a PHRASE: its
        tokens adjacent and in order in one window, with only stopwords/titles
        allowed between them (_phrase_present). Per-token set membership alone
        attests any label whose parts merely appear somewhere nearby, which is
        how a plausible-looking name assembled from two different characters'
        name parts passed the gate, entered the roster, and was then never
        re-checked for the rest of the book. Measured on one clean 8,081-entry
        artifact: 5 such labels carrying 11 entries occur nowhere in the
        790,543-character book as a phrase, while all 16 legitimate multi-token
        labels do and still pass.

        EXEMPT from the phrase requirement: a possessive composition ("X'S
        DAD"), which is a relation, not a name -- see _is_possessive_composition
        for the measurement that forces the exemption. The cost of the
        exemption is that a possessive label whose relation word is invented
        ("X'S FATHER" where the book only ever says "X's dad") still attests on
        per-token evidence; bounded, because the named half must still attest.

        When every token is present but not adjacent, missing_tokens is []
        and note="tokens_not_adjacent" -- attested=False with nothing missing.
        Callers must not read an empty missing_tokens as "nothing wrong"; that
        distinction is what attest_speaker turns into UNATTESTED.

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

    if missing or len(core_tokens) < 2 or _is_possessive_composition(label):
        return {
            "attested": not missing,
            "missing_tokens": missing,
            "core_tokens": core_tokens,
        }

    if _phrase_present(core_tokens, windows or []):
        return {"attested": True, "missing_tokens": [], "core_tokens": core_tokens}

    return {
        "attested": False,
        "missing_tokens": [],
        "core_tokens": core_tokens,
        "note": "tokens_not_adjacent",
    }


# Verdicts returned by attest_speaker(). Exported as constants so callers
# compare against these rather than re-spelling the strings.
ATTESTED = "attested"
UNATTESTED = "unattested"
UNVERIFIABLE = "unverifiable"


def _roster_core_tokens(roster_index) -> frozenset:
    """The union of the core tokens of every name established in ``roster_index``.

    Roster-AWARE, and therefore deliberately NOT part of canonicalize(): the
    roster arrives as an explicit argument, exactly like roster_key /
    remember_in_roster / resolve_against_roster. Pure; ``roster_index`` is
    read, never mutated.
    """
    tokens = set()
    for established in (roster_index or {}).values():
        tokens.update(_core_tokens(established))
    return frozenset(tokens)


def source_word_index(text: str) -> set:
    """The set of folded whole words the SOURCE TEXT contains, for callers that
    need book-wide (not window-local) evidence about a spelling.

    Same folding and same tokenization as attestation windows (_window_words),
    so a token looked up here compares like with like. Built once per book by
    the caller and passed down; this module never reads a file.
    """
    return _window_words(text or "")


def _is_distance_one(a: str, b: str) -> bool:
    """True iff ``a`` and ``b`` differ by exactly one Levenshtein edit
    (one substitution, one insertion, or one deletion).

    An explicit BOUNDED predicate, not a similarity ratio and not a library
    call: `difflib` and rapidfuzz both answer "how alike are these?", which is
    the question the banned-approaches list forbids acting on. This answers
    "is there exactly one edit between them?", which is decidable, exact, and
    cheap -- it early-exits on a length gap of 2 and otherwise makes a single
    linear pass. Compares Unicode code points, so it is script-agnostic.
    """
    if a == b:
        return False
    len_a, len_b = len(a), len(b)
    if abs(len_a - len_b) > 1:
        return False
    if len_a == len_b:
        differences = 0
        for left, right in zip(a, b):
            if left != right:
                differences += 1
                if differences > 1:
                    return False
        return differences == 1
    shorter, longer = (a, b) if len_a < len_b else (b, a)
    index = 0
    while index < len(shorter) and shorter[index] == longer[index]:
        index += 1
    return shorter[index:] == longer[index + 1:]


def attest_speaker(label, windows, roster_index=None):
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
      ATTESTED      every core token appears as a whole word in some window,
                    AND (for a multi-token label) they appear there adjacent
                    and in order -- see attest_label.
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

    ROSTER-NAME PARTIAL ATTESTATION (only when ``roster_index`` is given). A
    label with TWO OR MORE core tokens, at least one of which both (a) appears
    as a whole word in a window and (b) is itself a core token of a name this
    book has already established, is UNVERIFIABLE rather than UNATTESTED.
    Rationale: UNATTESTED is defined as positive evidence that the label is
    WRONG, and a label built around a name the book has established is not
    positively wrong -- "KIT'S FATHER" describes a real person the text names
    obliquely, and the honest answer is "cannot tell", not "refuted". The
    weaker rule "any attested token" was rejected: it is satisfied by any
    common noun the model copied out of the prose, which would gut the gate.
    Requiring a ROSTER token is what keeps it narrow.

    Its cost, stated plainly, and NARROWED but not closed by the adjacency
    rule: a label whose tokens all occur nearby but never adjacently is now
    refuted before this rule is consulted, so the commonest abuse -- a name
    recombined from two established characters' name parts -- no longer reaches
    it. What remains: a label with a token that occurs NOWHERE in the window is
    still rescued to UNVERIFIABLE by one roster token that does, and the roster
    is not itself guaranteed clean, so a junk name accepted earlier lends its
    tokens onward. The failure stays bounded -- such labels are
    ACCEPTED-and-counted rather than silently correct, and prose is never
    involved -- but it is a real widening of the gate on a polluted roster.

    ``roster_index`` is an explicit argument, never module state: this function
    stays a pure function of its inputs, and canonicalize() stays roster-free
    (contract 6). Passing no roster reproduces the previous behaviour exactly.

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

    # Non-adjacency is positive evidence, not absence of evidence: every token
    # IS in the window as a whole word, and the book still never puts them
    # together. Refuted outright, and deliberately BEFORE the roster path --
    # the recombined label's parts are usually roster names (that is what makes
    # it plausible), so letting the roster rule see it would rescue exactly the
    # case this refutes. Unsegmented scripts never reach here: their tokens are
    # individually missing, so they take the substring probe below as before.
    if result.get("note") == "tokens_not_adjacent":
        return UNATTESTED

    # Roster-name partial attestation (see docstring). Checked before the
    # substring probe because it is the cheaper and more specific rule.
    if roster_index and len(result["core_tokens"]) >= 2:
        missing = set(result["missing_tokens"])
        present = [token for token in result["core_tokens"] if token not in missing]
        if present:
            roster_tokens = _roster_core_tokens(roster_index)
            if any(token in roster_tokens for token in present):
                return UNVERIFIABLE

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


# ---------------------------------------------------------------------------
# Tier 3b: repair_speaker()
#
# HONEST FRAMING FIRST. This is a single-edit spelling repair, and it is NOT
# structurally immune to the "no fuzzy auto-merging of speaker names" ban in
# the same way roster_key() is. roster_key() cannot merge two names the book
# supports, as a matter of arithmetic: they have different letters, so they
# have different keys, full stop. Repair has no such proof available. The
# argument for it is a POLICY argument -- that a spelling which the author's
# text never uses is not a name the book supports, so folding it onto a
# spelling the text does use is a correction rather than a merge -- and the
# guards below are what make that argument hold often enough to be worth
# shipping. Every one of them can be defeated by a book with the right shape,
# and the ACCEPTED LIMITATIONS section names the shapes.
# ---------------------------------------------------------------------------


def _repair_candidate_pool(roster_index, window_words, source_words):
    """Core tokens of established roster names that are ALSO attested as whole
    words both in the label's own window and in the book at large.

    The book-wide condition is amendment (b): a repair must never target a
    roster name that was never source-attested, or roster pollution
    self-propagates -- a junk name accepted once would then start rewriting
    real names onto itself.
    """
    return {
        token for token in _roster_core_tokens(roster_index)
        if token in window_words and token in source_words
    }


def repair_speaker(canonical, windows, roster_index, source_words):
    """Repair a REFUTED speaker label onto an established spelling, or refuse.

    Call this ONLY on a label the attestation gate has already ruled
    UNATTESTED. Returns the repaired canonical name, or ``None`` to refuse --
    and refusing is the default: every condition below must hold.

      1. The label is non-empty, is not NARRATOR, and is not itself an
         established roster name (an established name is never refuted).
      2. A roster and a book-wide word index are both available. With neither,
         there is no evidence to repair from, so nothing is repaired.
      3. THE BOOK NEVER SPELLS IT THAT WAY. Every token being repaired must be
         absent, as a whole word, from the WHOLE BOOK -- not merely from the
         label's own window. This is the load-bearing guard and it replaces
         the minimum-token-length floor an earlier design proposed (see WHY NO
         LENGTH FLOOR below).
      4. Candidates are core tokens of names already established in this
         book's roster that also occur as whole words in this label's own
         attestation window AND in the book at large (_repair_candidate_pool).
      5. Exactly ONE candidate lies at Levenshtein distance 1 (_is_distance_one
         -- an exact bounded predicate, never a similarity ratio), AND no
         OTHER roster token anywhere in the book's roster lies at distance 1
         either. Two or more of either kind means the evidence does not pick a
         winner, so it refuses.
      6. Tokens that already attest are kept verbatim, in place, with their
         possessive intact; only refuted tokens are substituted.
      7. The rebuilt name must fully ATTEST against the same windows, or it is
         refused. A repair that still would not pass the gate is not a repair.

    WHY NO LENGTH FLOOR. A minimum token length ("only repair tokens of 4+
    characters") is a constant with no general justification: it is only ever
    chosen by measuring collisions on one book's roster, and it neither
    protects a long name (ANDRE/ANDREA are both long) nor is needed by a short
    one the book never spells (KITT). It was replaced by two guards that are
    properties of whatever book is loaded, not of any particular one:
      * condition 3 scales with the book: the shorter and more word-like a
        token is, the likelier the author's own prose contains it, and the
        likelier this guard fires. "TWIN" is refused on any book that uses the
        word "twin"; a 12-letter invented misspelling is refused on any book
        that happens to contain it. Nothing is tuned -- the book decides.
      * condition 5 scales with the cast: ambiguity is measured against the
        ENTIRE roster, not just the tokens visible in this window, so the
        denser a book's name-space is, the more often repair declines. A book
        with ANDREA and ANDRES in the cast repairs neither.
    DEGRADATION ON UNLIKE BOOKS: on a book with a very small vocabulary
    (short texts) guard 3 is weak because absence proves less; on a book whose
    cast names are near-anagrams of ordinary words, repair simply stops firing
    (a lost voice, never wrong prose). On a non-segmenting script (Chinese,
    Japanese, Thai) whole-word evidence is unavailable, so the gate returns
    UNVERIFIABLE and repair is never reached at all.

    ACCEPTED LIMITATIONS (adversarial cases; not papered over):
      * ANDRE -> ANDREA, where ANDRE is a real character who is never named in
        his own window (pronoun-only attribution) but IS named elsewhere in the
        book: PREVENTED by guard 3. If ANDRE is a real character whose name
        appears NOWHERE in the entire book, the repair still fires and is
        wrong. Accepted: the model cannot have read a name the book never
        contains, so such a label is far likelier invention than knowledge --
        but it can come from the model's memory of a famous text, and then this
        is a genuine mis-attribution.
      * MARIA -> MARIE with MARIE polluted into the roster by an earlier bad
        label: PREVENTED by guard 4's book-wide condition whenever the junk
        name is not a whole word of the book (which is how such names get in --
        via UNVERIFIABLE substring-only matches). NOT prevented when the junk
        target happens to be a genuine word of the book.
      * ANDRE with an acute accent -> ANDREA: PREVENTED by guard 3. Accents are
        folded on BOTH sides, so the accented spelling is found in the book's
        own word index and the label is left alone.
      * YUSUF -> YUSUP (transliteration variants): PREVENTED by guard 3 when
        the book uses both spellings. When the book only ever spells YUSUP, the
        repair fires -- and is then correct by construction, since the other
        spelling is the model's own invention.
      * TWIN -> TWINS (plural/singular): PREVENTED by guard 3 on any book that
        uses the singular word anywhere. A book that writes only "twins", never
        "twin", would repair it. Accepted, and bounded: the result is an
        existing roster name, so the cost is one line in the wrong existing
        voice, never a new voice and never altered prose.

    Pure: no I/O, no globals, and neither ``windows``, ``roster_index`` nor
    ``source_words`` is mutated. Deterministic and idempotent -- a repaired
    name attests, so feeding it back returns None (nothing left to repair).

    Args:
        canonical: the canonical UPPERCASE label the gate refused.
        windows: the label's own source windows (as passed to attest_speaker).
        roster_index: {roster_key: established_spelling} for this book.
        source_words: the whole book's folded word set (source_word_index()).

    Returns:
        The repaired canonical name, or None.
    """
    if not canonical or canonical == "NARRATOR":
        return None
    if not roster_index or not source_words:
        return None
    if roster_key(canonical) in roster_index:
        return None

    core_tokens = _core_tokens(canonical)
    if not core_tokens:
        return None

    window_words = set()
    for window in windows or []:
        if window:
            window_words |= _window_words(window)
    if not window_words:
        return None

    refuted = [token for token in core_tokens if token not in window_words]
    if not refuted:
        return None

    # Guard 3: never touch a spelling the author's own text uses.
    if any(token in source_words for token in refuted):
        return None

    roster_tokens = _roster_core_tokens(roster_index)
    pool = _repair_candidate_pool(roster_index, window_words, source_words)

    substitutions = {}
    for token in set(refuted):
        near_roster = [c for c in roster_tokens if _is_distance_one(token, c)]
        if len(near_roster) != 1:
            return None
        near_pool = [c for c in pool if _is_distance_one(token, c)]
        if len(near_pool) != 1:
            return None
        substitutions[token] = near_pool[0]

    rebuilt = []
    for raw_token in canonical.split():
        base, suffix = _split_possessive(raw_token)
        rebuilt.append(substitutions[base] + suffix if base in substitutions else raw_token)

    repaired = " ".join(rebuilt).strip()
    if not repaired or repaired == canonical:
        return None
    if attest_speaker(repaired, windows) != ATTESTED:
        return None
    return repaired


# ---------------------------------------------------------------------------
# Tier 3c: attribution-tag agreement
#
# THE DEFECT THIS ANSWERS. Attestation asks "does this NAME occur somewhere
# nearby?". It never asks "is it the name the adjacent attribution tag actually
# gives". Measured on one clean 8,081-entry production artifact: of 1,117
# dialogue spans followed by a machine-checkable name tag, 60 (5.4%) were
# labelled with a DIFFERENT established character than the tag names -- every
# one of them attested, so no existing check saw them.
#
# DETECT, NEVER REWRITE. These functions are read-only and return evidence.
# Relabelling a span from the tag would be a second repair_speaker: nearby
# evidence silently overriding the model. The caller routes a detection into
# the retry path (ask the model again, naming the tag) and, failing that, into
# the degradation report.
#
# ENGLISH-ONLY, BY CONSTRUCTION, AND IT DEGRADES TO SILENCE. Recognizing an
# attribution tag requires recognizing a speech verb, and there is no
# language-neutral way to do that -- so _SPEECH_VERBS below is an English word
# list, in the same acknowledged-limitation class as _RANK_TITLES (English
# titles) and span_tokenizer's quote-mark table. On a French or Japanese book
# NOTHING in the tag pattern matches: no token equals "said"/"asked"/..., so
# attribution_tag_name() returns None for every span, contradicts_attribution()
# returns None for every span, and the caller's counters stay at zero. The
# failure mode is zero detections -- never a false accusation and never a
# wrong retry. It is not book-fitted: it contains no name, idiom or phrasing
# from any particular novel, only the verbs any English narration uses.
# ---------------------------------------------------------------------------

# English speech verbs that introduce or close an attribution tag. Deliberately
# SHORT and unambiguous: every entry is a verb whose subject is the speaker of
# the adjacent quotation. Verbs that are commonly NOT attributive ("continued",
# "went on", "laughed") are left out -- this list is tuned for precision,
# because a false detection costs a wasted retry and a false degradation.
_SPEECH_VERBS = frozenset({
    "said", "says", "asked", "asks", "replied", "answered", "called",
    "shouted", "whispered", "murmured", "added", "muttered", "agreed",
    "snapped", "told",
})

# A word for tag-matching purposes: a letter-initial token, apostrophes and
# internal hyphens included ("O'Brien", "Jean-Luc", "Nita's"). `[^\W\d_]` is
# "any letter in any script", same intent as _is_name_char.
_TAG_WORD = r"[^\W\d_][\w'‘’\-]*"

# The two shapes an attribution tag takes right after a quotation:
# "<Name> said ..." and "said <Name> ...". Leading whitespace, commas, colons
# and dashes are skipped; anything else means this is not a tag.
_TAG_RE = re.compile(
    r"^[\s,;:\-–—]*(" + _TAG_WORD + r")\s+(" + _TAG_WORD + r")",
    re.UNICODE,
)

# A blank line before the tag means the narration starts a new paragraph, so it
# belongs to what FOLLOWS, not to the quotation before it.
_PARAGRAPH_BREAK_RE = re.compile(r"^[^\S\n]*\n\s*\n")

# Only the first line or so can hold the tag; scanning further invites matches
# deep inside unrelated narration.
_TAG_SCAN_CHARS = 120


def attribution_tag_name(text):
    """The proper-name word of an attribution tag at the START of ``text``.

    ``text`` is the narration immediately following a quotation. Returns the
    name as it is spelled in the source ("Nita", "McAllister"), or None when
    there is no recognizable tag. Pure, read-only, English-only (see the
    section comment: a non-English book yields None for every span).

    Refuses -- returns None -- in every ambiguous case:

      * a blank line before the tag (the narration is a new paragraph, so the
        tag introduces the NEXT quotation rather than closing the previous one);
      * no ``_SPEECH_VERBS`` verb adjacent to the candidate word;
      * a candidate that is not capitalized ("said the man", "said his
        mother") -- a common noun is not a name;
      * a POSSESSIVE candidate ("said Nita's dad"), where the name is a
        modifier of somebody else. This case is the reason the check needs a
        possessive test at all: the label "NITA'S DAD" is correct there, and a
        naive name-grab reads the tag as NITA and reports a contradiction.
    """
    if not text:
        return None
    if _PARAGRAPH_BREAK_RE.match(text):
        return None

    match = _TAG_RE.match(text[:_TAG_SCAN_CHARS])
    if not match:
        return None

    first, second = match.group(1), match.group(2)
    if first.lower() in _SPEECH_VERBS:
        candidate = second
    elif second.lower() in _SPEECH_VERBS:
        candidate = first
    else:
        return None

    if candidate.lower() in _SPEECH_VERBS:
        return None
    if not candidate[:1].isupper():
        return None
    if _split_possessive(candidate)[1]:
        return None
    return candidate


def contradicts_attribution(speaker, following_text, roster_index):
    """The established roster name an attribution tag gives, when it CONTRADICTS
    ``speaker``. Returns None whenever the two are compatible, or whenever the
    evidence does not decide -- which is most of the time, by design.

    ``speaker`` is a canonical label; ``following_text`` is the narration span
    immediately after that speaker's quotation; ``roster_index`` is a
    ``{roster_key: spelling}`` index (see remember_in_roster) used to turn a tag
    word into an established name. Roster-aware and therefore explicit-argument,
    like every other roster consumer here; nothing is mutated.

    Compatible -- and so NOT reported -- includes:

      * case and apostrophe-glyph differences (MCALLISTER vs McAllister,
        SKER'RET vs Sker’ret): both sides go through _fold_word;
      * GRANULARITY variants (TOM vs TOM SWALE, DARRYL vs DARRYL MCALLISTER):
        the label and the tagged name sharing any core token is agreement about
        a person, not a contradiction;
      * an AMBIGUOUS tag word matching two or more roster names (two characters
        sharing a first name): discarded, never guessed;
      * a tag word the roster does not know at all. Recall is deliberately
        sacrificed here: an unknown word cannot be compared with a label
        without guessing, and a guess is what this check exists to avoid.
    """
    if not speaker or speaker == "NARRATOR" or not roster_index:
        return None

    word = attribution_tag_name(following_text)
    if not word:
        return None

    base = _fold_word(word)
    if not base:
        return None

    speaker_tokens = set(_core_tokens(speaker))
    if base in speaker_tokens:
        return None

    tagged = {name for name in roster_index.values() if base in _core_tokens(name)}
    if len(tagged) > 1:
        # A bare tag word that IS an established name in its own right resolves
        # to that name, even though longer names contain it too: a roster
        # holding both "NITA" and "NITA'S DAD" is not ambiguous about who
        # "Nita said" means. Two names merely CONTAINING the word and neither
        # equal to it (two characters sharing a first name) stays ambiguous.
        tagged = {name for name in tagged if _core_tokens(name) == [base]}
    if len(tagged) != 1:
        return None

    tagged_name = next(iter(tagged))
    if speaker_tokens & set(_core_tokens(tagged_name)):
        return None
    return tagged_name


def near_spellings(canonical, windows, roster_index):
    """Established roster spellings within one edit of a refuted label token,
    for use as a PROMPT HINT when repair declines.

    Advisory only and deliberately unguarded compared with repair_speaker: it
    lists possibilities for the model to choose from (or ignore -- a prompt
    cannot enforce anything), it never changes a label. Returns a sorted list
    of candidate tokens, possibly empty. Pure; mutates nothing.
    """
    if not canonical or not roster_index:
        return []
    window_words = set()
    for window in windows or []:
        if window:
            window_words |= _window_words(window)
    roster_tokens = _roster_core_tokens(roster_index)
    candidates = set()
    for token in _core_tokens(canonical):
        if token in window_words:
            continue
        for candidate in roster_tokens:
            if _is_distance_one(token, candidate):
                candidates.add(candidate)
    return sorted(candidates)
