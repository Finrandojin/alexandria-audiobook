"""Character-by-character FSM that splits book text into quoted / unquoted spans.

WHY THIS EXISTS
---------------
Audiobooks are verbatim: the spoken word must match the printed word exactly.
The downstream pipeline sends only span *ids* to an LLM and receives only
*labels* back (``{"id", "speaker", "role", "instruct"}``) -- never book text.
Code then reassembles the audiobook script verbatim from the source string by
``(start, end)`` offsets. That makes this tokenizer the correctness keystone:
if the spans do not tile the source exactly, the reassembled text is wrong.

The one invariant every caller may rely on::

    "".join(source[s.start:s.end] for s in tokenize(source)) == source

Spans are contiguous (``spans[i].end == spans[i + 1].start``), the first starts
at 0, the last ends at ``len(source)``, ids are ``1..N`` in document order, and
no span is empty. Whitespace between a quotation and its attribution tag lives
in exactly one span, so concatenation is lossless.

Sentence segmentation libraries (pysbd, spaCy) are deliberately NOT used: they
group a quotation and its attribution tag into a single sentence, which is
precisely the failure mode this module fixes.

WHAT IT DOES *NOT* DO
---------------------
* It does not drop attribution tags. ``"Go," he said.`` yields a quoted span
  ``"Go,"`` and an unquoted span ``` he said.``` -- the narrator reads the tag.
* It does not classify speakers, does not detect em-dash or unquoted dialogue,
  and does not split narration into sentences. Those are the LLM's job.
* It does not normalize, strip or rewrite a single character of the source.

STATE MACHINE
-------------
Three states: ``NARRATION``, ``IN_PAIRED``, ``IN_SINGLE``. Only the *outer*
quote level delimits, so nested quotes stay inside their enclosing span
(``"She told me 'go away' yesterday," he said.`` is ONE quoted span plus one
unquoted span). A quotation left open at end of input becomes a quoted span
running to ``len(source)`` -- no crash, no lost text.

PAIRED (UNAMBIGUOUS) QUOTE DELIMITERS
-------------------------------------
``PAIRED_QUOTES`` maps every unambiguous OPENER character to the set of
characters that may close *that* opener. Adding a language's convention is a
one-line data change; the FSM never grows a branch for it. In ``IN_PAIRED``
only the closers of the opener that started the span end it, so a quotation
opened with one convention cannot be closed by another convention's mark and
inner quotes of a different convention stay inside the outer span.

Supported today:

===========  ================  =========================================
Opener       Closers           Convention
===========  ================  =========================================
``"``        ``"`` ``”``       English straight (also closes on curly)
``“``        ``"`` ``”``       English curly
``„``        ``“`` ``”``       German / Polish / Czech low-high
``«``        ``»``             French / Russian / Spanish guillemets
``‹``        ``›``             single guillemets
``「``       ``」``            CJK corner brackets
``『``       ``』``            CJK white corner brackets
``＂``       ``＂``            full-width straight (U+FF02)
===========  ================  =========================================

A ``"`` seen in NARRATION always opens -- double quotes are effectively never
apostrophes in prose. The same holds for every other character in the table:
none of them is ever an apostrophe, so none is routed through the ambiguous
single-quote machinery below, and none is subject to its length or paragraph
bounds.

``“`` IS BOTH AN OPENER AND A CLOSER -- how that is disambiguated
-----------------------------------------------------------------
``“`` opens an English curly quotation but closes a German ``„`` one. State,
not the character, decides: in NARRATION ``“`` opens (unchanged English
behaviour); inside a span opened by ``„`` it closes, because ``„``'s closer
set is the only one consulted there. An English document never contains ``„``
and so can never reach that state -- adding German cannot regress English.
The mirror case ``‚ ... ‘`` works the same way, and leaves ``‘``'s role as the
unambiguous single opener in NARRATION untouched.

Whitespace *inside* the marks is part of the quoted span and is never touched,
so French typography's inner spaces -- U+0020, U+00A0 or U+202F, as in
``« bonjour »`` -- survive byte-exactly.

WHAT IS DELIBERATELY NOT A DELIMITER
------------------------------------
``《 》`` and ``〈 〉`` mark work titles rather than speech in Chinese, and
``〝 〞`` is vanishingly rare in book text; treating them as dialogue would
quote non-dialogue, which is the unsafe direction. They stay narration.

SINGLE-QUOTE DISAMBIGUATION RULE (the ambiguous case, documented exactly)
------------------------------------------------------------------------
``‘`` (left single curly) is unambiguous and ALWAYS opens a span.

``'`` (U+0027) and ``’`` (right single curly) are ambiguous -- they are
also the apostrophe character. Such a character opens a quoted span only when
ALL FIVE of the following hold; otherwise it is treated as an apostrophe and
stays inside the surrounding UNQUOTED span:

1. It is at start-of-text, or the preceding character is whitespace, or the
   preceding character is opening punctuation (one of ``([{<"“”«—–-*_~``).
   This rejects ``don't``, ``O'Brien`` and possessive ``Jones'``.
2. It is not the last character of the input.
3. The following character is an ASCII/Unicode *letter*. Requiring a letter
   (not a digit) rejects decade elisions such as ``'90s``.
4. The alphabetic run that follows is not a known elision (``'tis``, ``'twas``,
   ``'em``, ``'round``, ``'til`` ... see ``ELISIONS``). ``'tis`` therefore never
   opens a span.
5. A syntactically valid CLOSER (see below) exists later in the input, and it
   falls within BOTH of these bounds:

   a. Before the next paragraph break -- a blank-line boundary, i.e. a newline
      separated from a following newline by at most horizontal whitespace.
      When no break follows, end of input is the bound.
   b. Within ``MAX_AMBIGUOUS_SINGLE_SPAN`` characters of the opener.

   All three parts earn their keep. Without "a closer exists", an unterminated
   single quote swallows the rest of the chunk on what is far more likely an
   apostrophe. Without (a), an opener plus a possessive in a later paragraph
   closes across the paragraph boundary. Without (b), the same thing happens
   *inside* one long paragraph, which is the case that actually bites:
   ``He said 'no more.`` followed by 40 sentences of narration and ``The dogs'
   tails wagged.`` -- one paragraph, no newline anywhere -- produced a single
   1139-character *quoted* span. No text is lost either way, but handing the
   LLM a page-long quotation invites it to give plain narration a character
   voice, which is fidelity-adverse.

   ``MAX_AMBIGUOUS_SINGLE_SPAN`` is 500 to match ``MAX_CHUNK_CHARS`` in
   ``project.py``: a quotation longer than the pipeline's own chunk cap gets
   split downstream regardless, so accepting one buys nothing and risks a
   runaway. Single-quoted dialogue longer than that, or spanning a paragraph
   break, is rare enough that narrator fallback is the right trade.

   Rule 5 applies ONLY to the ambiguous glyphs. ``‘`` is unambiguous and is
   not subject to it -- neither bound constrains ``‘`` dialogue.

A ``'`` or ``’`` closes an open single-quoted span only when the preceding
character is a non-space AND the following character is end-of-input,
whitespace, or any non-alphanumeric character. That keeps ``doesn't`` /
``doesn’t`` inside the quotation instead of closing it early.

Ambiguity that these rules cannot resolve resolves toward UNQUOTED, because an
unlabelled span falls back to NARRATOR downstream -- the safe failure mode.

KNOWN, ACCEPTED LIMITATIONS
---------------------------
* A single-quoted line that opens on an elision (``'Course you can,' he said.``)
  is read as narration. Rule 4 wins; narrator fallback is safe.
* A plural possessive inside single-quoted dialogue (``'the dogs' tails,' he
  said``) closes the quotation early at ``dogs'``. The remaining text becomes
  narration -- again the safe direction, and no text is lost.
* Straight-single-quoted dialogue that continues across a paragraph break, or
  that runs longer than ``MAX_AMBIGUOUS_SINGLE_SPAN``, is read as narration
  (rule 5's two bounds). ``‘`` dialogue is unaffected by both.
"""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

__all__ = [
    "Span",
    "tokenize",
    "reassemble",
    "validate_spans",
    "QUOTED",
    "UNQUOTED",
    "MAX_AMBIGUOUS_SINGLE_SPAN",
    "PAIRED_QUOTES",
]

QUOTED = "quoted"
UNQUOTED = "unquoted"

# --- quote character classes -------------------------------------------------

DOUBLE_OPENERS = '"“'            # "  and  “   (English; kept for reference)
DOUBLE_CLOSERS = '"”'            # "  and  ”

# Every unambiguous paired delimiter: OPENER -> the closers valid for IT.
# Adding a language's convention is a data change here, not a logic change.
# See the table in the module docstring for the rationale of each row, and in
# particular for why “ may appear as both a key and a value without ambiguity.
PAIRED_QUOTES = {
    '"': '"”',      # English straight double
    "“": '"”',      # English curly double
    "„": "“”",      # German / Polish / Czech low-high double
    "«": "»",       # French / Russian / Spanish guillemets
    "‹": "›",       # single guillemets
    "「": "」",     # CJK corner brackets
    "『": "』",     # CJK white corner brackets
    "＂": "＂",     # full-width straight double (U+FF02)
    "‚": "‘’",      # German / Polish low-high single
}

SINGLE_UNAMBIGUOUS_OPENER = "‘"  # ‘  (never an apostrophe)
SINGLE_AMBIGUOUS = "'’"          # '  and  ’  (also the apostrophe glyph)
SINGLE_CLOSERS = "'’"            # '  and  ’

# Characters that may legitimately sit immediately before an opening quote.
OPEN_CONTEXT_CHARS = "([{<\"“”«—–-*_~" + "".join(PAIRED_QUOTES)

# Longest span an AMBIGUOUS single quote (' or ’) may open. Matches
# MAX_CHUNK_CHARS in project.py: a quotation longer than the pipeline's own
# chunk cap is split downstream anyway, so accepting one buys nothing and
# risks a stray apostrophe swallowing a page of narration. Does not apply to
# the unambiguous ‘.
MAX_AMBIGUOUS_SINGLE_SPAN = 500

# Alphabetic runs that follow a leading apostrophe in an elision. A quote
# followed by one of these is an apostrophe, not an opening quote.
ELISIONS = frozenset(
    [
        "tis", "twas", "twere", "twill", "twould", "twixt", "tween",
        "em", "im", "er", "n", "un", "uns",
        "til", "till", "bout", "cause", "cept", "course",
        "gainst", "neath", "nother", "fore", "fraid",
        "round", "stead", "scuse", "specially",
        "way", "nuff", "ol", "ere", "kay", "lo", "pon", "gain", "bove",
        "bye", "sblood", "struth", "twarn", "tain",
    ]
)


@dataclass(frozen=True)
class Span:
    """A contiguous slice of the source string.

    ``id``    -- 1-based, contiguous, document order.
    ``start`` / ``end`` -- offsets such that ``source[start:end]`` is the text.
    ``kind``  -- ``"quoted"`` (a quotation *including* its quote marks) or
                 ``"unquoted"`` (everything else, attribution tags included).
    """

    id: int
    start: int
    end: int
    kind: str

    def text(self, source):
        """Return this span's verbatim text from the source string."""
        return source[self.start:self.end]

    def as_dict(self):
        return {"id": self.id, "start": self.start, "end": self.end, "kind": self.kind}


# --- single-quote heuristics -------------------------------------------------

def _alpha_run(source, index):
    """Return the lowercased alphabetic run starting at ``index``."""
    end = index
    n = len(source)
    while end < n and source[end].isalpha():
        end += 1
    return source[index:end].lower()


def _is_single_closer(source, index):
    """True when ``source[index]`` can close an open single-quoted span.

    Requires a non-space before it and end-of-input, whitespace or any
    non-alphanumeric character after it (so ``doesn't`` never closes).
    """
    if index <= 0:
        return False
    prev = source[index - 1]
    if prev.isspace():
        return False
    nxt_index = index + 1
    if nxt_index >= len(source):
        return True
    return not source[nxt_index].isalnum()


def _closer_index(source):
    """Every position in ``source`` that could close a single-quoted span.

    Precomputed once per tokenize() call so the opener rule's "does a closer
    exist?" test is a binary search rather than a rescan of the tail. Without
    it, a chunk full of failing opener candidates costs O(n^2) -- ~60s on
    200KB of adversarial text. The result is identical either way.
    """
    return [
        j
        for j, ch in enumerate(source)
        if ch in SINGLE_CLOSERS and _is_single_closer(source, j)
    ]


def _paragraph_break_index(source):
    """Start offsets of every blank-line boundary in ``source``.

    A paragraph break is a newline separated from a following newline by at
    most horizontal whitespace, so ``\\n\\n``, ``\\n   \\n`` and ``\\r\\n\\r\\n``
    all qualify. Precomputed once per tokenize() call and binary-searched, for
    the same reason as the closer index.
    """
    breaks = []
    n = len(source)
    for j, ch in enumerate(source):
        if ch != "\n":
            continue
        k = j + 1
        while k < n and source[k] in " \t\r":
            k += 1
        if k < n and source[k] == "\n":
            breaks.append(j)
    return breaks


def _paragraph_limit(source, index, breaks=None):
    """Offset of the next paragraph break after ``index``, else ``len(source)``."""
    if breaks is None:
        breaks = _paragraph_break_index(source)
    position = bisect_right(breaks, index)
    if position < len(breaks):
        return breaks[position]
    return len(source)


def _find_single_closer(source, open_index, closers=None):
    """Index of the first valid closer after ``open_index``, or -1."""
    # open_index + 1 is guaranteed to be a letter by the caller, so start after.
    if closers is None:
        closers = _closer_index(source)
    position = bisect_left(closers, open_index + 2)
    if position < len(closers):
        return closers[position]
    return -1


def _is_single_opener(source, index, closers=None, breaks=None):
    """Apply the five-part ambiguous-single-quote rule (see module docstring)."""
    # 1. preceded by start-of-text, whitespace or opening punctuation
    if index > 0:
        prev = source[index - 1]
        if not (prev.isspace() or prev in OPEN_CONTEXT_CHARS):
            return False
    # 2. not the final character
    nxt_index = index + 1
    if nxt_index >= len(source):
        return False
    # 3. followed by a letter (digits rejected: '90s is an elision)
    if not source[nxt_index].isalpha():
        return False
    # 4. not a known elision ('tis, 'twas, 'em ...)
    if _alpha_run(source, nxt_index) in ELISIONS:
        return False
    # 5. a valid closer must exist, and must land before BOTH the next
    #    paragraph break and the span-length cap -- otherwise a stray opener
    #    plus a distant possessive swallows a page of narration
    closer = _find_single_closer(source, index, closers)
    if closer == -1:
        return False
    limit = _paragraph_limit(source, index, breaks)
    if index + MAX_AMBIGUOUS_SINGLE_SPAN < limit:
        limit = index + MAX_AMBIGUOUS_SINGLE_SPAN
    if closer >= limit:
        return False
    return True


# --- the finite state machine ------------------------------------------------

_NARRATION = 0
_IN_PAIRED = 1
_IN_SINGLE = 2


def tokenize(source):
    """Split ``source`` into contiguous quoted / unquoted :class:`Span` records.

    Returns ``[]`` for empty input. Never raises on malformed quoting.
    """
    if not source:
        return []

    spans = []
    n = len(source)
    closers = _closer_index(source)
    breaks = _paragraph_break_index(source)

    def emit(start, end, kind):
        if end > start:
            spans.append(Span(id=len(spans) + 1, start=start, end=end, kind=kind))

    state = _NARRATION
    segment_start = 0     # start of the current unquoted run
    quote_start = -1      # start of the currently open quotation
    pending_closers = ""  # closers valid for the currently open paired quote
    i = 0

    while i < n:
        ch = source[i]

        if state == _NARRATION:
            if ch in PAIRED_QUOTES:
                emit(segment_start, i, UNQUOTED)
                state = _IN_PAIRED
                pending_closers = PAIRED_QUOTES[ch]
                quote_start = i
            elif ch == SINGLE_UNAMBIGUOUS_OPENER or (
                ch in SINGLE_AMBIGUOUS
                and _is_single_opener(source, i, closers, breaks)
            ):
                emit(segment_start, i, UNQUOTED)
                state = _IN_SINGLE
                quote_start = i

        elif state == _IN_PAIRED:
            # Only THIS opener's closers end the span, so a differently
            # punctuated inner quotation stays inside it.
            if ch in pending_closers:
                emit(quote_start, i + 1, QUOTED)
                segment_start = i + 1
                state = _NARRATION

        else:  # _IN_SINGLE -- nested double quotes are ignored on purpose
            if ch in SINGLE_CLOSERS and _is_single_closer(source, i):
                emit(quote_start, i + 1, QUOTED)
                segment_start = i + 1
                state = _NARRATION

        i += 1

    # Tail: an unterminated quotation runs to end of input rather than losing text.
    if state == _NARRATION:
        emit(segment_start, n, UNQUOTED)
    else:
        emit(quote_start, n, QUOTED)

    return spans


def reassemble(spans, source):
    """Rebuild the source text from spans by offset. Must equal ``source``."""
    return "".join(source[s.start:s.end] for s in spans)


def validate_spans(spans, source):
    """Raise ``ValueError`` unless the spans tile ``source`` exactly.

    Checks the full contract: 1..N ids in order, contiguity, no empty spans,
    full coverage, known kinds, and byte-exact reassembly.
    """
    if not source:
        if spans:
            raise ValueError("empty source must produce no spans")
        return True

    if not spans:
        raise ValueError("non-empty source produced no spans")

    if spans[0].start != 0:
        raise ValueError("first span does not start at 0: %d" % spans[0].start)
    if spans[-1].end != len(source):
        raise ValueError(
            "last span ends at %d, expected %d" % (spans[-1].end, len(source))
        )

    for index, span in enumerate(spans):
        if span.id != index + 1:
            raise ValueError("span id %d out of order at position %d" % (span.id, index))
        if span.end <= span.start:
            raise ValueError("empty or inverted span id %d" % span.id)
        if span.kind not in (QUOTED, UNQUOTED):
            raise ValueError("unknown span kind %r on id %d" % (span.kind, span.id))
        if index + 1 < len(spans) and span.end != spans[index + 1].start:
            raise ValueError(
                "gap or overlap between span %d and %d" % (span.id, spans[index + 1].id)
            )

    if reassemble(spans, source) != source:
        raise ValueError("reassembly does not reproduce the source verbatim")

    return True
