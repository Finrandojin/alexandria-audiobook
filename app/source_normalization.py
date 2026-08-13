"""Explicit, auditable normalization for known source-text corruption."""

import re


KNOWN_SOURCE_CORRUPTIONS = {"саге": "care", "пар": "nap"}
_KNOWN_RE = re.compile("|".join(re.escape(value) for value in KNOWN_SOURCE_CORRUPTIONS),
                       re.IGNORECASE)
_ILLUSTRATION_CAPTION_RE = re.compile(
    r"Illustration from Volume\s+\d+\s*,\s*coloring by\s+[^\n]{1,100}?\s*"
    r"\(source\)\s*", re.IGNORECASE)

_WORD_RE = re.compile(r"[^\W_]+(?:[’'][^\W_]+)?", re.UNICODE)
EXTREME_PHRASE_REPEAT_MIN = 20
EXTREME_PHRASE_REPEAT_KEEP = 3
EXTREME_PHRASE_MAX_WORDS = 5


def normalize_known_source_corruptions(text):
    """Return normalized text and location evidence without mutating its source file."""
    changes = []

    def replace(match):
        before = match.group(0)
        after = KNOWN_SOURCE_CORRUPTIONS[before.casefold()]
        if before[:1].isupper():
            after = after.capitalize()
        offset = match.start()
        line = text.count("\n", 0, offset) + 1
        line_start = text.rfind("\n", 0, offset) + 1
        changes.append({"offset": offset, "line": line, "column": offset - line_start + 1,
                        "before": before, "after": after})
        return after

    normalized = _KNOWN_RE.sub(replace, text)

    def remove_caption(match):
        offset = match.start()
        line = normalized.count("\n", 0, offset) + 1
        line_start = normalized.rfind("\n", 0, offset) + 1
        changes.append({"offset": offset, "line": line,
                        "column": offset - line_start + 1,
                        "before": match.group(0), "after": "",
                        "rule": "illustration_caption"})
        return ""

    return _ILLUSTRATION_CAPTION_RE.sub(remove_caption, normalized), changes


def normalize_extreme_phrase_repetitions(text):
    """Collapse only measured token-loop traps, returning location evidence.

    Ordinary emphasis and stutters stay untouched. Corpus calibration found a
    legitimate maximum of 13 contiguous repeats, while runs of 25, 42 and 43
    made the generation model loop to its token ceiling. A repeated unit may
    contain one to five words, but separators must be whitespace so unrelated
    prose cannot be joined into a match.
    """
    tokens = list(_WORD_RE.finditer(text))
    candidates = []
    for start in range(len(tokens)):
        for width in range(1, EXTREME_PHRASE_MAX_WORDS + 1):
            if start + width * EXTREME_PHRASE_REPEAT_MIN > len(tokens):
                break
            unit = [match.group(0).casefold()
                    for match in tokens[start:start + width]]
            repeats = 1
            while start + (repeats + 1) * width <= len(tokens):
                offset = start + repeats * width
                next_unit = [match.group(0).casefold()
                             for match in tokens[offset:offset + width]]
                separator = text[tokens[offset - 1].end():tokens[offset].start()]
                if next_unit != unit or not separator.isspace():
                    break
                repeats += 1
            if repeats < EXTREME_PHRASE_REPEAT_MIN:
                continue
            end_index = start + repeats * width - 1
            candidates.append((tokens[start].start(), tokens[end_index].end(),
                               width, repeats))

    selected = []
    for candidate in sorted(candidates, key=lambda item: (item[0], -item[1])):
        if selected and candidate[0] < selected[-1][1]:
            continue
        selected.append(candidate)
    if not selected:
        return text, []

    changes = []
    pieces = []
    cursor = 0
    for start, end, width, repeats in selected:
        repeated = text[start:end]
        repeated_tokens = list(_WORD_RE.finditer(repeated))
        kept_end = repeated_tokens[width * EXTREME_PHRASE_REPEAT_KEEP - 1].end()
        replacement = repeated[:kept_end] + "…"
        pieces.extend((text[cursor:start], replacement))
        line_start = text.rfind("\n", 0, start) + 1
        changes.append({
            "offset": start,
            "line": text.count("\n", 0, start) + 1,
            "column": start - line_start + 1,
            "before": repeated,
            "after": replacement,
            "rule": "extreme_phrase_repetition",
            "phrase_words": width,
            "repetitions": repeats,
        })
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), changes


# Cyrillic letters visually identical to Latin ones in upright fonts, per the
# Unicode TR39 confusables data (https://www.unicode.org/Public/security/latest/
# confusables.txt), vendored as a small explicit dict rather than a dependency
# so every possible rewrite stays enumerable and auditable.
_HOMOGLYPH_MAP = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ѕ": "s", "ј": "j",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X", "Ѕ": "S", "І": "I",
    "Ј": "J",
    # Italic-form confusables observed in this corpus's actual corrupted
    # sources (KNOWN_SOURCE_CORRUPTIONS: "саге" -> "care", "пар" -> "nap").
    "г": "r", "п": "n",
}
_CYRILLIC_CHAR_RE = re.compile(r"[Ѐ-ӿ]")
_CYRILLIC_WORD_RE = re.compile(r"\w*[Ѐ-ӿ]\w*", re.UNICODE)
# A genuinely bilingual/Cyrillic text must never be rewritten; stray OCR
# corruption in a Latin book stays far below this share of letters.
MAX_HOMOGLYPH_CYRILLIC_RATIO = 0.005


def normalize_homoglyph_words(text):
    """Map whole words of Cyrillic lookalike characters back to Latin,
    returning normalized text and location evidence without mutating the
    source file. Complements normalize_known_source_corruptions (which runs
    first and handles exact known words): a word is rewritten only when the
    document is overwhelmingly Latin and every Cyrillic character in the
    word has a homoglyph mapping - one unmappable character leaves the whole
    word untouched."""
    letter_count = sum(1 for char in text if char.isalpha())
    if not letter_count:
        return text, []
    cyrillic_count = sum(1 for char in text if _CYRILLIC_CHAR_RE.match(char))
    if not cyrillic_count or cyrillic_count / letter_count >= MAX_HOMOGLYPH_CYRILLIC_RATIO:
        return text, []

    changes = []

    def replace(match):
        word = match.group(0)
        if any(char not in _HOMOGLYPH_MAP
               for char in _CYRILLIC_CHAR_RE.findall(word)):
            return word
        after = "".join(_HOMOGLYPH_MAP.get(char, char) for char in word)
        offset = match.start()
        line = text.count("\n", 0, offset) + 1
        line_start = text.rfind("\n", 0, offset) + 1
        changes.append({"offset": offset, "line": line, "column": offset - line_start + 1,
                        "before": word, "after": after, "rule": "homoglyph"})
        return after

    return _CYRILLIC_WORD_RE.sub(replace, text), changes


_FRONT_MATTER_ANCHOR = re.compile(
    r"Original (?:Web Novel|Light Novel) Chapter\s*[―—-]\s*(?:In)?[Cc]omplete\.\s*\n+"
    r"Original Translation by [^\n]+\.\s*\n+"
)


def strip_known_front_matter(text):
    """Strip a known fan-compiler's non-narrative front matter (translator's
    note + table of contents) when present, returning the story text and
    evidence of what was removed without mutating the source file.

    Scoped to one observed, stable compiler template (confirmed across 5
    "wn" uploads): a "Manifesto." translator's essay and chapter listing,
    ending right before the first "Original ... Chapter - Complete." /
    "Original Translation by ..." marker pair, which is always immediately
    followed by the real chapter 1 prose. This content isn't dialogue or
    narration and was measured live to break chunk generation (near-zero
    recall no matter how many times retried or split) since the annotation
    model has no idea how to handle it. Returns the text unchanged (and
    None) whenever the shape doesn't match, rather than guessing.
    """
    if not text.lstrip("﻿ \t\r\n").startswith("Manifesto."):
        return text, None
    match = _FRONT_MATTER_ANCHOR.search(text)
    if not match:
        return text, None
    return text[match.end():], {"removed_chars": match.end(),
                                 "removed_lines": text.count("\n", 0, match.end())}


_YEAR_RE = re.compile(r"\s*(?:1[89]|20)\d{2}\b")
_REPLACEMENT = "�"


def _nearest_surviving(chars, index, step):
    """Return the closest non-U+FFFD neighbour in one direction.

    Consecutive U+FFFD are separate destroyed characters, so a neighbour
    lookup has to skip past them to find real context. Returns "\n" when it
    runs off either end, which makes start/end of file behave like a line
    boundary.
    """
    position = index + step
    while 0 <= position < len(chars) and chars[position] == _REPLACEMENT:
        position += step
    return chars[position] if 0 <= position < len(chars) else "\n"


def _infer_replacement(chars, index):
    """Infer one destroyed character from its surroundings, or None."""
    left = chars[index - 1] if index else "\n"
    right = chars[index + 1] if index + 1 < len(chars) else "\n"
    right_surviving = _nearest_surviving(chars, index, 1)
    if _YEAR_RE.match("".join(chars[index + 1:index + 7])):
        return "©"
    if right == _REPLACEMENT and (left.isalnum() or left in ".,!?"):
        return "…"
    if left == _REPLACEMENT and right == _REPLACEMENT:
        # Interior of a run of three or more, as in "\n���\n" -> "\n“…”\n".
        return "…"
    if left == _REPLACEMENT and right in "\n \t":
        return "”"
    if left in "\n \t" and (right_surviving.isalnum() or right == _REPLACEMENT):
        return "“"
    if left in ".!?,;:…" and right in "\n \t":
        return "”"
    if left.isalpha() and right.islower():
        return "’"
    if left.isalpha() and right.isupper():
        return "—"
    if left.isdigit():
        return "–"
    return None


def repair_lossy_replacements(text):
    """Infer characters destroyed into U+FFFD, returning text and evidence.

    Distinct from generate_script.fix_mojibake, which repairs the recoverable
    byte form (``â€™``). Here the original bytes are gone, so each U+FFFD is
    inferred from its neighbours. Inference is per character position because
    a run of U+FFFD is several destroyed characters, not one. Returns the text
    unchanged when there is nothing to repair, and never mutates the source
    file. Positions that cannot be inferred are left as U+FFFD for the caller's
    residual policy to handle.
    """
    if _REPLACEMENT not in text:
        return text, []
    chars = list(text)
    repairs = []
    for index, char in enumerate(chars):
        if char != _REPLACEMENT:
            continue
        inferred = _infer_replacement(chars, index)
        if inferred is not None:
            repairs.append({"offset": index, "before": _REPLACEMENT,
                            "after": inferred})
    for repair in repairs:
        chars[repair["offset"]] = repair["after"]
    return "".join(chars), repairs


def neutralize_lossy_residue(text, substitute="'"):
    """Replace U+FFFD that no rule could infer, returning text and a count.

    Applied only after repair_lossy_replacements. The residue is genuinely
    unrecoverable: destroyed letters (``coup d’état``), and cases ambiguous
    between a plural possessive and a closing quote (``knights’`` vs
    ``knights”``) that context cannot separate. A plain apostrophe is the most
    likely value across that residue. Callers record the count so the
    approximation is visible rather than silent.
    """
    if _REPLACEMENT not in text:
        return text, 0
    return text.replace(_REPLACEMENT, substitute), text.count(_REPLACEMENT)


# Signatures of a publisher colophon. Each is strong on its own: none of these
# appear in narrative prose, so one hit marks a paragraph as non-story.
_PUBLISHER_SIGNATURES = re.compile(
    r"©|\(c\)\s*\d{4}|all rights reserved|first published in|"
    r"english translation (?:rights|©|copyright)|"
    r"this book is a work of fiction|"
    r"scanning,? uploading,? and distribution|"
    r"\bISBN\b|ebook edition|printed in the united states|"
    r"yenpress\.com|j-novel\.club|yen (?:on|press)|j-novel club|"
    r"kadokawa|hayakawa|seven seas|vertical, inc|"
    r"library of congress|cataloging-in-publication|"
    r"^\s*copyright\b|translation by |cover art by |illustration by |"
    # Library-of-Congress cataloging block, printed verbatim by several
    # publishers: these field labels never occur in narrative prose.
    r"^\s*(?:subjects|identifiers|classification|description|names|title)\s*:|"
    r"\bLCCN\b|\bLCGFT\b|\bCYAC\b|\bLCC\s+P|"
    # Navigation scaffolding an epub extractor leaves at the head.
    r"^\s*(?:table of )?contents\b|^\s*cover page\b|^\s*landmarks\b|"
    r"^\s*color illustrations\b|^\s*(?:front|back)\s*matter\b",
    re.IGNORECASE | re.MULTILINE)

# A colophon paragraph is short. Long prose that happens to mention a publisher
# is story, not front matter.
_MAX_COLOPHON_PARAGRAPH_CHARS = 400
# Colophon blocks sit at the very ends. Never strip from the middle of a book.
# A colophon runs 10-30 paragraphs regardless of book length, so the share
# alone is too tight on a short book and the floor alone too loose on a long
# one; the scan window is the larger of the two, capped at half the book so a
# head and tail scan can never meet.
_MAX_COLOPHON_SHARE = 0.10
_MIN_COLOPHON_WINDOW = 40


# An epub extractor flattens a table of contents into one long paragraph, so
# the length cap above would exclude it. Repeated chapter markers or bullets
# identify it regardless of length - narrative prose does not enumerate
# "Chapter 1 Chapter 2 Chapter 3".
_TOC_MARKERS = re.compile(r"chapter\s+[0-9IVXLC]+|•|\u25a0\s*\d", re.IGNORECASE)
_MIN_TOC_MARKERS = 3


def _is_navigation_paragraph(text):
    return len(_TOC_MARKERS.findall(text)) >= _MIN_TOC_MARKERS


def _is_colophon_paragraph(paragraph):
    text = paragraph.strip()
    if not text:
        return False
    if _is_navigation_paragraph(text):
        return True
    if len(text) > _MAX_COLOPHON_PARAGRAPH_CHARS:
        return False
    return bool(_PUBLISHER_SIGNATURES.search(text))


def _colophon_run(paragraphs, limit, reverse=False):
    """Length of the colophon run anchored at one end.

    Tolerates ordinary lines between signature lines - a colophon interleaves
    the publisher's address and the book's title with its legal text - but only
    while a signature line keeps appearing, so the run cannot creep into prose.
    """
    order = range(len(paragraphs) - 1, -1, -1) if reverse else range(len(paragraphs))
    run, last_signature = 0, -1
    for step, index in enumerate(order):
        if step >= limit:
            break
        if not paragraphs[index].strip():
            # Blank paragraphs are neutral: an epub extractor leaves runs of
            # them inside a colophon, and counting them as evidence against
            # broke the run five blanks in.
            continue
        if _is_colophon_paragraph(paragraphs[index]):
            last_signature = step
            run = step + 1
        elif (len(paragraphs[index].strip()) > _MAX_COLOPHON_PARAGRAPH_CHARS
              and not _is_navigation_paragraph(paragraphs[index])):
            break                      # real prose: stop here
        elif step - last_signature > 4:
            break                      # drifted too far from the last signature
    return run if last_signature >= 0 else 0


def strip_publisher_matter(text):
    """Remove a publisher colophon from the head and tail of a book.

    Official epubs carry a copyright page, and often a closing colophon, that
    is not narration but was being read aloud: one book opened with 16 entries
    of copyright notice, translator credit and a New York street address,
    another closed with 12 including a URL and an ebook edition number.

    Distinct from strip_known_front_matter, which handles one fan-compiler's
    "Manifesto." template. This is the publisher pattern, and it appears at
    both ends rather than only the front.

    Only runs anchored at an end are removed, never a match in the middle, and
    the whole book is never stripped. Returns the text and what was removed.
    """
    paragraphs = text.split("\n\n")
    if len(paragraphs) < 8:
        return text, {"front_paragraphs": 0, "back_paragraphs": 0}
    limit = min(len(paragraphs) // 2,
                max(_MIN_COLOPHON_WINDOW,
                    int(len(paragraphs) * _MAX_COLOPHON_SHARE)))
    front = _colophon_run(paragraphs, limit)
    back = _colophon_run(paragraphs, limit, reverse=True)
    if front + back > len(paragraphs) * 0.8:
        # Nearly all of it looks like a colophon: a broken or truncated source,
        # not a book with front matter. Leave it for the caller's gate to
        # reject rather than silently emptying it.
        return text, {"front_paragraphs": 0, "back_paragraphs": 0}
    kept = paragraphs[front:len(paragraphs) - back] if back else paragraphs[front:]
    return "\n\n".join(kept), {"front_paragraphs": front, "back_paragraphs": back}
