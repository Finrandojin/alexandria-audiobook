"""Deterministic, source-backed repairs for annotated scripts."""

import copy
import re

from script_preflight import find_adjacent_duplicate_blocks, _normalize
from source_normalization import KNOWN_SOURCE_CORRUPTIONS


_WORD_WITH_CYRILLIC_RE = re.compile(r"[^\W\d_]*[\u0400-\u04ff][^\W\d_]*", re.UNICODE)
_CYRILLIC_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x",
    "у": "y", "к": "k", "м": "m", "т": "t", "в": "b", "г": "r",
    "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "Х": "X",
    "У": "Y", "К": "K", "М": "M", "Т": "T", "В": "B", "Г": "R",
})
EXPLICIT_SILENCE_MS = 1000


def build_deterministic_repair(entries, source_text, merge_empty_into_pause=True):
    """Return repaired copies and evidence; never mutate the supplied entries.

    merge_empty_into_pause (default True, single-pass behavior): convert an empty
    entry into a pause_after on its previous entry and drop it. The three-pass
    segment stage passes False: an empty {type,text} unit there should be surfaced
    by the segment gate's empty_text finding, not silently merged into a
    possibly different-typed neighbor before the gate ever sees it (finding #7)."""
    repaired = copy.deepcopy(entries)
    changes = []
    unresolved = []
    source_words = set(re.findall(r"\w+", _normalize(source_text), re.UNICODE))

    for index, entry in enumerate(repaired):
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
            continue
        original = entry["text"]
        replacements = []
        for match in _WORD_WITH_CYRILLIC_RE.finditer(original):
            old_word = match.group(0)
            new_word = KNOWN_SOURCE_CORRUPTIONS.get(old_word.casefold())
            if new_word and old_word[:1].isupper():
                new_word = new_word.capitalize()
            if not new_word:
                new_word = old_word.translate(_CYRILLIC_HOMOGLYPHS)
            if any("\u0400" <= char <= "\u04ff" for char in new_word):
                unresolved.append({"entry_number": index + 1, "text": old_word,
                                   "reason": "unsupported_cyrillic_character"})
            elif (new_word.casefold() not in source_words and
                  old_word.casefold() not in source_words):
                unresolved.append({"entry_number": index + 1, "text": old_word,
                                   "candidate": new_word, "reason": "candidate_not_in_source"})
            else:
                replacements.append((match.start(), match.end(), old_word, new_word))
        if replacements:
            updated = original
            for start, end, _old, new in reversed(replacements):
                updated = updated[:start] + new + updated[end:]
            entry["text"] = updated
            changes.append({
                "type": "unicode_homoglyph", "entry_number": index + 1,
                "before": original, "after": updated,
            })

    texts = [_normalize(entry.get("text")) if isinstance(entry, dict) else "" for entry in repaired]
    duplicate_findings = find_adjacent_duplicate_blocks(texts, source_text)
    removals = []
    for finding in duplicate_findings:
        details = finding["details"]
        occurrences = details.get("source_occurrences")
        # Three cases, and the old `!= 1` test collapsed two opposite ones.
        #
        #   0  the block is nowhere in the source: the model invented the
        #      repetition, and nothing here can decide which copy to keep.
        #      Genuinely unresolved.
        #   1  the block appears once but was emitted twice: the model
        #      duplicated it, so deleting the second copy restores the source.
        #  >=2 the SOURCE ITSELF repeats the block, so emitting it twice is
        #      faithful transcription. Deleting a copy would corrupt the text
        #      the model was asked to reproduce exactly.
        #
        # grimgar03 opens with its title 8 times - the source carries it 49
        # times in all - and every one of 17 attempts was rejected for
        # correctly reproducing that. The book could not be generated at all.
        if occurrences is None or occurrences == 0:
            unresolved.append({"entry_numbers": finding["entry_numbers"],
                               "reason": "duplicate_not_in_source"})
            continue
        if occurrences >= 2:
            changes.append({
                "type": "adjacent_duplicate_block_kept",
                "entry_numbers": finding["entry_numbers"],
                "source_occurrences": occurrences,
                "note": "source repeats this block; emitting it twice is "
                        "faithful, so nothing is removed",
            })
            continue
        block_size = details["block_size"]
        first = finding["entry_numbers"][0]
        second_start = first + block_size
        removals.extend(range(second_start - 1, second_start - 1 + block_size))
        changes.append({
            "type": "adjacent_duplicate_block",
            "kept_entry_numbers": list(range(first, first + block_size)),
            "removed_entry_numbers": list(range(second_start, second_start + block_size)),
            "source_occurrences": 1,
        })
    empty_indexes = ([index for index, entry in enumerate(repaired)
                      if isinstance(entry, dict) and not str(entry.get("text") or "").strip()]
                     if merge_empty_into_pause else [])
    for index in empty_indexes:
        if index == 0:
            unresolved.append({"entry_number": 1, "reason": "empty_first_entry"})
            continue
        previous = repaired[index - 1]
        if not isinstance(previous, dict):
            unresolved.append({"entry_number": index + 1, "reason": "invalid_previous_entry"})
            continue
        if not str(previous.get("text") or "").strip():
            unresolved.append({"entry_number": index + 1, "reason": "previous_entry_not_spoken"})
            continue
        if previous.get("pause_after") is not None:
            unresolved.append({"entry_number": index + 1, "reason": "previous_pause_already_set"})
            continue
        previous["pause_after"] = EXPLICIT_SILENCE_MS
        changes.append({
            "type": "empty_entry_to_pause", "removed_entry_number": index + 1,
            "pause_after_entry_number": index, "pause_ms": EXPLICIT_SILENCE_MS,
        })
        removals.append(index)
    for index in sorted(set(removals), reverse=True):
        del repaired[index]

    return {"entries": repaired, "changes": changes, "unresolved": unresolved}
