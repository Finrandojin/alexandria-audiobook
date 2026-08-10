"""Read-only, deterministic checks for annotated audiobook scripts."""

import difflib
import re
import unicodedata
from collections import Counter

from speech_text import get_speech_normalization


_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_FRONT_MATTER_RE = re.compile(
    r"\b(?:copyright|all rights reserved|isbn(?:-1[03])?|table of contents)\b",
    re.IGNORECASE,
)
_NARRATION_RE = re.compile(
    r"\b(?:he|she|they|subaru|emilia)\s+(?:said|asked|replied|thought|looked|felt|was|were)\b",
    re.IGNORECASE,
)
_SCRIPT_PREFIXES = ("LATIN", "CYRILLIC", "HIRAGANA", "KATAKANA", "CJK",
                    "ARABIC", "HEBREW", "HANGUL", "GREEK", "THAI")


def _character_script(character):
    name = unicodedata.name(character, "")
    return next((prefix for prefix in _SCRIPT_PREFIXES if name.startswith(prefix)), None)


def audit_unicode_text(text, source_text=None):
    """Describe Unicode scripts and source-unsupported characters without mutation."""
    text = str(text or "")
    source = str(source_text or "")
    scripts = sorted({script for char in text if (script := _character_script(char))})
    source_scripts = {script for char in source if (script := _character_script(char))}
    introduced = (sorted(set(scripts) - source_scripts) if source_text is not None
                  else sorted(set(scripts) - {"LATIN"}))
    controls = sorted({f"U+{ord(char):04X}" for char in text
                       if unicodedata.category(char) in {"Cc", "Cs"}
                       and char not in "\n\r\t"})
    mixed = []
    for match in _WORD_RE.finditer(text):
        word_scripts = sorted({script for char in match.group() if (script := _character_script(char))})
        if "LATIN" in word_scripts and len(word_scripts) > 1:
            mixed.append({"text": match.group(), "scripts": word_scripts,
                          "offset": match.start()})
    return {"normalization": "NFC", "is_nfc": text == unicodedata.normalize("NFC", text),
            "scripts": scripts, "introduced_scripts": introduced,
            "replacement_character_count": text.count("\ufffd"),
            "unsafe_controls": controls, "mixed_script_words": mixed}


def is_possible_misattributed_narration(text, speaker):
    return (bool(speaker) and speaker.casefold() not in {"narrator", "narration"}
            and bool(_NARRATION_RE.search(text)))


def _normalize(value):
    return " ".join(str(value or "").split()).casefold()


def _normalize_words(value):
    return " ".join(_WORD_RE.findall(_normalize(value)))


def _finding(severity, code, message, entry_numbers=None, **details):
    result = {"severity": severity, "code": code, "message": message}
    if entry_numbers:
        result["entry_numbers"] = entry_numbers
    if details:
        result["details"] = details
    return result


# ONE definition of how much decode damage a source may carry, used by every
# path that generates a script. Before this existed the same question had three
# different answers: generate_script graded it, three_pass_generate refused any
# count, and the per-entry audit blocked at any count - so a repaired book
# generated single-pass and was refused three-pass, for the same input.
#
# 0.5% sits between index18 corrupt (1.40%) and index18 after deterministic
# repair (0.26%), so a mis-decoded file is still rejected and a repaired one
# runs. Raising it is a decision about accepting damaged prose, and belongs
# here where every caller sees the same number.
MAX_REPLACEMENT_SHARE = 0.005


def replacement_load_is_acceptable(count, length):
    """True when a text's replacement characters are within policy.

    Callers must not re-derive this. A gate that answers the same question a
    second time is a gate that will eventually answer it differently.
    """
    if not count:
        return True
    return (count / max(1, length)) <= MAX_REPLACEMENT_SHARE


def find_adjacent_duplicate_blocks(texts, source_text):
    findings = []
    occupied = set()
    source_normalized = _normalize_words(source_text)
    for block_size in range(5, 1, -1):
        index = 0
        while index + (2 * block_size) <= len(texts):
            left = texts[index:index + block_size]
            right = texts[index + block_size:index + (2 * block_size)]
            positions = set(range(index, index + (2 * block_size)))
            if (positions.isdisjoint(occupied) and left == right and
                    all(len(text) >= 8 for text in left)):
                block_text = _normalize_words(" ".join(left))
                source_occurrences = source_normalized.count(block_text) if source_normalized else None
                # A block the SOURCE itself repeats is faithful
                # transcription, not a defect. grimgar03 opens with its
                # title eight times; the whole-book gate rejected the
                # finished book for reproducing it, after all 49 chunks had
                # generated cleanly. script_repair.py already made this
                # distinction for the repair path - having it in only one of
                # the two places is how the same book failed twice for the
                # same reason (Rule 15: one decision, one implementation).
                faithful = source_occurrences >= 2
                findings.append(_finding(
                    "manual_review" if faithful else "blocking",
                    "adjacent_duplicate_block",
                    f"Entries {index + 1}-{index + block_size} are repeated "
                    + ("immediately, and the source repeats them too."
                       if faithful else "immediately."),
                    list(range(index + 1, index + (2 * block_size) + 1)),
                    block_size=block_size,
                    source_occurrences=source_occurrences,
                ))
                occupied.update(positions)
                index += 2 * block_size
            else:
                index += 1
    return findings


def find_adjacent_near_duplicate_entries(texts, source_text, minimum_ratio=0.90):
    """Adjacent entry pairs that are near-duplicates of each other but not
    supported by the source - likely model re-generation at a seam.

    Non-blocking evidence only: findings are always ``manual_review``, never
    a rejection - a fallback net for paraphrased seam repeats that slip past
    the exact ``find_adjacent_duplicate_blocks`` check. Deliberately NOT
    wired into ``chunk_quality.validate_chunk_quality`` - any finding there
    sets ``passed: False`` and triggers a retry, which would turn this
    non-blocking net into a blocking gate.
    """
    findings = []
    occupied = set()
    for finding in find_adjacent_duplicate_blocks(texts, source_text):
        occupied.update(number - 1 for number in finding["entry_numbers"])

    source_normalized = _normalize_words(source_text) if source_text else ""
    for index in range(len(texts) - 1):
        if index in occupied or (index + 1) in occupied:
            continue
        first, second = texts[index], texts[index + 1]
        if len(first) < 8 or len(second) < 8:
            continue
        first_words = _WORD_RE.findall(first)
        second_words = _WORD_RE.findall(second)
        if len(first_words) < 5 or len(second_words) < 5:
            continue
        ratio = difflib.SequenceMatcher(None, first_words, second_words, autojunk=False).ratio()
        if ratio < minimum_ratio:
            continue

        if source_text:
            # Identical text used twice needs two occurrences in the source to be
            # "genuinely repeated prose"; one occurrence can't back both uses.
            required_occurrences = 2 if first == second else 1
            first_supported = source_normalized.count(_normalize_words(first)) >= required_occurrences
            second_supported = source_normalized.count(_normalize_words(second)) >= required_occurrences
            if first_supported and second_supported:
                continue
            source_checked = True
            source_supported = [first_supported, second_supported]
        else:
            source_checked = False
            source_supported = None

        findings.append(_finding(
            "manual_review", "adjacent_near_duplicate",
            f"Entries {index + 1}-{index + 2} are near-duplicates of each other.",
            [index + 1, index + 2],
            similarity=round(ratio, 4),
            source_checked=source_checked,
            source_supported=source_supported,
        ))
    return findings


def audit_script(entries, source_text=None, is_generic_speaker_fn=None):
    """Return deterministic findings without modifying ``entries``."""
    findings = []
    if not isinstance(entries, list):
        findings.append(_finding("blocking", "invalid_script", "The script root must be a JSON array."))
        return _build_report(0, findings)

    texts = []
    instructions = []
    valid_entries = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            findings.append(_finding("blocking", "invalid_entry", "Entry must be a JSON object.", [index]))
            texts.append("")
            continue

        valid_entries.append((index, entry))
        text = str(entry.get("text") or "").strip()
        speaker = str(entry.get("speaker") or "").strip()
        instruct = str(entry.get("instruct") or "").strip()
        texts.append(_normalize(text))
        instructions.append(_normalize(instruct))

        if not text:
            findings.append(_finding("blocking", "empty_text", "Entry has no speakable text.", [index]))
        if not speaker:
            findings.append(_finding("blocking", "missing_speaker", "Entry has no speaker.", [index]))
        if not instruct:
            findings.append(_finding("manual_review", "missing_instruction", "Entry has no delivery instruction.", [index]))
        unicode_report = audit_unicode_text(text, source_text)
        if unicode_report["introduced_scripts"]:
            findings.append(_finding(
                "blocking", "introduced_unicode_script",
                "Entry contains a writing system absent from the source.",
                [index], scripts=unicode_report["introduced_scripts"],
            ))
        if unicode_report["mixed_script_words"]:
            findings.append(_finding("blocking", "mixed_script_word",
                                     "A word combines multiple writing systems.", [index],
                                     words=unicode_report["mixed_script_words"]))
        if unicode_report["unsafe_controls"]:
            findings.append(_finding("blocking", "unsafe_unicode_character",
                                     "Entry contains unsafe control characters.",
                                     [index], unicode=unicode_report))
        elif unicode_report["replacement_character_count"]:
            # Non-blocking: the SOURCE gate already decided whether this book's
            # damage is acceptable. Re-refusing here would mean a source we
            # admitted could never produce an entry we accept - which is how
            # index18 passed the front door and then failed at chunk 31.
            findings.append(_finding("manual_review", "replacement_character",
                                     "Entry contains replacement characters "
                                     "the source gate already accepted.",
                                     [index], unicode=unicode_report))
        if index <= 30 and _FRONT_MATTER_RE.search(text):
            findings.append(_finding("manual_review", "front_matter", "Possible publication front matter.", [index]))
        if is_possible_misattributed_narration(text, speaker):
            findings.append(_finding(
                "manual_review", "possible_misattributed_narration",
                "Third-person narration may be assigned to a character.", [index], speaker=speaker,
            ))
        if speaker and is_generic_speaker_fn and is_generic_speaker_fn(speaker):
            findings.append(_finding(
                "manual_review", "generic_speaker", "Generic speaker label needs book-local review.",
                [index], speaker=speaker,
            ))
        speech = get_speech_normalization(text)
        if speech["risk_categories"]:
            findings.append(_finding(
                "manual_review", "nonprose_speech_risk",
                "Non-prose content has elevated TTS error risk; review the "
                "spoken preview and validate the generated audio.",
                [index], categories=speech["risk_categories"],
                normalized_preview=speech["text"],
                transformations=speech["transformations"],
                validation={
                    "recommended": True,
                    "method": "targeted_transcription_or_human_listening",
                    "status": "not_run",
                    "reason": "Script preflight does not generate or transcribe audio.",
                },
            ))

    findings.extend(find_adjacent_duplicate_blocks(texts, source_text))
    findings.extend(find_adjacent_near_duplicate_entries(texts, source_text))

    nonempty_instructions = [value for value in instructions if value]
    if len(nonempty_instructions) >= 20:
        uniqueness = len(set(nonempty_instructions)) / len(nonempty_instructions)
        if uniqueness >= 0.95:
            findings.append(_finding(
                "informational", "high_instruction_uniqueness",
                "Nearly every entry has a unique delivery instruction.",
                uniqueness=round(uniqueness, 4),
            ))

    if source_text:
        source_words = Counter(_WORD_RE.findall(_normalize(source_text)))
        script_words = Counter(word for text in texts for word in _WORD_RE.findall(text))
        source_total = sum(source_words.values())
        matched = sum(min(count, script_words.get(word, 0)) for word, count in source_words.items())
        coverage = matched / source_total if source_total else 1.0
        if source_total and coverage < 0.85:
            findings.append(_finding(
                "manual_review", "low_source_word_coverage",
                "Annotated text has low word coverage relative to the source.",
                coverage=round(coverage, 4),
            ))

    return _build_report(len(entries), findings)


def _build_report(entry_count, findings):
    counts = {severity: 0 for severity in ("blocking", "manual_review", "informational")}
    for finding in findings:
        counts[finding["severity"]] += 1
    return {
        "entry_count": entry_count,
        "counts": counts,
        "can_apply_repairs": counts["blocking"] == 0,
        "findings": findings,
    }
