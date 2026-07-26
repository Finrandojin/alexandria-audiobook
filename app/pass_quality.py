"""Validators for the three-pass generation flow (segment / attribute /
instruct). Segment reuses recall_core for source-fidelity scoring; attribute and
instruct enforce a hard per-entry text freeze (they may only add fields)."""

import re
import unicodedata

from recall_core import tokens, ngrams, counter_recall
from review_script import normalize_text
# Import the fidelity thresholds and the introduced-character regex from the
# single-pass gate so pass 1 tracks any recalibration there automatically
# (e.g. the 0.84->0.82 near-miss floor history) instead of drifting.
from chunk_quality import (MIN_SOURCE_TOKEN_RECALL, MIN_ORDERED_TRIGRAM_RECALL,
                           MIN_OUTPUT_SOURCE_RATIO, MAX_OUTPUT_SOURCE_RATIO,
                           _CYRILLIC_RE)
from script_preflight import find_adjacent_duplicate_blocks

_VALID_SEGMENT_TYPES = {"NARRATOR", "SPOKEN"}
_QUOTE_CHARS = {'"', '“', '”', '「', '」', '『', '』'}
_CURLY_AND_JAPANESE_OPEN = {'“', '「', '『'}
_CURLY_AND_JAPANESE_CLOSE = {'”', '」', '』'}


_REPORTING_VERB_TAIL = re.compile(
    r"\b(?:murmured|whispered|said|replied|answered|asked|cried|shouted),?\s+"
    r"([^\n]{1,120})$", re.IGNORECASE)
_SOURCE_LABEL_TAIL = re.compile(r"(?:^|\n\n)([^\n]{1,60})$")
_METADATA_PARAGRAPH = re.compile(
    r"^(?:Light Novel Adaptation found in|Original Web Novel Chapter|"
    r"Original Translation by)\b", re.IGNORECASE)


def _is_metadata_paragraph(current):
    paragraph = "".join(current).rsplit("\n\n", 1)[-1].strip()
    return bool(_METADATA_PARAGRAPH.match(paragraph))


def _pop_source_label(current):
    value = "".join(current)
    match = _SOURCE_LABEL_TAIL.search(value)
    if not match:
        return None
    candidate = match.group(1).strip().rstrip(":")
    words = candidate.split()
    if (not 1 <= len(words) <= 5 or
            not all(word == "???" or word[:1].isupper() for word in words) or
            (len(words) > 2 and candidate.isupper()) or
            any(char in candidate for char in ".!,;")):
        return None
    del current[match.start(1):]
    return candidate


def analyze_outer_quote_regions(text, initial_depth=0, allow_open_end=False):
    """Split source into narrator/spoken regions using outer quote boundaries.

    Curly quotes may be nested. Some source dialogue also uses an inner opening
    curly quote but shares the paragraph's only closing mark with the outer
    quote; in that case the inner mark is treated as decoration, not a new
    nesting level. Quote delimiters are intentionally omitted from speakable
    text.
    """
    regions, current, pending_source_label = [], [], None
    depth, saw_quote, repairs = initial_depth, bool(initial_depth), []

    def flush():
        nonlocal pending_source_label
        part = "".join(current).strip()
        if part:
            entry = {"type": "SPOKEN" if depth else "NARRATOR", "text": part}
            if depth and pending_source_label:
                entry["source_label"] = pending_source_label
                pending_source_label = None
            regions.append(entry)
        current.clear()

    for index, char in enumerate(text):
        if char == "\n" and text[index:index + 2] == "\n\n" and depth:
            flush()
            current.append(char)
            continue
        if char in _CURLY_AND_JAPANESE_OPEN:
            # Source credits sometimes quote a chapter title. It is metadata,
            # not unattributed dialogue, and must remain narrator material.
            if depth == 0 and _is_metadata_paragraph(current):
                saw_quote = True
                continue
            if depth == 0:
                pending_source_label = _pop_source_label(current)
                flush()
                depth = 1
                saw_quote = True
            elif not text[text.rfind("\n\n", 0, index) + 2:index].strip():
                # Publishers repeat an opening quote at the start of each
                # paragraph in one continuous, multi-paragraph speech.
                saw_quote = True
            elif char == '“':
                paragraph_tail = text[index + 1:].split("\n\n", 1)[0]
                if paragraph_tail.count('”') >= 2:
                    depth += 1
            else:
                depth += 1
            continue
        if char in _CURLY_AND_JAPANESE_CLOSE and depth:
            if char == '”' and depth == 1:
                paragraph_tail = text[index + 1:].split("\n\n", 1)[0]
                next_close = paragraph_tail.find('”')
                next_open = paragraph_tail.find('“')
                if next_close >= 0 and (next_open < 0 or next_close < next_open):
                    # Stylized cries may use one opening mark and decorative
                    # closers between fragments: “Ha!”Haha!”Hahaha!”
                    continue
            if depth == 1:
                flush()
            depth -= 1
            continue
        if char == '”':
            # Bounded recovery for a dropped opening delimiter in prose such as
            # "she murmured I see…”, keeping her eyes lowered". Only a short,
            # same-line phrase immediately following a reporting verb is safe
            # enough to infer; all other stray closers remain visible so the
            # quality gate fails closed.
            match = _REPORTING_VERB_TAIL.search("".join(current))
            if match:
                spoken = match.group(1).strip()
                del current[match.start(1):]
                flush()
                regions.append({"type": "SPOKEN", "text": spoken})
                repairs.append({"code": "inferred_missing_open_quote",
                                "offset": index, "text": spoken})
                saw_quote = True
                continue
            repairs.append({"code": "ignored_unmatched_close_quote",
                            "offset": index})
            saw_quote = True
            continue
        if char == '"':
            if depth and not text[text.rfind("\n\n", 0, index) + 2:index].strip():
                saw_quote = True
                continue
            flush()
            depth = 0 if depth else 1
            saw_quote = True
            continue
        current.append(char)
    if depth and not allow_open_end:
        flush()
        repairs.append({"code": "inferred_missing_close_quote",
                        "offset": len(text)})
        depth = 0
    else:
        flush()
    complete = depth == 0 or allow_open_end
    return {"regions": regions if saw_quote and complete else [],
            "repairs": repairs, "initial_depth": initial_depth,
            "final_depth": depth}


def split_outer_quote_regions(text):
    """Return only regions for callers that do not need repair telemetry."""
    return analyze_outer_quote_regions(text)["regions"]


def _split_quote_regions(source_text):
    """Return normalized text regions outside and inside outer dialogue quotes."""
    regions = split_outer_quote_regions(source_text)
    outside = [entry["text"] for entry in regions
               if entry["type"] == "NARRATOR"]
    inside = [entry["text"] for entry in regions
              if entry["type"] == "SPOKEN"]
    return outside, inside


def _region_contains_entry(regions, entry_text):
    needle = normalize_text(entry_text).split()
    if not needle:
        return True
    for region in regions:
        haystack = normalize_text(region).split()
        if len(needle) == 1:
            if haystack == needle:
                return True
        elif any(haystack[i:i + len(needle)] == needle
                 for i in range(len(haystack) - len(needle) + 1)):
            return True
    return False


def _quote_region_findings(source_text, entries, quote_analysis=None):
    if not quote_analysis and not any(char in source_text for char in _QUOTE_CHARS):
        return []
    if quote_analysis:
        source_regions = quote_analysis["regions"]
        outside = [entry["text"] for entry in source_regions
                   if entry["type"] == "NARRATOR"]
        inside = [entry["text"] for entry in source_regions
                  if entry["type"] == "SPOKEN"]
    else:
        outside, inside = _split_quote_regions(source_text)
    findings = []
    for number, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or "")
        if any(char in text for char in _QUOTE_CHARS):
            findings.append({"code": "mixed_quote_region", "entry_number": number,
                             "message": "An entry combines quote delimiters with narration or dialogue; split at the quote boundary."})
            continue
        expected = inside if entry.get("type") == "SPOKEN" else outside
        other = outside if entry.get("type") == "SPOKEN" else inside
        in_expected = _region_contains_entry(expected, text)
        if not in_expected and _region_contains_entry(other, text):
            findings.append({"code": "quote_region_misclassified", "entry_number": number,
                             "message": "SPOKEN text must originate inside quotes and NARRATOR text outside quotes."})
        elif not in_expected:
            findings.append({"code": "crosses_quote_boundary", "entry_number": number,
                             "message": "Entry text crosses a dialogue quote boundary; split narration and spoken text."})
    return findings


def _introduced_character_findings(source_text, output_text, entries):
    """Mirror the single-pass gate's introduced-character + adjacent-duplicate
    checks so pass 1 has the same fidelity bar (unsupported_cyrillic,
    unsupported_unicode_character, source_unsupported_duplicate). Text-only, so
    it is shape-agnostic across {type,text} and {speaker,text,instruct}."""
    findings = []
    source_cyrillic = set(_CYRILLIC_RE.findall(source_text))
    unsupported = sorted(set(_CYRILLIC_RE.findall(output_text)) - source_cyrillic)
    if unsupported:
        findings.append({"code": "unsupported_cyrillic", "characters": unsupported,
                         "message": "Response introduced Cyrillic characters absent from the source."})
    source_non_ascii = {char for char in unicodedata.normalize("NFC", source_text)
                        if ord(char) > 127 and unicodedata.category(char).startswith("L")}
    introduced = sorted({char for char in unicodedata.normalize("NFC", output_text)
                         if ord(char) > 127 and unicodedata.category(char).startswith("L")
                         and char not in source_non_ascii
                         and not ("Ѐ" <= char <= "ӿ")})
    if introduced:
        findings.append({"code": "unsupported_unicode_character",
                         "characters": [{"character": char, "codepoint": f"U+{ord(char):04X}",
                                         "name": unicodedata.name(char, "UNKNOWN")}
                                        for char in introduced],
                         "message": "Response introduced non-ASCII letters absent from the source."})
    entry_texts = [" ".join(str(e.get("text") or "").split()).casefold()
                   if isinstance(e, dict) else "" for e in entries]
    for duplicate in find_adjacent_duplicate_blocks(entry_texts, source_text):
        if duplicate.get("details", {}).get("source_occurrences") == 1:
            findings.append({"code": "source_unsupported_duplicate",
                             "entry_numbers": duplicate["entry_numbers"],
                             "message": "An adjacent repeated block occurs only once in the source."})
    return findings


def validate_segment_quality(source_text, entries, quote_analysis=None):
    """Fidelity gate for pass 1 output [{type, text}]. Same recall/trigram math
    as the single-pass gate, but validates the segment shape (type in
    {NARRATOR, SPOKEN}) rather than speaker/instruct."""
    findings = []
    if not isinstance(entries, list) or not entries:
        return _report(0, 0, 0.0, 0.0, 0.0,
                       [{"code": "missing_entries", "message": "Response contains no entries."}])
    output_parts = []
    for number, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            findings.append({"code": "invalid_entry", "entry_number": number,
                             "message": "Entry must be a JSON object."})
            continue
        missing = [k for k in ("type", "text") if k not in entry]
        if missing:
            findings.append({"code": "missing_fields", "entry_number": number,
                             "fields": missing, "message": "Entry is missing required fields."})
        if entry.get("type") not in _VALID_SEGMENT_TYPES:
            findings.append({"code": "invalid_type", "entry_number": number,
                             "value": entry.get("type"),
                             "message": "Each entry's type must be NARRATOR or SPOKEN."})
        text = str(entry.get("text") or "")
        if not text.strip():
            findings.append({"code": "empty_text", "entry_number": number,
                             "message": "Entry contains no speakable text."})
        source_label = str(entry.get("source_label") or "").strip()
        output_parts.append(f"{source_label} {text}".strip())

    source_tokens = tokens(source_text)
    output_text = " ".join(output_parts)
    output_tokens = tokens(output_text)
    sc, oc = len(source_tokens), len(output_tokens)
    recall = counter_recall(source_tokens, output_tokens)
    trigram = counter_recall(ngrams(source_tokens, 3), ngrams(output_tokens, 3))
    ratio = oc / sc if sc else (1.0 if not oc else 0.0)
    if sc and recall < MIN_SOURCE_TOKEN_RECALL:
        findings.append({"code": "low_source_token_recall", "value": round(recall, 4),
                         "minimum": MIN_SOURCE_TOKEN_RECALL,
                         "message": "Too much source text is absent from the response."})
    if sc >= 3 and trigram < MIN_ORDERED_TRIGRAM_RECALL:
        findings.append({"code": "low_ordered_trigram_recall", "value": round(trigram, 4),
                         "minimum": MIN_ORDERED_TRIGRAM_RECALL,
                         "message": "Too many ordered source phrases are absent."})
    if sc and not MIN_OUTPUT_SOURCE_RATIO <= ratio <= MAX_OUTPUT_SOURCE_RATIO:
        findings.append({"code": "output_source_ratio", "value": round(ratio, 4),
                         "minimum": MIN_OUTPUT_SOURCE_RATIO, "maximum": MAX_OUTPUT_SOURCE_RATIO,
                         "message": "Output length is implausible for the source chunk."})
    findings.extend(_quote_region_findings(source_text, entries, quote_analysis))
    findings.extend(_introduced_character_findings(source_text, output_text, entries))
    return _report(sc, oc, recall, trigram, ratio, findings)


def _report(source_count, output_count, recall, trigram, ratio, findings):
    return {
        "passed": not findings,
        "metrics": {
            "source_tokens": source_count, "output_tokens": output_count,
            "source_token_recall": round(recall, 4),
            "ordered_trigram_recall": round(trigram, 4),
            "output_source_ratio": round(ratio, 4),
        },
        "findings": findings,
    }


def _leading_tokens(text):
    return normalize_text(str(text or "")).split()


def index_head_check(frozen_entries, response_entries):
    """Return (ok, reason, ordered) for the immutable-index contract.

    Passes 2 and 3 provide a locally assigned integer ``n`` and accept exactly
    one response for every index. Text is never accepted back from the model.
    Older responses may still include ``head``; it is ignored because requiring
    the model to reproduce an anchor made otherwise-correct attribution fail on
    punctuation and repeated line openings."""
    k = len(frozen_entries)
    if not isinstance(response_entries, list) or len(response_entries) != k:
        got = len(response_entries) if isinstance(response_entries, list) else "non-list"
        return False, f"expected {k} entries, got {got}", None
    by_index = {}
    for item in response_entries:
        if not isinstance(item, dict):
            return False, "entry is not a JSON object", None
        n = item.get("n")
        if isinstance(n, float) and n.is_integer():
            n = int(n)
        if isinstance(n, bool) or not isinstance(n, int) or not (0 <= n < k):
            return False, f"entry has a missing or out-of-range index n={n!r}", None
        if n in by_index:
            return False, f"duplicate index n={n}", None
        by_index[n] = item
    ordered = [by_index[i] for i in range(k)]
    return True, "", ordered


# Entry-type tokens the model must never return as a speaker name. NARRATOR is
# absent on purpose: it is both a type and a legitimate speaker.
ENTRY_TYPE_NAMES = frozenset({"SPOKEN", "NARRATION", "DIALOGUE"})

MIN_NAME_ATTESTATIONS = 2
# Below this the source is a test fixture or a fragment, not a book, and a
# real name may legitimately appear once. Books in this corpus are 250k+.
MIN_SOURCE_FOR_ATTESTATION = 5000


def is_attested_name(name, source_text, min_attestations=MIN_NAME_ATTESTATIONS):
    """Whether a speaker name is written as a name in the source.

    A real character is capitalised nearly every time; an invented one is not
    in the text at all, or is a common word the model mistook for a name. The
    ratio rather than a flat "never lowercase" rule, so a character whose name
    is also a word (Rose, Hope) still passes on being named far more often than
    the word is used.

    The single implementation for both gates. Roster admission passes a higher
    min_attestations because a bad name there propagates to every later batch,
    while the output gate only has to judge one entry. They previously had
    separate implementations and drifted: the output gate was fixed for
    hyphenated names and the roster gate was not, so build_roster went on
    rejecting Bri-chan after validate_attribution had started accepting him.
    """
    if not source_text or not name:
        return True
    if len(source_text) < MIN_SOURCE_FOR_ATTESTATION:
        return True
    # Match case-insensitively and judge by the first letter, because str.title()
    # capitalises after every non-letter: "BRI-CHAN".title() is "Bri-Chan" but
    # the book writes "Bri-chan". That spelling mismatch rejected three real
    # grimgar03 characters - Bri-chan (55 mentions), Zodiac-kun, Barbara-sensei -
    # as inventions, which is every honorific-suffixed name in a translated work.
    occurrences = re.findall(r"\b" + re.escape(name) + r"\b", source_text,
                             re.IGNORECASE)
    capitalized = sum(1 for found in occurrences if found[:1].isupper())
    lowercase = len(occurrences) - capitalized
    return capitalized >= min_attestations and capitalized > lowercase * 2


def validate_attribution(frozen_entries, response_entries, source_text=None):
    """Pass 2 gate. Verifies the index+head alignment, then requires every SPOKEN
    span to have a non-empty speaker other than NARRATOR, and every NARRATOR span
    to stay NARRATOR."""
    ok, reason, ordered = index_head_check(frozen_entries, response_entries)
    if not ok:
        return {"passed": False,
                "findings": [{"code": "alignment_violated", "message": reason}]}
    findings = []
    for i, (frozen, item) in enumerate(zip(frozen_entries, ordered), 1):
        raw_speaker = item.get("speaker")
        speaker = raw_speaker.strip() if isinstance(raw_speaker, str) else ""
        if frozen.get("type") == "SPOKEN":
            # The prompt shows each entry as {"n", "type", "text"}, so the model
            # sometimes echoes the type back as the speaker. "SPOKEN" is
            # non-empty and is not NARRATOR, so it satisfied the checks below
            # and shipped as a character name - 326 entries in one book, and
            # present in 8 of 9 books measured. UNKNOWN is deliberately not
            # rejected here: it is the placeholder stabilize_speaker_identities
            # assigns for genuinely unresolved speakers.
            if (source_text and speaker
                    and speaker.upper() not in ("UNKNOWN", "NARRATOR")
                    and not is_attested_name(speaker, source_text)):
                # The roster gate filters what goes IN; nothing filtered what
                # came OUT. The model invented FUTURE_ME - the protagonist's
                # future self, a phrase the book never capitalises - and
                # shipped it on 250 entries, 13% of one book. With MAILMAN,
                # ARUMANFI and SWORD_GOD_GARU_FARION that was 279 entries,
                # 14.6% of the book, attributed to names not in the text.
                findings.append({"code": "speaker_not_in_source",
                                 "entry_number": i,
                                 "value": speaker,
                                 "message": "The speaker does not appear as a "
                                            "name in the source text."})
                continue
            if speaker.upper() in ENTRY_TYPE_NAMES:
                findings.append({"code": "speaker_is_entry_type",
                                 "entry_number": i,
                                 "value": speaker,
                                 "message": "The speaker repeats the entry type "
                                            "instead of naming a character."})
                continue
            if not speaker or speaker.upper() == "NARRATOR":
                findings.append({"code": "spoken_not_named", "entry_number": i,
                                 "message": "A spoken line was not assigned a character name."})
        else:  # NARRATOR (or any non-SPOKEN)
            if speaker.upper() != "NARRATOR":
                findings.append({"code": "narrator_renamed", "entry_number": i,
                                 "value": speaker,
                                 "message": "A narrator line must keep the speaker NARRATOR."})
    return {"passed": not findings, "findings": findings}


def validate_instruct(prior_entries, response_entries):
    """Pass 3 gate. Verifies the index+head alignment (speaker/text are supplied
    by the frozen entry, not the model, so they cannot change), and requires a
    non-empty instruct on every entry."""
    ok, reason, ordered = index_head_check(prior_entries, response_entries)
    if not ok:
        return {"passed": False,
                "findings": [{"code": "alignment_violated", "message": reason}]}
    findings = []
    for i, item in enumerate(ordered, 1):
        raw_instruct = item.get("instruct")
        if not isinstance(raw_instruct, str) or not raw_instruct.strip():
            findings.append({"code": "missing_instruct", "entry_number": i,
                             "message": "Every entry needs a non-empty instruct."})
    return {"passed": not findings, "findings": findings}
