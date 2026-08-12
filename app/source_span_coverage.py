"""Deterministic source-span tags and coverage checks for generation experiments."""

import re


_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n\s*\n+")


def get_source_spans(source_text):
    """Return ordered, non-empty sentence/paragraph spans without changing text."""
    spans = []
    start = 0
    for match in _BOUNDARY_RE.finditer(source_text):
        end = match.start()
        text = source_text[start:end].strip()
        if text:
            spans.append({"id": f"S{len(spans) + 1:03d}", "text": text})
        start = match.end()
    tail = source_text[start:].strip()
    if tail:
        spans.append({"id": f"S{len(spans) + 1:03d}", "text": tail})
    return spans


def format_tagged_source(spans):
    """Render spans for a prompt while keeping the original span records pure."""
    return "\n".join(f'[{span["id"]}] {span["text"]}' for span in spans)


def get_span_coverage_findings(spans, entries):
    """Report missing, unknown, and malformed source-span declarations."""
    expected = {span["id"] for span in spans}
    declared = set()
    findings = []
    for number, entry in enumerate(entries or [], 1):
        ids = entry.get("source_span_ids") if isinstance(entry, dict) else None
        if not isinstance(ids, list) or not ids or any(not isinstance(item, str) for item in ids):
            findings.append({"code": "invalid_source_span_ids", "entry_number": number})
            continue
        declared.update(ids)
    unknown = sorted(declared - expected)
    missing = sorted(expected - declared)
    if unknown:
        findings.append({"code": "unknown_source_span_ids", "span_ids": unknown})
    if missing:
        findings.append({"code": "uncovered_source_spans", "span_ids": missing})
    return findings
