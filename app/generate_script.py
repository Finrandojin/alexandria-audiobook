import argparse
import hashlib
import os
import sys
import json
import re
import time
import math
from dataclasses import dataclass
from openai import OpenAI
from config_settings import load_app_config
from chunk_quality import validate_chunk_quality, is_trigram_only_near_miss
from default_prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT
from lmstudio_settings import (ensure_ideal_settings, get_effective_max_tokens,
                               get_next_retry_max_tokens)
from repair_source_encoding import preflight_source
from script_repair import build_deterministic_repair
from source_normalization import (normalize_extreme_phrase_repetitions,
                                  normalize_homoglyph_words,
                                  normalize_known_source_corruptions,
                                  strip_known_front_matter,
                                  strip_publisher_matter)
from speaker_identity import (build_speaker_consistency_report,
                              stabilize_speaker_identities)
from script_preflight import (MAX_REPLACEMENT_SHARE, audit_script,
                              audit_unicode_text,
                              replacement_load_is_acceptable,
                              replacement_repair_hint)
from utils import (atomic_json_write, extract_balanced, get_runtime_data_dir,
                   get_app_config_path, is_generic_speaker, safe_load_json)


def get_generation_checkpoint_path(output_path):
    return output_path + ".generation_checkpoint.json"


def get_generation_quality_path(output_path):
    return output_path + ".generation_quality.json"


def get_response_log_path(log_name):
    """Return an isolated response log when a durable run id is available."""
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    run_id = os.environ.get("ALEXANDRIA_RUN_ID")
    if not run_id:
        return os.path.join(log_dir, log_name)
    stem, extension = os.path.splitext(log_name)
    response_dir = os.path.join(log_dir, "responses", run_id)
    os.makedirs(response_dir, exist_ok=True)
    return os.path.join(response_dir, stem + (extension or ".log"))


def build_generation_quality_manifest(status, fingerprint, accepted_chunks,
                                      source_normalizations, **details):
    return {
        "status": status,
        "fingerprint": fingerprint,
        "source_normalizations": source_normalizations,
        "accepted_chunk_count": len(accepted_chunks),
        # Persist how many chunks were salvaged by the exhaustion-only
        # trigram-only near-miss path (see chunk_quality.is_trigram_only_near_miss)
        # so completed-book rescue counts survive in the manifest instead of only
        # in the per-run checkpoint (cleared on success) and the stdout log.
        "near_miss_accepted_count": sum(
            1 for item in accepted_chunks if item.get("near_miss_accepted")),
        "chunks": [{
            "chunk_number": item["chunk_number"],
            "source_sha256": item["source_sha256"],
            "entry_count": len(item["entries"]),
            "quality": item["quality"],
            "adaptively_split": item.get("adaptively_split", False),
            "near_miss_accepted": item.get("near_miss_accepted", False),
            "attempts": item.get("attempts", []),
        } for item in accepted_chunks],
        **details,
    }


def save_generation_quality_manifest(output_path, manifest):
    atomic_json_write(manifest, get_generation_quality_path(output_path))


def passes_final_generation_gate(whole_quality, preflight, unresolved_repairs=None):
    return bool(whole_quality.get("passed")
                and not preflight.get("counts", {}).get("blocking")
                and not unresolved_repairs)


def build_final_generation_repair(entries, source_text):
    """Apply the same source-backed repair after chunks are reassembled.

    Chunk-local repair cannot see a duplicate block spanning a chunk boundary.
    Running it once more over the assembled book removes only blocks proven to
    occur once in the source and preserves unresolved findings for a loud final
    gate failure.
    """
    structural = build_deterministic_repair(entries, source_text)
    if structural["unresolved"]:
        return structural
    identities = stabilize_speaker_identities(structural["entries"])
    return {"entries": identities["entries"],
            "changes": structural["changes"] + identities["changes"],
            "unresolved": [], "review": identities["review"]}


def get_generation_fingerprint(source_text, chunks, model_name, base_url, params, chunk_size):
    settings = {
        "model_name": model_name, "base_url": base_url, "chunk_size": chunk_size,
        "system_prompt": params.system_prompt, "user_prompt_template": params.user_prompt_template,
        "max_tokens": params.max_tokens, "temperature": params.temperature,
        "top_p": params.top_p, "top_k": params.top_k, "min_p": params.min_p,
        "presence_penalty": params.presence_penalty, "banned_tokens": params.banned_tokens,
        "context_length": params.context_length, "hard_max_tokens": params.hard_max_tokens,
    }
    encoded = json.dumps(settings, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {
        "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "settings_sha256": hashlib.sha256(encoded).hexdigest(),
        "chunk_sha256": [hashlib.sha256(chunk.encode("utf-8")).hexdigest() for chunk in chunks],
    }


def get_valid_chunk_size(config_value, cli_value=None):
    """Return the effective single-pass chunk size after validating both inputs."""
    value = cli_value if cli_value is not None else config_value
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        source = "--chunk-size" if cli_value is not None else "generation.chunk_size"
        raise ValueError(f"{source} must be an integer >= 1 (got {value!r})")
    return value


def load_generation_checkpoint(output_path, fingerprint):
    checkpoint = safe_load_json(get_generation_checkpoint_path(output_path), None)
    if not isinstance(checkpoint, dict) or checkpoint.get("fingerprint") != fingerprint:
        return []
    accepted = checkpoint.get("accepted_chunks")
    if not isinstance(accepted, list) or len(accepted) > len(fingerprint["chunk_sha256"]):
        return []
    for index, item in enumerate(accepted):
        if (not isinstance(item, dict) or not isinstance(item.get("entries"), list)
                or not item.get("quality", {}).get("passed")
                or item.get("source_sha256") != fingerprint["chunk_sha256"][index]):
            return []
    return accepted


def save_generation_checkpoint(output_path, fingerprint, accepted_chunks):
    atomic_json_write({"fingerprint": fingerprint, "accepted_chunks": accepted_chunks},
                      get_generation_checkpoint_path(output_path))


def clear_generation_checkpoint(output_path):
    path = get_generation_checkpoint_path(output_path)
    if os.path.exists(path):
        os.remove(path)


class AdjacentArrayOverlapError(ValueError):
    """Adjacent LLM arrays repeat source words at their boundary."""


def _get_boundary_overlap(left_entries, right_entries, minimum_words=3):
    """Return the longest normalized suffix/prefix overlap between two arrays."""
    if (not left_entries or not right_entries
            or not isinstance(left_entries[-1], dict)
            or not isinstance(right_entries[0], dict)):
        return []
    left_text = str(left_entries[-1].get("text") or "")
    right_text = str(right_entries[0].get("text") or "")
    left_words = re.findall(r"\w+", left_text.casefold(), re.UNICODE)
    right_words = re.findall(r"\w+", right_text.casefold(), re.UNICODE)
    maximum = min(len(left_words), len(right_words))
    for size in range(maximum, minimum_words - 1, -1):
        if left_words[-size:] == right_words[:size]:
            return left_words[-size:]
    return []


def clean_json_string(text):
    """Clean and combine adjacent complete JSON arrays from an LLM response."""
    def parse_array_span(value):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = repair_json_array(value)
        return parsed if isinstance(parsed, list) else None

    # Remove thinking tags (various formats used by different models)
    # GLM, DeepSeek, Qwen, etc. use different thinking tag formats
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    text = re.sub(r'<thinking>[\s\S]*?</thinking>', '', text)
    text = re.sub(r'<reflection>[\s\S]*?</reflection>', '', text)
    text = re.sub(r'<reasoning>[\s\S]*?</reasoning>', '', text)
    # Handle unclosed thinking tags (model started thinking but didn't close)
    text = re.sub(r'<think>[\s\S]*$', '', text)
    text = re.sub(r'<thinking>[\s\S]*$', '', text)

    # Remove markdown code blocks
    if "```" in text:
        # Find content between ```json and ``` or just ``` and ```
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            text = match.group(1).strip()

    # Find the JSON array - match from first [ to its closing ]
    start = text.find('[')
    if start == -1:
        return None

    span = extract_balanced(text, '[', ']')
    if span is None:
        # No closing bracket found, try to salvage
        last_complete = text.rfind('},')
        if last_complete > start:
            return text[start:last_complete+1] + ']'
        return None

    spans = [span]
    cursor = start + len(span)
    while True:
        next_start = text.find('[', cursor)
        if next_start == -1:
            break
        next_span = extract_balanced(text, '[', ']', next_start)
        if next_span is None:
            if text[cursor:next_start].strip():
                break
            return None
        next_value = parse_array_span(next_span)
        if next_value is None:
            if re.match(r'^\[\s*\{', next_span):
                return None
            break
        if text[cursor:next_start].strip():
            # A later valid array behind prose is ambiguous: do not silently
            # discard either the intervening material or the array.
            return None
        spans.append(next_span)
        cursor = next_start + len(next_span)

    if len(spans) > 1:
        parsed_spans = [parse_array_span(item) for item in spans]
        if any(item is None for item in parsed_spans):
            parsed_spans = []
        for left_entries, right_entries in zip(parsed_spans, parsed_spans[1:]):
            overlap = _get_boundary_overlap(left_entries, right_entries)
            if overlap:
                raise AdjacentArrayOverlapError(" ".join(overlap))

    json_text = spans[0] if len(spans) == 1 else "[" + ",".join(
        item.strip()[1:-1] for item in spans) + "]"

    # Clean control characters inside strings (common LLM issue)
    # Replace literal newlines/tabs inside JSON strings with escaped versions
    def fix_control_chars(match):
        s = match.group(0)
        # Replace unescaped control characters
        s = s.replace('\n', '\\n')
        s = s.replace('\r', '\\r')
        s = s.replace('\t', '\\t')
        return s

    # Fix control characters inside string values
    json_text = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_control_chars, json_text)

    return json_text


def _extract_valid_json_objects(json_text):
    """Extract complete JSON objects without assuming a pass-specific schema."""
    entries = []
    cursor = 0
    while True:
        start = json_text.find("{", cursor)
        if start < 0:
            break
        span = extract_balanced(json_text, "{", "}", start)
        if span is None:
            cursor = start + 1
            continue
        try:
            value = json.loads(span)
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(value, dict):
            entries.append(value)
        cursor = start + len(span)
    return entries


def repair_json_array(json_text):
    """Attempt to repair common JSON array issues from LLM output."""
    if not json_text:
        return None

    def _filter_entries(lst):
        """Keep only dict entries; LLMs sometimes emit bare strings in the array."""
        filtered = [e for e in lst if isinstance(e, dict)]
        if len(filtered) < len(lst):
            print(f"  Warning: Dropped {len(lst) - len(filtered)} non-object entries from LLM JSON array")
        return filtered if filtered else None

    # Try parsing as-is first
    try:
        result = json.loads(json_text)
        if isinstance(result, list):
            return _filter_entries(result)
    except json.JSONDecodeError:
        pass

    # Fix 1: Add missing commas between objects (}\s*{" -> },\n{")
    fixed = re.sub(r'\}\s*\{', '},\n{', json_text)
    try:
        result = json.loads(fixed)
        if isinstance(result, list):
            return _filter_entries(result)
    except json.JSONDecodeError:
        pass

    # Fix 2: Remove trailing commas before ]
    fixed = re.sub(r',\s*\]', ']', fixed)
    try:
        result = json.loads(fixed)
        if isinstance(result, list):
            return _filter_entries(result)
    except json.JSONDecodeError:
        pass

    # Fix 3: Try to extract individual entries and rebuild. This is deliberately
    # schema-neutral: single-pass returns speaker/text/instruct, pass 2 returns
    # n/head/speaker, and pass 3 returns n/head/instruct.
    entries = _extract_valid_json_objects(json_text)

    if entries:
        return entries

    # Fix 4: Last resort - find last complete entry and truncate
    last_complete = json_text.rfind('},')
    if last_complete > 0:
        try:
            truncated = json_text[:last_complete+1] + ']'
            # Ensure it starts with [
            if not truncated.strip().startswith('['):
                truncated = '[' + truncated
            result = json.loads(truncated)
            if isinstance(result, list):
                return _filter_entries(result)
        except json.JSONDecodeError:
            pass

    return None

def salvage_json_entries(json_text):
    """Last resort: extract individual valid objects for any generation pass."""
    entries = _extract_valid_json_objects(json_text)
    return entries if entries else None


def fix_mojibake(text):
    """Fix common mojibake characters resulting from CP1252-as-UTF8."""
    replacements = [
        ('â€™', '\u2019'),  # Right single quote
        ('â€˜', '\u2018'),  # Left single quote
        ('â€œ', '\u201c'),  # Left double quote
        ('â€\x9d', '\u201d'),  # Right double quote
        ('â€?', '\u201d'),  # Sometimes ? if undefined
        ('â€“', '\u2013'),  # En dash (UTF-8 E2 80 93 read as CP1252)
        ('â€”', '\u2014'),  # Em dash (UTF-8 E2 80 94 read as CP1252)
        ('â€¦', '\u2026'),  # Ellipsis
    ]

    for bad, good in replacements:
        text = text.replace(bad, good)

    return text


def get_preprocessed_source(text, strip_front_matter=True):
    """Return source text and cleanup reports exactly as generation uses them."""
    text = fix_mojibake(text)
    text, source_normalizations = normalize_known_source_corruptions(text)
    text, homoglyph_normalizations = normalize_homoglyph_words(text)
    source_normalizations.extend(homoglyph_normalizations)
    text, repetition_normalizations = normalize_extreme_phrase_repetitions(text)
    source_normalizations.extend(repetition_normalizations)
    front_matter_removed = None
    if strip_front_matter:
        text, front_matter_removed = strip_known_front_matter(text)
    text, publisher_matter = strip_publisher_matter(text)
    return text, {
        "source_normalizations": source_normalizations,
        "front_matter_removed": front_matter_removed,
        "publisher_matter": publisher_matter,
    }

def split_into_chunk_records(text, max_size=3000):
    """Split text with explicit oversized-paragraph continuation metadata."""
    paragraphs = re.split(r'\n\s*\n', text)
    records = []
    current_chunk = ""

    def emit(text, continues_from=False, continues_to=False):
        if text.strip():
            records.append({"text": text.strip(),
                            "continues_paragraph_from_previous": continues_from,
                            "continues_paragraph_to_next": continues_to})

    def split_long_piece(piece):
        pieces = []
        remaining = piece.strip()
        while len(remaining) > max_size:
            cut = remaining.rfind(" ", 0, max_size + 1)
            if cut <= 0:
                cut = max_size
            pieces.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        if remaining:
            pieces.append(remaining)
        return pieces

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 > max_size:
            if current_chunk:
                emit(current_chunk)
                current_chunk = ""

            if len(para) > max_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                paragraph_pieces = []
                for sentence in sentences:
                    for piece in split_long_piece(sentence):
                        if (paragraph_pieces and
                                len(paragraph_pieces[-1]) + len(piece) + 1 <= max_size):
                            paragraph_pieces[-1] += " " + piece
                        else:
                            paragraph_pieces.append(piece)
                for index, piece in enumerate(paragraph_pieces):
                    emit(piece, continues_from=index > 0,
                         continues_to=index + 1 < len(paragraph_pieces))
            else:
                current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para

    if current_chunk:
        emit(current_chunk)

    return records


def split_into_chunks(text, max_size=3000):
    """Split text into strings at paragraph/sentence boundaries."""
    return [record["text"] for record in split_into_chunk_records(text, max_size)]

@dataclass
class LLMGenParams:
    """LLM prompt + sampling settings shared by script generation and review.

    Groups the knobs previously threaded as ~9 separate arguments through
    process_chunk()/review_batch() and their callers. `system_prompt` and
    `user_prompt_template` are optional overrides; each caller falls back to its
    own module default when they are None.

    NOTE: the sampling defaults below are a script-generation baseline only. The
    review pass uses different tuned values (higher max_tokens, lower temperature,
    top_k=20) and always constructs this explicitly, so don't rely on these
    defaults for review — pass review's values in.
    """
    system_prompt: str = None
    user_prompt_template: str = None
    max_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.8
    top_k: int = None
    min_p: float = None
    presence_penalty: float = 0.0
    banned_tokens: list = None
    context_length: int = None
    hard_max_tokens: int = 16384
    segment_temperature: float = None
    attribute_temperature: float = None
    instruct_temperature: float = None
    segment_output_ratio: float = 3.0
    presegment_quotes: bool = False
    reasoning_allowance: int = 0
    reasoning_effort: str = None
    # LM Studio honours seed: same seed twice is identical, a different seed
    # differs, no seed varies. That is what lets sampled voting stay
    # reproducible instead of trading determinism for accuracy.
    seed: int = None
    # Kept at the end so legacy positional constructors retain their meaning.
    attribute_system_prompt: str = None


def _rotate_log_if_large(log_path, max_bytes=10 * 1024 * 1024):
    """Rotate <log_path> to <log_path>.bak once it exceeds max_bytes (best-effort)."""
    if os.path.exists(log_path) and os.path.getsize(log_path) > max_bytes:
        backup_path = log_path + ".bak"
        try:
            if os.path.exists(backup_path):
                os.remove(backup_path)
            os.rename(log_path, backup_path)
        except OSError:
            pass  # If rotation fails, just append to existing log


def classify_length_finish(content, reasoning_tokens, already_escalated):
    """Decide what a finish_reason=length response means.

    Returns "reasoning_overflow" when a reasoning model spent its whole budget
    thinking and emitted no visible content, and has already been given one
    larger budget. Returns "escalate_once" for the first such response, and
    "truncated_output" for ordinary visible truncation, which keeps its existing
    retry and subdivision handling. Rule 10: one policy, applied the same way on
    every attempt.
    """
    if reasoning_tokens and not (content or "").strip():
        return "reasoning_overflow" if already_escalated else "escalate_once"
    return "truncated_output"


def build_extra_body(params):
    """Collect non-standard sampling options for the OpenAI-compatible call."""
    return {k: v for k, v in {
        "top_k": params.top_k,
        "min_p": params.min_p,
        "banned_tokens": params.banned_tokens if params.banned_tokens else None,
        "reasoning_effort": params.reasoning_effort,
        "seed": params.seed,
    }.items() if v is not None}


def get_quality_retry_policy(finish_reason, completion_tokens, effective_max,
                             quality):
    """Return one consistent retry action for a deterministic quality failure."""
    codes = {finding["code"] for finding in quality.get("findings", [])}
    incomplete = bool({"low_source_token_recall", "low_ordered_trigram_recall",
                       "output_source_ratio"} & codes
                      and quality.get("metrics", {}).get("output_source_ratio", 1.0) < 0.9)
    near_limit = (completion_tokens is not None and effective_max
                  and completion_tokens >= effective_max * 0.9)
    if finish_reason == "length" or (incomplete and near_limit):
        return "increase_tokens"
    return "retry_same_budget"


def is_severe_chunk_truncation(quality):
    """Return whether quality evidence matches the cheap early-stop failure."""
    metrics = quality.get("metrics", {})
    codes = {finding.get("code") for finding in quality.get("findings", [])}
    return (metrics.get("output_source_ratio", 1.0) < 0.6
            and "low_source_token_recall" in codes
            and "low_ordered_trigram_recall" in codes)


def get_chunk_retry_action(quality, consecutive_severe, allow_early_split=False):
    """Choose whether a failed generation attempt should retry or split."""
    if (allow_early_split and is_severe_chunk_truncation(quality)
            and consecutive_severe >= 2):
        return "split"
    return "retry"


def record_attempt_context(observer, attempt, phase, split_part=None):
    """Label one live attempt record before handing it to its observer."""
    attempt["phase"] = phase
    if split_part is not None:
        attempt["split_part"] = split_part
    observer(attempt)


NEAR_MISS_RECALL_THRESHOLD = 0.75


def _is_near_miss_recall(metrics):
    """True if source coverage was close to (but under) chunk_quality.py's
    0.90 pass threshold - distinguishes an almost-complete, imprecise
    response from catastrophic early-stop truncation. Shared by
    _build_retry_feedback_message (wording) and process_chunk (bonus-retry
    eligibility) so both use the same definition of "near miss" (Rule 15).

    Diagnosed live (2026-07-19): a chunk's retry sequence hit 11%, 11%,
    11%, then 86% recall on attempt 4 (its last) - a near-complete
    response one attempt away from passing - then regressed to 5% because
    the retry message told it "you barely started, redo everything" when
    it hadn't.
    """
    if not metrics:
        return False
    recall = metrics.get("source_token_recall")
    return recall is not None and recall >= NEAR_MISS_RECALL_THRESHOLD


def _describe_missing_spans(spans):
    """Render chunk_quality.py's missing_source_spans as a plain-English
    clause quoting where the dropped source content begins, so the retry
    message points at specific text instead of only a coverage percentage.
    Returns "" when there are no spans (message unchanged from before)."""
    clauses = [f'the passage beginning "{span["preview"]} ..." '
               f'(about {span["token_count"]} words)'
               for span in spans or [] if span.get("preview")]
    if not clauses:
        return ""
    return " The missing content includes " + "; ".join(clauses) + "."


def _build_retry_feedback_message(quality):
    """Plain-English retry instruction instead of a raw JSON findings dump -
    the model has to act on this, not parse it. Diagnosed from a real
    production incident (2026-07-19): fresh single attempts against a known
    failing chunk succeeded 3/3 live, but retries within a rejected session
    kept producing near-identical tiny output across several tries - the
    raw `json.dumps(quality["findings"])` dump this replaced gave the model
    a list of validation codes to parse, not something to act on.

    Falls back to each finding's own human-readable `message`
    (chunk_quality.py already attaches one to every finding) for finding
    types outside the dominant incomplete-output cluster, and to the
    original JSON dump only if no messages exist at all (never worse than
    before this change).
    """
    findings = quality.get("findings", [])
    truncation_codes = {"low_source_token_recall", "low_ordered_trigram_recall",
                        "output_source_ratio"}
    codes = {finding.get("code") for finding in findings}
    if codes & truncation_codes:
        metrics = quality.get("metrics", {})
        recall = metrics.get("source_token_recall")
        coverage = f"about {recall * 100:.0f}%" if recall is not None else "too little"
        spans_hint = _describe_missing_spans(quality.get("missing_source_spans"))
        if _is_near_miss_recall(metrics):
            return (f"Your previous response was close but only covered {coverage} "
                    "of the source text, missing some content or phrasing precisely. "
                    "Review it and produce a complete, precise conversion of the "
                    "entire chunk, filling in whatever was missing." + spans_hint)
        return (f"Your previous response covered only {coverage} of the source "
                "text before stopping early. Convert the ENTIRE source chunk "
                "from beginning to end this time - do not stop partway through "
                "and do not summarize any part of it." + spans_hint)
    messages = [finding["message"] for finding in findings if finding.get("message")]
    if messages:
        return " ".join(messages)
    return json.dumps(findings, ensure_ascii=False)


def call_llm_for_entries(client, model_name, sys_prompt, user_prompt, params,
                         log_name, label, max_retries=2, validate_entries=None,
                         transform_entries=None, attempt_observer=None,
                         retry_decider=None, near_miss_sink=None):
    """Call the LLM and parse a JSON array of entries, with retries.

    Shared by process_chunk() (script generation) and review_batch() (review):
    the two only differed in their log file/label and failure sentinel. Returns a
    list of entries, or [] if every attempt failed to produce parseable JSON.
    `log_name` is the raw-response log basename; `label` tags each block
    (e.g. "CHUNK 3/40" or "BATCH 2/10").
    """
    retry_feedback = None
    requested_max = params.max_tokens
    consecutive_severe = 0
    response_fingerprints = {}
    # Prompts already sent on this batch. At temperature 0 the model is
    # deterministic, so re-sending a prompt it has already rejected must
    # produce the same rejected answer - the retry cannot succeed and only
    # costs wall time. Observed on owarimonogatari3's pass-2: attempts 3 and 4
    # sent the same 764-token prompt and got the same 325-token completion,
    # then the batch was failed anyway. Rule 10: decide the policy once.
    attempted_prompts = set()
    # Persists across attempts: a reasoning model gets exactly one larger
    # budget before the batch is failed (Rule 10 - one retry policy).
    reasoning_escalated = False
    for attempt in range(max_retries + 1):
        t0 = time.time()
        truncation_retry_available = False
        attempt_record = None
        try:
            base_messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
            attempt_prompt = user_prompt
            if retry_feedback:
                attempt_prompt += ("\n\nYour previous response was rejected by deterministic "
                                   "quality checks. Return the complete source text exactly once. "
                                   f"Failures: {retry_feedback}")
            messages = base_messages if not retry_feedback else [
                base_messages[0], {"role": "user", "content": attempt_prompt}
            ]
            # Only when the sampler is deterministic. Above 0 a repeated prompt
            # is a legitimate second sample and must still be sent.
            if not params.temperature:
                if attempt_prompt in attempted_prompts:
                    print(f"  {label}: identical prompt already rejected at "
                          f"temperature 0; further retries cannot differ",
                          flush=True)
                    break
            effective_max = get_effective_max_tokens(
                requested_max, params.context_length, messages,
                params.hard_max_tokens, scale_to_context=False)
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=params.temperature,
                top_p=params.top_p,
                presence_penalty=params.presence_penalty,
                max_tokens=effective_max,
                extra_body=build_extra_body(params)
            )
            if not params.temperature:
                # Only a real response proves this deterministic prompt cannot
                # differ. Connection/API failures must retain their retry budget.
                attempted_prompts.add(attempt_prompt)

            choice = response.choices[0]
            # An OpenAI-compatible server may return JSON null for content when
            # a reasoning model spent its whole budget thinking. Calling
            # .strip() on that raised AttributeError, which the caller booked as
            # a generic api_error - so the one-escalation-then-overflow policy
            # never ran on the exact responses it exists for.
            text = (choice.message.content or "").strip()
            response_fingerprint = hashlib.sha256(
                " ".join(text.split()).encode("utf-8")).hexdigest()
            response_fingerprints[response_fingerprint] = (
                response_fingerprints.get(response_fingerprint, 0) + 1)
            repeat_count = response_fingerprints[response_fingerprint]
            finish_reason = choice.finish_reason
            usage = getattr(response, 'usage', None)
            # Reasoning models bill thinking tokens to completion_tokens but
            # return them in message.reasoning_content, so they are invisible
            # to any budget sized on the visible response.
            usage_details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = getattr(usage_details, "reasoning_tokens", None)

            # Log raw response for debugging (rotating to cap unbounded growth)
            log_path = get_response_log_path(log_name)
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            if not os.environ.get("ALEXANDRIA_RUN_ID"):
                _rotate_log_if_large(log_path)
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"\n{'='*80}\n")
                lf.write(f"{label} | attempt {attempt + 1} | finish_reason={finish_reason}\n")
                if usage:
                    lf.write(f"tokens: prompt={getattr(usage, 'prompt_tokens', '?')} completion={getattr(usage, 'completion_tokens', '?')}\n")
                lf.write(f"{'─'*80}\n")
                lf.write(text)
                lf.write(f"\n{'='*80}\n")

            print(f"  finish_reason={finish_reason}", end="")
            if usage:
                print(f" | tokens: prompt={getattr(usage, 'prompt_tokens', '?')} completion={getattr(usage, 'completion_tokens', '?')}", end="")
            print(f" | took {time.time() - t0:.1f}s")
            if attempt_observer:
                attempt_record = {
                    "attempt": attempt + 1,
                    "elapsed_seconds": round(time.time() - t0, 3),
                    "finish_reason": finish_reason,
                    "requested_max_tokens": requested_max,
                    "effective_max_tokens": effective_max,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                    "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                    "reasoning_tokens": reasoning_tokens,
                    "error": None,
                    "response_fingerprint": response_fingerprint,
                    "response_repeat_count": repeat_count,
                }
                attempt_observer(attempt_record)

            if finish_reason == "length":
                verdict = classify_length_finish(
                    text, reasoning_tokens, reasoning_escalated)
                if verdict == "reasoning_overflow":
                    print("  Reasoning overflow: the model spent its whole "
                          "budget thinking and returned no content. Failing "
                          "this batch instead of subdividing.")
                    if attempt_record is not None:
                        attempt_record["outcome"] = "response_rejected"
                        attempt_record["failure_codes"] = ["reasoning_overflow"]
                    return []
                print(f"  WARNING: Response was truncated (hit effective max_tokens={effective_max}). Consider optimizing LM Studio context.")
                if attempt < max_retries:
                    next_max = get_next_retry_max_tokens(
                        requested_max, "token_truncated", params.hard_max_tokens)
                    next_effective = get_effective_max_tokens(
                        next_max, params.context_length, base_messages,
                        params.hard_max_tokens, scale_to_context=False)
                    if next_effective > effective_max:
                        print(f"  Token budget: requested={requested_max}, effective={effective_max}, "
                              f"next_requested={next_max}, next_effective={next_effective}")
                        requested_max = next_max
                        retry_feedback = None
                        truncation_retry_available = True
                        if verdict == "escalate_once":
                            reasoning_escalated = True
                    else:
                        print(f"  Token escalation exhausted: effective budget cannot grow "
                              f"beyond {effective_max} in the loaded context.")

        except Exception as e:
            if attempt_observer:
                attempt_observer({"attempt": attempt + 1,
                                  "elapsed_seconds": round(time.time() - t0, 3),
                                  "finish_reason": None, "requested_max_tokens": requested_max,
                                  "effective_max_tokens": None, "prompt_tokens": None,
                                  "completion_tokens": None,
                                  "error": f"{type(e).__name__}: {e}",
                                  "outcome": "api_error",
                                  "failure_codes": ["api_error"]})
            print(f"Error calling LLM API (attempt {attempt + 1}) after {time.time() - t0:.1f}s: {e}")
            if attempt < max_retries:
                continue
            return []

        # Clean and extract JSON from response
        try:
            json_text = clean_json_string(text)
        except AdjacentArrayOverlapError as exc:
            if attempt_record is not None:
                attempt_record["outcome"] = "response_rejected"
                attempt_record["failure_codes"] = ["adjacent_array_overlap"]
            retry_feedback = f"adjacent_array_overlap: {exc}"
            print(f"Warning: {label} repeats text across adjacent arrays "
                  f"(attempt {attempt + 1}): {exc}")
            if attempt < max_retries and (finish_reason != "length" or truncation_retry_available):
                print("Retrying...")
                continue
            return []

        if not json_text:
            if attempt_record is not None:
                attempt_record["outcome"] = "response_rejected"
                attempt_record["failure_codes"] = ["missing_json_array"]
            print(f"Warning: Could not find JSON array in {label} response (attempt {attempt + 1})")
            if attempt < max_retries and (finish_reason != "length" or truncation_retry_available):
                print("Retrying...")
                continue
            # Last-attempt recovery: clean_json_string deliberately rejects
            # ambiguous multi-array structure, but repair_json_array can still
            # extract complete objects from the raw array region. It remains
            # safe only because the normal transform + deterministic quality
            # gates below must accept the reconstructed source in full.
            array_start = text.find("[")
            if array_start == -1:
                print(f"Response preview: {text[:300]}...")
                return []
            json_text = text[array_start:]
            print("  Trying quality-gated raw-array salvage after retries exhausted")

        # Try to parse, with repair attempts
        entries = repair_json_array(json_text)

        if entries and len(entries) > 0:
            if transform_entries:
                transformed = transform_entries(entries)
                if transformed.get("unresolved"):
                    if attempt_record is not None:
                        attempt_record["outcome"] = "response_rejected"
                        attempt_record["failure_codes"] = ["unresolved_deterministic_repairs"]
                    retry_feedback = "unresolved_safe_repair"
                    print(f"Warning: {label} has unresolved deterministic repairs "
                          f"(attempt {attempt + 1}): {transformed['unresolved']}")
                    if attempt < max_retries:
                        print("Retrying...")
                        continue
                    return []
                entries = transformed["entries"]
                if transformed.get("changes"):
                    print(f"  Applied {len(transformed['changes'])} deterministic chunk repair(s)")
                if transformed.get("review"):
                    print(f"  Speaker identity review: {transformed['review']}")
            if validate_entries:
                quality = validate_entries(entries)
                if not quality["passed"]:
                    if near_miss_sink is not None and is_trigram_only_near_miss(quality):
                        best = (near_miss_sink[0][1]["metrics"]["ordered_trigram_recall"]
                                if near_miss_sink else -1.0)
                        if quality["metrics"]["ordered_trigram_recall"] > best:
                            near_miss_sink[:] = [(entries, quality)]
                    if attempt_record is not None:
                        attempt_record["outcome"] = "quality_rejected"
                        attempt_record["failure_codes"] = [
                            finding.get("code") for finding in quality.get("findings", [])]
                        attempt_record["quality_metrics"] = quality.get("metrics", {})
                    retry_feedback = _build_retry_feedback_message(quality)
                    metrics = quality.get("metrics", {})
                    completion_tokens = (getattr(usage, "completion_tokens", None)
                                         if usage else None)
                    retry_policy = get_quality_retry_policy(
                        finish_reason, completion_tokens, effective_max, quality)
                    if retry_policy == "retry_same_budget" and is_severe_chunk_truncation(quality):
                        consecutive_severe += 1
                    else:
                        consecutive_severe = 0
                    if attempt < max_retries and retry_policy == "increase_tokens":
                        next_max = get_next_retry_max_tokens(
                            requested_max, "incomplete_output", params.hard_max_tokens)
                        if next_max != requested_max:
                            print(f"  Increasing token budget after incomplete output: "
                                  f"{requested_max} -> {next_max}")
                            requested_max = next_max
                    elif attempt < max_retries:
                        print(f"  Retry policy: {retry_policy}; token budget remains {requested_max}")
                    print(f"Warning: {label} failed quality validation "
                          f"(attempt {attempt + 1}): {retry_feedback}; metrics={metrics}")
                    retry_evidence = max(consecutive_severe, repeat_count)
                    if (retry_decider
                            and retry_decider(quality, retry_evidence) == "split"):
                        print("  Repeated unproductive response; switching to adaptive split")
                        return []
                    if attempt < max_retries:
                        print("Retrying...")
                        continue
                    return []
            if finish_reason == "length":
                if attempt_record is not None:
                    attempt_record["outcome"] = "response_rejected"
                    attempt_record["failure_codes"] = ["token_truncated"]
                if attempt < max_retries and truncation_retry_available:
                    print("Retrying truncated response with a larger token budget...")
                    continue
                return []
            if attempt > 0:
                print(f"  Succeeded on retry {attempt + 1}")
            if attempt_record is not None:
                if attempt_record.get("failure_codes"):
                    attempt_record["recovery_codes"] = attempt_record.pop("failure_codes")
                attempt_record["outcome"] = "accepted"
            return entries

        print(f"Warning: Could not parse {label} response as JSON (attempt {attempt + 1})")
        if attempt_record is not None:
            attempt_record["outcome"] = "response_rejected"
            attempt_record["failure_codes"] = ["malformed_json"]
        print(f"JSON preview: {json_text[:300]}...")

        if attempt < max_retries and (finish_reason != "length" or truncation_retry_available):
            print("Retrying...")
            continue
        if finish_reason == "length":
            return []

        # Last resort after every full-response retry is exhausted.
        salvaged_entries = salvage_json_entries(json_text)
        if salvaged_entries:
            if transform_entries:
                transformed = transform_entries(salvaged_entries)
                if transformed.get("unresolved"):
                    print("Regex-salvaged response has unresolved deterministic repairs")
                    return []
                salvaged_entries = transformed["entries"]
            if validate_entries:
                quality = validate_entries(salvaged_entries)
                if not quality["passed"]:
                    codes = ", ".join(finding["code"] for finding in quality["findings"])
                    print(f"Regex-salvaged response failed quality validation: {codes}")
                    return []
            print(f"Regex-salvaged {len(salvaged_entries)} entries from malformed response")
            if attempt_record is not None:
                attempt_record["outcome"] = "accepted"
                attempt_record["recovery_codes"] = attempt_record.pop(
                    "failure_codes", ["malformed_json"])
            return salvaged_entries

    return []


def _build_chunk_context(chunk_num, total_chunks, previous_entries):
    """Build the positional + character-roster context block prepended to a chunk."""
    context_parts = []

    if chunk_num == 1:
        context_parts.append("(Beginning of text)")
    elif chunk_num == total_chunks:
        context_parts.append("(End of text)")
    else:
        context_parts.append(f"(Part {chunk_num} of {total_chunks})")

    if previous_entries and len(previous_entries) > 0:
        # Build character roster for name consistency across chunks
        characters_seen = sorted(set(
            entry.get("speaker", "") for entry in previous_entries
            if entry.get("speaker", "") and entry.get("speaker", "") != "NARRATOR"
        ))
        if characters_seen:
            context_parts.append(f"Characters in this book: {', '.join(characters_seen)}")

        # Include last few entries so the model can maintain style and tone continuity
        tail = previous_entries[-3:]
        context_parts.append("\nPrevious section ended with:")
        for entry in tail:
            context_parts.append(json.dumps(entry, ensure_ascii=False))

    return "\n".join(context_parts)


def build_book_request_preflight(chunks, system_prompt, user_prompt_template,
                                 max_tokens, context_length, parallel,
                                 context_growth_chars=2000, reserve=512):
    """Estimate real prompt/completion size and per-slot context fit for a book."""
    requests = []
    total_chunks = len(chunks)
    for number, chunk in enumerate(chunks, 1):
        context = _build_chunk_context(number, total_chunks, None)
        user_prompt = user_prompt_template.format(context=context, chunk=chunk)
        prompt_tokens = math.ceil((len(system_prompt) + len(user_prompt)
                                   + context_growth_chars) / 3)
        predicted_completion = min(int(max_tokens), max(256, math.ceil(len(chunk) * 0.8)))
        requests.append({"chunk_number": number, "prompt_tokens": prompt_tokens,
                         "predicted_completion_tokens": predicted_completion,
                         "predicted_total_tokens": prompt_tokens + predicted_completion + reserve})
    totals = sorted(item["predicted_total_tokens"] for item in requests)
    per_slot = int(context_length or 0) // max(1, int(parallel or 1))
    worst = totals[-1] if totals else 0
    p95 = totals[max(0, math.ceil(len(totals) * 0.95) - 1)] if totals else 0
    return {"chunk_count": len(chunks), "context_length": context_length,
            "parallel": parallel, "per_slot_context": per_slot,
            "worst_predicted_tokens": worst, "p95_predicted_tokens": p95,
            "average_predicted_tokens": round(sum(totals) / len(totals), 1) if totals else 0,
            "predicted_fits": bool(per_slot and worst <= per_slot),
            "required_total_context": {str(level): worst * level for level in range(1, 5)},
            "requests": requests}


def process_chunk(client, model_name, chunk, chunk_num, total_chunks, params,
                  previous_entries=None, max_retries=4, attempt_observer=None,
                  allow_early_split=False, near_miss_sink=None):
    """Process a text chunk and return JSON script entries.

    max_retries=4 (5 total attempts) rather than the previous 2 (3 attempts):
    live reproduction of two real overnight batch failures measured a genuine
    ~40% single-attempt success rate on the specific class of content that
    fails (the model occasionally stops a few lines into a chunk, right after
    a short opening dialogue exchange, despite much more source remaining) --
    NOT a 0% "impossible" case. At that rate 3 attempts succeeds ~78% of the
    time; 5 attempts succeeds ~92%. This is worth the added budget because a
    failing attempt is cheap here (the model stops in 2-10s when it's going to
    fail, vs 60-100s for a real full-length generation) -- so the added
    attempts cost little even in the worst case, while meaningfully reducing
    how often a single unlucky run costs the rest of a book (checkpoint/resume
    requires a gapless accepted-chunk prefix, Rule 9).
    """
    sys_prompt = params.system_prompt or DEFAULT_SYSTEM_PROMPT
    usr_template = params.user_prompt_template or DEFAULT_USER_PROMPT

    context = _build_chunk_context(chunk_num, total_chunks, previous_entries)
    user_prompt = usr_template.format(context=context, chunk=chunk)
    established_speakers = [entry.get("speaker") for entry in (previous_entries or [])
                            if isinstance(entry, dict) and entry.get("speaker")]

    def prepare_entries(entries):
        structural = build_deterministic_repair(entries, chunk)
        if structural["unresolved"]:
            return structural
        identities = stabilize_speaker_identities(structural["entries"], established_speakers)
        return {"entries": identities["entries"],
                "changes": structural["changes"] + identities["changes"],
                "unresolved": [], "review": identities["review"]}

    local_attempts = []

    def observe(attempt):
        local_attempts.append(attempt)
        if attempt_observer:
            attempt_observer(attempt)

    def call(retries):
        return call_llm_for_entries(
            client, model_name, sys_prompt, user_prompt, params,
            log_name="llm_responses.log",
            label=f"CHUNK {chunk_num}/{total_chunks}",
            max_retries=retries,
            validate_entries=lambda entries: validate_chunk_quality(chunk, entries),
            transform_entries=prepare_entries,
            attempt_observer=observe,
            retry_decider=(
                (lambda quality, attempt_number: get_chunk_retry_action(
                    quality, attempt_number, allow_early_split=True))
                if allow_early_split and retries else None),
            near_miss_sink=near_miss_sink,
        )

    entries = call(max_retries)
    # One bonus attempt when the model came very close on its last try -
    # diagnosed live (2026-07-19): a chunk hit 86% recall on its final
    # attempt (one attempt away from the 90% pass threshold), then the
    # retry budget was already exhausted. See _is_near_miss_recall.
    # Two properties are deliberate, not oversights: this call passes no
    # retry_feedback (a fresh call_llm_for_entries, same as the first
    # attempt) - fresh samples measured 3/3 where feedback-looped retries
    # looped instead; and it can never trigger an early split, since
    # retry_decider is only wired up when `retries` is truthy above, and
    # this call always passes max_retries=0.
    if not entries and local_attempts and _is_near_miss_recall(
            local_attempts[-1].get("quality_metrics")):
        print(f"  CHUNK {chunk_num}/{total_chunks} near-miss on final attempt; "
              "granting one bonus retry")
        entries = call(0)
    return entries


def split_failed_chunk(chunk, minimum_chars=400):
    """Split near the midpoint at a natural boundary, or return [] if unsafe."""
    if len(chunk) < minimum_chars * 2:
        return []
    candidates = [match.end() for match in re.finditer(r"\n\s*\n", chunk)]
    if not candidates:
        candidates = [match.end() for match in re.finditer(r"(?<=[.!?])\s+", chunk)]
    valid = [offset for offset in candidates
             if offset >= minimum_chars and len(chunk) - offset >= minimum_chars]
    if not valid:
        return []
    offset = min(valid, key=lambda value: abs(value - len(chunk) / 2))
    return [chunk[:offset].strip(), chunk[offset:].strip()]


def process_chunk_adaptively(client, model_name, chunk, chunk_num, total_chunks,
                             params, previous_entries=None, attempt_observer=None):
    """Try a full chunk, then a bounded natural-boundary split on exhaustion.

    Each split half gets its own independent retry budget. Measured on real
    production failures: a single attempt on this class of content has roughly
    a 40-60% failure rate even at tuned sampling params (the model occasionally
    "stops" a few lines into a chunk despite the source having much more left),
    so one half failing is not rare and is not evidence the other half will
    fail too. Earlier code returned as soon as the first half failed without
    ever attempting the second half, which meant a single unlucky sample on
    part 1 could silently forfeit the model's independent, often-good chance
    on part 2 -- and since the caller treats any empty result as fatal for the
    whole book (checkpoint/resume requires a gapless accepted-chunk prefix,
    Rule 9), that one unlucky sample cost the rest of the book's chunks too.

    Splitting recurses: a part that itself exhausts its retry budget is split
    again rather than given up on. Diagnosed live (2026-07-19): a chunk failed
    identically across 3 independent full-book runs, oscillating between
    near-perfect (89-93% recall) and near-total collapse (1-11%) on otherwise
    identical retries -- not content-driven (the source text is unremarkable),
    so a smaller target has a real independent chance the original size didn't
    get. `split_failed_chunk`'s own minimum_chars floor (refuses to split
    below 800 chars) already bounds the recursion depth, so no new depth
    limit is needed here.
    """
    # Captures the best full-chunk trigram-only near-miss (see
    # is_trigram_only_near_miss). Only accepted below if BOTH the full-chunk
    # retries and the adaptive split fail, so it never changes a chunk that
    # recovers by any means (Rule 9): a chunk that passes at 0.90 - full retry
    # or split - always takes a success return before the fallback is reached.
    near_miss = []
    entries = process_chunk(client, model_name, chunk, chunk_num, total_chunks,
                            params, previous_entries=previous_entries,
                            allow_early_split=True, near_miss_sink=near_miss,
                            attempt_observer=(
                                (lambda attempt: record_attempt_context(
                                    attempt_observer, attempt, "full"))
                                if attempt_observer else None))
    if entries:
        return entries, False
    parts = split_failed_chunk(chunk)
    if not parts:
        return _accept_near_miss(near_miss, chunk_num, total_chunks), False
    print(f"  Adaptive split: chunk {chunk_num}/{total_chunks} -> "
          f"{len(parts[0])} + {len(parts[1])} chars")
    combined = []
    any_part_failed = False
    for part_number, part in enumerate(parts, 1):
        context_entries = list(previous_entries or []) + combined
        part_entries, _ = process_chunk_adaptively(
            client, model_name, part, chunk_num, total_chunks, params,
            previous_entries=context_entries,
            attempt_observer=(
                (lambda attempt, part_number=part_number: record_attempt_context(
                    attempt_observer, attempt, "split", part_number))
                if attempt_observer else None))
        if not part_entries:
            print(f"  Adaptive split part {part_number}/{len(parts)} failed")
            any_part_failed = True
            continue
        combined.extend(part_entries)
    if any_part_failed:
        return _accept_near_miss(near_miss, chunk_num, total_chunks), True
    combined_quality = validate_chunk_quality(chunk, combined)
    if not combined_quality["passed"]:
        # The recombined halves clear the 0.90 gate for most split recoveries.
        # When they instead land as a whole-chunk trigram-only near-miss - the
        # case where the model only reaches near-complete quality on the smaller
        # split targets, never on the full chunk (e.g. real books whose sole
        # near-miss telemetry is in the split phase) - accept it here rather than
        # forfeiting the whole book. Still exhaustion-only and zero blast radius:
        # a combined result that passes 0.90 takes the success return below
        # unchanged, so no already-accepted chunk is affected.
        if is_trigram_only_near_miss(combined_quality):
            print(f"  Adaptive split recombination accepted as trigram-only near-miss "
                  f"(ordered_trigram_recall="
                  f"{combined_quality['metrics']['ordered_trigram_recall']})")
            return combined, True
        print("  Adaptive split recombination failed original-chunk validation")
        return _accept_near_miss(near_miss, chunk_num, total_chunks), True
    return combined, True


def _accept_near_miss(near_miss, chunk_num, total_chunks):
    """Return a captured full-chunk trigram-only near-miss's entries as a last
    resort after full retries and adaptive split are both exhausted, else [].

    This is the ONLY place the near-miss is accepted, and only on failure-return
    paths, so a chunk that recovers by any means never reaches it (zero change
    to already-accepted chunks). The caller's post-return gate re-checks the
    same is_trigram_only_near_miss condition before committing the chunk.
    """
    if not near_miss:
        return []
    entries, quality = near_miss[0]
    print(f"  CHUNK {chunk_num}/{total_chunks} accepted as trigram-only near-miss "
          f"after full exhaustion (ordered_trigram_recall="
          f"{quality['metrics']['ordered_trigram_recall']})")
    return entries

def main():
    parser = argparse.ArgumentParser(description="Generate annotated script from a book file.")
    parser.add_argument("input_file", help="Path to the input text/epub file")
    parser.add_argument("--output", default=None, help="Output JSON path (default: ../annotated_script.json)")
    parser.add_argument("--strip-front-matter", action=argparse.BooleanOptionalAction, default=True,
                         help="Strip known non-narrative compiler front matter (translator's "
                              "note / table of contents) before generation")
    parser.add_argument("--chunk-size", type=int, default=None,
                         help="Override generation.chunk_size (chars per LLM call) for this "
                              "run only, without editing config.json. Smaller chunks shorten "
                              "each call's output. The chunk size is part of the generation "
                              "fingerprint, so a different value starts a fresh checkpoint.")
    args = parser.parse_args()

    input_file_path = args.input_file
    print(f"Processing book from: {input_file_path}")

    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found: {input_file_path}")
        sys.exit(1)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        book_content = f.read()

    book_content, preprocessing = get_preprocessed_source(
        book_content, strip_front_matter=args.strip_front_matter)
    source_normalizations = preprocessing["source_normalizations"]
    if source_normalizations:
        print(f"Normalized {len(source_normalizations)} known source corruption(s) in memory; "
              "the upload was not modified.")
    front_matter_removed = preprocessing["front_matter_removed"]
    # Publisher colophon: copyright page, ISBN/CIP block, imprint lines. The
    # three-pass path has always stripped this; the single-pass path did not,
    # so a published book's chunk 1 was its legal notice. The model has no way
    # to turn "This book is a work of fiction. Names, characters..." into
    # annotated dialogue, and the coverage gate then failed the chunk for not
    # reproducing it - index18 retried chunk 1 eight times before this.
    publisher_matter = preprocessing["publisher_matter"]
    if publisher_matter["front_paragraphs"] or publisher_matter["back_paragraphs"]:
        print(f"Stripped publisher matter: "
              f"{publisher_matter['front_paragraphs']} paragraph(s) from the "
              f"front, {publisher_matter['back_paragraphs']} from the back")
        if front_matter_removed:
            print(f"Stripped {front_matter_removed['removed_chars']} characters of known "
                  "front matter (translator's note / table of contents) before generation")
    # PREFLIGHT. Name every known damage class before spending GPU time.
    #
    # Damage in a source file otherwise surfaces as a generation failure
    # twenty minutes in, and reads like a model problem. Daisy Miller died at
    # chunk 2 of 22 because its apostrophes had been stripped - "She s got to
    # give me some candy" - and 17 of the 28 public-domain novels carry the
    # same damage. index18 needed five separate fixes before it would generate
    # at all. None of that was visible when the user picked the file.
    preflight = preflight_source(book_content)
    for line in preflight["messages"]:
        print(line)
    book_content = preflight["text"]

    source_unicode = audit_unicode_text(book_content)
    print(f"Source scripts: {', '.join(source_unicode['scripts']) or 'none'}; "
          f"NFC normalized: {source_unicode['is_nfc']}")
    # Unsafe control characters stay a hard refusal: they are never legitimate
    # prose and can break downstream parsing.
    if source_unicode["unsafe_controls"]:
        print("Error: source contains unsafe control characters; "
              f"details={source_unicode}")
        sys.exit(1)
    # Replacement characters are graded rather than absolute. A file decoded
    # with the wrong codec loses every non-ASCII character, and index18 - one
    # of the four gold-annotated books - was excluded from every measurement
    # because of 6,662 of them, 1.40% of the file. Deterministic repair brings
    # that to 0.26%, and the remainder is genuinely ambiguous: accented
    # letters, ellipses and dashes that no rule and no model resolved reliably.
    #
    # Refusing at any count meant a repaired book was as unusable as a corrupt
    # one. The threshold sits between the two, so it still refuses the raw file
    # and now accepts the repaired one. It is not a claim that stray
    # replacement characters are harmless - `verbalize_symbols` drops them at
    # the TTS boundary (goal 5.1), so they never reach the engine either way.
    replacement_count = source_unicode["replacement_character_count"]
    replacement_share = replacement_count / max(1, len(book_content))
    if not replacement_load_is_acceptable(replacement_count, len(book_content)):
        print(f"Error: source is {replacement_share:.2%} replacement "
              f"characters ({replacement_count}), above the "
              f"{MAX_REPLACEMENT_SHARE:.2%} limit.")
        print(replacement_repair_hint(input_file_path))
        sys.exit(1)
    if replacement_count:
        print(f"WARNING: source carries {replacement_count} replacement "
              f"characters ({replacement_share:.3%}), under the "
              f"{MAX_REPLACEMENT_SHARE:.2%} limit. These are positions where "
              "the original character is unrecoverable; they are dropped "
              "before synthesis, so affected words will be missing a letter "
              "or mark in the audio.")

    print(f"Read {len(book_content)} characters")

    # Load LLM config
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_dir = os.path.dirname(__file__)
    data_dir = get_runtime_data_dir(root)
    config_path = get_app_config_path(data_dir, root, app_dir)
    if not os.path.exists(config_path):
        print("Warning: config.json not found. Using defaults.")
    config = load_app_config(config_path)

    llm_config = config.get("llm", {})
    base_url = llm_config.get("base_url", "http://localhost:11434/v1")
    api_key = llm_config.get("api_key", "local")
    model_name = llm_config.get("model_name", "richardyoung/qwen3-14b-abliterated:Q8_0")
    llm_mode = config.get("llm_mode", "local")

    # The attribution adapter measured +9.8 cross-book and +14.6 through the
    # served path, but a server started WITHOUT it answers happily at base
    # quality and nothing fails. Say so at startup rather than discovering it
    # in the numbers weeks later - the seed bug was exactly this shape.
    try:
        from attribution_adapter import check_adapter, describe
        ok, detail = check_adapter(config, base_url)
        print(f"LLM adapter: {describe(config)}")
        if not ok:
            print(f"WARNING: {detail}", flush=True)
    except Exception as exc:                            # noqa: BLE001
        print(f"WARNING: adapter check failed: {str(exc)[:160]}", flush=True)

    # Load custom prompts or use defaults
    prompts_config = config.get("prompts") or {}
    system_prompt = prompts_config.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
    user_prompt_template = prompts_config.get("user_prompt") or DEFAULT_USER_PROMPT

    # Load generation settings
    generation_config = config.get("generation") or {}
    configured_chunk_size = generation_config.get("chunk_size", 3000)
    try:
        chunk_size = get_valid_chunk_size(configured_chunk_size, args.chunk_size)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    if args.chunk_size is not None:
        print(f"Chunk size overridden via --chunk-size: {configured_chunk_size} -> {chunk_size}")
    max_tokens = generation_config.get("max_tokens", 4096)
    temperature = generation_config.get("temperature", 0.6)
    top_p = generation_config.get("top_p", 0.8)
    # Default to None (not 0) so an unconfigured sampler is omitted from the
    # request, while an explicit 0 is preserved and sent through.
    top_k = generation_config.get("top_k")
    min_p = generation_config.get("min_p")
    presence_penalty = generation_config.get("presence_penalty", 0.0)
    banned_tokens = generation_config.get("banned_tokens", [])

    print(f"Connecting to: {base_url}")
    print(f"Using model: {model_name}")
    print(f"Chunk size: {chunk_size} chars, Max tokens: {max_tokens}")
    if banned_tokens:
        print(f"Banned tokens: {banned_tokens}")

    # Self-heal a stale/misconfigured local or remote LM Studio before making
    # any calls, mirroring review_script.py/find_nicknames.py. This file has
    # no VRAM watchdog or concurrency wave processing of its own (chunks are
    # processed strictly sequentially), so only the self-heal call applies
    # here - is_remote/lm_status aren't needed for anything else in this file.
    _, lm_status, heal_msg = ensure_ideal_settings(
        llm_mode, base_url, model_name, ssh_alias=config.get("llm_remote_ssh"))
    print(heal_msg)

    # Create OpenAI client with custom base URL
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    # Split into chunks at natural boundaries
    chunks = split_into_chunks(book_content, max_size=chunk_size)
    total_chunks = len(chunks)

    print(f"Split into {total_chunks} chunks at paragraph/sentence boundaries")
    request_preflight = build_book_request_preflight(
        chunks, system_prompt, user_prompt_template, max_tokens,
        lm_status.get("context_length"), lm_status.get("parallel"))
    print(f"Request preflight: worst={request_preflight['worst_predicted_tokens']} tokens, "
          f"p95={request_preflight['p95_predicted_tokens']}, "
          f"per-slot={request_preflight['per_slot_context']}, "
          f"fits={request_preflight['predicted_fits']}")

    output_path = args.output or os.path.join(data_dir, "annotated_script.json")
    response_log = os.path.relpath(get_response_log_path("llm_responses.log"), data_dir)

    all_entries = []
    chunk_times = []
    start_time = time.monotonic()

    # Sampling/prompt settings are constant across chunks - build once.
    gen_params = LLMGenParams(
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        banned_tokens=banned_tokens,
        context_length=lm_status.get("context_length"),
    )

    fingerprint = get_generation_fingerprint(
        book_content, chunks, model_name, base_url, gen_params, chunk_size)
    accepted_chunks = load_generation_checkpoint(output_path, fingerprint)
    if accepted_chunks:
        print(f"Resuming from generation checkpoint: {len(accepted_chunks)}/{total_chunks} validated chunks.")
        all_entries = [entry for item in accepted_chunks for entry in item["entries"]]

    for i, chunk in enumerate(chunks, 1):
        if i <= len(accepted_chunks):
            continue
        print(f"Processing chunk {i}/{total_chunks} ({len(chunk)} chars)...")

        chunk_start = time.monotonic()
        previous = all_entries if len(all_entries) > 0 else None
        chunk_attempts = []
        entries, adaptively_split = process_chunk_adaptively(
            client, model_name, chunk, i, total_chunks, gen_params,
            previous_entries=previous,
            attempt_observer=chunk_attempts.append,
        )
        chunk_elapsed = time.monotonic() - chunk_start
        chunk_times.append(chunk_elapsed)

        if not entries:
            save_generation_quality_manifest(output_path, build_generation_quality_manifest(
                "failed", fingerprint, accepted_chunks, source_normalizations,
                total_chunks=total_chunks, failed_chunk=i,
                failure="chunk_failed_after_retries", response_log=response_log,
                model_name=model_name, failed_chunk_attempts=chunk_attempts))
            print(f"Error: chunk {i}/{total_chunks} failed validation after retries; "
                  "preserving existing output and validated checkpoint")
            sys.exit(1)

        quality = validate_chunk_quality(chunk, entries)
        # process_chunk_adaptively only ever returns entries that either pass the
        # 0.90 gate or are an exhaustion-time trigram-only near-miss. Re-check the
        # same condition here (Rule 15) before committing: a hard failure still
        # aborts the book, a near-miss is accepted and flagged for telemetry.
        near_miss_accepted = not quality["passed"] and is_trigram_only_near_miss(quality)
        if not quality["passed"] and not near_miss_accepted:
            save_generation_quality_manifest(output_path, build_generation_quality_manifest(
                "failed", fingerprint, accepted_chunks, source_normalizations,
                total_chunks=total_chunks, failed_chunk=i,
                failure="post_return_validation_failed", failed_quality=quality,
                model_name=model_name, response_log=response_log))
            print(f"Error: chunk {i}/{total_chunks} failed post-return validation; "
                  "preserving existing output and validated checkpoint")
            sys.exit(1)
        if near_miss_accepted:
            print(f"  Chunk {i}/{total_chunks} accepted as trigram-only near-miss "
                  f"(ordered_trigram_recall={quality['metrics']['ordered_trigram_recall']})")
        all_entries.extend(entries)
        accepted_chunks.append({
            "chunk_number": i,
            "source_sha256": fingerprint["chunk_sha256"][i - 1],
            "entries": entries,
            "quality": quality,
            "adaptively_split": adaptively_split,
            "near_miss_accepted": near_miss_accepted,
            "attempts": chunk_attempts,
        })
        save_generation_checkpoint(output_path, fingerprint, accepted_chunks)
        print(f"  Got {len(entries)} entries (chunk took {chunk_elapsed:.0f}s)")

        remaining = total_chunks - i
        if remaining > 0:
            avg = sum(chunk_times) / len(chunk_times)
            eta_sec = avg * remaining
            if eta_sec < 60:
                eta_str = f"{eta_sec:.0f}s"
            elif eta_sec < 3600:
                eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
            else:
                eta_str = f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m"
            elapsed_total = time.monotonic() - start_time
            print(f"  ETA: ~{eta_str} remaining ({remaining} chunk(s) left, avg {avg:.0f}s/chunk, {elapsed_total:.0f}s elapsed)")

    if not all_entries:
        print("Error: No script entries generated")
        sys.exit(1)

    final_repair = build_final_generation_repair(all_entries, book_content)
    if not final_repair["unresolved"]:
        all_entries = final_repair["entries"]
    whole_quality = validate_chunk_quality(book_content, all_entries)
    preflight = audit_script(all_entries, book_content, is_generic_speaker)
    identity_review = stabilize_speaker_identities(all_entries)["review"]
    final_manifest = build_generation_quality_manifest(
        "verified" if passes_final_generation_gate(
            whole_quality, preflight, final_repair["unresolved"]) else "failed",
        fingerprint, accepted_chunks, source_normalizations,
        total_chunks=total_chunks, response_log=response_log,
        model_name=model_name,
        source_unicode=source_unicode,
        request_preflight=request_preflight,
        whole_book_quality=whole_quality,
        preflight=preflight,
        final_repairs={"changes": final_repair["changes"],
                       "unresolved": final_repair["unresolved"]},
        speaker_identity_review=identity_review,
        speaker_consistency=build_speaker_consistency_report(
            all_entries, identity_review),
    )
    save_generation_quality_manifest(output_path, final_manifest)
    if final_manifest["status"] != "verified":
        print("Error: final whole-book quality gate failed; preserving existing output and checkpoint")
        sys.exit(1)

    atomic_json_write(all_entries, output_path)
    save_generation_quality_manifest(output_path, {**final_manifest, "status": "complete"})
    clear_generation_checkpoint(output_path)

    # Only clear chunks when writing to the default annotated_script.json location
    if args.output is None:
        chunks_path = os.path.join(data_dir, "chunks.json")
        if os.path.exists(chunks_path):
            os.remove(chunks_path)
            print("Cleared old chunks.json")

    # Summary (check both "speaker" and "type" fields)
    speakers = set(entry.get("speaker") or entry.get("type") or "UNKNOWN" for entry in all_entries)
    print(f"\nGenerated {len(all_entries)} script entries")
    print(f"Speakers found: {', '.join(sorted(speakers))}")
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()
