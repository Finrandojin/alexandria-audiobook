"""Three-pass script generation orchestrator (segment -> attribute -> instruct).
A side-by-side alternative to generate_script.py's single pass; the single-pass
path is untouched. See docs/superpowers/specs/2026-07-21-three-pass-script-generation-design.md."""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import replace

from openai import OpenAI

from generate_script import (call_llm_for_entries, split_into_chunks,
                             split_into_chunk_records,
                             fix_mojibake, LLMGenParams,
                             split_failed_chunk, is_trigram_only_near_miss)
from source_normalization import (neutralize_lossy_residue,
                                  normalize_known_source_corruptions,
                                  repair_lossy_replacements,
                                  strip_known_front_matter,
                                  strip_publisher_matter)
from script_preflight import audit_unicode_text
from speaker_identity import stabilize_speaker_identities
from script_repair import build_deterministic_repair
from default_prompts import (load_segment_prompts, load_attribute_prompts,
                             load_instruct_prompts)
from pass_quality import (is_attested_name,
                          validate_segment_quality, validate_attribution,
                          validate_instruct, index_head_check,
                          analyze_outer_quote_regions, split_outer_quote_regions)
from review_script import normalize_text
from config_settings import load_app_config
from lmstudio_settings import (ensure_ideal_settings, get_effective_max_tokens,
                               TokenBudgetError)
from utils import (get_runtime_data_dir, get_app_config_path,
                   atomic_json_write, safe_load_json, is_nonverbal_text)

BATCH_SIZE = 25
NARRATOR_DEFAULT_INSTRUCT = "Neutral, even narration."
CHARACTER_DEFAULT_INSTRUCT = "Natural, in-character delivery."


def _record_resolution(sink, value):
    """Append a per-chunk pass-1 resolution to the telemetry sink, if one is
    provided. `sink` is a per-chunk list; callers read its last entry."""
    if sink is not None:
        sink.append(value)


def as_profile_mapping(profile):
    """Return a plain mapping for one three_pass_model_profiles entry.

    load_app_config validates that section into ThreePassModelProfile objects,
    but this module reads profiles with .get(), so a configured profile
    previously crashed the run with AttributeError. None-valued fields are
    dropped so an unset profile key falls through to the caller's default
    instead of overriding it with None.
    """
    if profile is None:
        return {}
    if hasattr(profile, "model_dump"):
        return {key: value for key, value in profile.model_dump().items()
                if value is not None}
    return profile


DEFAULT_MODEL_PROFILES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "three_pass_model_profiles.json")


def load_default_model_profiles(path=None):
    """Return the measured per-model profiles checked into the repo.

    app/config.json is gitignored and machine-local, so profiles set only
    there would not reproduce on another machine. These defaults ship with the
    code; config.json still wins per key for local experiments.
    """
    try:
        with open(path or DEFAULT_MODEL_PROFILES_PATH, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def resolve_model_profile(model_name, config_profiles, defaults):
    """Merge the checked-in profile for a model with any config override."""
    merged = dict(as_profile_mapping((defaults or {}).get(model_name)))
    merged.update(as_profile_mapping((config_profiles or {}).get(model_name)))
    return merged


def resolve_chunk_size(cli_value, config_value, model_value=None):
    """Resolve the effective chunk size (CLI overrides config) and validate it.
    Guards BOTH sources (finding #14): a bad config chunk_size previously slipped
    through because only the CLI value was checked. Raises ValueError on < 1."""
    chunk_size = (cli_value if cli_value is not None else
                  model_value if model_value is not None else config_value)
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(f"chunk_size must be an integer >= 1 (got {chunk_size!r})")
    return chunk_size


def iter_unique_entry_batches(entries, batch_size=BATCH_SIZE):
    """Yield index/entry batches with unique normalized text.

    Each consecutive `batch_size` window is greedily colored into the fewest
    duplicate-free calls. Unlike stopping at the first repeated short line, this
    keeps the other entries in the window batched and preserves bounded source
    locality. Returned indices let callers restore source order."""
    for window_start in range(0, len(entries), batch_size):
        batches = []
        for index in range(window_start, min(window_start + batch_size, len(entries))):
            entry = entries[index]
            if not isinstance(entry, dict):
                continue
            key = normalize_text(str(entry.get("text") or ""))
            for batch, seen in batches:
                if key not in seen:
                    batch.append((index, entry))
                    seen.add(key)
                    break
            else:
                batches.append(([(index, entry)], {key}))
        for batch, _ in batches:
            yield batch


MIN_ROSTER_ATTESTATIONS = 3


def build_roster(entries, source_text=None):
    """Ordered unique UPPERCASE speaker names seen so far, excluding NARRATOR and
    the UNKNOWN placeholder — fed to pass 2 for naming consistency.

    When source_text is given, a name must also be attested in the prose. The
    roster is the propagation vector for a hallucinated speaker: once a bad name
    is in it, every later batch is told that name is an established character.
    On mushoku16 a single invention at entry 11 spread to entry 1,106. Gating
    admission contains the damage to the one entry that produced it.
    """
    roster = []
    for entry in entries:
        speaker = (entry.get("speaker") or "").strip().upper()
        if not speaker or speaker in ("NARRATOR", "UNKNOWN") or speaker in roster:
            continue
        if not is_attested_name(speaker, source_text,
                                MIN_ROSTER_ATTESTATIONS):
            continue
        roster.append(speaker)
    return roster


def default_instruct(entry):
    speaker = (entry.get("speaker") or "").strip().upper()
    return NARRATOR_DEFAULT_INSTRUCT if speaker == "NARRATOR" else CHARACTER_DEFAULT_INSTRUCT


def get_deterministic_named_entry(entry):
    """Resolve entries whose speaker is explicit without invoking the LLM."""
    if entry.get("type") == "NARRATOR":
        return {"speaker": "NARRATOR", "text": entry["text"]}
    if is_nonverbal_text(entry.get("text")):
        return {"speaker": "NARRATOR", "text": entry["text"]}
    if entry.get("source_label"):
        label = str(entry["source_label"]).strip().rstrip(":")
        return {"speaker": (label.upper() if label.strip("?") else "UNKNOWN"),
                "text": entry["text"]}
    return None


class PassExhausted(Exception):
    """A pass-2/3 batch could not produce valid output within its retry budget.
    In testing mode (on_exhaustion='fail') this aborts the book so the real
    failure rate is visible."""


def attribute_batch(client, model_name, frozen_batch, params, roster,
                    max_retries=3, on_exhaustion="fail", neighbor_contexts=None,
                    attempt_observer=None, source_text=None):
    """Assign speakers to one batch of frozen {type,text} entries. Enforces the
    text freeze; retries on invalid output. On exhaustion: 'fail' raises
    PassExhausted (testing default); 'fallback' keeps frozen text and labels
    unresolved SPOKEN spans UNKNOWN via stabilize_speaker_identities."""
    sys_prompt, usr_template = load_attribute_prompts()
    if params.system_prompt:
        sys_prompt = params.system_prompt
    if params.user_prompt_template:
        usr_template = params.user_prompt_template
    neighbor_contexts = neighbor_contexts or [{} for _ in frozen_batch]
    batch_json = json.dumps([
        {"n": i, "type": e["type"], "text": e["text"], **neighbor_contexts[i]}
        for i, e in enumerate(frozen_batch)], ensure_ascii=False)
    user_prompt = usr_template.format(roster=", ".join(roster) or "(none yet)",
                                      batch=batch_json)
    validated = {}

    def validate(entries):
        report = validate_attribution(frozen_batch, entries, source_text)
        if report["passed"]:
            validated["ordered"] = index_head_check(frozen_batch, entries)[2]
        return report

    call_params = replace(params, temperature=(params.attribute_temperature
                                               if params.attribute_temperature is not None
                                               else params.temperature))
    named = call_llm_for_entries(
        client, model_name, sys_prompt, user_prompt, call_params,
        log_name="llm_responses.log", label="ATTRIBUTE", max_retries=max_retries,
        validate_entries=validate, attempt_observer=attempt_observer)
    if named:
        # The model returned only {n, head, speaker} (never full text, so it can't
        # corrupt it). Bind by the validated index order and keep the frozen text
        # byte-exact; take only the assigned speaker.
        ordered = validated.get("ordered")
        if ordered is None:
            raise RuntimeError("validated attribution response lost its index binding")
        return [{**{k: v for k, v in f.items() if k != "type"},
                 "speaker": item.get("speaker")}
                for f, item in zip(frozen_batch, ordered)]
    if on_exhaustion == "fail":
        raise PassExhausted(f"attribution failed for a {len(frozen_batch)}-entry batch")
    seeded = [{**{k: v for k, v in e.items() if k != "type"},
               "speaker": "NARRATOR" if e["type"] == "NARRATOR" else "UNKNOWN"}
              for e in frozen_batch]
    return stabilize_speaker_identities(seeded, established_speakers=roster)["entries"]


def build_instruct_request(prior_batch, params, neighbor_contexts=None):
    """Build the canonical pass-3 system and user prompts."""
    sys_prompt, usr_template = load_instruct_prompts()
    if params.system_prompt:
        sys_prompt = params.system_prompt
    if params.user_prompt_template:
        usr_template = params.user_prompt_template
    neighbor_contexts = neighbor_contexts or [{} for _ in prior_batch]
    batch_json = json.dumps([
        {"n": i, "speaker": e["speaker"], "text": e["text"], **neighbor_contexts[i]}
        for i, e in enumerate(prior_batch)], ensure_ascii=False)
    return sys_prompt, usr_template.format(batch=batch_json)


def instruct_batch(client, model_name, prior_batch, params, max_retries=3,
                   neighbor_contexts=None, exhaustion_sink=None,
                   attempt_observer=None):
    """Add instruct to one batch of {speaker,text} entries. Enforces the freeze
    on text+speaker. On exhaustion, attaches a default instruct per entry so
    pass 3 never fails the book."""
    sys_prompt, user_prompt = build_instruct_request(
        prior_batch, params, neighbor_contexts)
    validated = {}

    def validate(entries):
        report = validate_instruct(prior_batch, entries)
        if report["passed"]:
            validated["ordered"] = index_head_check(prior_batch, entries)[2]
        return report

    call_params = replace(params, temperature=(params.instruct_temperature
                                               if params.instruct_temperature is not None
                                               else params.temperature))
    annotated = call_llm_for_entries(
        client, model_name, sys_prompt, user_prompt, call_params,
        log_name="llm_responses.log", label="INSTRUCT", max_retries=max_retries,
        validate_entries=validate, attempt_observer=attempt_observer)
    if annotated:
        # The model returned only {n, head, instruct}. Keep speaker+text byte-exact
        # from prior (bound by validated index order); take only the instruct.
        ordered = validated.get("ordered")
        if ordered is None:
            raise RuntimeError("validated instruct response lost its index binding")
        return [{**p, "instruct": item.get("instruct")}
                for p, item in zip(prior_batch, ordered)]
    if exhaustion_sink is not None:
        exhaustion_sink.append(True)
    return [{**e, "instruct": default_instruct(e)} for e in prior_batch]


def does_instruct_batch_fit_context(prior_batch, params, neighbor_contexts=None):
    """Return whether an instruction request has room for a plausible response."""
    sys_prompt, user_prompt = build_instruct_request(
        prior_batch, params, neighbor_contexts)
    messages = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}]
    try:
        available = get_effective_max_tokens(
            params.max_tokens, params.context_length, messages,
            params.hard_max_tokens, scale_to_context=False)
    except TokenBudgetError:
        return False
    return available >= max(256, 48 * len(prior_batch))


def _call_segment(client, model_name, chunk, sys_prompt, user_prompt, params,
                  label, max_retries, near_miss_sink, validate=None,
                  attempt_observer=None, retry_decider=None):
    """Shared body for every pass-1 segment call (plain and context-rescue):
    the segment repair transform, the segment fidelity gate (optionally wrapped),
    and trigram-only near-miss capture. Callers build sys_prompt/user_prompt so
    the two paths can't diverge in how they invoke the gate (findings #10, #11)."""
    if validate is None:
        validate = lambda entries: validate_segment_quality(chunk, entries)
    # Segmentation only adds small JSON/type overhead around source text. Bound
    # both the first request and retry ceiling so a weak model cannot spend
    # 10k-16k tokens expanding a ~1k-token source chunk.
    source_words = max(1, len(chunk.split()))
    completion_ceiling = resolve_completion_ceiling(
        source_words, params, reasoning_allowance=params.reasoning_allowance)
    bounded_params = replace(
        params, max_tokens=min(params.max_tokens, completion_ceiling),
        hard_max_tokens=min(params.hard_max_tokens, completion_ceiling),
        temperature=(params.segment_temperature
                     if params.segment_temperature is not None else params.temperature))
    def repair(entries):
        repaired = build_deterministic_repair(
            entries, chunk, merge_empty_into_pause=False)
        quote_split = []
        for number, entry in enumerate(repaired["entries"], 1):
            text = str(entry.get("text") or "").strip()
            if entry.get("type") != "SPOKEN":
                if any(char in text for char in ('"', '“', '”')):
                    parts, current, quoted = [], [], False
                    for char in text:
                        opens = char in ('"', '“') and not quoted
                        closes = char in ('"', '”') and quoted
                        if opens or closes:
                            part = "".join(current).strip()
                            if part:
                                parts.append({**entry, "type": "SPOKEN" if quoted
                                              else "NARRATOR", "text": part})
                            current = []
                            quoted = not quoted
                        else:
                            current.append(char)
                    part = "".join(current).strip()
                    if part:
                        parts.append({**entry, "type": "SPOKEN" if quoted
                                      else "NARRATOR", "text": part})
                    if not quoted and len(parts) > 1:
                        quote_split.extend(parts)
                        repaired.setdefault("changes", []).append({
                            "entry_number": number, "code": "split_mixed_quote_regions"})
                        continue
            elif ((text.startswith('"') and text.endswith('"'))
                  or (text.startswith('“') and text.endswith('”'))):
                entry = {**entry, "text": text[1:-1]}
                repaired.setdefault("changes", []).append({
                    "entry_number": number, "code": "stripped_dialogue_delimiters"})
            quote_split.append(entry)
        repaired["entries"] = quote_split
        return repaired

    return call_llm_for_entries(
        client, model_name, sys_prompt, user_prompt, bounded_params,
        log_name="llm_responses.log", label=label, max_retries=max_retries,
        # Same deterministic structural repair (unicode-homoglyph fixups) the
        # single-pass path runs before its gate, so pass 1 doesn't waste a retry
        # on issues single-pass silently repairs. build_deterministic_repair is
        # text-only, so it applies unchanged to the {type,text} segment shape.
        # merge_empty_into_pause=False so empty units reach the gate (finding #7).
        transform_entries=repair,
        validate_entries=validate,
        attempt_observer=attempt_observer,
        retry_decider=retry_decider,
        near_miss_sink=near_miss_sink)


def segment_chunk(client, model_name, chunk, params, max_retries=4,
                  near_miss_sink=None, failure_sink=None, attempt_sink=None):
    """Pass 1 single attempt-budget over one chunk -> [{type,text}], via the
    segment fidelity gate. Captures a trigram-only near-miss into near_miss_sink
    (same mechanism call_llm_for_entries uses for single-pass). Returns [] on
    exhaustion."""
    sys_prompt, usr_template = load_segment_prompts()
    if params.system_prompt:
        sys_prompt = params.system_prompt
    if params.user_prompt_template:
        usr_template = params.user_prompt_template
    user_prompt = usr_template.format(chunk=chunk)
    attempts = []
    def observe(attempt):
        attempts.append(attempt)
        if attempt_sink is not None:
            attempt_sink.append(attempt)
    def decide(quality, repeat_evidence):
        codes = {finding.get("code") for finding in quality.get("findings", [])}
        splittable = {"low_source_token_recall", "low_ordered_trigram_recall",
                      "output_source_ratio", "mixed_quote_region",
                      "quote_region_misclassified", "crosses_quote_boundary"}
        return "split" if repeat_evidence >= 2 and codes & splittable else "retry"
    entries = _call_segment(
        client, model_name, chunk, sys_prompt, user_prompt, params, "SEGMENT",
        max_retries, near_miss_sink, attempt_observer=observe,
        retry_decider=decide)
    if not entries and failure_sink is not None and attempts:
        failure_sink[:] = [set(attempts[-1].get("failure_codes") or [])]
    return entries


def _accept_segment_near_miss(near_miss):
    if not near_miss:
        return []
    entries, quality = near_miss[0]
    print("  SEGMENT accepted as trigram-only near-miss "
          f"(ordered_trigram_recall={quality['metrics']['ordered_trigram_recall']})")
    return entries


def _resolved_near_miss(near_miss, resolution_sink):
    """Accept the exhaustion near-miss (if any) and record the resolution."""
    entries = _accept_segment_near_miss(near_miss)
    _record_resolution(resolution_sink, "near_miss" if entries else "fail")
    return entries


def segment_chunk_adaptively(client, model_name, chunk, params,
                             resolution_sink=None, failure_sink=None,
                             attempt_sink=None, quote_analysis=None):
    """Pass 1 with the full safety net: full-chunk attempt, then a
    natural-boundary split whose halves each recurse, and exhaustion-only
    trigram-only near-miss acceptance. Mirrors process_chunk_adaptively but for
    the segment gate. Returns [{type,text}] or [] (book failure). When
    resolution_sink is given, appends exactly one resolution string describing
    how the chunk was handled (clean / adaptive_split / recombination_near_miss /
    near_miss / fail). Only the top-level call should pass a sink; recursive
    part-calls do not, so inner resolutions don't pollute the record."""
    if params.presegment_quotes:
        quote_analysis = quote_analysis or analyze_outer_quote_regions(chunk)
        regions = quote_analysis["regions"]
        if len(regions) > 1:
            # Outer quotes already answer the only pass-1 question: inside is
            # spoken, outside is narration. Do not ask the model to rewrite
            # tiny attribution regions; live testing showed that invites
            # hallucinated expansion despite perfect source coverage.
            if validate_segment_quality(
                    chunk, regions, quote_analysis=quote_analysis)["passed"]:
                resolution = ("quote_presegmented_repaired"
                              if quote_analysis["repairs"]
                              else "quote_presegmented_continuation"
                              if (quote_analysis.get("initial_depth") or
                                  quote_analysis.get("final_depth"))
                              else "quote_presegmented")
                _record_resolution(resolution_sink, resolution)
                return regions
    near_miss = []
    local_failures = []
    entries = segment_chunk(client, model_name, chunk, params,
                            near_miss_sink=near_miss, failure_sink=local_failures,
                            attempt_sink=attempt_sink)
    if entries:
        _record_resolution(resolution_sink, "clean")
        return entries
    parts = split_failed_chunk(chunk)
    if not parts:
        if failure_sink is not None:
            failure_sink[:] = local_failures
        return _resolved_near_miss(near_miss, resolution_sink)
    print(f"  Adaptive split (segment): -> {len(parts[0])} + {len(parts[1])} chars")
    combined, any_failed = [], False
    for part in parts:
        part_entries = segment_chunk_adaptively(
            client, model_name, part, params, attempt_sink=attempt_sink)
        if not part_entries:
            any_failed = True
            continue
        combined.extend(part_entries)
    if any_failed:
        if failure_sink is not None:
            failure_sink[:] = local_failures
        return _resolved_near_miss(near_miss, resolution_sink)
    combined_quality = validate_segment_quality(chunk, combined)
    if not combined_quality["passed"]:
        codes = {f.get("code") for f in combined_quality["findings"]}
        m = combined_quality["metrics"]
        # Both halves already passed their own segment gate (we only reach here
        # when any_failed is False), so the recombined whole has adequate content
        # coverage. A whole-chunk trigram dip when trigram is the ONLY defect is a
        # split-seam artifact, not lost content - accept it if it still clears the
        # trigram-only near-miss floor rather than discarding two good halves.
        # Recall / ratio / cyrillic / duplicate defects are NOT waived (real).
        if is_trigram_only_near_miss(combined_quality):
            print(f"  Adaptive split (segment) recombination accepted: both halves "
                  f"passed, trigram-only near-miss at seam "
                  f"(trigram={m['ordered_trigram_recall']} recall={m['source_token_recall']})")
            _record_resolution(resolution_sink, "recombination_near_miss")
            return combined
        # Diagnostic: log exactly why a recombination was rejected so we can tell
        # trigram-seam brittleness from real content loss / duplication.
        print(f"  Adaptive split (segment) recombination REJECTED: codes={sorted(codes)} "
              f"metrics={m}")
        return _resolved_near_miss(near_miss, resolution_sink)
    _record_resolution(resolution_sink, "adaptive_split")
    return combined


def should_rescue_with_context(failure_codes):
    """Context is not a remedy for omission, truncation, or quote structure."""
    return bool(set(failure_codes or ()) & {"context_required"})


def select_preflight_chunks(source_text, chunk_size):
    """Select distinct first, middle, and dialogue-dense real-book chunks."""
    chunks = split_into_chunks(source_text, max_size=chunk_size)
    if not chunks:
        return []
    selected = [("first", 0)]
    middle = len(chunks) // 2
    if middle:
        selected.append(("middle", middle))
    # Endnote/reference sections often contain more quoted terms than the story
    # itself. They are useful source material, but are not a representative
    # dialogue qualification sample.
    prose_candidates = [i for i, chunk in enumerate(chunks)
                        if chunk.count("←") <= 2]
    dialogue_pool = prose_candidates or list(range(len(chunks)))
    dialogue = max(dialogue_pool, key=lambda i: sum(
        chunks[i].count(mark) for mark in ('"', '“', '”')))
    if dialogue not in {index for _, index in selected}:
        selected.append(("dialogue", dialogue))
    return [(label, index, chunks[index]) for label, index in selected]


# Escalating context windows (chars of surrounding source) tried, in order, as a
# last resort when a chunk exhausts normal retries + adaptive split. Defaults;
# overridable via generation config (context_rescue_windows / _retries).
_CONTEXT_RESCUE_WINDOWS = (2000, 4000, 6000)
_CONTEXT_RESCUE_MAX_RETRIES = 2
_CONTEXT_SEGMENT_USER = (
    "The text between the CONTEXT markers below is surrounding material from the "
    "same book, given ONLY as reference for narrative flow and continuity. DO NOT "
    "convert it and DO NOT include any of it in your output.\n\n"
    "=== CONTEXT BEFORE (reference only) ===\n{before}\n=== END CONTEXT ===\n\n"
    "=== CONTEXT AFTER (reference only) ===\n{after}\n=== END CONTEXT ===\n\n"
    "Now convert ONLY the SOURCE TEXT below into the JSON array of "
    '{{"type","text"}} units. Your output must cover exactly the SOURCE TEXT and '
    "nothing from the context.\n\nSOURCE TEXT:\n{chunk}"
)


_CONTEXT_BLEED_MIN_CHARS = 40


def _output_has_context_bleed(entries, chunk, before, after):
    """True if any entry's text clearly leaked from the reference context: it
    appears (normalized) in before+after but NOT in the target chunk. Conservative
    - only entries with >= _CONTEXT_BLEED_MIN_CHARS of normalized text count, so a
    short generic line ("Yes.") that legitimately recurs in both context and chunk
    doesn't trip a false rejection."""
    chunk_norm = normalize_text(chunk)
    context_norm = normalize_text((before or "") + " " + (after or ""))
    if not context_norm:
        return False
    context_tokens = context_norm.split()
    chunk_tokens = chunk_norm.split()
    chunk_spans = {tuple(chunk_tokens[i:i + 8])
                   for i in range(max(0, len(chunk_tokens) - 7))}
    context_spans = {tuple(context_tokens[i:i + 8])
                     for i in range(max(0, len(context_tokens) - 7))}
    for entry in entries:
        text_norm = normalize_text(str((entry or {}).get("text") or "")
                                   if isinstance(entry, dict) else "")
        if (len(text_norm) >= _CONTEXT_BLEED_MIN_CHARS
                and text_norm in context_norm and text_norm not in chunk_norm):
            return True
        entry_tokens = text_norm.split()
        entry_spans = {tuple(entry_tokens[i:i + 8])
                       for i in range(max(0, len(entry_tokens) - 7))}
        if (entry_spans & context_spans) - chunk_spans:
            return True
    return False


def segment_chunk_with_context(client, model_name, chunk, before, after, params,
                               max_retries=2, near_miss_sink=None):
    """Last-resort pass-1 retry: give the model surrounding SOURCE text (before /
    after the failing chunk, reference-only) for narrative flow, but validate that
    the output still covers ONLY the target chunk. Captures a trigram-only
    near-miss into near_miss_sink like the normal segment path. Returns
    [{type,text}] or []."""
    sys_prompt, _ = load_segment_prompts()
    if params.system_prompt:
        sys_prompt = params.system_prompt
    user_prompt = _CONTEXT_SEGMENT_USER.format(before=before or "(start of book)",
                                               after=after or "(end of book)",
                                               chunk=chunk)

    def validate(entries):
        # Fidelity gate PLUS a context-bleed guard: a target-correct output that
        # also pastes a reference-context sentence can otherwise pass recall /
        # trigram / ratio (the leaked sentence adds output but doesn't drop source
        # recall), so reject clear context-only entries as a validation failure.
        report = validate_segment_quality(chunk, entries)
        if _output_has_context_bleed(entries, chunk, before, after):
            report = dict(report)
            report["passed"] = False
            report["findings"] = list(report["findings"]) + [{
                "code": "context_bleed",
                "message": "An entry reproduced reference-context text absent from the target chunk."}]
        return report

    return _call_segment(client, model_name, chunk, sys_prompt, user_prompt,
                         params, "SEGMENT+CTX", max_retries, near_miss_sink,
                         validate=validate)


def _tail_join(parts, limit):
    """Join just enough trailing parts to cover `limit` chars from the end,
    instead of materializing the whole list (finding #9). The result may slightly
    exceed limit (the boundary part isn't cut); callers slice [-window:]."""
    acc, total = [], 0
    for part in reversed(parts):
        acc.append(part)
        total += len(part)
        if total >= limit:
            break
    return "".join(reversed(acc))


def _head_join(parts, limit):
    """Join just enough leading parts to cover `limit` chars from the start."""
    acc, total = [], 0
    for part in parts:
        acc.append(part)
        total += len(part)
        if total >= limit:
            break
    return "".join(acc)


def _rescue_prompt_fits(chunk, before, after, overhead_chars, params):
    """Estimate whether a context-rescue prompt for this window fits the model's
    context, leaving room to emit the chunk. Uses the pipeline's chars//3 token
    estimate. Returns True when context_length is unknown (keep prior behavior)."""
    ctx_len = getattr(params, "context_length", None)
    if not ctx_len:
        return True
    prompt_tokens = math.ceil((overhead_chars + len(chunk) + len(before) + len(after)) / 3)
    # Segment output reproduces the chunk text wrapped in JSON; reserve ~1.5x the
    # chunk's token estimate plus a small structural margin as the output budget.
    output_budget = math.ceil(len(chunk) / 3 * 1.5) + 64
    return prompt_tokens + output_budget <= ctx_len


def rescue_chunk_with_context(client, model_name, chunks, index, params,
                              resolution_sink=None, windows=None, max_retries=None):
    """When chunk `index` fails normal segmentation, retry it with escalating
    surrounding-source context. Accepts a clean pass, else the best trigram-only
    near-miss any window produced. Returns entries or []. When resolution_sink is
    given, appends the resolution (context_rescue:<window> /
    context_rescue_near_miss / fail). Windows whose prompt would exceed the
    model's context budget are skipped (finding #4). `windows` and `max_retries`
    default to the module constants when None (finding #12: config-tunable)."""
    windows = windows or _CONTEXT_RESCUE_WINDOWS
    if max_retries is None:
        max_retries = _CONTEXT_RESCUE_MAX_RETRIES
    max_window = max(windows)
    before_all = _tail_join(chunks[:index], max_window)
    after_all = _head_join(chunks[index + 1:], max_window)
    sys_prompt, _ = load_segment_prompts()
    if params.system_prompt:
        sys_prompt = params.system_prompt
    overhead_chars = len(sys_prompt) + len(_CONTEXT_SEGMENT_USER)
    best_near_miss = []  # holds the single best [(entries, quality)] seen so far
    for window in windows:
        before, after = before_all[-window:], after_all[:window]
        if not _rescue_prompt_fits(chunks[index], before, after, overhead_chars, params):
            print(f"  context rescue {window}-char window skipped "
                  "(prompt would exceed context budget)")
            continue
        near_miss = []
        seg = segment_chunk_with_context(
            client, model_name, chunks[index],
            before, after, params, max_retries=max_retries, near_miss_sink=near_miss)
        if seg:
            print(f"  chunk {index + 1}/{len(chunks)} rescued with "
                  f"{window}-char surrounding context (clean pass)")
            _record_resolution(resolution_sink, f"context_rescue:{window}")
            return seg
        if near_miss:
            trig = near_miss[0][1]["metrics"]["ordered_trigram_recall"]
            best = (best_near_miss[0][1]["metrics"]["ordered_trigram_recall"]
                    if best_near_miss else -1.0)
            if trig > best:
                best_near_miss = near_miss
            print(f"  context rescue at {window} chars: trigram-only near-miss "
                  f"{trig} captured; escalating")
        else:
            print(f"  context rescue at {window} chars did not pass; escalating")
    if best_near_miss:
        entries, quality = best_near_miss[0]
        print(f"  chunk {index + 1}/{len(chunks)} rescued with context as "
              f"trigram-only near-miss "
              f"(ordered_trigram_recall={quality['metrics']['ordered_trigram_recall']})")
        _record_resolution(resolution_sink, "context_rescue_near_miss")
        return entries
    _record_resolution(resolution_sink, "fail")
    return []


def three_pass_checkpoint_path(output_path):
    return output_path + ".threepass_checkpoint.json"


def three_pass_manifest_path(output_path):
    return output_path + ".threepass_manifest.json"


def _resolution_counts(resolutions):
    """Roll per-chunk resolution strings up into summary counts."""
    return {
        "near_miss_accepted": sum(r == "near_miss" for r in resolutions),
        "context_rescued": sum(r.startswith("context_rescue") for r in resolutions),
        "split_recombined": sum(r in ("adaptive_split", "recombination_near_miss")
                                for r in resolutions),
        "quote_repairs": sum(r == "quote_presegmented_repaired"
                             for r in resolutions),
        "quote_continuations": sum(r == "quote_presegmented_continuation"
                                   for r in resolutions),
    }


def _write_manifest(output_path, fingerprint, resolutions, passes, status,
                    failed_pass=None, failed_chunk=None, legacy_resume=False,
                    progress=None, diagnostic_failures=None, telemetry=None):
    """Persist the run manifest next to the output so results are analyzable from
    structured data instead of log-grepping."""
    if not output_path:
        return
    manifest = {
        "fingerprint": fingerprint,
        "status": status,
        "chunks": [{"index": i + 1, "resolution": r}
                   for i, r in enumerate(resolutions)],
        "counts": _resolution_counts(resolutions),
        "passes": passes,
        "legacy_resume": legacy_resume,
        "progress": progress or {},
        "diagnostic_failures": diagnostic_failures or [],
        "telemetry": telemetry or {},
    }
    if failed_pass is not None:
        manifest["failed_pass"] = failed_pass
    if failed_chunk is not None:
        manifest["failed_chunk"] = failed_chunk
    atomic_json_write(manifest, three_pass_manifest_path(output_path))


def three_pass_fingerprint(source_text, model_name, chunk_size, params=None,
                           on_exhaustion="fail", context_windows=None,
                           context_rescue_retries=None, endpoint=None,
                           collect_all_failures=False):
    digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    settings = {
        "model_name": model_name, "chunk_size": chunk_size,
        "endpoint": endpoint, "on_exhaustion": on_exhaustion,
        "context_windows": context_windows,
        "context_rescue_retries": context_rescue_retries,
        "collect_all_failures": collect_all_failures,
        "pipeline_version": 8,
        "default_prompts_sha256": hashlib.sha256("\n".join(
            sum((list(load_segment_prompts()), list(load_attribute_prompts()),
                 list(load_instruct_prompts())), [])).encode("utf-8")).hexdigest(),
    }
    if params is not None:
        settings.update({name: getattr(params, name, None) for name in (
            "system_prompt", "user_prompt_template", "max_tokens", "temperature",
            "top_p", "top_k", "min_p", "presence_penalty", "banned_tokens",
            "context_length", "hard_max_tokens", "segment_temperature",
            "attribute_temperature", "instruct_temperature",
            "segment_output_ratio", "presegment_quotes")})
    encoded = json.dumps(settings, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {"source_sha256": digest, "settings_sha256": hashlib.sha256(encoded).hexdigest(),
            "model_name": model_name, "pipeline": "three_pass"}


def _load_three_pass_checkpoint(output_path, fingerprint):
    data = safe_load_json(three_pass_checkpoint_path(output_path), None)
    if not isinstance(data, dict) or data.get("fingerprint") != fingerprint:
        return None
    return data


def _save_three_pass_checkpoint(output_path, fingerprint, stage, segmented,
                                chunks_done, named, annotated, resolutions=None,
                                elapsed_s=None, diagnostic_failures=None):
    atomic_json_write({"fingerprint": fingerprint, "stage": stage,
                       "chunks_done": chunks_done, "segmented": segmented,
                       "named": named, "annotated": annotated,
                       "resolutions": resolutions or [],
                       "elapsed_s": elapsed_s or {},
                       "diagnostic_failures": diagnostic_failures or []},
                      three_pass_checkpoint_path(output_path))


def run_three_pass(client, model_name, source_text, params, chunk_size,
                   on_exhaustion="fail", output_path=None,
                   context_windows=None, context_rescue_retries=None, endpoint=None,
                   collect_all_failures=False, thinking_mode=None,
                   unicode_report=None, attribution_votes=1,
                   vote_temperature=0.3):
    """Full flow. Returns the assembled [{speaker,text,instruct}] list, or raises
    RuntimeError if pass 1 exhausts a chunk. When output_path is given, saves a
    checkpoint after each pass-1 chunk and each pass-2/3 batch and resumes from
    it; when None, runs purely in memory. context_windows / context_rescue_retries
    override the context-rescue defaults (finding #12)."""
    chunk_records = split_into_chunk_records(source_text, max_size=chunk_size)
    chunks = [record["text"] for record in chunk_records]
    quote_analyses = []
    quote_depth = 0
    for index, record in enumerate(chunk_records):
        analysis = analyze_outer_quote_regions(
            record["text"], initial_depth=quote_depth,
            allow_open_end=index < len(chunk_records) - 1)
        quote_analyses.append(analysis)
        quote_depth = analysis["final_depth"]
    fingerprint = three_pass_fingerprint(
        source_text, model_name, chunk_size, params, on_exhaustion,
        context_windows, context_rescue_retries, endpoint, collect_all_failures)
    state = _load_three_pass_checkpoint(output_path, fingerprint) if output_path else None
    segmented = state["segmented"] if state else []
    chunks_done = state["chunks_done"] if state else 0
    named = state["named"] if state else []
    annotated = state["annotated"] if state else []
    legacy_resume = bool(state and "resolutions" not in state)
    resolutions = (list(state.get("resolutions", [])) if state
                   else [])
    # A failed chunk may have been checkpointed for timing. It is retried on
    # resume, so discard its provisional resolution and replace it with the
    # eventual outcome instead of emitting two manifest rows for one chunk.
    resolutions = resolutions[:chunks_done]
    if len(resolutions) < chunks_done:
        resolutions.extend(["resumed"] * (chunks_done - len(resolutions)))
    elapsed_s = dict(state.get("elapsed_s", {})) if state else {}
    diagnostic_failures = list(state.get("diagnostic_failures", [])) if state else []
    # Latest attempt seen by call_llm_for_entries, so a failure record can say
    # why the batch failed instead of only which entry it was.
    last_attempts = {}

    reasoning_allowance = ReasoningAllowance()

    def record_attempt(attempt):
        last_attempts["latest"] = attempt
        # Size the next call from what this model has actually shown. Stays at
        # zero for a model that never reports reasoning_tokens, so a
        # non-reasoning model keeps exactly today's ceiling.
        reasoning_allowance.observe(
            attempt.get("reasoning_tokens"),
            truncated=attempt.get("finish_reason") == "length")
        params.reasoning_allowance = reasoning_allowance.current()

    def last_attempt_for(_index):
        return last_attempts.get("latest")
    attempts = []

    def save(stage):
        if output_path:
            _save_three_pass_checkpoint(output_path, fingerprint, stage,
                                        segmented, chunks_done, named, annotated,
                                        resolutions, elapsed_s, diagnostic_failures)
    passes = {}

    def emit_manifest(status, failed_pass=None, failed_chunk=None):
        failure_counts = Counter(
            code for attempt in attempts
            for code in (attempt.get("failure_codes") or []))
        _write_manifest(output_path, fingerprint, resolutions, passes, status,
                        telemetry={
                            "model_name": model_name,
                            "thinking_mode": thinking_mode or "default",
                            "unicode": dict(unicode_report or {}),
                            "failure_reasons": dict(Counter(
                                f.get("reason") or "unknown"
                                for f in diagnostic_failures)),
                            "truncations": sum(
                                f.get("finish_reason") == "length"
                                for f in diagnostic_failures),
                        },
                        failed_pass=failed_pass, failed_chunk=failed_chunk,
                        legacy_resume=legacy_resume, progress={
                            "source_words": len(source_text.split()),
                            "source_words_total": len(source_text.split()),
                            "source_words_completed": sum(
                                len(chunks[j].split()) for j, resolution in
                                enumerate(resolutions) if resolution != "fail"),
                            "chunks_total": len(chunks),
                            "chunks_attempted": chunks_done,
                            "chunks_completed": sum(
                                resolution != "fail" for resolution in resolutions),
                            "segmented_entries": len(segmented),
                            "attributed_entries": sum(isinstance(e, dict) for e in named),
                            "instructed_entries": sum(isinstance(e, dict) for e in annotated),
                            "llm_calls": len(attempts),
                            "repeated_responses": sum(
                                a.get("response_repeat_count", 0) > 1 for a in attempts),
                            "completion_tokens": sum(
                                a.get("completion_tokens") or 0 for a in attempts),
                            "failure_codes": dict(sorted(failure_counts.items())),
                            "response_fingerprints": len({
                                a.get("response_fingerprint") for a in attempts
                                if a.get("response_fingerprint")}),
                        }, diagnostic_failures=diagnostic_failures)

    # Pass 1 — resume from chunks_done.
    seg_start = time.time()
    seg_base = elapsed_s.get("segment", 0)
    # resolve_completion_ceiling is consumed ONLY by pass 1, so pass 1 has to be
    # what feeds the allowance. Wiring it to the pass-2/3 observers alone left it
    # at zero for the pass that uses it, and thinking-on segmentation kept
    # reporting "cannot grow beyond 2700" - the visible-output budget with no
    # room for reasoning at all.
    observed_attempts = 0
    for i in range(chunks_done, len(chunks)):
        sink = []
        failures = []
        seg = segment_chunk_adaptively(client, model_name, chunks[i], params,
                                       resolution_sink=sink, failure_sink=failures,
                                       attempt_sink=attempts,
                                       quote_analysis=quote_analyses[i])
        for attempt in attempts[observed_attempts:]:
            reasoning_allowance.observe(
                attempt.get("reasoning_tokens"),
                truncated=attempt.get("finish_reason") == "length")
        observed_attempts = len(attempts)
        params.reasoning_allowance = reasoning_allowance.current()
        if not seg and should_rescue_with_context(failures[0] if failures else set()):
            # Last resort: retry with escalating surrounding-source context.
            print(f"  chunk {i + 1}/{len(chunks)} failed normal segmentation; "
                  "trying escalating surrounding-source context")
            seg = rescue_chunk_with_context(client, model_name, chunks, i, params,
                                            resolution_sink=sink,
                                            windows=context_windows,
                                            max_retries=context_rescue_retries)
        resolutions.append(sink[-1] if sink else ("clean" if seg else "fail"))
        if not seg:
            if collect_all_failures:
                diagnostic_failures.append(build_segment_failure_record(
                    i + 1, chunks[i], failures[0] if failures else []))
                chunks_done = i + 1
                elapsed_s["segment"] = seg_base + time.time() - seg_start
                save("segment_incomplete")
                continue
            elapsed_s["segment"] = seg_base + time.time() - seg_start
            passes["segment"] = {"elapsed_s": round(elapsed_s["segment"], 3),
                                 "status": "failed"}
            save("segment_failed")
            emit_manifest("failed", failed_pass="segment", failed_chunk=i + 1)
            raise RuntimeError(f"pass 1 (segment) failed on chunk {i + 1}/{len(chunks)}")
        segmented.extend(seg)
        chunks_done = i + 1
        elapsed_s["segment"] = seg_base + time.time() - seg_start
        save("segment")
    elapsed_s["segment"] = seg_base + time.time() - seg_start
    passes["segment"] = {"elapsed_s": round(elapsed_s["segment"], 3),
                         "status": ("incomplete" if any(
                             f["pass"] == "segment" for f in diagnostic_failures)
                             else "complete")}
    # Pass 2 — deterministic duplicate-free batches, restored to source order.
    # Maintain a running roster (set for O(1) membership + list for order) updated
    # per batch, instead of rescanning the whole `named` prefix every batch.
    named.extend([None] * (len(segmented) - len(named)))
    for index, entry in enumerate(segmented):
        if named[index] is not None:
            continue
        named[index] = get_deterministic_named_entry(entry)
    roster = build_roster(
        (entry for entry in named if isinstance(entry, dict)), source_text)
    roster_seen = set(roster)
    attr_start = time.time()
    attr_base = elapsed_s.get("attribute", 0)
    try:
        for indexed_batch in iter_unique_entry_batches(segmented):
            pending = [(index, entry) for index, entry in indexed_batch
                       if named[index] is None and not any(
                           f["pass"] == "attribute" and f.get("entry") == index
                           for f in diagnostic_failures)]
            if not pending:
                continue
            work = [pending]
            while work:
                current = work.pop(0)
                batch = [entry for _, entry in current]
                contexts = [{"previous_context": segmented[index - 1] if index else None,
                             "next_context": segmented[index + 1]
                             if index + 1 < len(segmented) else None}
                            for index, _ in current]
                try:
                    new_named, vote_confidences = attribute_batch_voted(
                        client, model_name, batch, params, roster=roster,
                        votes=attribution_votes,
                        vote_temperature=vote_temperature,
                        on_exhaustion=on_exhaustion, neighbor_contexts=contexts,
                        attempt_observer=record_attempt,
                        source_text=source_text)
                except PassExhausted:
                    if len(current) == 1:
                        if collect_all_failures:
                            index, entry = current[0]
                            diagnostic_failures.append(build_failure_record(
                                "attribute", index, entry["text"],
                                last_attempt_for(index)))
                            save("attribute_incomplete")
                            continue
                        raise
                    midpoint = len(current) // 2
                    print(f"  Attribution batch exhausted; subdividing "
                          f"{len(current)} -> {midpoint} + {len(current) - midpoint}")
                    work[0:0] = [current[:midpoint], current[midpoint:]]
                    continue
                for position, ((index, _), entry) in enumerate(
                        zip(current, new_named)):
                    if attribution_votes > 1 and position < len(vote_confidences):
                        # Only present when voting, so a default run's output
                        # shape is unchanged.
                        entry = {**entry,
                                 "attribution_confidence": round(
                                     vote_confidences[position], 3)}
                    named[index] = entry
                if on_exhaustion == "fallback":
                    roster = build_roster(
                        (entry for entry in named if isinstance(entry, dict)),
                        source_text)
                    roster_seen = set(roster)
                else:
                    for entry in new_named:
                        speaker = (entry.get("speaker") or "").strip().upper()
                        if (speaker and speaker not in ("NARRATOR", "UNKNOWN")
                                and speaker not in roster_seen):
                            roster_seen.add(speaker)
                            roster.append(speaker)
                elapsed_s["attribute"] = attr_base + time.time() - attr_start
                # Each accepted subdivision is durable; a later single-entry
                # failure resumes after this work instead of replaying the batch.
                save("attribute")
    except PassExhausted:
        elapsed_s["attribute"] = attr_base + time.time() - attr_start
        passes["attribute"] = {"elapsed_s": round(elapsed_s["attribute"], 3),
                               "status": "failed"}
        save("attribute_failed")
        emit_manifest("failed", failed_pass="attribute")
        raise
    elapsed_s["attribute"] = attr_base + time.time() - attr_start
    passes["attribute"] = {"elapsed_s": round(elapsed_s["attribute"], 3),
                           "status": ("incomplete" if any(
                               f["pass"] == "attribute" for f in diagnostic_failures)
                               else "complete")}
    # Pass 3 uses the same duplicate-free scheduling so ambiguous heads cannot
    # slip through there either (finding #5).
    annotated.extend([None] * (len(named) - len(annotated)))
    for index, entry in enumerate(named):
        if (annotated[index] is None and isinstance(entry, dict)
                and is_nonverbal_text(entry.get("text"))):
            annotated[index] = {**entry, "instruct": default_instruct(entry)}
    inst_start = time.time()
    inst_base = elapsed_s.get("instruct", 0)
    for indexed_batch in iter_unique_entry_batches(named):
        pending = [(index, entry) for index, entry in indexed_batch
                   if annotated[index] is None]
        if not pending:
            continue
        work = [pending]
        while work:
            current = work.pop(0)
            batch = [entry for _, entry in current]
            contexts = [{"previous_context": named[index - 1] if index else None,
                         "next_context": named[index + 1]
                         if index + 1 < len(named) else None}
                        for index, _ in current]
            if (len(current) > 1
                    and not does_instruct_batch_fit_context(batch, params, contexts)):
                midpoint = len(current) // 2
                print(f"  Instruction batch exceeds context budget; subdividing "
                      f"{len(current)} -> {midpoint} + {len(current) - midpoint}")
                work[0:0] = [current[:midpoint], current[midpoint:]]
                continue
            exhausted = []
            new_annotated = instruct_batch(
                client, model_name, batch, params, neighbor_contexts=contexts,
                exhaustion_sink=exhausted, attempt_observer=record_attempt)
            if exhausted and collect_all_failures and len(current) > 1:
                midpoint = len(current) // 2
                print(f"  Instruction batch exhausted; subdividing "
                      f"{len(current)} -> {midpoint} + {len(current) - midpoint}")
                work[0:0] = [current[:midpoint], current[midpoint:]]
                continue
            if exhausted and collect_all_failures:
                index, entry = current[0]
                diagnostic_failures.append(build_failure_record(
                    "instruct", index, entry["text"], last_attempt_for(index)))
            for (index, _), entry in zip(current, new_annotated):
                annotated[index] = entry
            elapsed_s["instruct"] = inst_base + time.time() - inst_start
            save("instruct")
    elapsed_s["instruct"] = inst_base + time.time() - inst_start
    passes["instruct"] = {"elapsed_s": round(elapsed_s["instruct"], 3),
                          "status": ("incomplete" if any(
                              f["pass"] == "instruct" for f in diagnostic_failures)
                              else "complete")}
    save("done")
    emit_manifest("incomplete" if diagnostic_failures else "complete")
    return [entry for entry in annotated if isinstance(entry, dict)]


def build_segment_failure_record(chunk_number, chunk_text, failure_codes):
    """Build a pass-1 failure record.

    Pass 1 fails per source chunk rather than per entry, so it cannot reuse
    build_failure_record's entry shape. It carries the same "reason" key so the
    manifest's failure_reasons rollup counts pass-1 failures instead of
    bucketing them all as unknown.
    """
    codes = sorted(failure_codes or [])
    return {"pass": "segment", "chunk": chunk_number,
            "source_sha256": hashlib.sha256(chunk_text.encode()).hexdigest(),
            "source_characters": len(chunk_text),
            "source_preview": chunk_text[:500],
            "failure_codes": codes,
            "reason": codes[0] if codes else "segment_exhausted"}


# Fixed, distinct seeds so a vote is reproducible: the same input yields the
# same samples and therefore the same majority on every run.
_VOTE_SEED_BASE = 1000


def vote_seeds(count):
    """Return `count` distinct, stable seeds for one voted attribution."""
    return [_VOTE_SEED_BASE + index for index in range(count)]


def majority_vote(votes):
    """Return (winner, confidence) for one entry's ballots.

    Confidence is the winning share, which is the signal greedy decoding cannot
    produce: measured on mushoku16, greedy agreed with a unanimous vote 81% of
    the time but with a split vote only 42%, so a split reliably marks a
    contested line. With no majority the first sample wins, keeping the result
    deterministic rather than dependent on tie ordering.
    """
    real = [v for v in votes if v is not None]
    if not real:
        return None, 0.0
    tally = Counter(real)
    best = max(tally.values())
    for vote in real:                      # first sample with the top count
        if tally[vote] == best:
            return vote, best / len(real)
    return real[0], best / len(real)


def attribute_batch_voted(client, model_name, frozen_batch, params, roster,
                          votes=1, vote_temperature=0.3, **kwargs):
    """Attribute one batch, optionally by majority vote across seeded samples.

    votes=1 is the greedy path and is byte-identical to calling attribute_batch
    directly, so this is inert unless voting is switched on.

    Above 1, the batch is attributed once per fixed seed at vote_temperature
    and the majority wins per entry. Greedy commits to a single path with no
    way to correct itself: measured on mushoku16, in 30 of 49 disagreements it
    chose a speaker that not one of three samples picked, and in one scene it
    scattered four lines addressed to Rudi across two characters while every
    sample said ROXY.

    Returns (entries, confidences). Each confidence is the winning share, so a
    caller can flag contested lines - a signal greedy cannot produce.
    """
    if votes <= 1:
        result = attribute_batch(client, model_name, frozen_batch, params,
                                 roster, **kwargs)
        return result, [1.0] * len(result or [])

    ballots = []
    for seed in vote_seeds(votes):
        sampled = replace(params, seed=seed, temperature=vote_temperature,
                          attribute_temperature=vote_temperature)
        result = attribute_batch(client, model_name, frozen_batch, sampled,
                                 roster, **kwargs)
        if result and len(result) == len(frozen_batch):
            ballots.append(result)
    if not ballots:
        return [], []
    if len(ballots) == 1:
        return ballots[0], [1.0] * len(ballots[0])

    entries, confidences = [], []
    for position in range(len(ballots[0])):
        speakers = [ballot[position].get("speaker") for ballot in ballots]
        winner, confidence = majority_vote(speakers)
        entry = dict(ballots[0][position])
        entry["speaker"] = winner
        entries.append(entry)
        confidences.append(confidence)
    return entries, confidences


def build_failure_record(pass_name, index, text, last_attempt=None):
    """Build a diagnostic failure record carrying why the batch failed.

    The earlier record shape (pass/entry/text_sha256/text_preview) said which
    entry failed but never why, so causes had to be recovered by grepping run
    logs. last_attempt is the final observed attempt dict from
    generate_script's attempt_observer, or None when no attempt was recorded.
    """
    attempt = last_attempt or {}
    codes = attempt.get("failure_codes") or []
    return {
        "pass": pass_name,
        "entry": index,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_preview": text[:500],
        "reason": codes[0] if codes else (attempt.get("outcome") or "unknown"),
        "finish_reason": attempt.get("finish_reason"),
        "prompt_tokens": attempt.get("prompt_tokens"),
        "completion_tokens": attempt.get("completion_tokens"),
        "reasoning_tokens": attempt.get("reasoning_tokens"),
        "effective_max_tokens": attempt.get("effective_max_tokens"),
        "attempt": attempt.get("attempt"),
    }


REASONING_ALLOWANCE_FLOOR = 1024


class ReasoningAllowance:
    """Track a model's observed thinking-token cost.

    Reasoning tokens bill to completion_tokens but are returned in
    message.reasoning_content, so a ceiling sized on visible output truncates a
    reasoning model mid-thought. A model that never reports reasoning_tokens
    keeps an allowance of zero, so non-reasoning behaviour is unchanged.
    """

    def __init__(self):
        self._observations = []

    def observe(self, reasoning_tokens, truncated=False):
        """Record one call's thinking cost.

        A truncated response reports only the reasoning it managed to emit
        before hitting the ceiling, not what it wanted. Taking that at face
        value makes the allowance converge upward one small step per chunk,
        truncating every chunk on the way, so a censored observation is
        treated as a lower bound and inflated instead.
        """
        if not reasoning_tokens:
            return
        tokens = int(reasoning_tokens)
        if truncated:
            tokens *= 2
        self._observations.append(tokens)

    def current(self):
        if not self._observations:
            return 0
        ordered = sorted(self._observations)
        index = min(len(ordered) - 1, int(len(ordered) * 0.95))
        return max(REASONING_ALLOWANCE_FLOOR, ordered[index])


def resolve_completion_ceiling(source_words, params, reasoning_allowance=0):
    """Bound segmentation output, leaving room for invisible reasoning.

    The visible-output bound is unchanged from the original: it stops a weak
    model spending 10k-16k tokens expanding a ~1k-token chunk. The reasoning
    allowance is added on top rather than carved out of it, so a reasoning
    model gets the same visible budget as everyone else.
    """
    visible = max(512, math.ceil(max(1, source_words) * params.segment_output_ratio))
    return visible + max(0, int(reasoning_allowance))


MAX_REPLACEMENT_DENSITY = 0.02


def read_source_text(path):
    """Read a book, detecting UTF-8 vs cp1252 rather than assuming.

    Measured on this corpus: a large share of .txt sources are cp1252, where
    smart quotes are single bytes (0x92, 0x93) that are not valid UTF-8.
    Decoding those as UTF-8 with replacement manufactures thousands of U+FFFD
    and would trip the damage gate on a file that is perfectly intact. Strict
    UTF-8 first, then cp1252, and only then replacement - so genuinely damaged
    sources (literal U+FFFD bytes, as in index18) still reach the gate.
    """
    raw = open(path, "rb").read()
    for encoding in ("utf-8", "cp1252"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8/replace"


def prepare_source_text(book):
    """Repair, neutralize and audit source text before any LLM call.

    Mirrors production's gate in generate_script.main (audit_unicode_text then
    hard failure) so the diagnostic CLI cannot spend hours on a source that
    production would reject outright. Raises ValueError rather than exiting so
    the behaviour is testable; main() turns it into a non-zero exit.
    """
    damaged = book.count("�")
    if damaged and damaged / max(len(book), 1) > MAX_REPLACEMENT_DENSITY:
        raise ValueError(
            f"source replacement-character density "
            f"{damaged / len(book):.1%} exceeds the "
            f"{MAX_REPLACEMENT_DENSITY:.0%} ceiling; refusing to process it")
    book, repairs = repair_lossy_replacements(book)
    book, residual = neutralize_lossy_residue(book)
    report = audit_unicode_text(book)
    if report["unsafe_controls"]:
        raise ValueError("source contains unsafe control characters: "
                         f"{report['unsafe_controls']}")
    if report["replacement_character_count"]:
        raise ValueError("source still contains replacement characters after "
                         "repair; refusing to process it")
    return book, {"repaired": len(repairs), "residual": residual,
                  "scripts": report["scripts"], "is_nfc": report["is_nfc"]}


def main():
    parser = argparse.ArgumentParser(description="Three-pass annotated script generation.")
    parser.add_argument("input_file")
    parser.add_argument("--output", default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--strip-front-matter", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--pass2-on-exhaustion", choices=["fail", "fallback"],
                        default="fail",
                        help="testing default 'fail' surfaces pass-2 failures; "
                             "'fallback' degrades gracefully (production).")
    parser.add_argument("--preflight", action="store_true",
                        help="Run first/middle/dialogue-heavy samples only.")
    parser.add_argument("--attribution-votes", type=int, default=1,
                        help="Attribute each batch this many times with fixed "
                             "seeds and take the majority (default 1 = greedy). "
                             "Greedy commits to one path and cannot recover; "
                             "measured on mushoku16 it chose a speaker no "
                             "sample agreed with in 61%% of disagreements.")
    parser.add_argument("--vote-temperature", type=float, default=0.3,
                        help="Sampling temperature for voted attribution "
                             "(default 0.3). At 0.7 the samples abstain to "
                             "UNKNOWN instead of committing.")
    parser.add_argument("--reasoning-effort", default=None,
                        help="Pass through to the model (e.g. 'none' to "
                             "disable thinking on a reasoning model).")
    parser.add_argument("--collect-all-failures", action="store_true",
                        help="Diagnostic mode: record exhausted work, continue, "
                             "write only a .partial.json result, and exit nonzero.")
    args = parser.parse_args()
    if args.preflight and args.collect_all_failures:
        parser.error("--collect-all-failures cannot be combined with --preflight")

    book, source_encoding = read_source_text(args.input_file)
    if source_encoding != "utf-8":
        print(f"Read {args.input_file} as {source_encoding} (not valid UTF-8)")
    book = fix_mojibake(book)
    book, _ = normalize_known_source_corruptions(book)
    if args.strip_front_matter:
        book, _ = strip_known_front_matter(book)
    book, publisher = strip_publisher_matter(book)
    if publisher["front_paragraphs"] or publisher["back_paragraphs"]:
        print(f"Stripped publisher matter: {publisher['front_paragraphs']} "
              f"paragraph(s) from the front, {publisher['back_paragraphs']} "
              "from the back (copyright page / colophon, not narration)")
    try:
        book, unicode_report = prepare_source_text(book)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    if unicode_report["repaired"] or unicode_report["residual"]:
        print(f"Repaired {unicode_report['repaired']} destroyed character(s); "
              f"neutralized {unicode_report['residual']} unrecoverable one(s). "
              "The source file was not modified.")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_dir = os.path.dirname(__file__)
    data_dir = get_runtime_data_dir(root)
    config = load_app_config(get_app_config_path(data_dir, root, app_dir))
    llm = config.get("llm", {})
    gen = config.get("generation") or {}
    model_name = llm.get("model_name")
    model_profile = resolve_model_profile(
        model_name, gen.get("three_pass_model_profiles"),
        load_default_model_profiles())
    try:
        chunk_size = resolve_chunk_size(
            args.chunk_size, gen.get("three_pass_chunk_size", 3000),
            model_profile.get("chunk_size"))
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    base_url = llm.get("base_url", "http://localhost:1234/v1")
    llm_mode = config.get("llm_mode", "local")
    # Self-heal LM Studio: load model_name at its verified context if nothing is
    # loaded / settings are stale, mirroring generate_script.py. Without this a
    # fresh `lms unload` leaves no model loaded and every call 400s.
    _, lm_status, heal_msg = ensure_ideal_settings(
        llm_mode, base_url, model_name, ssh_alias=config.get("llm_remote_ssh"))
    print(heal_msg)
    params = LLMGenParams(
        max_tokens=gen.get("max_tokens", 10000),
        temperature=gen.get("temperature", 0.6),
        top_p=gen.get("top_p", 0.8),
        top_k=gen.get("top_k"), min_p=gen.get("min_p"),
        context_length=lm_status.get("context_length"),
        segment_temperature=model_profile.get(
            # Segmentation and attribution are classification, not writing:
            # each has one right answer, so sampling only adds noise. Measured
            # on mushoku16, sending an identical attribution batch twice at
            # temperature 0.1 changed 23.6% of speakers; at 0.0 it changed 0%.
            # That noise was most of the 37.4% run-to-run disagreement that
            # made model comparison impossible, and it also meant regenerating
            # a book produced materially different speakers each time.
            # instruct stays at 0.1: it is the one genuinely generative pass,
            # writing delivery direction rather than choosing a label.
            "segment_temperature", gen.get("three_pass_segment_temperature", 0.0)),
        attribute_temperature=model_profile.get(
            "attribute_temperature", gen.get("three_pass_attribute_temperature", 0.0)),
        instruct_temperature=model_profile.get(
            "instruct_temperature", gen.get("three_pass_instruct_temperature", 0.1)),
        segment_output_ratio=model_profile.get(
            "segment_output_ratio", gen.get("three_pass_segment_output_ratio", 3.0)),
        presegment_quotes=model_profile.get(
            "presegment_quotes", gen.get("three_pass_presegment_quotes", True)),
        reasoning_effort=args.reasoning_effort)
    client = OpenAI(base_url=base_url, api_key=llm.get("api_key", "local"))

    # Context-rescue tuning (finding #12): config-overridable, else defaults.
    cfg_windows = gen.get("context_rescue_windows")
    context_windows = tuple(cfg_windows) if cfg_windows else None
    context_rescue_retries = gen.get("context_rescue_retries")

    output_path = args.output or os.path.join(root, "annotated_script.json")
    print(f"Three-pass generation: {len(book)} chars, chunk_size={chunk_size}, "
          f"model={model_name}, pass2_on_exhaustion={args.pass2_on_exhaustion}")
    if args.preflight:
        summary = {"status": "complete", "model_name": model_name, "samples": []}
        for label, index, sample in select_preflight_chunks(book, chunk_size):
            sample_out = f"{output_path}.preflight_{label}.json"
            try:
                sample_entries = run_three_pass(
                    client, model_name, sample, params, chunk_size,
                    on_exhaustion=args.pass2_on_exhaustion, output_path=sample_out,
                    context_windows=context_windows,
                    context_rescue_retries=context_rescue_retries,
                    endpoint=base_url)
                atomic_json_write(sample_entries, sample_out)
                summary["samples"].append({"label": label, "chunk_index": index,
                                           "status": "complete",
                                           "entries": len(sample_entries)})
            except (RuntimeError, PassExhausted) as exc:
                summary["status"] = "failed"
                summary["samples"].append({"label": label, "chunk_index": index,
                                           "status": "failed", "error": str(exc)})
                break
        atomic_json_write(summary, output_path + ".preflight_manifest.json")
        sys.exit(0 if summary["status"] == "complete" else 1)
    try:
        entries = run_three_pass(client, model_name, book, params, chunk_size,
                                 on_exhaustion=args.pass2_on_exhaustion,
                                 output_path=output_path,
                                 context_windows=context_windows,
                                 context_rescue_retries=context_rescue_retries,
                                 endpoint=base_url,
                                 collect_all_failures=args.collect_all_failures,
                                 unicode_report=unicode_report,
                                 thinking_mode=args.reasoning_effort,
                                 attribution_votes=args.attribution_votes,
                                 vote_temperature=args.vote_temperature)
    except (RuntimeError, PassExhausted) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    manifest = safe_load_json(three_pass_manifest_path(output_path), {})
    if manifest.get("status") == "incomplete":
        partial_path = output_path + ".partial.json"
        atomic_json_write(entries, partial_path)
        print(f"Diagnostic run found {len(manifest.get('diagnostic_failures', []))} "
              f"failure(s); wrote {len(entries)} successful entries to {partial_path}")
        sys.exit(1)
    atomic_json_write(entries, output_path)
    print(f"Wrote {len(entries)} entries to {output_path}")


if __name__ == "__main__":
    main()
