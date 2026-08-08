import os
import sys
import json
import re
import argparse
from openai import OpenAI
from review_prompts import REVIEW_SYSTEM_PROMPT, REVIEW_USER_PROMPT
from generate_script import clean_json_string, repair_json_array, salvage_json_entries
from speaker_canon import canonicalize, remember_in_roster, resolve_against_roster

# Marker embedded in both halves of review_prompts.txt (see the file itself)
# identifying prompts written for the current positional-overlay review
# schema: the reviewer only ever corrects "speaker"/"instruct" and must
# return exactly as many entries as it was given. This is review_script.py's
# own equivalent of generate_script.py's PROMPT_SCHEMA_MARKER
# ("span-labels-v1") for the classifier stage -- kept separate (rather than
# sharing generate_script.select_prompt) because that marker is specific to
# the span-classifier schema, not the review schema, and generate_script.py
# is out of scope for this fix.
REVIEW_PROMPT_SCHEMA_MARKER = "verbatim-review-v1"


def select_review_prompt(custom_prompt, default_prompt, config_key):
    """Choose between a saved custom review prompt and the built-in default.

    A custom prompt is honoured only when it carries
    REVIEW_PROMPT_SCHEMA_MARKER, i.e. when it was written for the current
    verbatim/positional-overlay review schema (speaker/instruct-only
    corrections, same entry count out as in). A prompt saved before that
    refactor still instructs the old "strip attribution tags, rephrase,
    split/merge, rewrite front-matter" methodology -- the overlay makes that
    incapable of damaging annotated_script.json's text, but it still causes
    entry-count mismatches (failed batches) and degrades speaker/instruct
    fixes, so a stale prompt is rejected loudly rather than used silently.
    """
    if not custom_prompt or not custom_prompt.strip():
        return default_prompt

    if REVIEW_PROMPT_SCHEMA_MARKER in custom_prompt:
        return custom_prompt

    print(f"  {'!' * 60}")
    print(f"  WARNING: saved custom prompt '{config_key}' predates the verbatim-review")
    print(f"  pipeline (missing '{REVIEW_PROMPT_SCHEMA_MARKER}' marker) -- using the built-in")
    print("  default instead. Re-customize starting from the new default if needed.")
    print(f"  {'!' * 60}")
    return default_prompt


def _canonicalize_speakers(entries):
    """Return a copy of `entries` with each "speaker" field canonicalized.

    Never touches "text" -- only the speaker label. Safety net against
    casing/formatting drift introduced by the review LLM (or older,
    pre-canonicalization scripts) so the `!= "NARRATOR"` checks in
    merge_consecutive_narrators stay reliable.
    """
    result = []
    for entry in entries:
        new_entry = dict(entry)
        if "speaker" in new_entry:
            new_entry["speaker"] = canonicalize(new_entry["speaker"])
        result.append(new_entry)
    return result


def build_script_roster(entries):
    """Roster index of the speaker spellings already present in a script.

    Spelling collisions are settled by remember_in_roster's rule (most
    boundary marks wins, ties to the incumbent), which is
    order-independent -- so this roster and generation's agree even though
    they are built in different orders. Pass the result to
    apply_positional_overlay so a reviewer's spelling variant is snapped back
    onto the established name instead of forking the roster.
    """
    index = {}
    for entry in entries or ():
        speaker = entry.get("speaker") if isinstance(entry, dict) else None
        if speaker:
            remember_in_roster(index, speaker)
    return index


def apply_positional_overlay(batch, corrected, roster=None):
    """Overlay LLM speaker/instruct corrections onto the ORIGINAL batch text.

    This is the hard, structural guarantee against text damage: "text" is
    ALWAYS taken from the original entry, never from the LLM's output, so no
    amount of accidental rewording/splitting/merging/attribution-stripping in
    a model response can touch the verbatim script. Only "speaker"
    (canonicalized) and "instruct" are taken from the LLM's corresponding
    entry, matched positionally (same index in `batch` and `corrected`).

    Returns None if `corrected` has a different number of entries than
    `batch`. The reviewer is no longer allowed to split, merge, add, or
    remove entries -- a count mismatch means its response can't be aligned
    positionally, so the caller must treat this as a failed batch (keep the
    original entries) rather than guessing an alignment.
    """
    if len(corrected) != len(batch):
        return None

    accepted = []
    for orig, corr in zip(batch, corrected):
        new_entry = dict(orig)
        corr = corr if isinstance(corr, dict) else {}

        # Speaker: prefer the LLM's correction, but only if it canonicalizes
        # to something non-empty. An empty string, None, or a value that
        # canonicalizes to "" (e.g. a stray "(shouting)") must NOT blank out
        # a good original label -- that would silently drop the entry from
        # the voices roster and mute it at render time. Fall back to the
        # original (canonicalized) speaker whenever the correction is empty,
        # not only when the "speaker" key is absent entirely.
        #
        # `roster` (optional, see build_script_roster) additionally snaps a
        # correction onto an established spelling from the same script when
        # the two differ only in their boundary marks -- whitespace, hyphens,
        # apostrophes ("ABBEMARIGNAN" -> "ABBE MARIGNAN", "OBRIEN" ->
        # "O'BRIEN"). Exact roster-key equality only -- similar-but-distinct
        # names (JON/JOHN, ELLA/BELLA) are never merged.
        corrected_speaker = resolve_against_roster(corr.get("speaker") or "", roster or {})
        new_entry["speaker"] = corrected_speaker or resolve_against_roster(
            orig.get("speaker", ""), roster or {})

        # Instruct: only accept a non-empty string from the LLM. None, a
        # list/dict, or a whitespace-only string is rejected in favor of the
        # original instruct (already present via dict(orig) above) rather
        # than writing a malformed value into annotated_script.json.
        corrected_instruct = corr.get("instruct")
        if isinstance(corrected_instruct, str) and corrected_instruct.strip():
            new_entry["instruct"] = corrected_instruct

        new_entry["text"] = orig.get("text", "")
        accepted.append(new_entry)
    return accepted


def _is_section_break(text):
    """Check if text looks like a chapter heading or section title."""
    stripped = text.strip()
    # "CHAPTER ONE", "CHAPTER II", "Chapter Three", etc.
    if re.match(r'(?i)^chapter\b', stripped):
        return True
    # All-caps short text = likely a title ("A SCANDAL IN BOHEMIA", "THE RED-HEADED LEAGUE")
    if stripped == stripped.upper() and len(stripped) < 80 and stripped.isascii():
        return True
    return False


def _join_narrator_texts(left, right):
    """Join two narrator texts across a merge boundary.

    Verbatim entries already carry their own boundary whitespace (a
    trailing/leading space or newline copied straight from the source), so
    unconditionally inserting an extra " " would introduce a byte that was
    never in the original text. Only inject a space when NEITHER side
    already has boundary whitespace -- this keeps legacy scripts (older
    annotated_script.json files whose entries were .strip()'d) merging
    readably, while verbatim entries stay byte-exact.
    """
    if not left or not right:
        return left + right
    if left[-1].isspace() or right[0].isspace():
        return left + right
    return left + " " + right


def merge_consecutive_narrators(entries, max_merged_length=800):
    """Merge consecutive NARRATOR entries that share the same instruct value.

    Skips merging across section/chapter breaks. Caps merged text at
    max_merged_length characters to avoid creating overly long TTS entries.
    """
    if not entries:
        return entries, 0

    merged = []
    merges = 0
    i = 0
    while i < len(entries):
        entry = entries[i]

        if entry.get("speaker") != "NARRATOR" or _is_section_break(entry.get("text", "")):
            merged.append(entry)
            i += 1
            continue

        # Start a narrator run — accumulate consecutive NARRATORs with same instruct
        combined_text = entry["text"]
        instruct = entry.get("instruct", "")
        run_count = 1
        j = i + 1

        while j < len(entries):
            next_entry = entries[j]
            if next_entry.get("speaker") != "NARRATOR":
                break
            if next_entry.get("instruct", "") != instruct:
                break
            if _is_section_break(next_entry.get("text", "")):
                break
            candidate = _join_narrator_texts(combined_text, next_entry["text"])
            if len(candidate) > max_merged_length:
                break
            combined_text = candidate
            run_count += 1
            j += 1

        merged.append({
            "speaker": "NARRATOR",
            "text": combined_text,
            "instruct": instruct
        })
        if run_count > 1:
            merges += run_count - 1
        i = j

    return merged, merges


def _rendered_len(entries_list):
    """Length (chars) of the JSON that would actually be sent to the LLM for
    this list of entries -- i.e. exactly what review_batch() renders via
    `json.dumps(entries_list, indent=2, ensure_ascii=False)`."""
    return len(json.dumps(entries_list, indent=2, ensure_ascii=False))


def build_review_batches(entries, batch_size, char_budget):
    """Split entries into review batches bounded by BOTH entry count and
    rendered JSON size.

    A fixed entry-count batch (the old behavior) can silently overflow a
    small LLM serving context window when individual entries are unusually
    large (e.g. huge Gutenberg front-matter blocks) -- the server then
    truncates the prompt instead of erroring, the model never sees its
    instructions or most of its entries, and review silently becomes a
    no-op on exactly the batches with the longest entries.

    Entries accumulate into the current batch until EITHER:
      - the batch already has `batch_size` entries, or
      - adding the next entry would push the batch's rendered
        `json.dumps(batch, indent=2, ensure_ascii=False)` length past
        `char_budget`,
    whichever comes first. A single entry never gets split across batches
    -- if one entry's own rendered size alone exceeds `char_budget`, it
    becomes a singleton batch by itself (the positional-overlay review
    contract requires each entry to survive review as one unit; splitting
    an entry is not an option).
    """
    if not entries:
        return []

    batches = []
    current = []
    for entry in entries:
        candidate = current + [entry]
        if current and (len(candidate) > batch_size or _rendered_len(candidate) > char_budget):
            batches.append(current)
            current = [entry]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def _truncate_context_entry(entry, max_text_chars=300):
    """Return a copy of `entry` for CONTEXT-ONLY display, with "text" capped
    to `max_text_chars` (plus an ellipsis) so a single giant neighbor entry
    (e.g. a huge front-matter block sitting just outside the target batch)
    can't blow the prompt's char budget from the context side.

    NEVER apply this to the target batch itself -- only to the +/-N
    neighbor entries shown for context in contextual review mode. The
    reviewer never edits context entries anyway (only the TARGET BATCH is
    returned), so truncating their display text has no effect on output
    correctness, only on prompt size.
    """
    text = entry.get("text", "")
    if isinstance(text, str) and len(text) > max_text_chars:
        truncated = dict(entry)
        truncated["text"] = text[:max_text_chars] + "..."
        return truncated
    return entry


# Best-effort truncation-tripwire heuristic (see _maybe_print_truncation_hint):
# English text averages roughly this many characters per token. If the
# actual rendered prompt is more than _TRUNCATION_HINT_MULTIPLIER times this
# baseline ratio away from what the reported prompt_tokens would imply, the
# server is probably silently truncating its context window rather than
# processing the whole prompt.
_CHARS_PER_TOKEN_BASELINE = 4.0
_TRUNCATION_HINT_MULTIPLIER = 3.5


def _maybe_print_truncation_hint(prompt_chars, prompt_tokens):
    """Print a one-line, best-effort hint when a response looks unusable
    because the LLM server silently truncated its context window.

    Symptom (observed in production on a 494k-word script): a locally
    served model (e.g. Ollama) with a serving context window smaller than
    the actual prompt truncates the prompt instead of erroring. The model
    never sees its instructions/entries and returns garbage (wrong format,
    wrong entry count) -- the overlay keeps annotated_script.json safe
    either way, but review silently becomes a no-op on exactly the batches
    containing the longest entries. The signature: reported prompt_tokens
    stays roughly flat across wildly different batch sizes, because the
    server only counts what it kept after truncating.

    This is purely a diagnostic print -- it never changes control flow or
    what gets returned/written.
    """
    if not prompt_tokens:
        return
    if prompt_chars > _TRUNCATION_HINT_MULTIPLIER * _CHARS_PER_TOKEN_BASELINE * prompt_tokens:
        print(f"  HINT: response unusable and prompt_tokens ({prompt_tokens}) far below prompt "
              f"size ({prompt_chars} chars) -- the LLM server may be truncating its context "
              f"window; reduce review_batch_char_budget or raise the server's context length.")


def review_batch(client, model_name, batch_entries, batch_num, total_batches,
                 previous_tail=None, source_context=None, max_retries=2,
                 system_prompt=None, user_prompt_template=None,
                 max_tokens=8000, temperature=0.4, top_p=0.8, top_k=20,
                 min_p=0, presence_penalty=0.0, banned_tokens=None):
    """Send a batch of script entries through the LLM for review and correction."""
    sys_prompt = system_prompt or REVIEW_SYSTEM_PROMPT
    usr_template = user_prompt_template or REVIEW_USER_PROMPT

    # Build context
    context_parts = []
    context_parts.append(f"Batch {batch_num} of {total_batches}.")

    if previous_tail:
        context_parts.append("\nPrevious batch ended with:")
        for entry in previous_tail:
            context_parts.append(json.dumps(entry, ensure_ascii=False))

    # Optional extra context (e.g. source snippet or surrounding entries)
    if source_context:
        context_parts.append(f"\nADDITIONAL REVIEW CONTEXT:\n{source_context}")

    context = "\n".join(context_parts)
    batch_json = json.dumps(batch_entries, indent=2, ensure_ascii=False)
    user_prompt = usr_template.format(context=context, batch=batch_json)
    prompt_chars = len(sys_prompt) + len(user_prompt)

    usage = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
                extra_body={
                    k: v for k, v in {
                        "top_k": top_k,
                        "min_p": min_p,
                        "banned_tokens": banned_tokens if banned_tokens else None,
                    }.items() if v is not None
                }
            )

            choice = response.choices[0]
            text = choice.message.content.strip()
            finish_reason = choice.finish_reason
            usage = getattr(response, 'usage', None)

            # Log raw response
            log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "review_responses.log")
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"\n{'='*80}\n")
                lf.write(f"BATCH {batch_num}/{total_batches} | attempt {attempt + 1} | finish_reason={finish_reason}\n")
                if usage:
                    lf.write(f"tokens: prompt={getattr(usage, 'prompt_tokens', '?')} completion={getattr(usage, 'completion_tokens', '?')}\n")
                lf.write(f"{'─'*80}\n")
                lf.write(text)
                lf.write(f"\n{'='*80}\n")

            print(f"  finish_reason={finish_reason}", end="")
            if usage:
                print(f" | tokens: prompt={getattr(usage, 'prompt_tokens', '?')} completion={getattr(usage, 'completion_tokens', '?')}", end="")
            print()

            if finish_reason == "length":
                print(f"  WARNING: Response was truncated (hit max_tokens={max_tokens}). Consider increasing max_tokens or reducing batch size.")

        except Exception as e:
            print(f"Error calling LLM API (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                continue
            return None

        # Clean and parse JSON response
        json_text = clean_json_string(text)

        prompt_tokens = getattr(usage, 'prompt_tokens', None) if usage else None

        if not json_text:
            print(f"Warning: Could not find JSON array in batch {batch_num} response (attempt {attempt + 1})")
            if attempt < max_retries:
                print("Retrying...")
                continue
            print(f"Response preview: {text[:300]}...")
            _maybe_print_truncation_hint(prompt_chars, prompt_tokens)
            return None

        entries = repair_json_array(json_text)

        if entries and len(entries) > 0:
            if len(entries) != len(batch_entries):
                _maybe_print_truncation_hint(prompt_chars, prompt_tokens)
            if attempt > 0:
                print(f"  Succeeded on retry {attempt + 1}")
            return entries

        print(f"Warning: Could not parse batch {batch_num} response as JSON (attempt {attempt + 1})")

        if attempt < max_retries:
            print("Retrying...")

        # Last resort
        salvaged = salvage_json_entries(json_text)
        if salvaged:
            print(f"Regex-salvaged {len(salvaged)} entries from malformed response")
            if len(salvaged) != len(batch_entries):
                _maybe_print_truncation_hint(prompt_chars, prompt_tokens)
            return salvaged

    _maybe_print_truncation_hint(prompt_chars, getattr(usage, 'prompt_tokens', None) if usage else None)
    return None


def normalize_text(text):
    """Normalize text for comparison: lowercase, collapse whitespace, strip punctuation.

    Punctuation is replaced with a SPACE (not deleted outright) before the
    whitespace collapse. Deleting it outright used to fuse words across a
    punctuation boundary that abuts two words with no space of its own
    (e.g. `he said:--"Sicut` -> `saidsicut`). That fusion happens
    differently depending on whether the text was normalized as one long
    string (whole source) or as separately-normalized, then word-joined,
    entries (per span) -- because entry/span boundaries often fall exactly
    at a quote mark. The mismatch is a false alarm: no characters are
    actually gained or lost, only whether two words end up glued together
    by this normalization. Replacing with a space first means both sides of
    every comparison tokenize identically regardless of where the
    normalization boundary falls, since a punctuation mark that used to
    silently disappear now always leaves a token separator behind.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def check_text_loss(original_entries, corrected_entries, threshold=0.95, upper_bound=None):
    """Check if corrected entries lost or gained significant text.

    Returns (passed, original_text, corrected_text, ratio).
    passed is True if the corrected word count ratio falls within
    [threshold, upper_bound]. If upper_bound is None, it defaults to
    1.0 + (1.0 - threshold), i.e. symmetric around 1.0.
    """
    orig_words = []
    for e in original_entries:
        orig_words.extend(normalize_text(e.get("text", "")).split())

    corr_words = []
    for e in corrected_entries:
        corr_words.extend(normalize_text(e.get("text", "")).split())

    if not orig_words:
        return True, "", "", 1.0

    orig_joined = " ".join(orig_words)
    corr_joined = " ".join(corr_words)

    ratio = len(corr_words) / len(orig_words) if orig_words else 1.0

    if upper_bound is None:
        upper_bound = 1.0 + (1.0 - threshold)
    passed = threshold <= ratio <= upper_bound
    return passed, orig_joined, corr_joined, ratio


def diff_entries(original, corrected):
    """Compare original and corrected entries, return a summary dict."""
    stats = {
        "text_changed": 0,
        "speaker_changed": 0,
        "instruct_changed": 0,
        "entries_original": len(original),
        "entries_corrected": len(corrected),
    }

    # Compare entry-by-entry up to the shorter length
    compare_len = min(len(original), len(corrected))
    for i in range(compare_len):
        orig = original[i]
        corr = corrected[i]
        if orig.get("text") != corr.get("text"):
            stats["text_changed"] += 1
        if orig.get("speaker") != corr.get("speaker"):
            stats["speaker_changed"] += 1
        if orig.get("instruct") != corr.get("instruct"):
            stats["instruct_changed"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Review and fix annotated audiobook script")
    parser.add_argument("--source", help="Path to original source text for comparison (mode 2, not yet implemented)")
    parser.add_argument("--context-window", type=int, default=0,
                        help="If > 0, review each entry with +/- N neighboring entries for better segmentation and speaker fixes")
    args = parser.parse_args()

    # Locate annotated_script.json
    script_path = os.path.join(os.path.dirname(__file__), "..", "annotated_script.json")
    if not os.path.exists(script_path):
        print("Error: annotated_script.json not found. Generate a script first.")
        sys.exit(1)

    with open(script_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    print(f"Loaded {len(entries)} script entries for review")

    # Established speaker spellings for this script, built once. Threaded
    # into every apply_positional_overlay() call so a reviewer's spelling
    # variant is snapped back onto the name the script already uses.
    script_roster = build_script_roster(entries)

    # Load source text if provided (mode 2 prep)
    source_text = None
    if args.source:
        if os.path.exists(args.source):
            with open(args.source, "r", encoding="utf-8") as f:
                source_text = f.read()
            print(f"Loaded source text: {len(source_text)} chars")
        else:
            print(f"Warning: Source file not found: {args.source}")

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config.json: {e}")
    else:
        print("Warning: config.json not found. Using defaults.")

    llm_config = config.get("llm", {})
    base_url = llm_config.get("base_url", "http://localhost:11434/v1")
    api_key = llm_config.get("api_key", "local")
    model_name = llm_config.get("model_name", "local-model")

    # Load custom review prompts or use defaults from review_prompts.txt
    prompts_config = config.get("prompts", {})
    review_sys = select_review_prompt(
        prompts_config.get("review_system_prompt"), REVIEW_SYSTEM_PROMPT, "review_system_prompt"
    )
    review_usr = select_review_prompt(
        prompts_config.get("review_user_prompt"), REVIEW_USER_PROMPT, "review_user_prompt"
    )

    generation_config = config.get("generation", {})
    batch_size = generation_config.get("review_batch_size", 25)
    # Dual-budget batching: a fixed entry count alone can silently overflow
    # a small LLM serving context window when individual entries are huge
    # (e.g. Gutenberg front-matter). 12000 chars (~3k tokens) is
    # conservative -- it leaves room for the system prompt plus contextual
    # neighbor windows even inside a 4096-token serving window. See
    # build_review_batches().
    batch_char_budget = generation_config.get("review_batch_char_budget", 12000)
    max_tokens = generation_config.get("max_tokens", 8000)
    temperature = generation_config.get("temperature", 0.4)
    top_p = generation_config.get("top_p", 0.8)
    top_k = generation_config.get("top_k", 20)
    min_p = generation_config.get("min_p", 0)
    presence_penalty = generation_config.get("presence_penalty", 0.0)
    banned_tokens = generation_config.get("banned_tokens", [])

    print(f"Connecting to: {base_url}")
    print(f"Using model: {model_name}")
    print(f"Batch size: up to {batch_size} entries or {batch_char_budget:,} chars per batch, Max tokens: {max_tokens}")
    if banned_tokens:
        print(f"Banned tokens: {banned_tokens}")

    client = OpenAI(base_url=base_url, api_key=api_key)

    all_corrected = []
    # "text_changed"/"entries_added"/"entries_removed" are intentionally
    # absent: the positional overlay (apply_positional_overlay) guarantees
    # every accepted entry keeps its original text and every accepted batch
    # keeps its original entry count, so those counters could only ever be
    # 0 on batches that made it through. A batch that would have caused
    # either is instead rejected outright and counted in batches_failed.
    total_stats = {
        "speaker_changed": 0,
        "instruct_changed": 0,
        "batches_failed": 0,
    }

    if args.context_window and args.context_window > 0:
        window = max(1, args.context_window)
        batches = build_review_batches(entries, batch_size, batch_char_budget)
        total_batches = len(batches)
        print(f"Contextual review mode enabled: batching up to {batch_size} entries or "
              f"{batch_char_budget:,} chars per LLM call, with +/-{window} neighbors "
              f"({total_batches} batches)")

        previous_tail = None
        pos = 0
        for batch_index, batch in enumerate(batches, 1):
            start = pos
            end = start + len(batch)
            pos = end
            before = entries[max(0, start - window):start]
            after = entries[end:min(len(entries), end + window)]

            batch_chars = _rendered_len(batch)
            print(f"\nReviewing batch {batch_index}/{total_batches} ({len(batch)} entries, {batch_chars:,} chars)...")

            contextual_lines = [
                "Contextual batch review mode.",
                "The 'SCRIPT ENTRIES TO REVIEW' below is your TARGET BATCH.",
                "Use the following PREVIOUS and NEXT entries for context, but DO NOT include them in your output. Only return the corrected TARGET BATCH.",
                "Context entries' \"text\" may be truncated with '...' for brevity -- that truncation applies ONLY to this context section, never to your TARGET BATCH.",
            ]
            if before:
                contextual_lines.append("\n--- PREVIOUS ENTRIES (Context Only) ---")
                contextual_lines.extend(json.dumps(_truncate_context_entry(e), ensure_ascii=False) for e in before)
            if after:
                contextual_lines.append("\n--- NEXT ENTRIES (Context Only) ---")
                contextual_lines.extend(json.dumps(_truncate_context_entry(e), ensure_ascii=False) for e in after)

            corrected = review_batch(
                client, model_name, batch, batch_index, total_batches,
                previous_tail=None,  # contextual mode uses explicit before/after window instead
                source_context="\n".join(contextual_lines),
                system_prompt=review_sys,
                user_prompt_template=review_usr,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                banned_tokens=banned_tokens
            )

            if corrected is None:
                print(f"  FAILED — keeping original entries for batch {batch_index}")
                all_corrected.extend(_canonicalize_speakers(batch))
                total_stats["batches_failed"] += 1
                previous_tail = batch[-2:] if len(batch) >= 2 else batch
                continue

            if len(corrected) != len(batch):
                print(f"  FAILED — reviewer returned {len(corrected)} entries for a {len(batch)}-entry "
                      f"batch (splitting/merging is not permitted); keeping original entries for batch {batch_index}")
                all_corrected.extend(_canonicalize_speakers(batch))
                total_stats["batches_failed"] += 1
                previous_tail = batch[-2:] if len(batch) >= 2 else batch
                continue

            accepted = apply_positional_overlay(batch, corrected, roster=script_roster)

            # Safety net only: the positional overlay always takes "text"
            # from the original batch, so this can never actually fail. If
            # it does, that's a structural bug -- surface it loudly instead
            # of silently keeping originals.
            passed, orig_text, corr_text, ratio = check_text_loss(batch, accepted, threshold=1.0, upper_bound=1.0)
            if not passed:
                print(f"  BUG: text-loss safety net tripped despite positional overlay (ratio {ratio:.4f})! "
                      f"Keeping original entries for batch {batch_index}.")
                all_corrected.extend(_canonicalize_speakers(batch))
                total_stats["batches_failed"] += 1
                previous_tail = batch[-2:] if len(batch) >= 2 else batch
                continue

            stats = diff_entries(batch, accepted)
            total_stats["speaker_changed"] += stats["speaker_changed"]
            total_stats["instruct_changed"] += stats["instruct_changed"]

            changes = stats["speaker_changed"] + stats["instruct_changed"]
            if changes > 0:
                print(f"  Changes: {stats['speaker_changed']} speaker, {stats['instruct_changed']} instruct")
            else:
                print("  No changes")

            all_corrected.extend(accepted)
            previous_tail = accepted[-2:] if len(accepted) >= 2 else accepted
    else:
        # Split entries into batches, bounded by both count and rendered
        # JSON size (see build_review_batches).
        batches = build_review_batches(entries, batch_size, batch_char_budget)

        total_batches = len(batches)
        print(f"Split into {total_batches} batches (up to {batch_size} entries or {batch_char_budget:,} chars each)")

        previous_tail = None

        for i, batch in enumerate(batches, 1):
            batch_chars = _rendered_len(batch)
            print(f"\nReviewing batch {i}/{total_batches} ({len(batch)} entries, {batch_chars:,} chars)...")

            corrected = review_batch(
                client, model_name, batch, i, total_batches,
                previous_tail=previous_tail,
                source_context=None,  # Mode 2: would pass source text chunk here
                system_prompt=review_sys,
                user_prompt_template=review_usr,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                banned_tokens=banned_tokens
            )

            if corrected is None:
                print(f"  FAILED — keeping original entries for batch {i}")
                all_corrected.extend(_canonicalize_speakers(batch))
                total_stats["batches_failed"] += 1
                previous_tail = batch[-2:] if len(batch) >= 2 else batch
                continue

            if len(corrected) != len(batch):
                print(f"  FAILED — reviewer returned {len(corrected)} entries for a {len(batch)}-entry "
                      f"batch (splitting/merging is not permitted); keeping original entries for batch {i}")
                all_corrected.extend(_canonicalize_speakers(batch))
                total_stats["batches_failed"] += 1
                previous_tail = batch[-2:] if len(batch) >= 2 else batch
                continue

            accepted = apply_positional_overlay(batch, corrected, roster=script_roster)

            # Safety net only: the positional overlay always takes "text"
            # from the original batch, so this can never actually fail. If
            # it does, that's a structural bug -- surface it loudly instead
            # of silently keeping originals.
            passed, orig_text, corr_text, ratio = check_text_loss(batch, accepted, threshold=1.0, upper_bound=1.0)
            if not passed:
                print(f"  BUG: text-loss safety net tripped despite positional overlay (ratio {ratio:.4f})! "
                      f"Keeping original entries for batch {i}.")
                all_corrected.extend(_canonicalize_speakers(batch))
                total_stats["batches_failed"] += 1
                previous_tail = batch[-2:] if len(batch) >= 2 else batch
                continue

            # Diff stats
            stats = diff_entries(batch, accepted)
            total_stats["speaker_changed"] += stats["speaker_changed"]
            total_stats["instruct_changed"] += stats["instruct_changed"]

            changes = stats["speaker_changed"] + stats["instruct_changed"]
            if changes > 0:
                print(f"  Changes: {stats['speaker_changed']} speaker, {stats['instruct_changed']} instruct")
            else:
                print(f"  No changes")

            all_corrected.extend(accepted)
            previous_tail = accepted[-2:] if len(accepted) >= 2 else accepted

    # Post-processing: merge consecutive NARRATOR entries with same instruct
    merge_narrators_enabled = generation_config.get("merge_narrators", False)
    narrator_merges = 0
    if merge_narrators_enabled:
        pre_merge_count = len(all_corrected)
        all_corrected, narrator_merges = merge_consecutive_narrators(all_corrected, max_merged_length=800)
        if narrator_merges > 0:
            print(f"\nPost-processing: merged {narrator_merges} consecutive narrator entries "
                  f"({pre_merge_count} -> {len(all_corrected)} entries)")
    else:
        print("\nNarrator merging: disabled (enable in Setup > Advanced)")

    # Write corrected script
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump(all_corrected, f, indent=2, ensure_ascii=False)

    # Delete chunks.json so editor regenerates
    chunks_path = os.path.join(os.path.dirname(__file__), "..", "chunks.json")
    if os.path.exists(chunks_path):
        os.remove(chunks_path)
        print("Cleared old chunks.json")

    # Final summary
    # Text-loss and entry-count changes are structurally impossible on
    # accepted batches (see apply_positional_overlay / total_stats comment
    # above), so they're not tracked here -- only speaker/instruct edits and
    # narrator merges count as "changes".
    total_changes = (total_stats["speaker_changed"] +
                     total_stats["instruct_changed"] + narrator_merges)

    print(f"\n{'='*60}")
    print(f"Review complete: {len(entries)} -> {len(all_corrected)} entries")
    print(f"  Speaker changed: {total_stats['speaker_changed']}")
    print(f"  Instruct changed:{total_stats['instruct_changed']}")
    print(f"  Narrators merged:{narrator_merges}")
    if total_stats["batches_failed"] > 0:
        print(f"  Batches failed:  {total_stats['batches_failed']}")
    print(f"  Total changes:   {total_changes}")
    print(f"{'='*60}")

    if total_changes == 0:
        print("No issues found -- script looks clean.")
    else:
        print(f"Fixed {total_changes} issues across {total_batches} batches.")

    print(f"Output saved to: {script_path}")
    print("Task review completed successfully.")


if __name__ == "__main__":
    main()
