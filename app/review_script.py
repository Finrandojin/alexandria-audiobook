import os
import sys
import json
import re
import argparse
from openai import OpenAI
from review_prompts import REVIEW_SYSTEM_PROMPT, REVIEW_USER_PROMPT
from generate_script import clean_json_string, repair_json_array, salvage_json_entries


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

    # Mode 2 future: inject source text
    if source_context:
        context_parts.append(f"\nORIGINAL SOURCE TEXT (for reference):\n{source_context}")

    context = "\n".join(context_parts)
    batch_json = json.dumps(batch_entries, indent=2, ensure_ascii=False)
    user_prompt = usr_template.format(context=context, batch=batch_json)

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

        if not json_text:
            print(f"Warning: Could not find JSON array in batch {batch_num} response (attempt {attempt + 1})")
            if attempt < max_retries:
                print("Retrying...")
                continue
            print(f"Response preview: {text[:300]}...")
            return None

        entries = repair_json_array(json_text)

        if entries and len(entries) > 0:
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
            return salvaged

    return None


def normalize_text(text):
    """Normalize text for comparison: lowercase, collapse whitespace, strip punctuation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def check_text_loss(original_entries, corrected_entries, threshold=0.95):
    """Check if corrected entries lost significant text from the original.

    Returns (passed, original_text, corrected_text, ratio).
    passed is True if the corrected text covers at least `threshold` of the original words.
    """
    orig_words = []
    for e in original_entries:
        orig_words.extend(normalize_text(e.get("text", "")).split())

    corr_words = []
    for e in corrected_entries:
        corr_words.extend(normalize_text(e.get("text", "")).split())

    if not orig_words:
        return True, "", "", 1.0

    # Check what fraction of original words appear in corrected text
    orig_joined = " ".join(orig_words)
    corr_joined = " ".join(corr_words)

    # Simple word-level coverage: count original words that appear in corrected
    orig_word_set = set(orig_words)
    corr_word_set = set(corr_words)

    # For a more robust check, compare total word counts
    # (a dropped sentence means fewer total words)
    ratio = len(corr_words) / len(orig_words) if orig_words else 1.0

    passed = ratio >= threshold
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
    args = parser.parse_args()

    # Locate annotated_script.json
    script_path = os.path.join(os.path.dirname(__file__), "..", "annotated_script.json")
    if not os.path.exists(script_path):
        print("Error: annotated_script.json not found. Generate a script first.")
        sys.exit(1)

    with open(script_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    print(f"Loaded {len(entries)} script entries for review")

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

    generation_config = config.get("generation", {})
    batch_size = generation_config.get("review_batch_size", 25)
    max_tokens = generation_config.get("max_tokens", 8000)
    temperature = generation_config.get("temperature", 0.4)
    top_p = generation_config.get("top_p", 0.8)
    top_k = generation_config.get("top_k", 20)
    min_p = generation_config.get("min_p", 0)
    presence_penalty = generation_config.get("presence_penalty", 0.0)
    banned_tokens = generation_config.get("banned_tokens", [])

    print(f"Connecting to: {base_url}")
    print(f"Using model: {model_name}")
    print(f"Batch size: {batch_size} entries, Max tokens: {max_tokens}")
    if banned_tokens:
        print(f"Banned tokens: {banned_tokens}")

    client = OpenAI(base_url=base_url, api_key=api_key)

    # Split entries into batches
    batches = []
    for i in range(0, len(entries), batch_size):
        batches.append(entries[i:i + batch_size])

    total_batches = len(batches)
    print(f"Split into {total_batches} batches of ~{batch_size} entries")

    all_corrected = []
    total_stats = {
        "text_changed": 0,
        "speaker_changed": 0,
        "instruct_changed": 0,
        "entries_added": 0,
        "entries_removed": 0,
        "batches_failed": 0,
    }

    previous_tail = None

    for i, batch in enumerate(batches, 1):
        print(f"\nReviewing batch {i}/{total_batches} ({len(batch)} entries)...")

        corrected = review_batch(
            client, model_name, batch, i, total_batches,
            previous_tail=previous_tail,
            source_context=None,  # Mode 2: would pass source text chunk here
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
            all_corrected.extend(batch)
            total_stats["batches_failed"] += 1
            previous_tail = batch[-2:] if len(batch) >= 2 else batch
            continue

        # Text-loss safety check
        passed, orig_text, corr_text, ratio = check_text_loss(batch, corrected)
        if not passed:
            print(f"  WARNING: Text loss detected! Word ratio: {ratio:.2f} (threshold: 0.95)")
            print(f"  Original words: {len(orig_text.split())}, Corrected words: {len(corr_text.split())}")
            print(f"  Keeping original entries for batch {i} to prevent data loss.")
            all_corrected.extend(batch)
            total_stats["batches_failed"] += 1
            previous_tail = batch[-2:] if len(batch) >= 2 else batch
            continue

        # Diff stats
        stats = diff_entries(batch, corrected)
        entry_diff = len(corrected) - len(batch)

        if entry_diff > 0:
            total_stats["entries_added"] += entry_diff
        elif entry_diff < 0:
            total_stats["entries_removed"] += abs(entry_diff)

        total_stats["text_changed"] += stats["text_changed"]
        total_stats["speaker_changed"] += stats["speaker_changed"]
        total_stats["instruct_changed"] += stats["instruct_changed"]

        changes = stats["text_changed"] + stats["speaker_changed"] + stats["instruct_changed"]
        if changes > 0 or entry_diff != 0:
            print(f"  Changes: {stats['text_changed']} text, {stats['speaker_changed']} speaker, {stats['instruct_changed']} instruct", end="")
            if entry_diff > 0:
                print(f", +{entry_diff} entries (splits)")
            elif entry_diff < 0:
                print(f", {entry_diff} entries (merges)")
            else:
                print()
        else:
            print(f"  No changes")

        all_corrected.extend(corrected)
        previous_tail = corrected[-2:] if len(corrected) >= 2 else corrected

    # Write corrected script
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump(all_corrected, f, indent=2, ensure_ascii=False)

    # Delete chunks.json so editor regenerates
    chunks_path = os.path.join(os.path.dirname(__file__), "..", "chunks.json")
    if os.path.exists(chunks_path):
        os.remove(chunks_path)
        print("Cleared old chunks.json")

    # Final summary
    total_changes = (total_stats["text_changed"] + total_stats["speaker_changed"] +
                     total_stats["instruct_changed"] + total_stats["entries_added"] +
                     total_stats["entries_removed"])

    print(f"\n{'='*60}")
    print(f"Review complete: {len(entries)} -> {len(all_corrected)} entries")
    print(f"  Text changed:    {total_stats['text_changed']}")
    print(f"  Speaker changed: {total_stats['speaker_changed']}")
    print(f"  Instruct changed:{total_stats['instruct_changed']}")
    print(f"  Entries added:   {total_stats['entries_added']}")
    print(f"  Entries removed: {total_stats['entries_removed']}")
    if total_stats["batches_failed"] > 0:
        print(f"  Batches failed:  {total_stats['batches_failed']}")
    print(f"  Total changes:   {total_changes}")
    print(f"{'='*60}")

    if total_changes == 0:
        print("No issues found — script looks clean.")
    else:
        print(f"Fixed {total_changes} issues across {total_batches} batches.")

    print(f"Output saved to: {script_path}")
    print("Task review completed successfully.")


if __name__ == "__main__":
    main()
