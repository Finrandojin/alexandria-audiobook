"""A/B production retries against bounded targeted missing-passage repair."""

import argparse
import json
import os
import sys
import time

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

REPAIR_SYSTEM_PROMPT = """Convert only the supplied audiobook passage into a JSON
array. Preserve every word exactly once. Each object must contain speaker, text,
and instruct. Return JSON only; do not include text outside the supplied passage."""
REPAIR_USER_PROMPT = """The earlier conversion omitted the passage below. Convert
ONLY this exact passage, from beginning to end, for insertion into the existing
script. Do not repeat surrounding material.

MISSING PASSAGE:
{passage}"""


def get_missing_passage(source, entries):
    """Return exact low-recall sentence spans, bounded to three per repair."""
    from chunk_quality import validate_chunk_quality
    from source_span_coverage import get_source_spans

    ranked = []
    for index, span in enumerate(get_source_spans(source)):
        metrics = validate_chunk_quality(span["text"], entries)["metrics"]
        recall = metrics["source_token_recall"]
        if recall < 0.80:
            ranked.append((recall, index, span["text"]))
    ranked.sort(key=lambda item: (item[0], item[1]))
    selected = sorted(ranked[:3], key=lambda item: item[1])
    return "\n\n".join(item[2] for item in selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--order", choices=("baseline-first", "repair-first"),
                        default="baseline-first")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from openai import OpenAI
    import generate_script
    from chunk_quality import validate_chunk_quality
    from default_prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT
    from utils import atomic_json_write

    with open(args.source, encoding="utf-8") as handle:
        book = handle.read()
    with open(os.path.join(APP, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    generation = config.get("generation") or {}
    book, _ = generate_script.get_preprocessed_source(book)
    chunks = generate_script.split_into_chunks(
        book, max_size=generation.get("chunk_size", 6000))
    if not 1 <= args.chunk <= len(chunks):
        raise SystemExit(f"chunk {args.chunk} outside 1..{len(chunks)}")
    source = chunks[args.chunk - 1]
    mode = config.get("llm_mode", "local")
    llm = (config.get("llm_remote" if mode == "remote" else "llm_local")
           or config.get("llm") or {})
    client = OpenAI(base_url=llm.get("base_url"),
                    api_key=llm.get("api_key") or "local")
    params = generate_script.LLMGenParams(
        max_tokens=generation.get("max_tokens", 10000),
        temperature=generation.get("temperature", 0.6),
        top_p=generation.get("top_p", 0.8), top_k=generation.get("top_k"),
        min_p=generation.get("min_p"),
        presence_penalty=generation.get("presence_penalty", 0.0))

    arms = (["baseline", "targeted_repair"] if args.order == "baseline-first"
            else ["targeted_repair", "baseline"])
    runs = []
    for arm in arms:
        started = time.time()
        attempts = []
        repair_rounds = []
        if arm == "baseline":
            entries, adaptively_split = generate_script.process_chunk_adaptively(
                client, llm.get("model_name"), source, args.chunk, len(chunks),
                params, attempt_observer=attempts.append)
        else:
            context = generate_script._build_chunk_context(
                args.chunk, len(chunks), None)
            entries = generate_script.call_llm_for_entries(
                client, llm.get("model_name"), DEFAULT_SYSTEM_PROMPT,
                DEFAULT_USER_PROMPT.format(context=context, chunk=source), params,
                "targeted_missing_repair_responses.log", "INITIAL", max_retries=0,
                attempt_observer=attempts.append)
            for repair_round in range(1, 4):
                quality = validate_chunk_quality(source, entries)
                if quality["passed"]:
                    break
                passage = get_missing_passage(source, entries)
                if not passage:
                    break
                additions = generate_script.call_llm_for_entries(
                    client, llm.get("model_name"), REPAIR_SYSTEM_PROMPT,
                    REPAIR_USER_PROMPT.format(passage=passage), params,
                    "targeted_missing_repair_responses.log",
                    f"REPAIR {repair_round}", max_retries=1,
                    validate_entries=lambda value, passage=passage:
                    validate_chunk_quality(passage, value),
                    attempt_observer=attempts.append)
                repair_rounds.append({"round": repair_round,
                                      "passage_chars": len(passage),
                                      "entry_count": len(additions)})
                if not additions:
                    break
                entries.extend(additions)
            adaptively_split = False
        quality = validate_chunk_quality(source, entries)
        runs.append({
            "arm": arm, "passed": bool(entries and quality["passed"]),
            "adaptively_split": adaptively_split, "entry_count": len(entries),
            "attempt_count": len(attempts), "attempts": attempts,
            "repair_rounds": repair_rounds, "quality": quality,
            "seconds": round(time.time() - started, 1),
        })
        print(f"{arm}: {'PASS' if runs[-1]['passed'] else 'FAIL'}", flush=True)

    atomic_json_write({
        "experiment": "targeted_missing_repair", "source": os.path.abspath(args.source),
        "chunk": args.chunk, "total_chunks": len(chunks), "order": args.order,
        "model": llm.get("model_name"), "runs": runs,
    }, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
