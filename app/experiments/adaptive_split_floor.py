"""A/B the adaptive-split safety floor on one exact production chunk."""

import argparse
import json
import os
import sys
import time

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)


def get_floor_splitter(original_splitter, floor):
    """Bind one floor without changing production's splitter signature."""
    return lambda chunk, minimum_chars=floor: original_splitter(
        chunk, minimum_chars=minimum_chars)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--order", choices=("old-first", "new-first"),
                        default="old-first")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from openai import OpenAI
    import generate_script
    from chunk_quality import validate_chunk_quality
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
        top_p=generation.get("top_p", 0.8),
        top_k=generation.get("top_k"), min_p=generation.get("min_p"),
        presence_penalty=generation.get("presence_penalty", 0.0))

    floors = [800, 400] if args.order == "old-first" else [400, 800]
    original_splitter = generate_script.split_failed_chunk
    runs = []
    for floor in floors:
        generate_script.split_failed_chunk = get_floor_splitter(
            original_splitter, floor)
        started = time.time()
        attempts = []
        try:
            entries, adaptively_split = generate_script.process_chunk_adaptively(
                client, llm.get("model_name"), source, args.chunk, len(chunks),
                params, attempt_observer=attempts.append)
        finally:
            generate_script.split_failed_chunk = original_splitter
        quality = validate_chunk_quality(source, entries)
        runs.append({
            "floor": floor, "passed": bool(entries and quality["passed"]),
            "adaptively_split": adaptively_split,
            "entry_count": len(entries), "attempt_count": len(attempts),
            "attempts": attempts, "quality": quality,
            "seconds": round(time.time() - started, 1),
        })
        print(f"floor {floor}: {'PASS' if runs[-1]['passed'] else 'FAIL'}",
              flush=True)

    atomic_json_write({
        "experiment": "adaptive_split_floor",
        "source": os.path.abspath(args.source), "chunk": args.chunk,
        "total_chunks": len(chunks), "order": args.order,
        "model": llm.get("model_name"), "runs": runs,
    }, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
