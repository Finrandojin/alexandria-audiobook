"""How often does one chunk actually pass? Goal 3.1.

WHY A SINGLE CHUNK. grimgar03 reached 49/49 on one run and died at chunk 11 on
the next, both on qwen3-14b against the same source. Script generation runs at
temperature 0.6, so chunk outcomes are a distribution, not a fact - and a
whole-book rerun costs 2.5 hours to sample that distribution exactly once.

Running one chunk repeatedly separates two explanations a whole-book run
cannot:

    the chunk usually passes and run 2 was unlucky
        -> the book is fragile, and a retry budget is the lever

    the chunk rarely passes and run 1 was lucky
        -> the coverage validator is refusing output the model cannot produce,
           which is a defect in the gate rather than in the book

It reuses the real pipeline - the same chunker, the same prompts, the same
validate_chunk_quality - because a probe that approximates the gate would
answer a question nobody asked.
"""
import argparse
import collections
import json
import os
import sys
import time

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(APP)
sys.path.insert(0, APP)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--chunk", type=int, required=True,
                        help="1-based chunk number to repeat")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "chunk_retry_probe.json"))
    args = parser.parse_args()

    from openai import OpenAI
    from chunk_quality import validate_chunk_quality
    from generate_script import (LLMGenParams, split_into_chunks,
                                 process_chunk)

    with open(args.source, encoding="utf-8") as handle:
        book = handle.read()
    with open(os.path.join(APP, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)

    generation = config.get("generation") or {}
    chunk_size = generation.get("chunk_size", 6000)
    chunks = split_into_chunks(book, max_size=chunk_size)
    if not 1 <= args.chunk <= len(chunks):
        raise SystemExit(f"chunk {args.chunk} outside 1..{len(chunks)}")
    target = chunks[args.chunk - 1]

    mode = config.get("llm_mode", "local")
    llm = (config.get("llm_remote" if mode == "remote" else "llm_local")
           or config.get("llm") or {})
    client = OpenAI(base_url=llm.get("base_url"),
                    api_key=llm.get("api_key") or "local")
    model_name = llm.get("model_name")
    params = LLMGenParams(
        max_tokens=generation.get("max_tokens", 10000),
        temperature=generation.get("temperature", 0.6))

    print(f"  {os.path.basename(args.source)} chunk {args.chunk}/{len(chunks)}"
          f"  ({len(target)} chars)  model={model_name}")
    print(f"  repeating {args.repeats}x at temperature {params.temperature}")

    outcomes = []
    for attempt in range(1, args.repeats + 1):
        started = time.time()
        try:
            entries = process_chunk(client, model_name, target, args.chunk,
                                    len(chunks), params)
        except Exception as exc:                            # noqa: BLE001
            outcomes.append({"run": attempt, "passed": False,
                             "error": str(exc)[:200],
                             "seconds": round(time.time() - started, 1)})
            print(f"    run {attempt}: ERROR {str(exc)[:80]}", flush=True)
            continue
        elapsed = round(time.time() - started, 1)
        if not entries:
            outcomes.append({"run": attempt, "passed": False,
                             "reason": "no entries after retries",
                             "seconds": elapsed})
            print(f"    run {attempt}: FAILED (exhausted) {elapsed}s",
                  flush=True)
            continue
        quality = validate_chunk_quality(target, entries)
        outcomes.append({
            "run": attempt, "passed": bool(quality.get("passed")),
            "entries": len(entries), "seconds": elapsed,
            "failures": [f.get("code") or f.get("type")
                         for f in (quality.get("failures") or [])][:5],
            "metrics": {k: v for k, v in (quality.get("metrics") or {}).items()
                        if isinstance(v, (int, float))},
        })
        print(f"    run {attempt}: {'PASS' if quality.get('passed') else 'FAIL'}"
              f"  {len(entries)} entries  {elapsed}s", flush=True)

    passed = sum(1 for o in outcomes if o["passed"])
    reasons = collections.Counter(
        code for o in outcomes for code in o.get("failures", []))
    result = {
        "scope": "one chunk through the real pipeline, repeated; temperature "
                 "0.6 makes this a distribution rather than a single fact",
        "source": os.path.abspath(args.source),
        "chunk": args.chunk, "total_chunks": len(chunks),
        "repeats": args.repeats, "passed": passed,
        "pass_rate_pct": round(100.0 * passed / max(1, len(outcomes)), 1),
        "model": model_name, "temperature": params.temperature,
        "failure_codes": dict(reasons),
        "runs": outcomes,
    }
    from utils import atomic_json_write
    atomic_json_write(result, args.out)
    print(f"\n  passed {passed}/{len(outcomes)} "
          f"({result['pass_rate_pct']}%)  failures={dict(reasons)}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
