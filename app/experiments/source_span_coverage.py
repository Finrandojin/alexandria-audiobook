"""A/B explicit source-span accounting on one known-fragile generation chunk."""

import argparse
import json
import os
import sys
import time

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(APP)
sys.path.insert(0, APP)


SPAN_SYSTEM_PROMPT = """Convert audiobook prose into a JSON array. Preserve every
word exactly once and assign speakers accurately. Each object must contain
speaker, text, instruct, and source_span_ids. source_span_ids must list every
[Snnn] source span represented by that object. Return JSON only."""

SPAN_USER_PROMPT = """Convert the complete tagged source below from beginning to
end. Tags identify source positions and must not appear in text. Every tag must
occur in at least one source_span_ids list. An entry may cite multiple tags and
a tag may be cited by multiple entries when dialogue splits a sentence.

TAGGED SOURCE:
{tagged_source}"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from openai import OpenAI
    from chunk_quality import validate_chunk_quality
    from generate_script import (LLMGenParams, call_llm_for_entries, process_chunk,
                                 split_into_chunks)
    from source_span_coverage import (format_tagged_source, get_source_spans,
                                      get_span_coverage_findings)
    from utils import atomic_json_write

    with open(args.source, encoding="utf-8") as handle:
        book = handle.read()
    with open(os.path.join(APP, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    generation = config.get("generation") or {}
    chunks = split_into_chunks(book, max_size=generation.get("chunk_size", 6000))
    if not 1 <= args.chunk <= len(chunks):
        raise SystemExit(f"chunk {args.chunk} outside 1..{len(chunks)}")
    source = chunks[args.chunk - 1]
    spans = get_source_spans(source)
    mode = config.get("llm_mode", "local")
    llm = (config.get("llm_remote" if mode == "remote" else "llm_local")
           or config.get("llm") or {})
    client = OpenAI(base_url=llm.get("base_url"), api_key=llm.get("api_key") or "local")
    model = llm.get("model_name")
    params = LLMGenParams(max_tokens=generation.get("max_tokens", 10000),
                          temperature=generation.get("temperature", 0.6))

    runs = []
    for repeat in range(1, args.repeats + 1):
        for arm in ("baseline", "span_ids"):
            started = time.time()
            if arm == "baseline":
                entries = process_chunk(client, model, source, args.chunk, len(chunks), params)
                coverage_findings = None
            else:
                def validate(entries):
                    quality = validate_chunk_quality(source, entries)
                    coverage = get_span_coverage_findings(spans, entries)
                    if coverage:
                        quality = dict(quality)
                        quality["passed"] = False
                        quality["findings"] = list(quality["findings"]) + coverage
                    return quality

                entries = call_llm_for_entries(
                    client, model, SPAN_SYSTEM_PROMPT,
                    SPAN_USER_PROMPT.format(tagged_source=format_tagged_source(spans)),
                    params, "source_span_coverage_responses.log",
                    f"SPAN CHUNK {args.chunk}/{len(chunks)}", max_retries=4,
                    validate_entries=validate)
                coverage_findings = get_span_coverage_findings(spans, entries)
            quality = validate_chunk_quality(source, entries)
            runs.append({
                "repeat": repeat, "arm": arm,
                "passed": bool(entries and quality["passed"] and not coverage_findings),
                "entry_count": len(entries), "quality": quality,
                "coverage_findings": coverage_findings,
                "seconds": round(time.time() - started, 1),
            })
            print(f"{arm} repeat {repeat}: {'PASS' if runs[-1]['passed'] else 'FAIL'}",
                  flush=True)

    result = {
        "experiment": "source_span_coverage", "source": os.path.abspath(args.source),
        "chunk": args.chunk, "total_chunks": len(chunks), "span_count": len(spans),
        "model": model, "temperature": params.temperature, "repeats": args.repeats,
        "arms": ["baseline", "span_ids"], "runs": runs,
    }
    atomic_json_write(result, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
