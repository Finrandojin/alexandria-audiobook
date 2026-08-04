"""Does removing the quote-guessing actually improve attribution?

`quote_aware_chunking` measured that a depth-aware chunker removes 45% of
`inferred_missing_close_quote` repairs - the code admitting it had to guess
where a quote closed. That was carefully NOT claimed as an accuracy gain, and
this is the experiment that decides it.

The link it has to survive is `quote_repair_risk`'s: chunks whose quote state
was repaired or carried across a boundary misfile narration as speech at 16.0%
and 14.3% against 2.3% elsewhere. If that mechanism is real, removing repairs
should move attribution. If it is not, this comes out flat.

WHY owarimonogatari3 AND NOT grimgar03. The chunker changes grimgar03 not at
all - 4 repairs before, 4 after - so running it there would produce a
guaranteed null that says nothing about the chunker. owarimonogatari3 drops 15
to 10, the largest English change available. Choosing the book where the
independent variable actually varies is the difference between a test and a
ritual.

POWER, STATED BEFORE RUNNING. Five repairs removed, each misfiling a region at
roughly 16%, against 396 gold lines. The expected effect is single-digit lines.
That is very likely below what this n can resolve, so a null here means BELOW
DETECTION AT n=396, not "no effect", and must be reported that way. The honest
reason to run it anyway is that the alternative is leaving a shipped-code
decision resting on a proxy.

The stronger signal is Japanese, where repairs drop 51 to 26 - but there is no
Japanese gold, so it cannot be scored at all.

BOTH ARMS ARE IDENTICAL except the chunker: same model, same decoding, same
prompts, same gold, same three-pass path. The only swap is
split_into_chunk_records.
"""
import argparse, json, os, re, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

from experiments.scoring import alias_groups, same_speaker
from experiments.stats import exact_mcnemar
from experiments.quote_aware_chunking import quote_aware_chunks
from experiments.prose_vs_nonprose import nonprose_score

SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE"}


def quote_aware_records(text, max_size=3000, slack=1.5):
    """Chunk records from the quote-aware chunker.

    The continuation flags are False because this chunker never cuts a
    paragraph mid-way when it can avoid it; where it must, the cut is at a
    priority punctuation point rather than an arbitrary offset, so there is no
    dangling half-paragraph to mark.
    """
    return [{"text": c.strip(),
             "continues_paragraph_from_previous": False,
             "continues_paragraph_to_next": False}
            for c in quote_aware_chunks(text, max_size=max_size, slack=slack)
            if c.strip()]


def trim_front_matter(text, threshold=0.45, min_chars=200):
    """Drop the leading non-prose block before chunking.

    NOT a convenience. The first chunk of owarimonogatari3 is "Cover /
    [Owarimonogatari 3] Mayoi Hell 001 / Mayoi Hell / 001" and the segmenter
    cannot process it - four attempts on two different models both died on
    chunk 1/222, which blocks the whole comparison before it starts. That is
    the same non-prose pathology `nonprose_gate` measured today at 2.6% of the
    library, arriving from a completely different direction.

    Front matter carries no gold dialogue lines, so removing it costs the
    comparison nothing, and it is removed from BOTH arms identically so it
    cannot favour either. Only the LEADING block is trimmed - a mid-book run of
    non-prose stays, because dropping it would silently change what is being
    compared.
    """
    # Splitting on a blank line is not enough: this book's front matter is
    # separated by whitespace-only lines, so "Cover / [Owarimonogatari 3] /
    # Mayoi Hell / 001" arrives as one paragraph and only "Cover" gets dropped.
    # The first KEPT block therefore has to look like real narration - prose by
    # score AND long enough to be a paragraph rather than a heading.
    paragraphs = re.split(r"\n\s*\n", text)
    start = 0
    for i, para in enumerate(paragraphs):
        body = para.strip()
        if len(body) >= min_chars and nonprose_score(body) < threshold:
            start = i
            break
    return "\n\n".join(paragraphs[start:])


def score(segmented, gold):
    """Alias-aware accuracy of segmented output against gold lines."""
    import collections, re
    norm = lambda t: re.sub(r"\W+", "", t or "").lower()
    groups = alias_groups(gold)
    by_text = collections.defaultdict(list)
    for entry in segmented:
        by_text[norm(entry.get("text"))].append(entry)
    rows = []
    for g in gold["entries"]:
        if g["expected_speaker"].upper() in SPECIAL:
            continue
        hits = by_text.get(norm(g["line"]))
        # A line the segmenter split differently cannot be scored; counting it
        # wrong would charge the arm for a segmentation difference rather than
        # an attribution one, which is the very thing under test.
        if not hits or len(hits) > 1:
            continue
        pred = (hits[0].get("speaker") or "").upper()
        rows.append({"id": g["id"], "expected": g["expected_speaker"].upper(),
                     "predicted": pred,
                     "correct": same_speaker(g["expected_speaker"], pred, groups)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--book", default="owarimonogatari3")
    ap.add_argument("--text")
    ap.add_argument("--model", default="qwen/qwen3-14b")
    ap.add_argument("--base_url", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--chunk-size", type=int, default=3000)
    ap.add_argument("--segment-temperature", type=float, default=0.1,
                    dest="segment_temperature",
                    help="must be > 0 or retries are identical; matches "
                         "config_settings.three_pass_segment_temperature")
    ap.add_argument("--out")
    args = ap.parse_args()

    text_path = args.text or os.path.join(
        REPO, "ab_test_runtime", "results", "collect_all_20260722-155801",
        "inputs", f"{args.book}.txt")
    gold_path = os.path.join(APP, "fixtures",
                             f"attribution_gold_{args.book}.json")
    for p in (text_path, gold_path):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")
    source = open(text_path, encoding="utf-8").read()
    source = trim_front_matter(source)
    gold = json.load(open(gold_path, encoding="utf-8"))

    import generate_script, three_pass_generate
    from generate_script import LLMGenParams
    from openai import OpenAI

    baseline_records = generate_script.split_into_chunk_records
    client = OpenAI(base_url=args.base_url, api_key="local")
    # segment_temperature MUST be non-zero. At 0.0 the model is deterministic,
    # so the segmenter's four retries are four byte-identical calls - observed
    # here as metrics repeating exactly (0.8363 twice, 0.9476 three times)
    # before the run aborted on chunk 1. Production sets
    # three_pass_segment_temperature=0.1 in config_settings for this reason;
    # matching it is what makes a retry a retry.
    params = LLMGenParams(max_tokens=2000, context_length=32768,
                          temperature=0.0, attribute_temperature=0.0,
                          segment_temperature=args.segment_temperature,
                          top_p=0.8, reasoning_effort="none")

    results, arms = {}, {}
    for arm, chunker in (("production", baseline_records),
                         ("quote_aware", quote_aware_records)):
        three_pass_generate.split_into_chunk_records = chunker
        n_chunks = len(chunker(source, args.chunk_size))
        print(f"\n=== {arm}: {n_chunks} chunks ===", flush=True)
        started = time.time()
        out = three_pass_generate.run_three_pass(
            client, args.model, source, params, args.chunk_size)
        segmented = out.get("segmented") if isinstance(out, dict) else out
        rows = score(segmented, gold)
        hit = sum(r["correct"] for r in rows)
        arms[arm] = rows
        results[arm] = {"chunks": n_chunks, "scored": len(rows),
                        "correct": hit,
                        "accuracy": hit / max(len(rows), 1),
                        "seconds": round(time.time() - started)}
        print(f"  {arm:12} {hit}/{len(rows)} = "
              f"{hit/max(len(rows),1)*100:.1f}%  "
              f"({results[arm]['seconds']}s)", flush=True)
    three_pass_generate.split_into_chunk_records = baseline_records

    a = {r["id"]: r["correct"] for r in arms["production"]}
    b = {r["id"]: r["correct"] for r in arms["quote_aware"]}
    shared = sorted(set(a) & set(b))
    gained = sum(1 for i in shared if b[i] and not a[i])
    lost = sum(1 for i in shared if a[i] and not b[i])
    p = exact_mcnemar(lost, gained)[0] if shared else 1.0
    print(f"\n  paired on {len(shared)} lines scoreable in BOTH arms")
    print(f"  quote_aware gains {gained}, loses {lost}, exact McNemar p={p:.4g}")
    if gained == lost == 0:
        print("\n  IDENTICAL on every shared line. The chunker changed 5 repairs\n"
              "  and none of them moved an attribution here.")
    print("\n  BELOW DETECTION is the expected outcome at this n and is NOT\n"
          "  evidence of no effect - see the power note in the docstring.")

    out_path = args.out or os.path.join(
        REPO, "ab_test_runtime", "experiments",
        f"chunker_attribution__{args.book}.json")
    json.dump({"book": args.book, "arms": results,
               "paired": {"n": len(shared), "gained": gained,
                          "lost": lost, "p": p},
               "rows": {k: v for k, v in arms.items()}},
              open(out_path, "w"), indent=1)
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
