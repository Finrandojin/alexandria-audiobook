"""Would asking the model beat guessing at broken quote structure?

`quote_repair_risk` found that chunks where quote state was GUESSED - a missing
delimiter inferred (`repaired`), or a quote spanning the chunk boundary
(`continuation`) - misfile narration as speech at 16.0% and 14.3% against 2.3%
elsewhere. 7% of judged rows carry 41% of the errors.

The indicated change is to stop trusting the pre-segmentation shortcut on those
chunks and fall through to the model. This tests whether that actually helps
BEFORE changing a Rule 9 path, because "the shortcut is unreliable here" does
not imply "the model is better here" - the same broken punctuation that defeats
the parser may defeat the model.

METHOD. Take only the chunks whose resolution was repaired or continuation and
which contain a judged line. Segment each one twice:

    presegmented   what shipped: analyze_outer_quote_regions on the chunk
    model          params.presegment_quotes = False, so pass 1 asks the LLM

Then score both against the human labels on the lines they contain. Same
chunks, same gold, one difference.

READINGS, fixed before running:

  model >> presegmented   the fallthrough is worth its cost and the change
                          should be made
  model ~ presegmented    broken quotes defeat both; the shortcut is not the
                          problem and the change buys only latency
  model << presegmented   the parser is better than the model even on the text
                          it struggles with, and this line of enquiry closes

The cost of the change, if made, is model latency on ~7% of chunks.
"""
import argparse, collections, glob, json, os, re, sys

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
sys.path.insert(0, REPO + "/app")
from openai import OpenAI
from generate_script import LLMGenParams, split_into_chunks
from pass_quality import analyze_outer_quote_regions
from three_pass_generate import segment_chunk_adaptively

M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
RISKY = ("quote_presegmented_repaired", "quote_presegmented_continuation")


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="qwen/qwen3-14b")
    ap.add_argument("--base_url", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--max_chunks", type=int, default=40)
    ap.add_argument("--out", default=REPO + "/ab_test_runtime/experiments/quote_fallthrough.json")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="local")
    params = LLMGenParams(max_tokens=12000, context_length=32768, temperature=0.0,
                          top_p=0.8, reasoning_effort="none",
                          presegment_quotes=False)

    jobs = []
    for path in sorted(glob.glob(REPO + "/ab_test_runtime/fixtures_draft/labelling_bundle__*.json")):
        book = os.path.basename(path).split("__")[1].replace(".json", "")
        src_path, cp = M + f"inputs/{book}.txt", M + INPUT_RUN + f"/{book}/result.json.threepass_checkpoint.json"
        if not (os.path.exists(src_path) and os.path.exists(cp)):
            continue
        src = open(src_path, encoding="utf-8").read()
        res = json.load(open(cp)).get("resolutions") or []
        chunks = split_into_chunks(src, max_size=3000)
        if len(chunks) != len(res):
            continue
        starts, off = [], 0
        for c in chunks:
            i = src.find(c, off)
            starts.append(i if i >= 0 else off)
            off = (i if i >= 0 else off) + len(c)
        judged = {}
        for e in json.load(open(path))["entries"]:
            probe = (e.get("line") or "").strip()[:50]
            if len(probe) < 12:
                continue
            pos = src.find(probe)
            if pos < 0:
                continue
            k = max(0, sum(1 for s in starts if s <= pos) - 1)
            judged.setdefault(k, []).append(
                (e.get("line"), e.get("expected_speaker") == "NOT_DIALOGUE"))
        for k, lines in judged.items():
            if k < len(res) and res[k] in RISKY:
                jobs.append({"book": book, "chunk": chunks[k], "lines": lines,
                             "resolution": res[k]})

    jobs = jobs[:args.max_chunks]
    print(f"{len(jobs)} risky chunks carrying "
          f"{sum(len(j['lines']) for j in jobs)} judged lines\n")
    if not jobs:
        return

    score = {"presegmented": [0, 0], "model": [0, 0]}
    for n, job in enumerate(jobs, 1):
        pre = analyze_outer_quote_regions(job["chunk"])["regions"]
        try:
            mod = segment_chunk_adaptively(client, args.model, job["chunk"], params)
        except Exception as exc:
            print(f"  chunk {n}: model path {type(exc).__name__}")
            mod = []
        for arm, entries in (("presegmented", pre), ("model", mod)):
            if not entries:
                continue
            types = {}
            for e in entries:
                types[norm(e.get("text"))] = e.get("type")
            for line, is_narration in job["lines"]:
                key = norm(line)
                got = types.get(key)
                if got is None:
                    got = next((t for k, t in types.items() if key and key in k), None)
                if got is None:
                    continue
                # correct = narration labelled NARRATOR, speech labelled SPOKEN
                want = "NARRATOR" if is_narration else "SPOKEN"
                score[arm][0] += 1
                score[arm][1] += (got == want)
        if n % 10 == 0:
            print(f"  {n}/{len(jobs)} chunks ...", flush=True)

    print()
    for arm in ("presegmented", "model"):
        n, ok = score[arm]
        print(f"  {arm:14} {ok}/{n} lines typed correctly = "
              f"{ok/max(n,1)*100:5.1f}%")
    pn, po = score["presegmented"]
    mn, mo = score["model"]
    if pn and mn:
        print(f"\n  model - presegmented: "
              f"{(mo/mn - po/pn)*100:+.1f} points on the risky chunks")
        print("  A null means broken quotes defeat both and the change buys only")
        print("  latency. Only a clear win justifies touching the Rule 9 path.")
    json.dump({"chunks": len(jobs), "score": score}, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
