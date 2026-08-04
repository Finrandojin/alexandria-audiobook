"""Does the adapter generalise beyond four translated light novels?

Every accuracy in this ledger comes from grimgar03, index18, mushoku16 and
owarimonogatari3 - four contemporary Japanese web novels in English
translation, 772 rows. Whether +5.4 is a property of the adapter or of that
corpus has never been tested, and I wrongly described testing it as blocked on
hand-labelling.

PDNC supplies public-domain English novels with speaker-attributed quotations:
Austen, Doyle, Chopin. Different century, different register, untranslated, and
a different annotation team.

WHAT IS DIFFERENT ABOUT THIS TEST, and why the numbers are not comparable to
the ledger's:

  no segmentation   PDNC gives quotation spans directly, so this measures
                    attribution on already-correct spans. The ledger's numbers
                    include lines the segmenter misfiled.
  richer labels     quote type (explicit / implicit / anaphoric) and character
                    category come from PDNC, so results stratify in ways our
                    own fixtures never could.
  larger roster     Pride and Prejudice has 74 characters against grimgar03's
                    21, so the selection problem is harder per line.

READINGS, fixed before running:

  adapter helps here too    the gain is a property of the adapter, and the
                            light-novel corpus was not doing the work
  adapter is flat here      what it learned is corpus-specific: it was trained
                            on two light novels and it transfers to light
                            novels. That is the most important negative
                            available and it would qualify every headline in
                            this ledger.
  adapter HURTS here        worse than flat - it has overfitted to a register,
                            and shipping it as a default would be wrong for any
                            book outside that register.
"""
import argparse, collections, glob, json, os, re, sys, time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = REPO + "/app/"
sys.path.insert(0, APP)
from openai import OpenAI
from experiments.scoring import alias_groups, same_speaker
from experiments.stats import clopper_pearson, paired
from generate_script import LLMGenParams
from three_pass_generate import attribute_batch

BATCH = 25


def set_scale(base_url, scale):
    root = base_url.rsplit("/v1", 1)[0]
    body = json.dumps([{"id": 0, "scale": scale}]).encode()
    req = urllib.request.Request(root + "/lora-adapters", data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    urllib.request.urlopen(req, timeout=30).read()
    with urllib.request.urlopen(root + "/lora-adapters", timeout=30) as fh:
        got = float(json.loads(fh.read())[0].get("scale", -1))
    if abs(got - scale) > 1e-6:
        raise RuntimeError(f"adapter scale did not take: {got} != {scale}")
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fixtures", nargs="+", default=sorted(
        glob.glob(APP + "fixtures/attribution_gold_pdnc_*.json")))
    ap.add_argument("--model", default="qwen/qwen3-14b")
    ap.add_argument("--base_url", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--limit", type=int, default=300,
                    help="quotations per novel; PDNC is large and this is a "
                         "generalisation probe, not a full scoring run")
    ap.add_argument("--out", default=REPO + "/ab_test_runtime/experiments/pdnc_eval.json")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="local")
    params = LLMGenParams(max_tokens=2000, context_length=32768, temperature=0.0,
                          attribute_temperature=0.0, top_p=0.8,
                          reasoning_effort="none")
    results = {}
    for path in args.fixtures:
        fx = json.load(open(path))
        book = fx["book"]
        entries = fx["entries"][:args.limit]
        groups = alias_groups({"aliases": fx["aliases"]})
        roster = fx["roster"]
        print(f"\n{book}: {len(entries)} quotations, {len(roster)} characters",
              flush=True)
        per_arm = {}
        for arm, scale in (("base", 0.0), ("lora", 1.0)):
            set_scale(args.base_url, scale)
            started, rows = time.time(), []
            for s in range(0, len(entries), BATCH):
                block = entries[s:s + BATCH]
                frozen = [{"type": "SPOKEN", "text": e["line"]} for e in block]
                ctx = [{"previous_context": {"type": "NARRATOR",
                                             "text": e["prev_context"]},
                        "next_context": {"type": "NARRATOR",
                                         "text": e["next_context"]}}
                       for e in block]
                try:
                    out = attribute_batch(client, args.model, frozen, params,
                                          roster, neighbor_contexts=ctx)
                except Exception as exc:
                    print(f"  {arm} block {s//BATCH}: {type(exc).__name__}",
                          flush=True)
                    rows.extend({"e": e, "ok": False, "pred": None}
                                for e in block)
                    continue
                for off, e in enumerate(block):
                    sp = (out[off] or {}).get("speaker") if off < len(out) else None
                    rows.append({"e": e, "pred": sp, "ok": same_speaker(
                        e["expected_speaker"], sp, groups)})
            hit = sum(1 for r in rows if r["ok"])
            lo, hi = clopper_pearson(hit, max(len(rows), 1))
            per_arm[arm] = rows
            print(f"  {arm:5} {hit}/{len(rows)} = {hit/max(len(rows),1)*100:5.1f}%  "
                  f"[{lo:.1f}-{hi:.1f}]  {time.time()-started:.0f}s", flush=True)
        # stratify by the labels our own fixtures never had
        for key in ("quote_type", "category"):
            buckets = collections.defaultdict(lambda: [0, 0, 0])
            for arm in ("base", "lora"):
                for r in per_arm[arm]:
                    b = buckets[r["e"].get(key, "?")]
                    if arm == "base":
                        b[0] += 1
                        b[1] += r["ok"]
                    else:
                        b[2] += r["ok"]
            print(f"  by {key}:")
            for name, (n, b, l) in sorted(buckets.items()):
                if n:
                    print(f"    {name:14}{n:5} base {b/n*100:5.1f}%  "
                          f"lora {l/n*100:5.1f}%  {(l-b)/n*100:+5.1f}")
        # Row level, not just counts. The previous artifact stored only n and
        # correct, which made per-row agreement with any other method - a
        # BookNLP ensemble, a second model - impossible to compute without
        # paying for the whole run again.
        results[book] = {a: {"n": len(r), "correct": sum(1 for x in r if x["ok"]),
                             "rows": [{"id": x["e"].get("id"),
                                       "expected": x["e"].get("expected_speaker"),
                                       "predicted": x.get("pred"),
                                       "correct": bool(x["ok"]),
                                       "quote_type": x["e"].get("quote_type"),
                                       "category": x["e"].get("category")}
                                      for x in r]}
                         for a, r in per_arm.items()}

    if results:
        tb = sum(v["base"]["correct"] for v in results.values())
        tn = sum(v["base"]["n"] for v in results.values())
        tl = sum(v["lora"]["correct"] for v in results.values())
        print(f"\n  pooled  base {tb}/{tn} = {tb/max(tn,1)*100:.1f}%   "
              f"lora {tl}/{tn} = {tl/max(tn,1)*100:.1f}%   "
              f"{(tl-tb)/max(tn,1)*100:+.1f}")
        print("\n  Not comparable to the ledger's +5.4: PDNC supplies correct")
        print("  quotation spans, so segmentation error is absent here.")
    json.dump(results, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
