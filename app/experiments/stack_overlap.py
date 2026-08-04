"""Why does the SAME model score 11 points higher through llama.cpp?

The base Qwen3-14B scores 68.8% on grimgar03 in bf16 through transformers
(`distill_eval`) and 79.7% at Q4_K_M through llama.cpp (`lora_serving_eval`),
on identical gold rows. Quantisation is supposed to COST accuracy, not add
eleven points, so one of those numbers is describing the harness rather than
the model - and the bf16 one was mine, written for this investigation, while
the llama.cpp path is the one every other result in the ledger used.

This decides it from the row level, which the summary numbers cannot:

  Q4 errors are a near-subset of bf16 errors
      the transformers path was simply handicapped. It got a strictly harder
      version of the same task, most likely from the generation configuration
      in the shim. The bf16 numbers should stop being quoted, INCLUDING the
      +11.7 headline, which was measured against that depressed baseline.

  the two disagree in both directions
      they are genuinely different behaviour, not one being crippled, and the
      stack is a real variable that every cross-stack comparison in this
      ledger has to account for.

WHY IT MATTERS BEYOND TIDINESS. The +11.7 distillation result and the +4.2
shippable result are both real within their stacks. But if the bf16 baseline
was artificially low, then +11.7 overstates what the adapter does, and +4.2 is
the number that should be carried forward - which is what I have been saying
since the discrepancy appeared, on the strength of a guess. This checks it.
"""
import argparse, collections, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO + "/app")

LEDGER = REPO + "/ab_test_runtime/experiments"


def arm(path, name):
    """gold id -> correct, for one arm of one artifact."""
    doc = json.load(open(path))
    return {r["id"]: bool(r.get("correct")) for r in doc["rows"]
            if r["arm"] == name}, \
           {r["id"]: (r.get("predicted") or "") for r in doc["rows"]
            if r["arm"] == name}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bf16", default=LEDGER + "/distill_eval__thunder-a6000-distill.json")
    ap.add_argument("--q4", default=LEDGER + "/lora_serving_eval__local-rocm-lora.json")
    ap.add_argument("--out", default=LEDGER + "/stack_overlap.json")
    args = ap.parse_args()

    for path in (args.bf16, args.q4):
        if not os.path.exists(path):
            print("missing:", path)
            return

    bf, bf_pred = arm(args.bf16, "base")
    q4, q4_pred = arm(args.q4, "base")
    shared = sorted(set(bf) & set(q4))
    if not shared:
        print("no shared gold ids between the two artifacts")
        return

    per_book = collections.defaultdict(lambda: collections.Counter())
    for gid in shared:
        book = gid.split(":", 1)[0]
        key = ("both" if bf[gid] and q4[gid] else
               "bf16_only" if bf[gid] else
               "q4_only" if q4[gid] else "neither")
        per_book[book][key] += 1
        per_book["POOLED"][key] += 1

    print(f"Base model only, no adapter, {len(shared)} shared gold rows\n")
    print(f"  {'book':18}{'n':>6}{'both':>8}{'bf16 only':>11}"
          f"{'Q4 only':>9}{'neither':>9}")
    for book in sorted(per_book):
        c = per_book[book]
        n = sum(c.values())
        print(f"  {book:18}{n:6}{c['both']/n*100:7.1f}%{c['bf16_only']/n*100:10.1f}%"
              f"{c['q4_only']/n*100:8.1f}%{c['neither']/n*100:8.1f}%")

    pooled = per_book["POOLED"]
    n = sum(pooled.values())
    bf_only = pooled["bf16_only"]
    q4_only = pooled["q4_only"]
    print(f"\n  rows bf16 got right and Q4 got wrong: {bf_only} "
          f"({bf_only/n*100:.1f}%)")
    print(f"  rows Q4 got right and bf16 got wrong: {q4_only} "
          f"({q4_only/n*100:.1f}%)")

    # A near-subset means one stack is strictly better, not merely different.
    ratio = bf_only / max(q4_only, 1)
    print(f"\n  ratio {ratio:.2f} (bf16-only / Q4-only)")
    if bf_only <= 0.25 * q4_only:
        print("  -> Q4's errors are close to a SUBSET of bf16's. The "
              "transformers path\n     was handicapped rather than the model "
              "being different, so the bf16\n     numbers describe my harness. "
              "+11.7 overstates the adapter; +4.2 in\n     the shippable stack "
              "is the number to carry forward.")
    else:
        print("  -> the two stacks disagree in BOTH directions, so this is real "
              "behavioural\n     difference rather than one being crippled. "
              "The stack is a variable\n     every cross-stack comparison must "
              "account for, and neither number is\n     simply wrong.")

    # What does bf16 do on the rows Q4 gets right? A blank answer points at
    # generation config; a wrong-but-plausible name points at the model.
    blanks = sum(1 for gid in shared
                 if q4[gid] and not bf[gid] and not bf_pred.get(gid))
    print(f"\n  of the {q4_only} rows only Q4 answered correctly, bf16 returned "
          f"NOTHING on {blanks}\n  ({blanks/max(q4_only,1)*100:.0f}%). A high "
          f"share points at the generation setup -\n  truncation or a stop "
          f"condition - rather than at attribution ability.")

    json.dump({"shared_rows": n,
               "per_book": {b: dict(c) for b, c in per_book.items()},
               "bf16_only": bf_only, "q4_only": q4_only,
               "bf16_blank_on_q4_correct": blanks},
              open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
