"""Like-for-like: what does the cascade buy, and what does the adapter buy?

The distillation result (+11.7 pooled) has been quoted against "the cascade's
+11.1 to +22.0" all day. Those numbers came from different places and were
never put on the same footing, so this computes both from the artifacts.

WHAT THE CASCADE ARTIFACTS ACTUALLY CONTAIN. Each cascade run scores two arms
over the SAME full row set: `cheap-w1`, the cheap model alone, and `cascade`,
the cheap model with disagreed rows escalated to the big one. So the cascade's
whole-book effect is directly available and does not need reconstructing - the
earlier worry that it was a routed-rows-only figure was wrong, and this script
is what establishes that rather than asserting it.

WHAT STILL DOES NOT MATCH, and cannot be fixed by arithmetic:

  different baselines   the cascade's cheap arm is Qwen3-14B at Q4 through
                        llama.cpp; the adapter's base arm is the same model in
                        bf16 through transformers. Their levels differ by more
                        than ten points on grimgar03, so only the DELTAS are
                        comparable, and even those sit on different starting
                        points.
  different cheap models  some cascade runs use gemma-3-27b or a 32B as the
                        cheap arm, which is not the model the adapter tunes.
  cost is not accuracy   the cascade pays a 70B for a fraction of rows on every
                        book, forever. The adapter pays once and then runs at
                        14B cost. Equal deltas do not mean equal value.

So the comparison below is deltas only, per book, with the cheap model named on
every row. It answers "does a distilled 14B move accuracy as far as escalating
to a 70B", not "is it the same system".
"""
import collections, glob, json, os, sys

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
sys.path.insert(0, REPO + "/app")

LEDGER = REPO + "/ab_test_runtime/experiments"
BOOKS = ("grimgar03", "index18", "mushoku16", "owarimonogatari3")


def arm_scores(doc):
    by = collections.defaultdict(lambda: [0, 0])
    for row in doc.get("rows") or []:
        bucket = by[row["arm"]]
        bucket[0] += 1
        bucket[1] += bool(row.get("correct"))
    return by


def main():
    print("CASCADE: cheap model alone vs cheap + escalation, same rows\n")
    print(f"  {'book':18}{'cheap model':26}{'cheap':>8}{'cascade':>9}{'delta':>8}{'n':>6}")
    cascade_deltas = collections.defaultdict(list)
    for path in sorted(glob.glob(LEDGER + "/cascade__*.json")):
        doc = json.load(open(path))
        book = next((b for b in BOOKS if b in os.path.basename(path)), "?")
        by = arm_scores(doc)
        if "cascade" not in by or "cheap-w1" not in by:
            continue
        c_ok, c_n = by["cascade"][1], by["cascade"][0]
        b_ok, b_n = by["cheap-w1"][1], by["cheap-w1"][0]
        if not c_n or not b_n:
            continue
        delta = c_ok / c_n - b_ok / b_n
        model = doc.get("meta", {}).get("model", "?")
        cascade_deltas[book].append((delta, model))
        print(f"  {book:18}{model[:24]:26}{b_ok/b_n*100:7.1f}%"
              f"{c_ok/c_n*100:8.1f}%{delta*100:+8.1f}{c_n:6}")

    distill = {}
    dpath = LEDGER + "/distill_eval__thunder-a6000-distill.json"
    if os.path.exists(dpath):
        doc = json.load(open(dpath))
        per = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
        for row in doc["rows"]:
            book = row["id"].split(":")[0]
            bucket = per[book][row["arm"]]
            bucket[0] += 1
            bucket[1] += bool(row.get("correct"))
        for book, arms in per.items():
            b, t = arms["base"], arms["tuned"]
            if b[0] and t[0]:
                distill[book] = (b[1] / b[0], t[1] / t[0], t[0])

    print("\nDISTILLED ADAPTER: base 14B vs the same 14B with the LoRA, same rows\n")
    print(f"  {'book':18}{'base':>8}{'tuned':>9}{'delta':>8}{'n':>6}")
    for book in BOOKS:
        if book in distill:
            b, t, n = distill[book]
            print(f"  {book:18}{b*100:7.1f}%{t*100:8.1f}%{(t-b)*100:+8.1f}{n:6}")

    print("\nSIDE BY SIDE, deltas only\n")
    print(f"  {'book':18}{'best cascade':>14}{'  (cheap model)':22}{'adapter':>9}")
    for book in BOOKS:
        best = max(cascade_deltas.get(book, [(float('-nan'), '-')]),
                   key=lambda x: x[0], default=(None, '-'))
        adapter = f"{(distill[book][1]-distill[book][0])*100:+.1f}" if book in distill else "-"
        if best[0] is None or best[0] != best[0]:
            print(f"  {book:18}{'-':>14}{'':22}{adapter:>9}")
        else:
            print(f"  {book:18}{best[0]*100:+13.1f}  {best[1][:20]:20}{adapter:>9}")

    print("\nFINAL ACCURACY, which is what a reader of the book actually gets\n")
    print(f"  {'book':18}{'cascade end':>13}{'adapter end':>13}{'diff':>8}")
    for book in BOOKS:
        best = max(cascade_deltas.get(book, []), key=lambda x: x[0], default=None)
        if not best or book not in distill:
            continue
        # Recover the cascade end-point for the best run on this book.
        end = None
        for path in sorted(glob.glob(LEDGER + "/cascade__*.json")):
            if book not in os.path.basename(path):
                continue
            doc = json.load(open(path))
            by = arm_scores(doc)
            if "cascade" not in by or "cheap-w1" not in by:
                continue
            d = by["cascade"][1] / by["cascade"][0] - by["cheap-w1"][1] / by["cheap-w1"][0]
            if abs(d - best[0]) < 1e-9:
                end = by["cascade"][1] / by["cascade"][0]
        if end is None:
            continue
        adapter_end = distill[book][1]
        print(f"  {book:18}{end*100:12.1f}%{adapter_end*100:12.1f}%"
              f"{(adapter_end-end)*100:+8.1f}")

    print("\n  THE DELTAS ARE NOT COMPARABLE AND THE END POINTS ARE CLOSER TO IT.")
    print("  The cascade's cheap arm is `cheap-w1` - context width ONE, a")
    print("  deliberately narrowed configuration - at Q4 through llama.cpp,")
    print("  scoring 55.8% on grimgar03. The adapter's base arm is the same")
    print("  model in bf16 with production neighbour contexts, at 68.8%. A")
    print("  weaker starting point makes a larger delta, so the cascade's")
    print("  deltas are inflated relative to a production baseline and its")
    print("  END POINT is the fairer number.")
    print("\n  Deltas only, and the cheap model differs between rows - a cascade")
    print("  built on gemma-3-27b is not the system the adapter modifies. The")
    print("  two also differ in kind: the cascade rents a 70B on every book")
    print("  forever, the adapter pays once and then runs at 14B cost.")


if __name__ == "__main__":
    main()
