"""Do the adapter and the cascade fix the SAME rows, or different ones?

End-to-end the distilled 14B matches or beats the 70B cascade on three of four
books. That says they reach the same PLACE; it does not say they get there by
fixing the same lines. The answer decides the architecture:

  high overlap    the cascade is redundant - ship one tuned 14B and retire the
                  escalation path entirely
  low overlap     they are complementary, the ceiling is above either alone,
                  and running the cascade on top of the adapter is worth a test

This needs no GPU. Both systems were scored against the same gold ids on the
same books, so the row-level agreement is already on disk.

THE TWO SYSTEMS DO NOT SHARE A BASELINE. The cascade's cheap arm is `cheap-w1`
at context width one through llama.cpp Q4; the adapter's base is the same model
in bf16 with production contexts, and it scores about thirteen points higher on
grimgar03. So "rows each system FIXED" is measured against different starting
points and those two sets are not directly comparable. What IS comparable is
which rows each system finally gets RIGHT, so that is the headline, with the
fix-sets reported underneath and labelled.

THE UNION IS AN ORACLE. "Either system is right" requires knowing which to
trust per row, which is the routing problem `realizable_router` showed cannot
be solved from book-level features (-0.96 against a fixed choice). It is a
ceiling, not a plan.
"""
import collections, glob, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO + "/app")

LEDGER = REPO + "/ab_test_runtime/experiments"
BOOKS = ("grimgar03", "index18", "mushoku16", "owarimonogatari3")


def rows_by_arm(path):
    doc = json.load(open(path))
    out = collections.defaultdict(dict)
    for row in doc.get("rows") or []:
        out[row["arm"]][row["id"]] = bool(row.get("correct"))
    return out, doc


def main():
    # The adapter's rows are keyed "book:id"; the cascade's are bare ids.
    adapter = collections.defaultdict(dict)
    for tag in ("thunder-a6000-distill",):
        path = LEDGER + f"/distill_eval__{tag}.json"
        if not os.path.exists(path):
            continue
        arms, _ = rows_by_arm(path)
        for arm in ("base", "tuned"):
            for key, ok in arms[arm].items():
                book, _, gid = key.partition(":")
                adapter[(book, arm)][gid] = ok

    cascade = {}
    for path in sorted(glob.glob(LEDGER + "/cascade__*.json")):
        book = next((b for b in BOOKS if b in os.path.basename(path)), None)
        arms, doc = rows_by_arm(path)
        if not book or "cascade" not in arms or "cheap-w1" not in arms:
            continue
        gain = (sum(arms["cascade"].values()) / max(len(arms["cascade"]), 1)
                - sum(arms["cheap-w1"].values()) / max(len(arms["cheap-w1"]), 1))
        # Keep the strongest cascade run per book: the one the ledger quotes.
        if book not in cascade or gain > cascade[book][0]:
            cascade[book] = (gain, arms, doc.get("meta", {}).get("model", "?"))

    print("Rows both systems scored, and who gets them right\n")
    print(f"  {'book':18}{'n':>5}{'both':>7}{'adapter':>9}{'cascade':>9}"
          f"{'neither':>9}{'union':>8}")
    totals = collections.Counter()
    for book in BOOKS:
        if (book, "tuned") not in adapter or book not in cascade:
            continue
        tuned = adapter[(book, "tuned")]
        casc = cascade[book][1]["cascade"]
        shared = sorted(set(tuned) & set(casc))
        if not shared:
            continue
        both = sum(1 for k in shared if tuned[k] and casc[k])
        only_a = sum(1 for k in shared if tuned[k] and not casc[k])
        only_c = sum(1 for k in shared if casc[k] and not tuned[k])
        neither = sum(1 for k in shared if not tuned[k] and not casc[k])
        union = both + only_a + only_c
        n = len(shared)
        totals.update({"n": n, "both": both, "only_a": only_a,
                       "only_c": only_c, "neither": neither})
        print(f"  {book:18}{n:5}{both/n*100:6.1f}%{only_a/n*100:8.1f}%"
              f"{only_c/n*100:8.1f}%{neither/n*100:8.1f}%{union/n*100:7.1f}%")

    n = totals["n"]
    if not n:
        print("\nNo book has both systems scored on shared ids.")
        return
    print(f"\n  pooled {n} rows")
    print(f"    both right          {totals['both']/n*100:5.1f}%")
    print(f"    adapter only        {totals['only_a']/n*100:5.1f}%")
    print(f"    cascade only        {totals['only_c']/n*100:5.1f}%")
    print(f"    neither             {totals['neither']/n*100:5.1f}%")
    union = (totals["both"] + totals["only_a"] + totals["only_c"]) / n
    adapter_acc = (totals["both"] + totals["only_a"]) / n
    cascade_acc = (totals["both"] + totals["only_c"]) / n
    print(f"\n    adapter alone       {adapter_acc*100:5.1f}%")
    print(f"    cascade alone       {cascade_acc*100:5.1f}%")
    print(f"    ORACLE union        {union*100:5.1f}%   "
          f"(+{(union-max(adapter_acc,cascade_acc))*100:.1f} over the better one)")

    print("\n  Reading it:")
    print("    'cascade only' near zero  -> the cascade fixes nothing the "
          "adapter misses,\n                                 and the "
          "escalation path can be retired.")
    print("    'cascade only' large      -> they are complementary and the "
          "union is worth\n                                 chasing, subject "
          "to routing being solvable -\n                                 "
          "which realizable_router says it is not, from\n"
          "                                 book-level features at least.")
    print("\n  The union is an ORACLE: collecting it needs a per-row choice "
          "nobody has\n  shown how to make. It bounds the prize, it is not a "
          "design.")


if __name__ == "__main__":
    main()
