"""Should the adapter be the default? Where does it make things WORSE?

Every number so far is an average, and an average can hide a regression that a
listener would notice immediately: a book that gets worse, or a specific
character whose lines start going to someone else. +5.4 pooled is not a reason
to ship if one lead character collapses.

This is the check before the decision, over the shippable stack only (Q4 base,
f16 LoRA, llama.cpp - the configuration a user would actually run):

  per book        paired McNemar, so a book that improves by luck is visible
  per character   every lead, with the count of lines that changed direction
  regressions     any book or lead character where the adapter loses ground

WHAT WOULD BLOCK SHIPPING. A book that gets significantly worse; a lead
character losing more lines than they gain; or a gain that rests on one book
while others regress. None of those are visible in a pooled figure.

WHAT WOULD NOT. A minor character regressing by a line or two - they are a
small share of what is heard, and the noise on a handful of lines is larger
than the effect.
"""
import argparse, collections, glob, json, os, sys

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
APP = REPO + "/app/"
sys.path.insert(0, APP)
from experiments.stats import exact_mcnemar, clopper_pearson

LEDGER = REPO + "/ab_test_runtime/experiments"
BOOKS = ("grimgar03", "index18", "mushoku16", "owarimonogatari3")
SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE"}
LEAD_SHARE = 0.05


def leads_for(book):
    path = APP + f"fixtures/attribution_gold_{book}.json"
    if not os.path.exists(path):
        return set()
    entries = json.load(open(path))["entries"]
    counts = collections.Counter(e["expected_speaker"].upper() for e in entries)
    total = sum(counts.values())
    return {s for s, c in counts.items()
            if c / total >= LEAD_SHARE and s not in SPECIAL}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=LEDGER + "/shipping_readiness.json")
    args = ap.parse_args()

    base, lora, expected = {}, {}, {}
    for path in sorted(glob.glob(LEDGER + "/lora_serving_eval__*.json")):
        doc = json.load(open(path))
        for row in doc["rows"]:
            target = base if row["arm"] == "base" else lora
            target[row["id"]] = bool(row.get("correct"))
            expected[row["id"]] = (row.get("expected") or "").upper()
    shared = sorted(set(base) & set(lora))
    if not shared:
        print("no shippable-stack artifact with both arms yet")
        return

    print(f"Shippable stack (Q4 base + f16 LoRA via llama.cpp), "
          f"{len(shared)} rows\n")
    print(f"  {'book':18}{'base':>8}{'lora':>8}{'delta':>8}"
          f"{'gained':>8}{'lost':>7}{'p':>10}")
    per_book, blockers = {}, []
    for book in BOOKS:
        ids = [i for i in shared if i.startswith(book + ":")]
        if not ids:
            continue
        b = sum(base[i] for i in ids)
        l = sum(lora[i] for i in ids)
        gained = sum(1 for i in ids if lora[i] and not base[i])
        lost = sum(1 for i in ids if base[i] and not lora[i])
        p = exact_mcnemar(lost, gained)[0]   # returns (p, b, c)
        per_book[book] = {"n": len(ids), "base": b / len(ids),
                          "lora": l / len(ids), "gained": gained,
                          "lost": lost, "p": p}
        print(f"  {book:18}{b/len(ids)*100:7.1f}%{l/len(ids)*100:7.1f}%"
              f"{(l-b)/len(ids)*100:+7.1f}{gained:8}{lost:7}{p:10.4g}")
        if l < b:
            blockers.append(f"{book} regresses ({(l-b)/len(ids)*100:+.1f})")

    print("\n  lead characters (>=5% of a book's lines)")
    print(f"  {'book':16}{'character':22}{'lines':>6}{'base':>8}{'lora':>8}{'delta':>8}")
    per_char = {}
    for book in BOOKS:
        for lead in sorted(leads_for(book)):
            ids = [i for i in shared
                   if i.startswith(book + ":") and expected.get(i) == lead]
            if len(ids) < 8:      # too few lines to say anything
                continue
            b = sum(base[i] for i in ids)
            l = sum(lora[i] for i in ids)
            delta = (l - b) / len(ids)
            per_char[f"{book}/{lead}"] = {"n": len(ids), "delta": delta}
            flag = "  <-- REGRESSION" if l < b else ""
            print(f"  {book[:14]:16}{lead[:20]:22}{len(ids):6}"
                  f"{b/len(ids)*100:7.1f}%{l/len(ids)*100:7.1f}%"
                  f"{delta*100:+7.1f}{flag}")
            if l < b:
                blockers.append(f"{book}/{lead} regresses ({delta*100:+.1f})")

    total_b = sum(base[i] for i in shared)
    total_l = sum(lora[i] for i in shared)
    gained = sum(1 for i in shared if lora[i] and not base[i])
    lost = sum(1 for i in shared if base[i] and not lora[i])
    p = exact_mcnemar(lost, gained)[0]
    lo, hi = clopper_pearson(total_l, len(shared))
    print(f"\n  pooled  base {total_b/len(shared)*100:.1f}%  "
          f"lora {total_l/len(shared)*100:.1f}%  "
          f"({(total_l-total_b)/len(shared)*100:+.1f})  "
          f"+{gained}/-{lost}  p={p:.4g}  [{lo:.1f}-{hi:.1f}]")

    print("\n  VERDICT")
    if not blockers:
        print("    No book and no lead character regresses. The gain is broad "
              "rather than\n    carried by one book, so making the adapter the "
              "default is supported by\n    this evidence.")
    else:
        print("    Regressions found - these are what a listener would notice:")
        for b in blockers:
            print(f"      {b}")
        print("    A pooled gain does not settle it; decide against these.")

    json.dump({"per_book": per_book, "per_character": per_char,
               "pooled": {"base": total_b / len(shared),
                          "lora": total_l / len(shared),
                          "gained": gained, "lost": lost, "p": p},
               "blockers": blockers}, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
