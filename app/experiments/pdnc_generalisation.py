"""Goal 1.3: a clean held-out attribution number on books never developed on.

WHY THE EXISTING PDNC NUMBER DOES NOT ANSWER THIS. `pdnc_eval_full.json` scores
28 novels twice, base and LoRA, and the LoRA arm is contaminated: 25 of the 28
were in its training data. The pooled 71.6% it reports is therefore mostly a
memory check.

The base arm is not contaminated. It is the shipped local model with no PDNC
training at all, so every one of the 28 books is held out for it, and the three
that also appear in this project's development work are held out in the weaker
sense of never having been trained on. That distinction is the whole analysis:

    base on the 25 books NOTHING in this project has ever looked at
        = generalisation to unseen books
    base on the 3 development books (Pride and Prejudice, The Awakening,
      The Sign of the Four)
        = the figure those books produced during development

Goal 1.3 asks whether the first is within 5 points of the second.

THE COMPARISON THIS FILE REFUSES TO MAKE. Goal 1.1 records PDNC scoring far
higher than the light novels (80.5% against a 46-67% median) and warns at
length that the gap is *sampling*, not difficulty: the light-novel gold keeps
only lines the deterministic namer failed on, while PDNC takes `entries[:limit]`
unfiltered. Nothing here compares a PDNC number to a light-novel number. Both
sides of every comparison below are PDNC rows drawn the same way.

`quote_type` is reported alongside, because PDNC labels each quote Explicit,
Implicit or Anaphoric, and a book's mix of those is a plausible confound for
any per-book difference. Reporting the mix does not control for it - it makes
the confound visible instead of leaving it implied.
"""
import argparse
import collections
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if not os.path.isdir(os.path.join(REPO, "ab_test_runtime")):
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(REPO, "app"))

FULL = os.path.join(REPO, "ab_test_runtime", "experiments",
                    "pdnc_eval_full.json")

# The three PDNC novels this project already reported figures for in goal 1.1.
# They are "development books" in the sense that their numbers informed
# decisions here - not that anything was trained on them.
DEVELOPMENT_BOOKS = ("PrideAndPrejudice", "TheAwakening", "TheSignOfTheFour")


def book_rows(payload, book, arm):
    return (payload.get(book, {}).get(arm, {}) or {}).get("rows", []) or []


def accuracy(rows):
    if not rows:
        return None
    return 100.0 * sum(1 for r in rows if r.get("correct")) / len(rows)


def quote_mix(rows):
    counts = collections.Counter(r.get("quote_type") or "?" for r in rows)
    total = sum(counts.values()) or 1
    return {k: round(100.0 * v / total, 1) for k, v in counts.most_common()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", default="base",
                    choices=("base", "lora"),
                    help="base is the uncontaminated arm; lora is included "
                         "only so the contamination can be shown, not used")
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "pdnc_generalisation.json"))
    args = ap.parse_args()

    with open(FULL, encoding="utf-8") as handle:
        payload = json.load(handle)

    books = sorted(payload)
    per_book = {}
    for book in books:
        rows = book_rows(payload, book, args.arm)
        acc = accuracy(rows)
        if acc is None:
            continue
        per_book[book] = {"accuracy": round(acc, 1), "n": len(rows),
                          "quote_mix": quote_mix(rows)}

    held_out = {b: v for b, v in per_book.items()
                if b not in DEVELOPMENT_BOOKS}
    development = {b: v for b, v in per_book.items()
                   if b in DEVELOPMENT_BOOKS}

    def pooled(group):
        rows = [r for b in group for r in book_rows(payload, b, args.arm)]
        return {"books": len(group), "n": len(rows),
                "accuracy": round(accuracy(rows), 1) if rows else None}

    ho, dev = pooled(held_out), pooled(development)
    gap = (round(ho["accuracy"] - dev["accuracy"], 1)
           if ho["accuracy"] is not None and dev["accuracy"] is not None
           else None)

    values = sorted(v["accuracy"] for v in per_book.values())
    result = {
        "scope": f"PDNC only, {args.arm} arm; every comparison is PDNC rows "
                 "sampled the same way. No light-novel figure appears here.",
        "arm": args.arm,
        "development_books": list(DEVELOPMENT_BOOKS),
        "held_out": ho, "development": dev, "gap_points": gap,
        "per_book_spread": {
            "min": values[0], "max": values[-1],
            "median": round(statistics.median(values), 1),
            "iqr": [round(statistics.quantiles(values, n=4)[0], 1),
                    round(statistics.quantiles(values, n=4)[2], 1)],
        },
        "per_book": per_book,
    }

    from utils import atomic_json_write
    atomic_json_write(result, args.out)

    print(f"=== PDNC {args.arm} arm ===")
    print(f"  held out ({ho['books']} books, n={ho['n']}): {ho['accuracy']}%")
    print(f"  development ({dev['books']} books, n={dev['n']}): "
          f"{dev['accuracy']}%")
    print(f"  gap: {gap:+} points   target: within 5")
    s = result["per_book_spread"]
    print(f"  per-book spread: {s['min']}% to {s['max']}%, "
          f"median {s['median']}%, IQR {s['iqr']}")
    print("\n  worst five books:")
    for book, v in sorted(per_book.items(), key=lambda kv: kv[1]["accuracy"])[:5]:
        print(f"    {book[:30]:32} {v['accuracy']:5.1f}%  {v['quote_mix']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
