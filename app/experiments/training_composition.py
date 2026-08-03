"""Does WHICH books the teacher labelled predict where the adapter helps?

The learning curve saturates near 800 rows, so volume is exhausted and
composition is the remaining variable. One pattern is visible in what we have:

    training books        grimgar06, mushoku18 (+ arc4_volume10wn, mushoku23)
    mushoku16             same series as mushoku18 - largest gain, +9.0
    grimgar03             same series as grimgar06 - +4.2
    owarimonogatari3      no same-series data - +4.3, and 3 of the 5 character
                          regressions
    index18               no same-series data - +7.6, the SECOND LARGEST gain

index18 already breaks it at the book level. That is why this is run at
character level, where there are roughly twenty points rather than four, and
why the result is reported as suggestive.

THIS IS UNDERPOWERED BY CONSTRUCTION AND THE FINDING CANNOT BE PROMOTED. Four
books, two series with in-training data and two without, is not enough to
establish a compositional effect - and a four-point pattern is exactly what was
just retracted for the contiguity mechanism. A positive result here means "worth
testing with a fifth book from an untrained series", nothing more. A negative
result is more informative: it would say composition does not obviously matter
either, which together with saturation would close the training-data question
entirely.
"""
import argparse, collections, glob, json, os, sys

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
APP = REPO + "/app/"
sys.path.insert(0, APP)
from experiments.stats import exact_mcnemar

LEDGER = REPO + "/ab_test_runtime/experiments"
BOOKS = ("grimgar03", "index18", "mushoku16", "owarimonogatari3")
SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE"}
LEAD_SHARE = 0.05

# Series membership, by the book naming already used throughout the corpus.
SERIES = {"grimgar03": "grimgar", "grimgar06": "grimgar",
          "mushoku16": "mushoku", "mushoku18": "mushoku",
          "mushoku23": "mushoku", "index18": "index",
          "owarimonogatari3": "owarimonogatari",
          "arc4_volume10wn": "arc4"}
TRAINED_ON = ("grimgar06", "mushoku18", "arc4_volume10wn", "mushoku23")


def leads_for(book):
    path = APP + f"fixtures/attribution_gold_{book}.json"
    if not os.path.exists(path):
        return {}
    entries = json.load(open(path))["entries"]
    counts = collections.Counter(e["expected_speaker"].upper() for e in entries)
    total = sum(counts.values())
    return {s: c for s, c in counts.items()
            if c / total >= LEAD_SHARE and s not in SPECIAL}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=LEDGER + "/training_composition.json")
    args = ap.parse_args()

    trained_series = {SERIES[b] for b in TRAINED_ON if b in SERIES}
    base, lora, expected = {}, {}, {}
    for path in sorted(glob.glob(LEDGER + "/lora_serving_eval__local-rocm*.json")):
        doc = json.load(open(path))
        for row in doc["rows"]:
            target = base if row["arm"] == "base" else lora
            target[row["id"]] = bool(row.get("correct"))
            expected[row["id"]] = (row.get("expected") or "").upper()
    shared = sorted(set(base) & set(lora))
    if not shared:
        print("no shippable-stack artifact with both arms")
        return

    print(f"training books: {', '.join(TRAINED_ON)}")
    print(f"series with in-training data: {', '.join(sorted(trained_series))}\n")

    print(f"  {'book':18}{'series seen':>13}{'base':>8}{'lora':>8}{'delta':>8}")
    buckets = {True: [0, 0, 0], False: [0, 0, 0]}   # n, base_ok, lora_ok
    for book in BOOKS:
        ids = [i for i in shared if i.startswith(book + ":")]
        if not ids:
            continue
        seen = SERIES.get(book) in trained_series
        b = sum(base[i] for i in ids)
        l = sum(lora[i] for i in ids)
        buckets[seen][0] += len(ids)
        buckets[seen][1] += b
        buckets[seen][2] += l
        print(f"  {book:18}{str(seen):>13}{b/len(ids)*100:7.1f}%"
              f"{l/len(ids)*100:7.1f}%{(l-b)/len(ids)*100:+7.1f}")

    print(f"\n  pooled by whether the series appears in training")
    for seen in (True, False):
        n, b, l = buckets[seen]
        if n:
            print(f"    series seen={str(seen):<5} {n:4} rows  "
                  f"base {b/n*100:5.1f}%  lora {l/n*100:5.1f}%  "
                  f"{(l-b)/n*100:+.1f}")

    # Character level: more points, and the regressions live here.
    print(f"\n  lead characters, by whether their series was in training")
    rows_by_seen = {True: [], False: []}
    for book in BOOKS:
        seen = SERIES.get(book) in trained_series
        for lead, count in leads_for(book).items():
            ids = [i for i in shared
                   if i.startswith(book + ":") and expected.get(i) == lead]
            if len(ids) < 8:
                continue
            b = sum(base[i] for i in ids)
            l = sum(lora[i] for i in ids)
            rows_by_seen[seen].append({"book": book, "lead": lead,
                                       "n": len(ids),
                                       "delta": (l - b) / len(ids)})
    for seen in (True, False):
        group = rows_by_seen[seen]
        if not group:
            continue
        deltas = [g["delta"] for g in group]
        regressions = sum(1 for d in deltas if d < 0)
        mean = sum(deltas) / len(deltas)
        print(f"    series seen={str(seen):<5} {len(group):2} characters  "
              f"mean {mean*100:+5.1f}  regressions {regressions}/{len(group)}")

    seen_mean = (sum(g["delta"] for g in rows_by_seen[True])
                 / max(len(rows_by_seen[True]), 1))
    unseen_mean = (sum(g["delta"] for g in rows_by_seen[False])
                   / max(len(rows_by_seen[False]), 1))
    print(f"\n  character-level difference: "
          f"{(seen_mean - unseen_mean)*100:+.1f} points in favour of "
          f"{'seen' if seen_mean > unseen_mean else 'unseen'} series")
    print("\n  SUGGESTIVE ONLY. Two series with in-training data and two "
          "without cannot\n  establish a compositional effect, and index18 - "
          "no same-series data, second\n  largest gain - already contradicts "
          "the simple story at book level. A four\n  point pattern is exactly "
          "what was retracted for the contiguity mechanism.\n  Testing this "
          "properly needs a fifth book from an untrained series.")

    json.dump({"trained_on": list(TRAINED_ON),
               "book_level": {b: None for b in BOOKS},
               "character_level": {"seen": rows_by_seen[True],
                                   "unseen": rows_by_seen[False]},
               "caveat": "underpowered; 2 series in-training vs 2 not"},
              open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
