"""What does each book score for free, before any model runs?

`owarimonogatari3` was found to collapse to the frequency prior at 36.9% while
other books sit between 1.2% and 17.3%. That number has been quoted as a
property of the book, but the floors themselves were never computed, so a 55%
on one book and a 55% on another have been compared as though they meant the
same thing. They may not.

Four baselines, none of which reads a single line of text:

    majority        always the most frequent speaker in the book
    alternate       strict alternation between the two most frequent speakers,
                    which is what a two-hander conversation looks like
    previous        the speaker of the previous scored line - the "no speaker
                    change" baseline
    round_robin     cycle through the book's speakers in frequency order, as a
                    control that is structured but carries no information

BASELINES ARE COMPUTED ON THE SAME ROWS THE HARNESSES SCORE - single-occurrence
lines with a real speaker - so they are directly comparable to every accuracy
in the ledger rather than to a different denominator.

WHAT THIS IS FOR. A model beating `majority` by two points on a book whose
majority is 37% has demonstrated almost nothing; the same two points over a 5%
floor is a different claim. This does not measure any model. It measures how
much of each book is free.
"""
import argparse, collections, json, os, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = REPO + "/app/"
sys.path.insert(0, APP)
from experiments.scoring import alias_groups, same_speaker
from experiments.stats import clopper_pearson

M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
BOOKS = ("grimgar03", "index18", "mushoku16", "owarimonogatari3")
SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE"}


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


def scored_rows(book):
    """The same row set the harnesses use: unique lines with a real speaker,
    in reading order."""
    gold = json.load(open(APP + f"fixtures/attribution_gold_{book}.json"))
    seg = json.load(open(
        M + INPUT_RUN + f"/{book}/result.json.threepass_checkpoint.json"))["segmented"]
    occ = collections.Counter(norm(e.get("text")) for e in seg)
    want = {norm(g["line"]): g for g in gold["entries"]
            if occ[norm(g["line"])] == 1
            and g["expected_speaker"].upper() not in SPECIAL}
    ordered = []
    for entry in seg:
        g = want.get(norm(entry.get("text")))
        if g is not None:
            ordered.append(g["expected_speaker"].upper())
    return ordered, alias_groups(gold)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=REPO + "/ab_test_runtime/experiments/trivial_baselines.json")
    args = ap.parse_args()

    out = {}
    print(f"  {'book':18}{'n':>5}{'majority':>10}{'alternate':>11}"
          f"{'previous':>10}{'round_robin':>13}{'speakers':>10}")
    for book in BOOKS:
        truth, groups = scored_rows(book)
        if not truth:
            continue
        n = len(truth)
        freq = collections.Counter(truth)
        ranked = [s for s, _ in freq.most_common()]

        def score(predictions):
            return sum(1 for p, t in zip(predictions, truth)
                       if same_speaker(t, p, groups)) / n

        majority = score([ranked[0]] * n)
        pair = ranked[:2] if len(ranked) > 1 else ranked * 2
        alternate = score([pair[i % 2] for i in range(n)])
        # "previous" cannot answer the first line; it is counted wrong rather
        # than dropped, so every baseline shares one denominator.
        previous = score([None] + truth[:-1])
        robin = score([ranked[i % len(ranked)] for i in range(n)])
        lo, hi = clopper_pearson(round(majority * n), n)
        out[book] = {"n": n, "majority": majority, "majority_ci": [lo, hi],
                     "alternate": alternate, "previous": previous,
                     "round_robin": robin, "speakers": len(freq),
                     "top_speaker": ranked[0]}
        print(f"  {book:18}{n:5}{majority*100:9.1f}%{alternate*100:10.1f}%"
              f"{previous*100:9.1f}%{robin*100:12.1f}%{len(freq):10}")

    print("\n  strongest free baseline per book, and what it implies")
    for book, d in out.items():
        best = max(("majority", "alternate", "previous", "round_robin"),
                   key=lambda k: d[k])
        print(f"    {book:18} {best:11} {d[best]*100:5.1f}%   "
              f"(top speaker {d['top_speaker']}, {d['speakers']} speakers)")
    print("\n  Read every accuracy in the ledger against its book's floor. "
          "Two books\n  with the same score are not the same result if their "
          "floors differ, and\n  the floors here differ by more than 20 points.")

    json.dump(out, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
