"""Can a router PREDICT the right method per book, or only explain it after?

Every routing claim in this investigation has been fitted: the best arm per
book was read off the results, and the gap to a single fixed method was called
the value of routing. That number is not achievable by anything that has to
decide before seeing the answers.

This evaluates the realizable version. For each held-out book, both the fixed
method and the router's rule are chosen using ONLY the other books, and scored
on the held-out one. Nothing here is allowed to see the held-out book's gold
when making its choice.

    loo-fixed    the arm with the best mean accuracy on the OTHER books, applied
                 blindly. This is what shipping one setting looks like.
    loo-router   1-nearest-neighbour on a book-level text feature: find the most
                 similar other book and copy the arm that won there.
    oracle       the best arm on the held-out book itself. NOT achievable, and
                 present only to bound the gap.

FEATURES ARE COMPUTED FROM THE TEXT, NEVER FROM GOLD - first-person pronoun
density, speech-tag density, dialogue fraction, mean spoken length. A router
using gold-derived features could not run in production, which is the whole
failure this script exists to avoid repeating.

WHAT THIS CAN AND CANNOT SHOW. Four books means four test points per family.
That is enough to show routing FAILING - if loo-router does not beat loo-fixed
across families, the idea is dead on the evidence available. It is not enough
to establish that routing works: a win on four points, each a different book,
would be suggestive and nothing more, and must not be reported otherwise.
`style_routing` already found that two section-level features route the wrong
way, so the prior here is unfavourable.

The honest headline is the oracle gap. If oracle barely beats loo-fixed, there
is nothing for any router to win and the question closes regardless of feature
choice.
"""
import argparse, collections, glob, json, os, re, statistics, sys

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
APP = REPO + "/app/"
sys.path.insert(0, APP)

M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
LEDGER = REPO + "/ab_test_runtime/experiments"
BOOKS = ("grimgar03", "index18", "mushoku16", "owarimonogatari3")
FIRST = re.compile(r"\b(i|me|my|mine|myself|we|us|our)\b", re.I)
TAG = re.compile(r"\b(said|asked|replied|shouted|whispered|muttered|called|"
                 r"answered|cried|added|continued|murmured|yelled)\b", re.I)


def book_features(book):
    """Book-level descriptors available at inference time. No gold is touched."""
    path = M + INPUT_RUN + f"/{book}/result.json.threepass_checkpoint.json"
    seg = json.load(open(path))["segmented"]
    narration = [e for e in seg if e.get("type") == "NARRATOR"]
    spoken = [e for e in seg if e.get("type") != "NARRATOR"]
    words = " ".join((e.get("text") or "") for e in narration).split()
    per_1k = (len(FIRST.findall(" ".join(words))) / max(len(words), 1)) * 1000
    tags = (len(TAG.findall(" ".join(words))) / max(len(words), 1)) * 1000
    return {
        "first_person_per_1k": per_1k,
        "speech_tags_per_1k": tags,
        "dialogue_fraction": len(spoken) / max(len(seg), 1),
        "mean_spoken_chars": statistics.mean(
            [len(e.get("text") or "") for e in spoken]) if spoken else 0.0,
    }


def load_ledger():
    """(experiment, book) -> {arm: {row_id: correct}} from the written artifacts.

    Arms are compared on the intersection of row ids within a book, because an
    arm that answered fewer rows would otherwise look better for having skipped
    the hard ones - the defect that made w4 read 11 points high.
    """
    table = collections.defaultdict(lambda: collections.defaultdict(dict))
    for path in sorted(glob.glob(LEDGER + "/*.json")):
        name = os.path.basename(path)
        if name.endswith(".ckpt"):
            continue
        book = next((b for b in BOOKS if b in name), None)
        if not book:
            continue
        try:
            doc = json.load(open(path))
        except Exception:
            continue
        rows = doc.get("rows")
        if not rows or "summary" not in doc:
            continue
        family = doc.get("meta", {}).get("experiment") or name.split("__")[0]
        for row in rows:
            arm = row.get("arm")
            rid = row.get("id")
            if arm is None or rid is None:
                continue
            table[(family, book)][arm][rid] = bool(row.get("correct"))
    return table


def paired_accuracy(arms):
    """Accuracy per arm over rows every arm answered."""
    if len(arms) < 2:
        return {}
    shared = set.intersection(*(set(v) for v in arms.values()))
    if len(shared) < 25:
        return {}
    return {a: sum(v[r] for r in shared) / len(shared) for a, v in arms.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--min_books", type=int, default=3,
                    help="families measured on fewer books cannot be leave-one-out tested")
    ap.add_argument("--out", default=REPO + "/ab_test_runtime/experiments/realizable_router.json")
    args = ap.parse_args()

    feats = {b: book_features(b) for b in BOOKS}
    print("Book features (text-derived, no gold):")
    print(f"  {'book':20}{'1st/1k':>9}{'tags/1k':>9}{'dialog%':>9}{'spoken ch':>11}")
    for b in BOOKS:
        f = feats[b]
        print(f"  {b:20}{f['first_person_per_1k']:9.1f}{f['speech_tags_per_1k']:9.1f}"
              f"{f['dialogue_fraction']*100:8.1f}%{f['mean_spoken_chars']:11.1f}")

    table = load_ledger()
    families = collections.defaultdict(dict)
    for (family, book), arms in table.items():
        acc = paired_accuracy(arms)
        if acc:
            families[family][book] = acc

    keys = [k for k in FEATURES_USED]
    results, rows_out = [], []
    print(f"\n{'family':28}{'books':>6}{'loo-fixed':>11}{'loo-router':>12}"
          f"{'oracle':>9}{'router-fixed':>14}")
    for family, per_book in sorted(families.items()):
        # Only arms present in every book can be compared across books; an arm
        # measured on one book cannot be chosen for another.
        common = set.intersection(*(set(a) for a in per_book.values())) \
            if per_book else set()
        usable = {b: {a: v for a, v in acc.items() if a in common}
                  for b, acc in per_book.items() if len(common) >= 2}
        usable = {b: a for b, a in usable.items() if len(a) >= 2}
        if len(usable) < args.min_books:
            continue
        fixed_s, router_s, oracle_s = [], [], []
        for held in usable:
            others = {b: a for b, a in usable.items() if b != held}
            # loo-fixed: best mean arm on the other books
            mean = {a: statistics.mean(o[a] for o in others.values())
                    for a in common}
            fixed_arm = max(mean, key=mean.get)
            # loo-router: copy the winner from the most similar other book
            def distance(b):
                return sum(abs(feats[held][k] - feats[b][k]) / (
                    max(abs(feats[x][k]) for x in BOOKS) or 1) for k in keys)
            near = min(others, key=distance)
            router_arm = max(others[near], key=others[near].get)
            oracle_arm = max(usable[held], key=usable[held].get)
            fixed_s.append(usable[held][fixed_arm])
            router_s.append(usable[held][router_arm])
            oracle_s.append(usable[held][oracle_arm])
            rows_out.append({"family": family, "held_out": held,
                             "nearest": near, "fixed_arm": fixed_arm,
                             "router_arm": router_arm, "oracle_arm": oracle_arm,
                             "fixed": usable[held][fixed_arm],
                             "router": usable[held][router_arm],
                             "oracle": usable[held][oracle_arm]})
        f, r, o = (statistics.mean(fixed_s), statistics.mean(router_s),
                   statistics.mean(oracle_s))
        results.append({"family": family, "books": len(usable), "arms": sorted(common),
                        "loo_fixed": f, "loo_router": r, "oracle": o})
        print(f"  {family:26}{len(usable):>6}{f*100:10.1f}%{r*100:11.1f}%"
              f"{o*100:8.1f}%{(r-f)*100:+13.1f}")

    if not results:
        print("\nNo family is measured on enough books with shared arms.")
        return
    d_router = statistics.mean(x["loo_router"] - x["loo_fixed"] for x in results)
    d_oracle = statistics.mean(x["oracle"] - x["loo_fixed"] for x in results)
    wins = sum(1 for x in results if x["loo_router"] > x["loo_fixed"])
    print(f"\n  across {len(results)} families")
    print(f"    router - fixed   {d_router*100:+.2f} points   "
          f"(router ahead in {wins}/{len(results)})")
    print(f"    oracle - fixed   {d_oracle*100:+.2f} points   "
          f"<- the whole prize any router competes for")

    # Families where one arm wins in every book need no router at all, and
    # averaging them in drags both figures toward zero for a reason that has
    # nothing to do with routing. The families with a non-zero oracle gap are
    # the only ones where the question is live.
    live = [x for x in results if x["oracle"] - x["loo_fixed"] > 1e-9]
    if live:
        lr = statistics.mean(x["loo_router"] - x["loo_fixed"] for x in live)
        lo = statistics.mean(x["oracle"] - x["loo_fixed"] for x in live)
        lw = sum(1 for x in live if x["loo_router"] > x["loo_fixed"])
        print(f"\n  restricted to the {len(live)} families where the best arm "
              f"is NOT the same in every book")
        print(f"    router - fixed   {lr*100:+.2f} points   "
              f"(router ahead in {lw}/{len(live)})")
        print(f"    oracle - fixed   {lo*100:+.2f} points")
        print(f"    The other {len(results)-len(live)} families need no router: "
              f"one arm wins everywhere.")
    print("\n  The oracle gap bounds every possible router. If it is small, no "
          "feature\n  choice rescues routing. A router below fixed is choosing "
          "worse than not\n  choosing, which is the outcome style_routing's "
          "section-level features had.")
    print("  Four books means four test points per family: this can show "
          "routing failing,\n  it cannot establish that routing works.")

    json.dump({"features": feats, "families": results, "folds": rows_out,
               "caveat": "Leave-one-out over 4 books. Negative results are "
                         "informative; a positive result here is suggestive "
                         "only and must not be reported as established."},
              open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


FEATURES_USED = ("first_person_per_1k", "speech_tags_per_1k",
                 "dialogue_fraction", "mean_spoken_chars")

if __name__ == "__main__":
    main()
