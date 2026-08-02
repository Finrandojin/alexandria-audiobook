"""Does the model assume speakers alternate when they do not?

owarimonogatari3's base 14B scored 40.1% against a previous-speaker floor of
50.0%: worse than repeating the last speaker. `trivial_baselines` found 50 of
its 63 measured arms below that floor, and `committed_history` found the TRUE
previous speaker worth +9.3 there while the model's own prior answer cost 3.1.

All three point at one mechanism: the model changes speaker more often than the
text does. This measures that directly, with no GPU.

THE METRIC IS THE CONTINUATION RATE - P(this line has the same speaker as the
previous one). Gold has a true value per book; every arm has its own. If the
predicted rate is systematically below gold, the model over-alternates, and a
continuation bias is the indicated fix. If it matches, over-alternation is the
wrong explanation and the below-floor result needs another one.

ADJACENCY IS TEXTUAL, NOT ROW-ORDER. Scored rows are a subset of segments, so
two consecutive rows in the fixture can sit far apart in the book with
narration between them. Counting those as adjacent would measure the sampling
of the fixture rather than the turn-taking of the text. Only pairs separated by
at most `--max-gap` segments count, and the pair count is reported so a book
with few genuine adjacencies is visible rather than silently thin.

READINGS, fixed before running:

  predicted << gold      the model over-alternates; a continuation prior is
                         indicated, and it explains the below-floor scores
  predicted ~ gold       over-alternation is not the mechanism, and the
                         below-floor result on owarimonogatari3 is caused by
                         something else - most likely wrong names on correctly
                         grouped lines, which cluster_vs_name can already see
  predicted >> gold      the model under-alternates, sticking to one speaker
                         through genuine turn changes
"""
import argparse, collections, glob, json, os, re, sys

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
APP = REPO + "/app/"
sys.path.insert(0, APP)
from experiments.scoring import alias_groups, same_speaker

M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
LEDGER = REPO + "/ab_test_runtime/experiments"
BOOKS = ("grimgar03", "index18", "mushoku16", "owarimonogatari3")
SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE", ""}


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


def book_context(book):
    """Segment position for every scored line, so adjacency is textual."""
    gold = json.load(open(APP + f"fixtures/attribution_gold_{book}.json"))
    seg = json.load(open(
        M + INPUT_RUN + f"/{book}/result.json.threepass_checkpoint.json"))["segmented"]
    occ = collections.Counter(norm(e.get("text")) for e in seg)
    position, by_id = {}, {}
    for index, entry in enumerate(seg):
        key = norm(entry.get("text"))
        if occ[key] == 1:
            position[key] = index
    for g in gold["entries"]:
        if g["expected_speaker"].upper() in SPECIAL:
            continue
        key = norm(g["line"])
        if key in position:
            by_id[g["id"]] = (position[key], g["expected_speaker"].upper())
    return by_id, alias_groups(gold)


def continuation(pairs, groups):
    """Fraction of adjacent pairs whose speaker does not change."""
    if not pairs:
        return float("nan"), 0
    same = sum(1 for a, b in pairs if same_speaker(a, b, groups))
    return same / len(pairs), len(pairs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-gap", type=int, default=2,
                    help="max segment distance for two scored lines to count "
                         "as adjacent in the text")
    ap.add_argument("--min-pairs", type=int, default=25)
    ap.add_argument("--out", default=REPO + "/ab_test_runtime/experiments/run_lengths.json")
    args = ap.parse_args()

    context = {b: book_context(b) for b in BOOKS}

    print("Gold continuation rate - how often the speaker does NOT change")
    print(f"  {'book':20}{'gold':>8}{'pairs':>8}{'floor':>8}")
    floors = {}
    gold_rate = {}
    baseline_path = REPO + "/ab_test_runtime/experiments/trivial_baselines.json"
    baseline = json.load(open(baseline_path)) if os.path.exists(baseline_path) else {}
    for book in BOOKS:
        by_id, groups = context[book]
        ordered = sorted(by_id.values())
        pairs = [(a[1], b[1]) for a, b in zip(ordered, ordered[1:])
                 if b[0] - a[0] <= args.max_gap]
        rate, n = continuation(pairs, groups)
        gold_rate[book] = rate
        floors[book] = (baseline.get(book, {}) or {}).get("previous")
        shown = f"{floors[book]*100:.1f}%" if floors[book] is not None else "-"
        print(f"  {book:20}{rate*100:7.1f}%{n:8}{shown:>8}")

    results = []
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
        if not isinstance(rows, list) or not rows:
            continue
        if not all(isinstance(r, dict) and "arm" in r for r in rows):
            continue
        by_id, groups = context[book]
        per_arm = collections.defaultdict(list)
        for row in rows:
            info = by_id.get(row.get("id"))
            predicted = (row.get("predicted") or "").upper()
            if info and predicted not in SPECIAL:
                per_arm[row["arm"]].append((info[0], predicted))
        for arm, items in per_arm.items():
            items.sort()
            pairs = [(a[1], b[1]) for a, b in zip(items, items[1:])
                     if b[0] - a[0] <= args.max_gap]
            rate, n = continuation(pairs, groups)
            if n < args.min_pairs:
                continue
            results.append({
                "artifact": name, "book": book, "arm": arm,
                "experiment": doc.get("meta", {}).get("experiment", ""),
                "predicted_continuation": rate, "gold_continuation": gold_rate[book],
                "delta": rate - gold_rate[book], "pairs": n})

    if not results:
        print("\nNo arm had enough textually adjacent pairs.")
        return

    print(f"\n{len(results)} arms with >={args.min_pairs} adjacent pairs")
    print(f"  {'book':20}{'arms':>5}{'gold':>9}{'predicted':>11}{'delta':>8}")
    summary = {}
    for book in BOOKS:
        sub = [r for r in results if r["book"] == book]
        if not sub:
            continue
        mean = sum(r["predicted_continuation"] for r in sub) / len(sub)
        summary[book] = {"gold": gold_rate[book], "predicted_mean": mean,
                         "delta": mean - gold_rate[book], "arms": len(sub)}
        print(f"  {book:20}{len(sub):5}{gold_rate[book]*100:8.1f}%"
              f"{mean*100:10.1f}%{(mean-gold_rate[book])*100:+8.1f}")

    print("\n  most over-alternating arms")
    for r in sorted(results, key=lambda r: r["delta"])[:8]:
        print(f"    {r['book']:18}{r['experiment'][:18]:20}{r['arm'][:12]:14}"
              f"{r['predicted_continuation']*100:6.1f}% vs "
              f"{r['gold_continuation']*100:5.1f}% gold  {r['delta']*100:+6.1f}")

    print("\n  A negative delta means the model changes speaker more often than "
          "the text\n  does. Read it next to that book's previous-speaker "
          "floor: the book where\n  the floor is highest is the book where "
          "over-alternation costs the most.")

    json.dump({"per_book": summary, "arms": results,
               "max_gap": args.max_gap}, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
