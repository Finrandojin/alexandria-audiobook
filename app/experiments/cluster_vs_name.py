"""Does the model know WHO is speaking but not WHAT they are called?

"Selection not recall" has driven this investigation for days: the roster holds
the right name for about 85% of lines and the model picks it about 30% of the
time. Every intervention aimed at that gap has failed - roster quality +2.2
(ns), scene cast null, closed-set -9.4. That pattern is what you would expect
if the interventions were aimed at the wrong thing.

This asks a question none of them asked: ignore the names entirely and ask
whether the model PARTITIONS the lines correctly. If it consistently assigns
one wrong-but-stable label to every line a single character speaks, it is
tracking the conversation and failing only to bind a name - a different and
much cheaper problem than not knowing who is talking.

THREE NUMBERS PER ARM, over exactly the same rows:

    accuracy      what the ledger already reports
    renamed       accuracy after relabelling each predicted cluster with the
                  gold speaker it most often covers
    ARI           adjusted Rand index of the predicted partition against the
                  gold partition, names discarded entirely

`renamed` IS AN ORACLE AND NOT SHIPPABLE. The mapping is fitted on the answers,
so it cannot be achieved by anything that has to run before seeing them. It is
an upper bound on what fixing name-binding alone could buy, and reporting it as
an achievable score would repeat exactly the closed-oracle mistake this ledger
already had to retract. ARI is the honest structural measure: it is corrected
for chance and uses no gold labels at assignment time.

READINGS, fixed before running:

  renamed >> accuracy, ARI high     the model tracks speakers and misnames
                                    them; name-binding is the problem and the
                                    roster interventions were aimed wrong
  renamed ~ accuracy, ARI low       it does not know who is speaking; every
                                    name-supply intervention was doomed, which
                                    would explain why all of them failed
  renamed >> accuracy, ARI low      the gain is an artifact of collapsing many
                                    predicted labels onto few gold ones - check
                                    the cluster count before believing it

That last reading is the trap. A model that answers with one name everywhere
gets a high `renamed` on a book with one dominant speaker, so the number of
distinct predicted clusters is printed next to every row.
"""
import argparse, collections, glob, json, os, sys

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
APP = REPO + "/app/"
sys.path.insert(0, APP)
from experiments.scoring import alias_groups, normalize, same_speaker

LEDGER = REPO + "/ab_test_runtime/experiments"
BOOKS = ("grimgar03", "index18", "mushoku16", "owarimonogatari3")
SPECIAL = {"UNKNOWN", "UNNAMED", "NOT_DIALOGUE", ""}


def canonical(name, groups):
    """Collapse aliases so KUZAK and KUZAKU are one gold speaker, not two."""
    up = (name or "").upper()
    for group in groups:
        if any(same_speaker(up, member, groups) for member in group):
            return sorted(group)[0]
    return normalize(up) or up


def adjusted_rand(labels_a, labels_b):
    """ARI between two partitions of the same items. Chance-corrected, so a
    model that guesses at the base rate scores 0 rather than the base rate."""
    table = collections.Counter(zip(labels_a, labels_b))
    rows = collections.Counter(labels_a)
    cols = collections.Counter(labels_b)
    n = len(labels_a)
    if n < 2:
        return float("nan")

    def c2(x):
        return x * (x - 1) / 2

    sum_ij = sum(c2(v) for v in table.values())
    sum_i = sum(c2(v) for v in rows.values())
    sum_j = sum(c2(v) for v in cols.values())
    expected = sum_i * sum_j / c2(n) if c2(n) else 0.0
    maximum = (sum_i + sum_j) / 2
    return (sum_ij - expected) / (maximum - expected) if maximum != expected else 0.0


def gold_groups():
    groups = {}
    for book in BOOKS:
        path = APP + f"fixtures/attribution_gold_{book}.json"
        if os.path.exists(path):
            groups[book] = alias_groups(json.load(open(path)))
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--min_rows", type=int, default=80)
    ap.add_argument("--out", default=REPO + "/ab_test_runtime/experiments/cluster_vs_name.json")
    args = ap.parse_args()

    groups_by_book = gold_groups()
    results = []
    for path in sorted(glob.glob(LEDGER + "/*.json")):
        name = os.path.basename(path)
        if name.endswith(".ckpt"):
            continue
        book = next((b for b in BOOKS if b in name), None)
        if not book or book not in groups_by_book:
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
        groups = groups_by_book[book]
        by_arm = collections.defaultdict(list)
        for row in rows:
            by_arm[row["arm"]].append(row)
        for arm, arm_rows in by_arm.items():
            # Rows the model did not answer carry no partition information, and
            # keeping them would let an arm that answered less look structurally
            # cleaner. They stay in `accuracy` and leave the clustering.
            usable = [r for r in arm_rows
                      if (r.get("predicted") or "").upper() not in SPECIAL
                      and (r.get("expected") or "").upper() not in SPECIAL]
            if len(usable) < args.min_rows:
                continue
            truth = [canonical(r["expected"], groups) for r in usable]
            pred = [canonical(r["predicted"], groups) for r in usable]
            acc = sum(1 for r in arm_rows if r.get("correct")) / len(arm_rows)

            # Best possible relabelling of each predicted cluster. Fitted on
            # the answers - an upper bound, never a score.
            majority = {}
            for cluster in set(pred):
                counts = collections.Counter(
                    t for p, t in zip(pred, truth) if p == cluster)
                majority[cluster] = counts.most_common(1)[0][0]
            renamed = sum(1 for p, t in zip(pred, truth)
                          if majority[p] == t) / len(usable)
            results.append({
                "artifact": name, "experiment": doc.get("meta", {}).get("experiment", ""),
                "book": book, "arm": arm, "model": doc.get("meta", {}).get("model", ""),
                "n": len(arm_rows), "n_clustered": len(usable),
                "accuracy": acc, "renamed": renamed,
                "ari": adjusted_rand(pred, truth),
                "pred_clusters": len(set(pred)), "gold_clusters": len(set(truth))})

    if not results:
        print("No artifact had enough answered rows.")
        return

    results.sort(key=lambda r: r["renamed"] - r["accuracy"], reverse=True)
    print(f"{len(results)} arms with >={args.min_rows} answered rows\n")
    print(f"  {'book':17}{'experiment':22}{'arm':14}{'acc':>7}{'renamed':>9}"
          f"{'gain':>7}{'ARI':>7}{'clusters':>10}")
    for r in results[:22]:
        print(f"  {r['book']:17}{r['experiment'][:20]:22}{r['arm'][:12]:14}"
              f"{r['accuracy']*100:6.1f}%{r['renamed']*100:8.1f}%"
              f"{(r['renamed']-r['accuracy'])*100:+7.1f}{r['ari']:7.3f}"
              f"{r['pred_clusters']:6}/{r['gold_clusters']:<4}")

    print(f"\n  by book (mean over arms)")
    print(f"  {'book':18}{'arms':>5}{'acc':>8}{'renamed':>9}{'gain':>7}{'ARI':>8}")
    for book in BOOKS:
        sub = [r for r in results if r["book"] == book]
        if not sub:
            continue
        m = lambda k: sum(r[k] for r in sub) / len(sub)
        print(f"  {book:18}{len(sub):5}{m('accuracy')*100:7.1f}%"
              f"{m('renamed')*100:8.1f}%{(m('renamed')-m('accuracy'))*100:+7.1f}"
              f"{m('ari'):8.3f}")

    overall = sum(r["renamed"] - r["accuracy"] for r in results) / len(results)
    mean_ari = sum(r["ari"] for r in results) / len(results)
    print(f"\n  mean name-binding headroom {overall*100:+.1f} points, "
          f"mean ARI {mean_ari:.3f}")
    print("  ARI near 0 means the partition is no better than chance, and a "
          "large\n  renamed-gain beside it is cluster collapse rather than "
          "structure - read\n  the cluster counts before believing any gain.")
    print("  `renamed` is fitted on the answers. It bounds what fixing "
          "name-binding\n  could buy; it is not a score anything can achieve.")

    json.dump({"arms": results,
               "mean_gain": overall, "mean_ari": mean_ari,
               "caveat": "renamed is an oracle relabelling fitted on gold; it "
                         "is an upper bound on name-binding fixes, not an "
                         "achievable accuracy."},
              open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
