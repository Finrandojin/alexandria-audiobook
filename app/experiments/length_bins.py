"""Does the fixture flatter the models, or penalise them?

§13.8.2 of the addendum proved a mismatch and then asserted a direction it had
not measured: the scored rows are longer than the population of spoken lines,
and I wrote that this "skews the flattering way" on the reasoning that longer
lines carry more evidence. That reasoning is exactly the kind that has gone 0/3
in this investigation. The mismatch is a fact; the direction is a measurement,
and this is the measurement.

Two numbers come out of it:

  raw          accuracy on the rows we actually scored - every number in the
               ledger, including the ~50-70% "plateau"
  reweighted   the same per-bin accuracies, reweighted to the length
               distribution of ALL spoken lines in the book

If reweighted < raw, the fixture flatters and the plateau is optimistic. If
reweighted > raw, it penalises. If they agree within their intervals, the
mismatch exists but does not matter, which is a real result and the one that
would let §13.8.2 stand as written.

Standing assumption, stated because it is the load-bearing one: reweighting
assumes accuracy within a length bin is the same for scored and unscored lines.
That is not verifiable without labelling more lines. What it buys is that
length can no longer be the explanation - any residual bias has to run through
some other property that correlates with length inside a bin.

Offline. Consumes committed artifacts, needs no GPU, disturbs nothing.
"""
import collections
import json, os, re, statistics, sys
sys.path.insert(0, "/home/fakemitch/pinokio/api/alexandria-audiobook2.git/app")
from experiments.stats import clopper_pearson

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
E = REPO + "/ab_test_runtime/experiments/"
M = REPO + "/ab_test_runtime/results/matrix_20260725-115148/"
INPUT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"
BOOKS = (("grimgar03", "attribution_gold_grimgar03_provisional.json"),
         ("mushoku16", "attribution_gold_random.json"))
# Fixed before looking at any accuracy number, so the bin edges cannot be
# chosen to produce a preferred answer.
EDGES = (40, 80, 160)
BIN_NAMES = ("<40", "40-79", "80-159", "160+")


def norm(t):
    return re.sub(r"\W+", "", t or "").lower()


def bin_of(text):
    n = len(text or "")
    for i, edge in enumerate(EDGES):
        if n < edge:
            return BIN_NAMES[i]
    return BIN_NAMES[-1]


def population(book):
    """Every spoken segment in the book, binned. This is the target the fixture
    is supposed to represent."""
    cp = json.load(open(M + INPUT_RUN + f"/{book}/result.json.threepass_checkpoint.json"))
    spoken = [e for e in cp["segmented"] if e.get("type") != "NARRATOR"]
    counts = collections.Counter(bin_of(e.get("text")) for e in spoken)
    total = sum(counts.values())
    return counts, total, spoken


def scored_bins(book, goldfile):
    """Bin every gold row, and record which ones the unique-text filter drops.

    The filter is part of the story: repeated text is short text, so dropping
    it removes short lines and is itself a source of the mismatch.
    """
    gold = json.load(open(REPO + "/app/fixtures/" + goldfile))
    cp = json.load(open(M + INPUT_RUN + f"/{book}/result.json.threepass_checkpoint.json"))
    occ = collections.Counter(norm(e.get("text")) for e in cp["segmented"])
    kept, dropped = {}, []
    for g in gold["entries"]:
        if occ[norm(g["line"])] == 1:
            kept[g["id"]] = bin_of(g["line"])
        else:
            dropped.append(g["line"])
    return kept, dropped


def artifacts_for(book, goldfile):
    """Every artifact scoring this book, as {label: {id: correct}} per arm.

    Matched on the gold file recorded in the artifact's own metadata, never on
    the filename - mushoku16's fixture is named `attribution_gold_random.json`
    and a filename match silently found zero artifacts for it.
    """
    out = {}
    for name in sorted(os.listdir(E)):
        if not name.endswith(".json"):
            continue
        try:
            doc = json.load(open(E + name))
        except (ValueError, OSError):
            continue
        meta = doc.get("meta") or {}
        if os.path.basename(str(meta.get("gold_path", ""))) != goldfile:
            continue
        if meta.get("validation") not in ("ok", None) and not isinstance(
                meta.get("validation"), str):
            continue
        for row in doc.get("rows", []):
            out.setdefault(f"{name[:-5]} :: {row['arm']}", {})[row["id"]] = row["correct"]
    return out


def report(book, goldfile):
    print("=" * 78)
    print(f"{book}")
    print("=" * 78)
    pop, pop_n, spoken = population(book)
    kept, dropped = scored_bins(book, goldfile)

    print(f"\n  population: {pop_n} spoken lines, median "
          f"{statistics.median([len(e.get('text') or '') for e in spoken]):.0f} chars")
    print(f"  fixture   : {len(kept)} scored, {len(dropped)} dropped by the "
          f"unique-text filter"
          + (f" (median {statistics.median([len(d) for d in dropped]):.0f} chars)"
             if dropped else ""))
    print(f"\n  {'bin':8} {'population':>12} {'scored':>10}   share pop / share scored")
    scored_counts = collections.Counter(kept.values())
    for name in BIN_NAMES:
        ps = pop[name] / pop_n * 100 if pop_n else 0
        ss = scored_counts[name] / len(kept) * 100 if kept else 0
        print(f"  {name:8} {pop[name]:12} {scored_counts[name]:10}   "
              f"{ps:6.1f}% / {ss:6.1f}%   {'OVER' if ss > ps + 5 else 'under' if ss < ps - 5 else ''}")

    runs = artifacts_for(book, goldfile)
    if not runs:
        print("\n  no artifacts for this book")
        return
    print(f"\n  {'run :: arm':58} {'raw':>7} {'reweighted':>11} {'delta':>7}")
    shifts = []
    for label, correct in sorted(runs.items()):
        ids = [i for i in correct if i in kept]
        if len(ids) < 40:
            continue
        per_bin = {}
        for name in BIN_NAMES:
            b = [i for i in ids if kept[i] == name]
            if b:
                per_bin[name] = (sum(1 for i in b if correct[i]), len(b))
        raw_k = sum(1 for i in ids if correct[i])
        raw = raw_k / len(ids) * 100
        # Reweight only over bins the fixture actually covers, renormalised, so
        # an uncovered bin cannot be silently imputed at the mean.
        cover = sum(pop[n] for n in per_bin)
        rew = sum(pop[n] / cover * (k / t) for n, (k, t) in per_bin.items()) * 100 \
            if cover else raw
        lo, hi = clopper_pearson(raw_k, len(ids))
        shifts.append(rew - raw)
        print(f"  {label[:58]:58} {raw:6.1f}% {rew:10.1f}% {rew-raw:+6.1f}"
              f"   [{lo:.1f}-{hi:.1f}]")

    if shifts:
        med = statistics.median(shifts)
        agree = sum(1 for s in shifts if (s < 0) == (med < 0))
        print(f"\n  median shift {med:+.1f} points over {len(shifts)} runs, "
              f"{agree}/{len(shifts)} in the same direction")
        print("  " + ("fixture FLATTERS: representative dialogue scores lower"
                      if med < -0.5 else
                      "fixture PENALISES: representative dialogue scores higher"
                      if med > 0.5 else
                      "no material bias: the length mismatch does not move accuracy"))

    print("\n  accuracy by bin, pooled over runs (is length even predictive?)")
    pooled = collections.defaultdict(lambda: [0, 0])
    for label, correct in runs.items():
        for i in correct:
            if i in kept:
                pooled[kept[i]][0] += 1
                pooled[kept[i]][1] += bool(correct[i])
    print(f"    {'bin':8} {'n':>7} {'accuracy':>10}   95% CI")
    for name in BIN_NAMES:
        n, k = pooled[name]
        if not n:
            continue
        lo, hi = clopper_pearson(k, n)
        print(f"    {name:8} {n:7} {k/n*100:9.1f}%   [{lo:.1f}-{hi:.1f}]")
    print("    Pooling over runs is for the SHAPE only - the runs are not "
          "independent,\n    so these intervals are narrower than the truth. "
          "The per-run table above is\n    what carries the direction claim.")


if __name__ == "__main__":
    for book, goldfile in BOOKS:
        report(book, goldfile)
        print()
    print("=" * 78)
    print("Closes §13.8.2 only for LENGTH. A fixture can still be unrepresentative")
    print("in ways that correlate with length inside a bin - scene type, speaker")
    print("frequency, chapter position. Those need their own reweighting.")
