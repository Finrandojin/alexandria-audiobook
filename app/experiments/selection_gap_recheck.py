"""Goal 1.2's selection gap, re-measured on the model that actually ships.

WHY THIS EXISTS. Goal 1.2 is built on two numbers: the candidate roster
contains the true speaker 85% of the time, and the model picks it 29.9% of the
time. That 29.9% came from a run whose `source_run` is **qwen3.5-9b**, and a
later six-model comparison put the shipped qwen3-14b about 17 points ahead on
this exact task. The goal itself says, in bold: re-measure before spending
anything on it. A 55-point gap between recall and selection is a different
research programme from a 25-point one, and the difference decides whether the
goal is the largest opportunity in the app or a mostly-closed one.

NO GPU REQUIRED. Every qwen3-14b arm already on disk records `in_candidates`
per row - whether the true speaker was in the roster the model was shown - so
selection is recoverable from artifacts already committed. The measurement was
sitting in the evidence tree the whole time.

WHICH ARM, AND WHY IT MATTERS. `closed_set`'s **open** arm is the one to read:
the model gets its ordinary generous roster and answers freely. Its sibling
arms are diagnostic instruments, not shipping behaviour - `closed-oracle` is
handed a shortlist built from the answer, and `closed-6` is truncated to six
names and demonstrably loses the true one. Pooling those with `open` would
mix three different questions and produce a number describing none of them.

WHAT THE TWO RATES MEAN, KEPT SEPARATE.

    roster recall = rows where the true speaker was in the roster at all
    selection     = OF THOSE ROWS, how often the model chose it

Selection is conditional on recall. Reporting overall accuracy in its place is
the error the goal exists to prevent, because it blends "the name was missing"
with "the name was there and was passed over" - and only the second is what
1.2 is about.
"""
import argparse
import collections
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if not os.path.isdir(os.path.join(REPO, "ab_test_runtime")):
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "app"))

EXPERIMENTS = os.path.join(REPO, "ab_test_runtime", "experiments")
OPEN_ARM = "open"
BOOKS = ("grimgar03", "index18", "mushoku16", "owarimonogatari3")

# The qwen3.5-9b figures goal 1.2 was written around.
LEGACY_RECALL = 85.0
LEGACY_SELECTION = 29.9


def candidate_files(book):
    """closed_set artifacts for one book, local runs preferred.

    Several runs of the same book exist across backends. They are not averaged:
    a backend difference is a real difference in conditions, and averaging over
    it would hide any disagreement. One file is chosen and the rest are
    reported as a spread check.
    """
    hits = sorted(glob.glob(os.path.join(
        EXPERIMENTS, f"closed_set__{book}__qwen__qwen3-14b*.json")))
    local = [p for p in hits if "local" in os.path.basename(p)]
    return (local or hits), hits


def rates(rows):
    """-> (recall %, selection %, overall %, n, n_in_roster)."""
    total = len(rows)
    in_roster = [r for r in rows if r.get("in_candidates")]
    if not total:
        return None
    picked = sum(1 for r in in_roster if r.get("correct"))
    overall = sum(1 for r in rows if r.get("correct"))
    return {
        "n": total,
        "n_in_roster": len(in_roster),
        "roster_recall_pct": round(100.0 * len(in_roster) / total, 1),
        "selection_pct": (round(100.0 * picked / len(in_roster), 1)
                          if in_roster else None),
        "overall_accuracy_pct": round(100.0 * overall / total, 1),
    }


def open_rows(path):
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle).get("rows") or []
    return [r for r in rows if r.get("arm") == OPEN_ARM]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(
        EXPERIMENTS, "selection_gap_recheck.json"))
    args = ap.parse_args()

    per_book, spread, pooled_rows = {}, {}, []
    for book in BOOKS:
        chosen, every = candidate_files(book)
        if not chosen:
            continue
        rows = open_rows(chosen[0])
        if not rows:
            continue
        per_book[book] = rates(rows)
        per_book[book]["artifact"] = os.path.basename(chosen[0])
        pooled_rows.extend(rows)
        variants = {}
        for path in every:
            other = open_rows(path)
            if other:
                r = rates(other)
                variants[os.path.basename(path)] = r["selection_pct"]
        spread[book] = variants

    pooled = rates(pooled_rows)
    result = {
        "scope": "qwen3-14b, closed_set OPEN arm only; selection is "
                 "conditional on the true speaker being in the roster",
        "legacy": {"model": "qwen3.5-9b",
                   "roster_recall_pct": LEGACY_RECALL,
                   "selection_pct": LEGACY_SELECTION},
        "pooled": pooled, "per_book": per_book,
        "selection_by_artifact": spread,
    }

    from utils import atomic_json_write
    atomic_json_write(result, args.out)

    print("=== selection gap on qwen3-14b (shipped) ===")
    print(f"  pooled n={pooled['n']}  roster recall "
          f"{pooled['roster_recall_pct']}%  selection "
          f"{pooled['selection_pct']}%  overall "
          f"{pooled['overall_accuracy_pct']}%")
    print(f"  legacy (qwen3.5-9b): recall {LEGACY_RECALL}%  "
          f"selection {LEGACY_SELECTION}%")
    delta = pooled["selection_pct"] - LEGACY_SELECTION
    print(f"  selection moved {delta:+.1f} points\n")
    for book, r in per_book.items():
        print(f"  {book:20} recall {r['roster_recall_pct']:5.1f}%  "
              f"selection {r['selection_pct']:5.1f}%  "
              f"(n={r['n']}, in roster {r['n_in_roster']})")
    print("\n  same-book spread across backends (selection %):")
    for book, variants in spread.items():
        values = sorted(v for v in variants.values() if v is not None)
        if len(values) > 1:
            print(f"    {book:20} {values}  range "
                  f"{max(values) - min(values):.1f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
