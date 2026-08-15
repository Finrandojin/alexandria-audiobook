"""Explain the weakest PDNC books using saved predictions and gold context."""
import argparse
import collections
import glob
import json
import os
import re
import sys


REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

from experiments.scoring import alias_groups, normalize, same_speaker  # noqa: E402
from experiments.pdnc_fixture import build as build_fixture  # noqa: E402
from utils import atomic_json_write  # noqa: E402

DEFAULT_BOOKS = ("MansfieldPark", "TheGambler", "Persuasion",
                 "AnneOfGreenGables", "TheSunAlsoRises")


def roster_match(name, roster, groups):
    return any(same_speaker(name, candidate, groups) for candidate in roster)


def context_mentions(name, context, groups):
    wanted = normalize(name)
    aliases = {wanted}
    for group in groups:
        if wanted in group:
            aliases.update(group)
    normalized_context = normalize(" ".join(context.split()))
    return any(re.search(r"(?:^| )" + re.escape(alias) + r"(?: |$)",
                         normalized_context) for alias in aliases if alias)


def classify_error(gold_in_roster, prediction, prediction_in_roster,
                   gold_in_context):
    if not gold_in_roster:
        return "gold_missing_from_roster"
    if not prediction:
        return "missing_prediction_or_batch_failure"
    if not prediction_in_roster:
        return "invalid_or_out_of_roster_prediction"
    if gold_in_context:
        return "missed_explicit_context_evidence"
    return "valid_candidate_selection_error"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--artifact", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "pdnc_eval_full.json"))
    parser.add_argument("--arm", default="base")
    parser.add_argument("--books", nargs="+", default=list(DEFAULT_BOOKS))
    parser.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments",
        "pdnc_failure_telemetry.json"))
    args = parser.parse_args()

    predictions = json.load(open(args.artifact, encoding="utf-8"))
    fixtures = {}
    for path in glob.glob(os.path.join(
            APP, "fixtures", "attribution_gold_pdnc_*.json")):
        fixture = json.load(open(path, encoding="utf-8"))
        fixtures[fixture["book"]] = fixture
    pdnc_data = os.path.join(REPO, "ab_test_runtime", "pdnc", "data")

    report = {"arm": args.arm, "route": "LLM attribution on gold spans",
              "source_artifact": os.path.relpath(args.artifact, REPO),
              "books": {}, "failure_classes": {}}
    totals = collections.Counter()
    for book in args.books:
        fixture = fixtures.get(book) or build_fixture(pdnc_data, book)
        roster = fixture["roster"]
        groups = alias_groups(fixture)
        gold = {entry["id"]: entry for entry in fixture["entries"]}
        rows = predictions[book][args.arm]["rows"]
        errors = []
        classes = collections.Counter()
        for row in rows:
            if row.get("correct"):
                continue
            entry = gold[row["id"]]
            previous = entry.get("prev_context") or ""
            following = entry.get("next_context") or ""
            gold_in_roster = roster_match(row["expected"], roster, groups)
            prediction_in_roster = roster_match(
                row.get("predicted"), roster, groups)
            gold_previous = context_mentions(
                row["expected"], previous, groups)
            gold_next = context_mentions(row["expected"], following, groups)
            failure_class = classify_error(
                gold_in_roster, row.get("predicted"), prediction_in_roster,
                gold_previous or gold_next)
            classes[failure_class] += 1
            totals[failure_class] += 1
            errors.append({
                "id": row["id"], "expected": row["expected"],
                "predicted": row.get("predicted"),
                "quote_type": row.get("quote_type"),
                "category": row.get("category"),
                "candidate_count": len(roster),
                "gold_in_roster": gold_in_roster,
                "prediction_in_roster": prediction_in_roster,
                "gold_mentioned_previous": gold_previous,
                "gold_mentioned_next": gold_next,
                "previous_context": previous,
                "line": entry.get("line"),
                "next_context": following,
                "failure_class": failure_class,
            })
        report["books"][book] = {
            "candidate_roster": roster, "candidate_count": len(roster),
            "n": len(rows), "errors": len(errors),
            "failure_classes": dict(classes), "error_rows": errors,
        }
    report["failure_classes"] = dict(totals)
    atomic_json_write(report, args.out)

    print(f"=== PDNC failure telemetry: {args.arm} ===")
    print(f"  {sum(totals.values())} errors across {len(report['books'])} books")
    for name, count in totals.most_common():
        print(f"  {name:38} {count:4}")
    for book, result in report["books"].items():
        print(f"  {book:28} {result['errors']:3}/{result['n']} errors, "
              f"{result['candidate_count']} candidates")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
