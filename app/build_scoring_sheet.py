"""Build one shared hand-scoring sheet across every model in a matrix run.

Pairwise arm comparison cannot answer the quality question: two identical runs
disagree on 37.4% of speaker assignments, which is larger than the difference
between any two configurations. So instead of comparing models to each other,
this samples a fixed set of lines once and shows what every model said about
those same lines, leaving one blank column for the correct answer.

Sampling is by text rather than entry index because models segment differently
- two runs of one book produced 1,995 and 2,038 entries.
"""

import argparse
import glob
import json
import os
import random

from compare_attribution_arms import normalize


def load_named(checkpoint_path):
    with open(checkpoint_path, encoding="utf-8") as handle:
        data = json.load(handle)
    return [entry for entry in (data.get("named") or []) if entry]


def find_model_runs(results_dir, book_tag):
    """Return {model_name: entries} for every model that finished this book."""
    runs = {}
    pattern = os.path.join(results_dir, "*", book_tag,
                           "result.json.threepass_checkpoint.json")
    for path in sorted(glob.glob(pattern)):
        model = path.split(os.sep)[-3]
        entries = load_named(path)
        if entries:
            runs[model] = entries
    return runs


def build_sheet(runs, size=50, seed=7):
    """Sample spoken lines and collect every model's answer for each.

    Lines are keyed on normalized text, so a line only enters the sheet if it
    survived segmentation identically across every model - otherwise the models
    would be scored on different text and the comparison would be meaningless.
    """
    if not runs:
        return []
    models = sorted(runs)
    indexed = {}
    for model in models:
        by_text = {}
        for entry in runs[model]:
            key = normalize(entry.get("text"))
            # First occurrence wins; duplicate lines are ambiguous to align.
            by_text.setdefault(key, entry)
        indexed[model] = by_text

    shared = set(indexed[models[0]])
    for model in models[1:]:
        shared &= set(indexed[model])

    # Only lines somebody attributed to a character: NARRATOR-only lines carry
    # no judgement to score.
    candidates = [
        key for key in shared
        if any((indexed[model][key].get("speaker") or "").upper()
               not in ("NARRATOR", "", "UNKNOWN") for model in models)
    ]
    candidates.sort()
    chosen = (candidates if len(candidates) <= size
              else random.Random(seed).sample(candidates, size))

    rows = []
    for key in chosen:
        answers = {model: indexed[model][key].get("speaker") for model in models}
        rows.append({
            "text": indexed[models[0]][key].get("text", "")[:400],
            "answers": answers,
            "models_agree": len(set(answers.values())) == 1,
            "correct_speaker": "",          # <- you fill this in
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    parser.add_argument("book_tag")
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="scoring_sheet.json")
    args = parser.parse_args()

    runs = find_model_runs(args.results_dir, args.book_tag)
    if not runs:
        print(f"no completed runs for {args.book_tag} under {args.results_dir}")
        return
    rows = build_sheet(runs, args.size, args.seed)
    disputed = [r for r in rows if not r["models_agree"]]
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"book": args.book_tag, "models": sorted(runs),
                   "entries_per_model": {m: len(e) for m, e in runs.items()},
                   "sampled": len(rows), "disputed": len(disputed),
                   "rows": rows}, handle, indent=2, ensure_ascii=False)
    print(f"models: {', '.join(sorted(runs))}")
    print(f"entries per model: {[len(e) for e in runs.values()]}")
    print(f"sampled {len(rows)} shared lines; {len(disputed)} disputed "
          f"({len(disputed)/max(len(rows),1):.0%})")
    print(f"\nFill in correct_speaker for each row in {args.output}.")
    print("Scoring the same lines for every model avoids pairwise comparison, "
          "which the 37.4% identical-run disagreement makes unusable.")


if __name__ == "__main__":
    main()
