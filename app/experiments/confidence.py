"""Risk/coverage from cross-model agreement — no GPU, no perturbation.

The project needs to know how much of a book can ship unreviewed. Temperature-0
self-consistency cannot answer it: repeats are byte-identical, so agreement with
yourself is guaranteed and carries no information.

Six models have already answered the same 147 gold lines under identical frozen
inputs. Agreement *between* them is a real signal, obtained from artifacts that
already exist. This measures whether it separates correct answers from wrong
ones well enough to define an auto-accept subset.

Also scores two deterministic text features for comparison, since a signal that
needs six models is expensive to run in production:
  - the answer appears in a speech-verb tag near the line;
  - the answer is scene-local rather than only in the global roster.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E = os.path.join(REPO, "ab_test_runtime", "experiments")
MODELS = {
    "qwen3.5-9b": "closed_set__qwen3.5-9b-uncensored-hauhaucs-aggressive.json",
    "gemma-4-e4b": "closed_set__gemma-4-e4b-uncensored-hauhaucs-aggressive.json",
    "ministral-14b": "closed_set__ministral-3-14b-instruct-2512.json",
    "heresy-14b": "closed_set__ministral-3-14b-instruct-2512-absolute-heresy-i1.json",
    "phi-4": "closed_set__microsoft__phi-4.json",
    "qwen3-14b": "closed_set__qwen__qwen3-14b.json",
}
ARM = "open"          # the realistic configuration, not the oracle diagnostic


def _alias_groups():
    """Alias sets from the fixture, so RUDI and RUDEUS are not a disagreement.

    The first version compared raw prediction strings. Two models naming the
    same character differently counted as disagreement, which deflates agreement
    and therefore the coverage of any confidence threshold built on it - the
    exact defect that cost the scorer 14 of 147 lines.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "fixtures", "attribution_gold_random.json")
    with open(path, encoding="utf-8") as handle:
        return [{n.upper() for n in group}
                for group in json.load(handle).get("aliases", [])]


ALIASES = _alias_groups()


def canonical(name):
    """One label per character, so votes are counted per person not per spelling."""
    upper = (name or "").strip().upper()
    for group in ALIASES:
        if upper in group:
            return sorted(group)[0]
    return upper


def load(arm=ARM):
    """{gold_id: {model: (prediction, correct)}} plus the gold answer."""
    rows = collections.defaultdict(dict)
    truth = {}
    for name, filename in MODELS.items():
        path = os.path.join(E, filename)
        if not os.path.exists(path):
            continue
        for row in json.load(open(path, encoding="utf-8"))["rows"]:
            if row["arm"] != arm:
                continue
            rows[row["id"]][name] = (row["predicted"], row["correct"])
            truth[row["id"]] = row["expected"]
    return rows, truth


def risk_coverage(scored, label):
    """Accuracy of the auto-accepted subset as the threshold is relaxed.

    scored: {gold_id: (confidence, correct)}. Reports, for each threshold, how
    much of the book is accepted and how accurate that slice is - the only
    framing that answers "can this ship".
    """
    if not scored:
        return
    print(f"\n{label}")
    print(f"  {'threshold':>10} {'coverage':>10} {'accepted':>9} {'accuracy':>9}")
    total = len(scored)
    for threshold in sorted({c for c, _ in scored.values()}, reverse=True):
        subset = [ok for c, ok in scored.values() if c >= threshold]
        if not subset:
            continue
        print(f"  {threshold:>10.2f} {len(subset)/total*100:9.1f}% "
              f"{len(subset):9} {sum(subset)/len(subset)*100:8.1f}%")


def main():
    rows, truth = load()
    names = sorted({m for v in rows.values() for m in v})
    print(f"{len(rows)} gold lines, {len(names)} models: {', '.join(names)}")

    # 1. how many models chose the majority answer
    agreement = {}
    for gold_id, per_model in rows.items():
        if len(per_model) < len(names):
            continue
        votes = collections.Counter(canonical(p) for p, _ in per_model.values())
        winner, count = votes.most_common(1)[0]
        # Correct means the majority answer matches what that model was scored
        # against - reuse the recorded correctness of any model that said it.
        correct = any(ok for p, ok in per_model.values() if canonical(p) == winner)
        agreement[gold_id] = (count / len(names), correct)
    risk_coverage(agreement, "A. majority-vote answer, confidence = share of models agreeing")

    # 2. the strongest single model, gated on whether the others concur
    best = "qwen3-14b" if "qwen3-14b" in names else names[0]
    gated = {}
    for gold_id, per_model in rows.items():
        if best not in per_model or len(per_model) < len(names):
            continue
        pick, ok = per_model[best]
        concur = sum(1 for p, _ in per_model.values()
                     if canonical(p) == canonical(pick)) / len(names)
        gated[gold_id] = (concur, ok)
    risk_coverage(gated, f"B. {best}'s answer, confidence = how many models concur")

    # 3. unanimity as a plain switch, which is what a pipeline would implement
    unanimous = [ok for c, ok in gated.values() if c == 1.0]
    if unanimous:
        print(f"\nC. all {len(names)} models agree: {len(unanimous)}/{len(gated)} lines "
              f"({len(unanimous)/len(gated)*100:.1f}% coverage), "
              f"{sum(unanimous)/len(unanimous)*100:.1f}% accurate")
    split = [ok for c, ok in gated.values() if c < 1.0]
    if split:
        print(f"   the rest: {len(split)} lines, "
              f"{sum(split)/len(split)*100:.1f}% accurate")


if __name__ == "__main__":
    main()
