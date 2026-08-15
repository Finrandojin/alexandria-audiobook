"""Explain clone-duration variation in the same-speaker Japanese probe."""
import argparse
import json
import math
import os
import statistics
import sys


APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(APP)
sys.path.insert(0, APP)

from experiments.provenance import input_sha256, provenance  # noqa: E402
from utils import atomic_json_write  # noqa: E402


def get_audio_seconds(path):
    import soundfile as sf
    full = path if os.path.isabs(path) else os.path.join(REPO, path)
    info = sf.info(full)
    return info.frames / info.samplerate


def get_ranks(values):
    """Return average ranks for ties, using zero-based ranks."""
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + end - 1) / 2
        for index in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def get_pearson(left, right):
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean)
                    for a, b in zip(left, right))
    left_ss = sum((a - left_mean) ** 2 for a in left)
    right_ss = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else None


def get_spearman(left, right):
    if len(left) != len(right) or len(left) < 2:
        return None
    return get_pearson(get_ranks(left), get_ranks(right))


def get_text_features(text, human_seconds):
    nonspace = sum(not char.isspace() for char in text)
    punctuation = sum(char in "、。！？!?，,.：:；;「」『』（）()" for char in text)
    return {
        "characters": nonspace,
        "punctuation": punctuation,
        "human_chars_per_second": nonspace / human_seconds,
    }


def get_bin_summary(rows, feature):
    ordered = sorted(rows, key=lambda row: row[feature])
    bins = []
    for number, group in enumerate((ordered[:10], ordered[10:20], ordered[20:])):
        bins.append({
            "bin": number + 1,
            "n": len(group),
            "feature_min": round(group[0][feature], 4),
            "feature_max": round(group[-1][feature], 4),
            "median_duration_ratio": round(statistics.median(
                row["duration_ratio"] for row in group), 4),
        })
    return bins


def analyze(rows):
    enriched = []
    for row in rows:
        human = get_audio_seconds(row["human_wav"])
        clone = get_audio_seconds(row["clone_wav"])
        current = dict(row)
        current.update(get_text_features(row["text"], human))
        current.update({"human_seconds_measured": human,
                        "clone_seconds_measured": clone,
                        "duration_ratio": clone / human})
        enriched.append(current)

    features = ("human_seconds_measured", "characters", "punctuation",
                "human_chars_per_second")
    ratios = [row["duration_ratio"] for row in enriched]
    correlations = {
        feature: round(get_spearman(
            [row[feature] for row in enriched], ratios), 4)
        for feature in features
    }
    return enriched, {
        "n": len(enriched),
        "duration_ratio_median": round(statistics.median(ratios), 4),
        "duration_ratio_min": round(min(ratios), 4),
        "duration_ratio_max": round(max(ratios), 4),
        "spearman_vs_duration_ratio": correlations,
        "tertiles": {feature: get_bin_summary(enriched, feature)
                      for feature in features},
        "largest_absolute_errors": [row["id"] for row in sorted(
            enriched, key=lambda row: abs(row["duration_ratio"] - 1),
            reverse=True)[:5]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=os.path.join(
        REPO, "ab_test_runtime", "experiments",
        "kokoro_same_speaker_generate.json"))
    parser.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments",
        "duration_outlier_analysis.json"))
    args = parser.parse_args()
    with open(args.input, encoding="utf-8") as handle:
        source = json.load(handle)
    rows, summary = analyze(source["rows"])
    result = {
        "question": "Which measured properties explain Japanese clone-duration variation?",
        "summary": summary,
        "rows": rows,
        "provenance": provenance(__file__, args,
                                 inputs=input_sha256([args.input])),
    }
    atomic_json_write(result, args.out)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
