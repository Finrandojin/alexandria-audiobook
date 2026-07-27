"""Score a run's attributions against hand-judged answers.

Every gate in the pipeline checks form - valid JSON, byte-exact text, non-empty
speaker - and none checks whether the speaker is *right*. So correctness bugs
stayed invisible until someone read the output: the SPOKEN type leak, the
WEARING hallucination and the roster feedback loop all passed their unit tests.

This gives correctness a number that moves when the pipeline changes.

The fixture holds only HARD lines - every one is a case where two decoders
disagreed - so absolute accuracy is expected to be low. Track the delta between
runs, not the absolute figure.
"""

import argparse
import collections
import json
import os
import re

GOLD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "fixtures", "attribution_gold.json")


def load_gold(path=None):
    with open(path or GOLD_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def normalize_speaker(value):
    return " ".join(str(value or "").split()).upper()


def alias_groups(gold):
    """Normalized alias sets declared by a gold fixture.

    A character named two ways is one character. mushoku16 calls the
    protagonist both RUDEUS and RUDI, and 14 of 147 gold lines were scored
    wrong purely for picking the other true name - a 9.5-point penalty for
    being right. Aliases live in the fixture, not in code, because they are
    facts about one book that a reader can check against the text, and because
    a scorer that invents equivalences can hide real errors.
    """
    return [{normalize_speaker(name) for name in group}
            for group in gold.get("aliases", [])]


def same_person(expected, actual, groups):
    """Whether two speaker names refer to the same character."""
    if not actual:
        return False
    if actual == expected:
        return True
    return any(actual in group and expected in group for group in groups)


def romaji_key(name):
    """Phonetic key collapsing Japanese-romanization spelling variance.

    These corpora are translated from Japanese, so a name has systematically
    variable romanizations rather than random typos. Two kinds were observed
    from magistral-small on mushoku16:

      RUDEUS / RUDIUS / RUDIEUS / RUDUEUS   medial vowels are unstable
      ALMANFI / ARUMANFI                    one liquid (L=R), and consonant
                                            clusters take an epenthetic vowel

    The first vowel and the consonant sequence survive both. Dropping ALL
    vowels also merges them, but it collides REIDA with RUDI - two distinct
    mushoku16 characters - so the first vowel is retained as a discriminator.
    Verified: zero collisions between distinct characters across both fixtures'
    full name sets (mushoku16 21 names, grimgar03 27 names).

    This is deliberately NOT wired into same_person. Exact match measures
    whether a name is usable downstream - a misspelled speaker fragments the
    cast list and breaks voice assignment - while this measures whether the
    model identified the right character. They are different questions, and
    the penalty is model-specific (magistral-small loses 7.9 points of oracle
    accuracy to it; three other models lose nothing), so silently normalizing
    would change every cross-model comparison in the ledger without saying so.
    Report both.
    """
    text = re.sub(r"[^A-Z]", "", normalize_speaker(name))
    if not text:
        return ""
    text = text.replace("L", "R")
    first_vowel = next((char for char in text if char in "AEIOU"), "")
    consonants = re.sub(r"(.)\1+", r"\1", re.sub(r"[AEIOU]", "", text))
    return f"{first_vowel}|{consonants}"


def same_person_phonetic(expected, actual, groups):
    """same_person, plus romanization-variant tolerance. See romaji_key."""
    if same_person(expected, actual, groups):
        return True
    if not actual:
        return False
    key = romaji_key(actual)
    return bool(key) and key == romaji_key(expected)


def normalize_line(value):
    return " ".join(str(value or "").split())


def find_entry(named_entries, item, by_text):
    """Locate the gold line in a run, by index first and then by text.

    Index alone is not enough: segmentation is not identical across runs (two
    runs of one book produced 1,995 and 2,038 entries), so a harness that only
    matched positions could not score the very comparisons it exists for.
    The recorded index is tried first because it is exact when it holds, and
    the nearest text match is used otherwise.
    """
    index = item["entry_index"]
    # Full normalized text, not a prefix. A 60-character prefix is not an
    # identity: measured on the corpus, grimgar03 and grimgar06 each contain
    # distinct lines sharing one, and matching on the prefix would score a gold
    # answer against a different line entirely.
    target = normalize_line(item["line"])
    if index < len(named_entries):
        entry = named_entries[index]
        if normalize_line(entry.get("text")) == target:
            return entry
    # Prefer the occurrence nearest the recorded index: short lines repeat.
    candidates = by_text.get(target)
    if not candidates:
        return None
    return min(candidates, key=lambda pair: abs(pair[0] - index))[1]


def score_run(named_entries, gold, include_disputed=False):
    """Compare a run's named entries against the gold answers.

    Alias-equivalent answers count as correct; see alias_groups.
    """
    groups = alias_groups(gold)
    by_text = {}
    for position, entry in enumerate(named_entries):
        by_text.setdefault(normalize_line(entry.get("text")),
                           []).append((position, entry))
    results = []
    for item in gold["entries"]:
        if item.get("disputed") and not include_disputed:
            continue
        entry = find_entry(named_entries, item, by_text)
        actual = normalize_speaker(entry.get("speaker")) if entry else ""
        expected = normalize_speaker(item["expected_speaker"])
        results.append({
            "id": item["id"],
            "expected": expected,
            "actual": actual,
            "correct": same_person(expected, actual, groups),
            # Reported alongside, never instead of. See romaji_key: exact match
            # is the product number, this is the attribution-ability number.
            "correct_phonetic": same_person_phonetic(expected, actual, groups),
            "aligned": entry is not None,
        })
    return results


def summarize(results):
    aligned = [r for r in results if r["aligned"]]
    correct = [r for r in aligned if r["correct"]]
    confusion = collections.Counter(
        (r["expected"], r["actual"]) for r in aligned if not r["correct"])
    missed = collections.Counter(
        r["expected"] for r in aligned if not r["correct"])
    phonetic = [r for r in aligned if r.get("correct_phonetic")]
    return {
        "scored": len(results),
        "aligned": len(aligned),
        "correct": len(correct),
        "accuracy": len(correct) / len(aligned) if aligned else 0.0,
        # The gap between these two is a per-model spelling penalty, not noise:
        # it was 7.9 points for magistral-small and 0.0 for three other models
        # on the same fixture. A comparison that quotes only one hides it.
        "correct_phonetic": len(phonetic),
        "accuracy_phonetic": (len(phonetic) / len(aligned) if aligned else 0.0),
        "spelling_penalty": ((len(phonetic) - len(correct)) / len(aligned)
                             if aligned else 0.0),
        "confusion": confusion,
        "missed": missed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint",
                        help="a run's result.json.threepass_checkpoint.json")
    parser.add_argument("--gold", default=None)
    parser.add_argument("--include-disputed", action="store_true",
                        help="include entries the judge and the user disagreed on")
    parser.add_argument("--baseline", default=None,
                        help="another checkpoint to compare against, so a change "
                             "shows as a delta rather than an absolute")
    args = parser.parse_args()

    gold = load_gold(args.gold)

    def named(path):
        with open(path, encoding="utf-8") as handle:
            return [e for e in (json.load(handle).get("named") or []) if e]

    results = score_run(named(args.checkpoint), gold, args.include_disputed)
    stats = summarize(results)

    print(f"gold set: {stats['scored']} lines "
          f"({len(gold['entries']) - stats['scored']} withheld)")
    if stats["aligned"] < stats["scored"]:
        print(f"WARNING: only {stats['aligned']} aligned - this run segmented "
              "differently, so the rest could not be scored")
    print(f"correct : {stats['correct']}/{stats['aligned']} "
          f"({stats['accuracy']:.1%})")
    # Printed only when it differs, so the common case stays a one-line answer
    # but a model paying a romanization penalty cannot be compared without it.
    if stats["correct_phonetic"] != stats["correct"]:
        print(f"phonetic: {stats['correct_phonetic']}/{stats['aligned']} "
              f"({stats['accuracy_phonetic']:.1%})  "
              f"spelling penalty {stats['spelling_penalty']:+.1%} - "
              f"right character, romanized differently (see romaji_key)")

    if args.baseline:
        base = summarize(score_run(named(args.baseline), gold,
                                   args.include_disputed))
        delta = stats["accuracy"] - base["accuracy"]
        print(f"baseline: {base['correct']}/{base['aligned']} "
              f"({base['accuracy']:.1%})")
        print(f"DELTA   : {delta:+.1%}  "
              f"({stats['correct'] - base['correct']:+d} lines)")

    if stats["missed"]:
        print("\nspeakers missed most:")
        for speaker, count in stats["missed"].most_common(6):
            print(f"  {speaker:12} {count}")
    if stats["confusion"]:
        print("\ntop confusions (expected -> got):")
        for (expected, actual), count in stats["confusion"].most_common(6):
            print(f"  {count:2}x  {expected:12} -> {actual or '(none)'}")


if __name__ == "__main__":
    main()
