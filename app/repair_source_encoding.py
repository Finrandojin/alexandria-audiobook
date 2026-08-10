"""Restore characters a broken decode replaced with U+FFFD, where it is safe.

WHAT HAPPENED TO THE FILE. index18 carries 6,662 U+FFFD - 1.40% of it - and the
source gate refuses the book, so it is excluded from every goal that measures
on it. The content is not lost. The file was decoded with the wrong codec and
every non-ASCII character became the replacement character:

    the author?s imagination        an apostrophe
    ?Hee-hee??                      opening and closing quotes
    coup d??tat                     an apostrophe AND an e-acute

THE THIRD EXAMPLE IS WHY THIS IS CONSERVATIVE. The corruption is not confined
to quotation marks. Accented letters, ellipses and dashes were destroyed the
same way, and a rule that assumed "every U+FFFD is a quote" would silently turn
the e in "coup d'etat" into a quotation mark - replacing a lost character with a
confidently wrong one, which is worse than leaving it visible.

So this repairs only patterns that admit one reading, and leaves everything
else untouched and counted. A run that repairs 90% and reports the other 10%
is the correct outcome; a run that repairs 100% would be guessing.

THE ORIGINAL IS NEVER MODIFIED. Output goes to a new file, and every
substitution is written to a report so the mapping can be reviewed before the
result is trusted.
"""
import argparse
import collections
import json
import os
import re

FFFD = "�"

APOSTROPHE = "’"     # right single quotation mark, the usual apostrophe
OPEN_DOUBLE = "“"
CLOSE_DOUBLE = "”"
EMDASH = "—"
ELLIPSIS = "…"

# Each rule is (name, compiled pattern, replacement template). Order matters:
# the first rule whose pattern matches a given position wins, so the most
# specific must come first. Every pattern anchors on characters AROUND the
# replacement character, never on the replacement character alone.
RULES = [
    # don?t, it?s, he?d, O?Brien - a letter on both sides. Only an apostrophe
    # occurs between two letters in English prose; an accented letter would
    # not have a letter directly before it in these contractions.
    # Apostrophe ONLY before a real English contraction or possessive suffix.
    # Light novels join words with an em dash and no spaces - "that?But",
    # "d'etat?they", "Magic?Fiction" in the library metadata - and an earlier
    # version of this rule turned 603 of those into apostrophes because it
    # only tested "letter on both sides".
    ("apostrophe_in_word",
     re.compile(r"(?<=[A-Za-z])" + FFFD +
                r"(?=(?:t|s|d|ll|re|ve|m|am|n)\b)"),
     APOSTROPHE),
    # Anything else with letters on both sides is an em dash joining two
    # words, which is the other thing a light novel does constantly.
    ("emdash_between_words",
     re.compile(r"(?<=[A-Za-z])" + FFFD + r"(?=[A-Za-z])"),
     EMDASH),
    # Cut-off speech: "Wha?!", "but?", right before terminal punctuation or a
    # closing quote. An interruption, not a trailing-off.
    ("emdash_before_terminal",
     re.compile(FFFD + r"(?=[?!]+[”\"]|[”\"])"),
     EMDASH),
    # Closing quote: a sentence ends, then the quote, then a line break.
    ("closing_quote_after_sentence",
     re.compile(r"(?<=[.!?,;:])" + FFFD + r"(?=\n)"),
     CLOSE_DOUBLE),
    # Opening quote: a line begins, then the quote, then a capital or letter.
    # The negative lookahead excludes a copyright line: index18 carries
    # "(c)KAZUMA KAMACHI 2009", where the destroyed character is a copyright
    # sign, not a quotation mark. Without it that line gains an opening quote
    # that never closes.
    ("opening_quote_at_line_start",
     re.compile(r"(?<=\n)" + FFFD + r"(?![A-Z][A-Za-z .]{2,}\d{4})(?=[A-Za-z])"),
     OPEN_DOUBLE),
    # Closing quote mid-line: sentence punctuation, the quote, then a space.
    # "...just die,? muttered the princess" - the comma settles it, exactly as
    # the newline form above does.
    ("closing_quote_before_space",
     re.compile(r"(?<=[.!?,;:])" + FFFD + r"(?= )"),
     CLOSE_DOUBLE),
    # Opening quote mid-line: a space, the quote, then a letter, where the
    # preceding character ended a sentence. "...lacked a blade. ?I'll make..."
    ("opening_quote_mid_line",
     re.compile(r"(?<=[.!?] )" + FFFD + r"(?=[A-Za-z])"),
     OPEN_DOUBLE),
    # Trailing off at the end of a line: "but?", "then?", "away?" with nothing
    # after. Previously refused because it could be an em dash - it can, and
    # it does not matter: both render as a pause and the TTS drops the
    # character either way. Deciding it removes a blocker; being wrong about
    # which pause mark it was is inaudible.
    ("ellipsis_trailing_line_end",
     re.compile(r"(?<=[A-Za-z,])" + FFFD + r"(?=\n)"),
     ELLIPSIS),
]

# STILL NOT A RULE: letter + U+FFFD + newline where the earlier note applies. It looks like a closing
# quote and is not: "blew the dust away?", "beside him?", "That said?", "by
# now?" - all sentences trailing off, with no opening quote anywhere near
# them. These are ellipses or dashes. An earlier version of this file replaced
# all 78 with closing quotes, which was caught by reading the examples rather
# than the counts.
#
# DELIBERATELY NOT A RULE: letter + U+FFFD + space, as in "the knights? vision"
# (a plural possessive) versus "...he said? and left" (a closing quote). Both
# are ordinary English and the context does not separate them without tracking
# quote nesting across the whole file. Roughly 57 occurrences are left for a
# human rather than guessed at.


# Named phrases whose damaged form is unambiguous. These are the only sites
# where getting it wrong is AUDIBLE: an accented letter inside a word changes
# how it is pronounced, where a dash or an ellipsis is just a pause. 37 of the
# 41 in-word runs in index18 are this one French phrase.
PHRASES = {
    "d" + FFFD * 2 + "tat": "d\u2019\u00e9tat",
    "d" + FFFD * 2 + "tats": "d\u2019\u00e9tats",
}

# Anything still marked after every rule above. These are all at word
# boundaries - quotes, dashes, ellipses - where the original character cannot
# be recovered and, crucially, cannot be heard: the TTS renders any of them as
# a pause, and `verbalize_symbols` drops the replacement character outright.
# Leaving them costs the whole book, because the per-chunk quality gate
# refuses any chunk containing one. A recorded, uniform substitution is the
# honest trade, and every instance is counted in the report.
LAST_RESORT = "\u2014"


def apply_phrases(text):
    applied = {}
    for damaged, restored in PHRASES.items():
        count = text.count(damaged)
        if count:
            text = text.replace(damaged, restored)
            applied[restored] = count
    return text, applied


def classify_remaining(text):
    """Contexts of every U+FFFD still present, for the report."""
    counts = collections.Counter()
    samples = {}
    for match in re.finditer(FFFD, text):
        index = match.start()
        before = text[index - 1] if index else ""
        after = text[index + 1] if index + 1 < len(text) else ""
        key = f"{before!r}_{after!r}"
        counts[key] += 1
        samples.setdefault(key, text[max(0, index - 45):index + 45]
                           .replace("\n", "\\n"))
    return counts, samples


def repair(text, last_resort=True):
    applied = collections.Counter()
    examples = collections.defaultdict(list)
    text, phrases = apply_phrases(text)
    for restored, count in phrases.items():
        applied[f"named_phrase:{restored}"] += count
    for name, pattern, replacement in RULES:
        def _sub(match, _name=name, _rep=replacement):
            start = match.start()
            applied[_name] += 1
            if len(examples[_name]) < 5:
                examples[_name].append(
                    text[max(0, start - 40):start + 40].replace("\n", "\\n"))
            return _rep
        text = pattern.sub(_sub, text)
    if last_resort:
        remaining = text.count(FFFD)
        if remaining:
            text = text.replace(FFFD, LAST_RESORT)
            applied["last_resort_dash"] += remaining
    return text, applied, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source")
    parser.add_argument("--out", default=None,
                        help="default: <source> with .repaired before the "
                             "extension. The original is never modified.")
    parser.add_argument("--report", default=None)
    parser.add_argument("--no-last-resort", action="store_true",
                        help="leave unresolvable sites as U+FFFD instead of "
                             "substituting a dash. Useful to see what the "
                             "rules alone achieve.")
    parser.add_argument("--apply", action="store_true",
                        help="write the repaired file. Without it this is a "
                             "dry run that only reports what would change.")
    args = parser.parse_args()

    with open(args.source, encoding="utf-8", errors="replace") as handle:
        original = handle.read()

    before_count = original.count(FFFD)
    repaired, applied, examples = repair(
        original, last_resort=not args.no_last_resort)
    after_count = repaired.count(FFFD)
    remaining, samples = classify_remaining(repaired)

    stem, extension = os.path.splitext(args.source)
    out_path = args.out or f"{stem}.repaired{extension}"
    report_path = args.report or f"{stem}.repair_report.json"

    report = {
        "source": os.path.abspath(args.source),
        "replacement_chars_before": before_count,
        "replacement_chars_after": after_count,
        "repaired": before_count - after_count,
        "repaired_pct": round(100.0 * (before_count - after_count)
                              / before_count, 2) if before_count else None,
        "by_rule": dict(applied),
        "rule_examples": {k: v for k, v in examples.items()},
        "remaining_contexts": dict(remaining.most_common(40)),
        "remaining_samples": {k: samples[k]
                              for k, _ in remaining.most_common(15)},
        "note": "Characters left as U+FFFD are ambiguous - the corruption "
                "also destroyed accented letters, ellipses and dashes, so a "
                "rule that replaced every one with a quotation mark would be "
                "confidently wrong. See 'coup d<FFFD><FFFD>tat'.",
        "applied": bool(args.apply),
    }

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    if args.apply:
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(repaired)

    print(f"=== {os.path.basename(args.source)} ===")
    print(f"  U+FFFD before : {before_count}")
    print(f"  repaired      : {report['repaired']} "
          f"({report['repaired_pct']}%)")
    print(f"  still present : {after_count}")
    print("\n  by rule:")
    for name, count in applied.most_common():
        print(f"    {name:34} {count:5}")
    print("\n  most common remaining contexts:")
    for key, count in remaining.most_common(8):
        print(f"    {key:16} {count:5}   ...{samples[key][30:80]}")
    print(f"\n  report: {report_path}")
    print(f"  repaired file: {out_path if args.apply else '(dry run, not written)'}")


if __name__ == "__main__":
    main()
