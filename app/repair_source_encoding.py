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
     re.compile(r"(?<=[.!?,] )" + FFFD + r"(?=[A-Za-z])"),
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


# A run of the same damaged pair repeated - "?!?!?!..." twenty-five times over
# - is a stylised scream in the source. Substituting one character for another
# preserves the repetition, and a long repeating token sequence sends the model
# into a generation loop: index18's chunk 10 ran to the 16,384-token ceiling on
# every attempt, three runs in a row, on a 50-character run of exactly this.
#
# Collapsing keeps the meaning (a drawn-out cry) and removes the trap. The
# count is reported, because shortening the author's text is a change a reader
# should be able to see.
_REPEATED_PAIR = re.compile(r"(?:" + FFFD + r"([?!]))(?:" + FFFD + r"[?!]){2,}")
_LONG_RUN = re.compile(FFFD + r"{3,}")


def collapse_repetitions(text):
    """Shorten repeated damaged patterns before anything is substituted."""
    collapsed = {}

    def _pair(match):
        collapsed["repeated_pair"] = collapsed.get("repeated_pair", 0) + 1
        return ELLIPSIS + match.group(1)

    def _run(match):
        collapsed["long_run"] = collapsed.get("long_run", 0) + 1
        return ELLIPSIS

    text = _REPEATED_PAIR.sub(_pair, text)
    text = _LONG_RUN.sub(_run, text)

    # Repetitions that were already in the file before this decode error -
    # index18 was damaged twice, and an earlier lossy conversion turned
    # non-ASCII characters into literal "?", leaving runs like "O?o?o?o?h?h?h".
    # They are the same failure mode and are recorded separately, because
    # shortening them is a change to text this tool did not damage.
    def _existing(match):
        collapsed["pre_existing_repetition"] = (
            collapsed.get("pre_existing_repetition", 0) + 1)
        unit = match.group(0)[:2]
        return unit + unit[-1] * 2
    text = re.sub(r"((?:[?!][A-Za-z])|(?:[A-Za-z][?!]))\1{5,}", _existing, text)
    return text, collapsed


def apply_phrases(text):
    applied = {}
    for damaged, restored in PHRASES.items():
        count = text.count(damaged)
        if count:
            text = text.replace(damaged, restored)
            applied[restored] = count
    return text, applied


# A repair that removes every replacement character and leaves the prose
# structurally broken is not a repair. The first version of this file did
# exactly that: substituting a dash for a run that was "dash + closing quote"
# left 69 dialogue lines open, quote imbalance went to 25.1% against 0.5-3.4%
# in undamaged books, and the model then generated until it hit its token
# ceiling and failed the chunk 9 times over.
#
# So the repairer measures its own output. Anything worse than this is
# reported as a failure of the repair, not accepted silently.
# Calibrated across the whole corpus, not two books. An earlier value of 5%
# came from grimgar03 and mushoku16 alone and would have condemned healthy
# files: owarimonogatari3 sits at 11.7% and generates 110/110 chunks,
# mushoku23 at 8.8%, mushoku18 at 20.3%. Repaired index18 is 8.2% - below a
# book that demonstrably works. 25% leaves room for that spread while still
# catching a file whose speech structure has genuinely collapsed.
MAX_UNBALANCED_QUOTE_SHARE = 0.25


STRAIGHT_DOUBLE = '"'


def quote_balance(text):
    """-> (unbalanced_lines, lines_containing_quotes, share).

    Counts straight quotes as well as curly ones. Several books in this corpus
    mix them - mushoku18 opens with a straight quote and closes with a curly
    one - and an earlier version counting only curly marks scored it 61.4%
    unbalanced, which was the metric failing rather than the book being
    damaged. A straight quote is its own opener and closer, so an even count
    on the line is balanced.
    """
    unbalanced = quoted = 0
    # PARAGRAPHS, NOT PHYSICAL LINES. Public-domain text is hard-wrapped at
    # ~70 columns, so one spoken sentence spans several lines and each carries
    # an odd quote count. Measuring physical lines scored all 28 PDNC novels
    # 34.5-84.3% "unbalanced" with nothing wrong: Emma is 84.3% and has no
    # defect at all. The light novels this was calibrated on happen to put one
    # paragraph per line, which is why the error was invisible for three
    # rounds of calibration.
    for line in re.split(r"\n\s*\n", text):
        opens, closes = line.count(OPEN_DOUBLE), line.count(CLOSE_DOUBLE)
        straight = line.count(STRAIGHT_DOUBLE)
        if not (opens or closes or straight):
            continue
        quoted += 1
        # Pair curly marks against each other, then let straight quotes absorb
        # whatever curly mark is left over on the line.
        leftover = abs(opens - closes)
        if (straight + leftover) % 2 != 0:
            unbalanced += 1
    share = unbalanced / quoted if quoted else 0.0
    return unbalanced, quoted, share


# A repeating two-character sequence this long makes the model generate until
# it hits its token ceiling. Measured: 25 repetitions did it every time.
# Only repetitions containing punctuation. "deeeeeeeep" is the author
# elongating a word and is harmless; "?o?o?o?h?h?h" and "-?-?-?" are damage,
# and a long run of either makes the model generate to its token ceiling.
# The unit must be TWO DIFFERENT characters, at least one of them punctuation.
# A run of the SAME character is ordinary typography and appears in books that
# generate perfectly: grimgar03 has a 22-dash scene break, mushoku16 has
# ellipsis leader dots. Flagging those told a healthy book it was damaged.
# What actually loops the model is an ALTERNATING pattern - index18's
# "-?-?-?-?" repeated 25 times ran every attempt to the 16,384-token ceiling.
# THRESHOLD FROM MEASUREMENT, not intuition. Across 36 books:
#   runs that demonstrably broke generation (index18)  25 and 42 repetitions
#   legitimate typography, maximum observed            13 repetitions
#     grimgar06  "I-I-I-I-"      a stutter, 7
#     arc4       "*  *  *  * "   Japanese scene break, 10-12
#     Mysterious ". . . . . ."   spaced ellipsis, 6-13
# 15 sits in the gap. Two earlier versions of this detector flagged scene
# breaks and ellipses in books that generate 100% of their chunks, which is
# worse than not checking: it tells a user their healthy book is damaged.
_REPETITION_TRAP = re.compile(r"((?![\w\s]{2})(.)(?!\2)(.))\1{14,}")


def structural_regressions(text):
    """Specific shapes that mean a substitution destroyed sentence structure."""
    open_ended_with_dash = 0
    close_without_open = 0
    repetition_traps = len(_REPETITION_TRAP.findall(text))
    for line in text.split("\n"):
        opens, closes = line.count(OPEN_DOUBLE), line.count(CLOSE_DOUBLE)
        if opens > closes and line.rstrip().endswith(LAST_RESORT):
            open_ended_with_dash += 1
        if closes > opens and line.lstrip().startswith(LAST_RESORT):
            close_without_open += 1
    return {"open_quote_ended_with_dash": open_ended_with_dash,
            "close_quote_started_with_dash": close_without_open,
            "repetition_traps": repetition_traps}


# Contractions whose apostrophe an earlier lossy conversion removed. Seven of
# the 28 public-domain novels have ZERO apostrophes and hundreds of broken
# forms - "don t", "it s", "Winterbourne s" - which is why Daisy Miller failed
# generation at chunk 2 of 22 while a clean novel finished 41 of 41.
#
# The stems are whitelisted rather than inferred. "word + space + s" is only
# safely a possessive when the word is a pronoun, a known auxiliary, or a
# capitalised name; a bare rule would rewrite ordinary prose.
_NEGATIONS = ("can", "don", "doesn", "didn", "isn", "wasn", "won", "couldn",
              "wouldn", "shouldn", "haven", "hasn", "hadn", "aren", "weren",
              "mustn", "needn", "ain", "shan", "oughtn")
_PRONOUN_S = ("it", "he", "she", "that", "there", "what", "who", "here",
              "let", "one", "everybody", "somebody", "nobody")
_PRONOUN_OTHER = ("i", "you", "we", "they", "he", "she", "it", "that", "who",
                  "there")

_CONTRACTION_RULES = [
    ("negation", re.compile(r"\b(" + "|".join(_NEGATIONS) + r") t\b",
                            re.IGNORECASE), r"\1’t"),
    ("pronoun_is", re.compile(r"\b(" + "|".join(_PRONOUN_S) + r") s\b",
                              re.IGNORECASE), r"\1’s"),
    ("i_am", re.compile(r"\bI m\b"), "I’m"),
    ("pronoun_other", re.compile(
        r"\b(" + "|".join(_PRONOUN_OTHER) + r") (ll|ve|re|d)\b",
        re.IGNORECASE), r"\1’\2"),
    # A capitalised name followed by a bare "s" is a possessive: "Winterbourne
    # s hat". Restricted to capitalised words so ordinary lowercase prose is
    # untouched.
    ("name_possessive", re.compile(r"\b([A-Z][a-z]{2,}) s\b"), r"\1’s"),
]


def restore_contractions(text):
    """-> (text, counts). Rebuild apostrophes a lossy conversion removed."""
    counts = {}
    for name, pattern, replacement in _CONTRACTION_RULES:
        text, n = pattern.subn(replacement, text)
        if n:
            counts[name] = n
    return text, counts


def check_source_health(text):
    """Every known damage class in one report, before anything is generated.

    This is what a user should see when they add a book - not a refusal after
    twenty minutes of generation. Each entry says what is wrong, how much, and
    whether this tool can fix it.
    """
    replacement = text.count(FFFD)
    broken = sum(len(p.findall(text)) for _n, p, _r in _CONTRACTION_RULES)
    traps = len(_REPETITION_TRAP.findall(text))
    _unb, _q, imbalance = quote_balance(text)
    findings = []
    if replacement:
        findings.append({
            "issue": "replacement_characters", "count": replacement,
            "share": round(replacement / max(1, len(text)), 5),
            "repairable": True,
            "detail": "decoded with the wrong codec; non-ASCII characters lost"})
    if broken:
        findings.append({
            "issue": "stripped_apostrophes", "count": broken,
            "repairable": True,
            "detail": "contractions split by a lossy conversion (don t, it s)"})
    if traps:
        findings.append({
            "issue": "repetition_traps", "count": traps,
            "repairable": True,
            "detail": "a long repeating run makes the model generate to its "
                      "token ceiling"})
    if imbalance > MAX_UNBALANCED_QUOTE_SHARE:
        findings.append({
            "issue": "unbalanced_quotes", "share": round(imbalance, 4),
            "repairable": False,
            "detail": "speech structure may be damaged; not auto-repairable"})
    return {"healthy": not findings, "findings": findings,
            "characters": len(text)}


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
    text, contractions = restore_contractions(text)
    for kind, count in contractions.items():
        applied[f"contraction_{kind}"] += count
    text, collapsed = collapse_repetitions(text)
    for kind, count in collapsed.items():
        applied[f"collapsed_{kind}"] += count
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
        # Close the speech first. A dialogue line that opens a quote and ends
        # in a damaged run ended with a dash AND a closing quote - cut-off
        # speech, "...my Florice-". Replacing both with dashes leaves the
        # quote open, and 69 lines in index18 came out that way: the model
        # then generated until it hit the 16k token ceiling, failed coverage,
        # and retried for 6.5 minutes a time. Structure the reader depends on
        # has to survive a substitution that cannot recover the character.
        lines = text.split("\n")
        # Pair an unmatched closing quote with an opener earlier in the SAME
        # line. "dodging one or two ?cannon shots,? he couldn't" is a quoted
        # phrase mid-sentence, not speech at a line boundary, so none of the
        # positional rules above match it and the marker fell through to a
        # dash - leaving "—cannon shots,”", which combines a quote delimiter
        # with narration and is exactly what the segment gate rejects. 45
        # lines in index18 had this shape and it failed pass 1 on chunk 98.
        paired = 0
        for index, line in enumerate(lines):
            if line.count(CLOSE_DOUBLE) <= line.count(OPEN_DOUBLE):
                continue
            match = re.search(r"(?<=[\s(\[])" + FFFD + r"(?=[A-Za-z])", line)
            if match:
                lines[index] = (line[:match.start()] + OPEN_DOUBLE
                                + line[match.end():])
                paired += 1
        if paired:
            applied["last_resort_paired_opening_quote"] += paired

        # Open the speech too, symmetrically. A line that ends with a closing
        # quote and begins with a damaged run began with an opening quote:
        # "--More men to die." was '"-More men to die."'. Without this the
        # book ends up with 232 lines that close a quote never opened, which
        # is as confusing to the model as one that never closes.
        opened = 0
        for index, line in enumerate(lines):
            if (line.count(CLOSE_DOUBLE) > line.count(OPEN_DOUBLE)
                    and line.lstrip().startswith(FFFD)):
                pad = line[:len(line) - len(line.lstrip())]
                stripped = line.lstrip()
                lines[index] = pad + OPEN_DOUBLE + stripped[1:]
                opened += 1
        if opened:
            applied["last_resort_opening_quote"] += opened
        closed = 0
        for index, line in enumerate(lines):
            if (line.count(OPEN_DOUBLE) > line.count(CLOSE_DOUBLE)
                    and line.rstrip().endswith(FFFD)):
                stripped = line.rstrip()
                pad = line[len(stripped):]
                lines[index] = stripped[:-1] + CLOSE_DOUBLE + pad
                closed += 1
        if closed:
            applied["last_resort_closing_quote"] += closed
        # Join whenever ANY of the three line-level fixes ran. This used to sit
        # under `if closed:`, so paired and opened repairs were computed,
        # counted in `applied`, and then discarded whenever no closing quote
        # happened to need fixing - the counter reported work that never
        # reached the text.
        if paired or opened or closed:
            text = "\n".join(lines)
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
    parser.add_argument("--force", action="store_true",
                        help="write even when the structural check fails")
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

    before_unbal, _bq, before_share = quote_balance(original)
    after_unbal, after_quoted, after_share = quote_balance(repaired)
    regressions = structural_regressions(repaired)
    # Two signals, and only one of them is proven harmful.
    #
    # The REGRESSIONS are shapes measured to break generation: a dialogue line
    # that opens a quote and ends in a substituted dash left index18's chunk 10
    # generating until it hit the token ceiling, nine times. Any of these
    # refuses the write.
    #
    # The SHARE is a heuristic. Undamaged books run 0.5-3.4%, so a much higher
    # figure means residual damage - but a quote spanning paragraphs is normal
    # prose and counts as unbalanced too, so the number cannot be driven to
    # zero and should not block on its own. It warns.
    structurally_sound = not any(regressions.values())
    share_elevated = after_share > MAX_UNBALANCED_QUOTE_SHARE

    report = {
        "source": os.path.abspath(args.source),
        "quote_balance": {
            "unbalanced_before": before_unbal,
            "unbalanced_after": after_unbal,
            "lines_with_quotes": after_quoted,
            "share_after": round(after_share, 4),
            "limit": MAX_UNBALANCED_QUOTE_SHARE,
            "structural_regressions": regressions,
            "structurally_sound": structurally_sound,
        },
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

    if args.apply and not structurally_sound and not args.force:
        print(f"\nREFUSING TO WRITE: {regressions}. A substitution destroyed "
              "sentence structure - a dialogue line that opens a quote and "
              "ends in a dash, or closes one it never opened. This exact shape "
              "left index18 generating until it hit its token ceiling, nine "
              "times on one chunk. Removing every replacement character while "
              "breaking the prose is not a repair. Use --force only after "
              "reading the report.")
        return 1
    if share_elevated:
        print(f"\nWARNING: {after_share:.1%} of quoted lines are unbalanced "
              f"(undamaged books run 0.5-3.4%). No proven-harmful shapes "
              "remain, so this is written, but the text still carries "
              "residual damage and may generate less reliably.")
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
    print(f"\n  quote balance: {after_unbal}/{after_quoted} lines "
          f"({after_share:.1%}, limit {MAX_UNBALANCED_QUOTE_SHARE:.0%})  "
          f"{'SOUND' if structurally_sound else 'BROKEN'}")
    if any(regressions.values()):
        print(f"  structural regressions: {regressions}")
    print(f"\n  report: {report_path}")
    print(f"  repaired file: {out_path if args.apply else '(dry run, not written)'}")


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)


def preflight_source(text):
    """Check a source for known damage and repair what can be repaired.

    ONE definition, called by both generation paths. The single-pass and
    three-pass gates have disagreed three times already - on the replacement
    limit, on duplicate blocks, and on publisher matter - each time because a
    capability lived on one path only, and each time the symptom was the same
    book behaving differently depending on how it was generated.

    Returns a new dict; the caller's text is not modified in place. The user's
    file on disk is never rewritten - a repair is a best guess at an original
    that cannot be recovered, and editing someone's book on a guess is not a
    decision this function gets to make.

    Damage found here is a risk signal, not proof of failure, so an
    unrepairable finding is reported and does NOT refuse the book. The hard
    refusals stay where they are: unsafe control characters and replacement
    load above the shared limit.
    """
    before = check_source_health(text)
    result = {"text": text, "healthy": before["healthy"],
              "findings": before["findings"], "applied": {}, "messages": []}
    if before["healthy"]:
        return result

    result["messages"].append(
        f"Source health: {len(before['findings'])} issue(s) found")
    for finding in before["findings"]:
        suffix = "" if finding["repairable"] else "  [not auto-repairable]"
        result["messages"].append(
            f"  - {finding['issue']}: {finding['detail']}{suffix}")

    repaired, applied, _examples = repair(text)
    after = check_source_health(repaired)
    result["applied"] = applied
    if applied:
        result["messages"].append(
            f"  repaired in memory ({sum(applied.values())} fixes; "
            "your file on disk is unchanged)")

    if after["healthy"]:
        result.update(text=repaired, healthy=True, findings=[])
        result["messages"].append("  source is now clean")
    elif len(after["findings"]) < len(before["findings"]):
        result.update(text=repaired, findings=after["findings"])
        result["messages"].append(
            f"  {len(after['findings'])} issue(s) remain; continuing")
    else:
        result["messages"].append(
            "  repair changed nothing; continuing with the original")
    return result
