"""Build and merge speaker-attribution judgement batches for an external judge.

The gold set is the only instrument in this project that has been reliable, and
what made it trustworthy was two independent readers agreeing at 94-97% with
every disagreement falling on lines whose true answer is not a character. This
scales that protocol: emit batches a large model can judge, merge them back with
validation, and report concordance when two judges have seen the same lines.

    build   sample lines from a book and write judgement batches
    merge   validate filled batches and write a gold fixture
    agree   compare two judges' answers and list only their disagreements

Design choices that come from things which went wrong:

- Lines whose normalized text repeats in the book are excluded. Such a line
  cannot be aligned to one position, and scoring it at each occurrence counts
  one judgement several times - a bug that reached two separate harnesses.
- Context is taken from the segmented entries themselves, in reading order, not
  by searching the source text. A source search failed on 35 of 50 lines when
  it was tried; the segmented text is byte-frozen from the source and exact.
- Batches are small. An earlier 147-line request was truncated by the judge at
  row 85, and the loss was invisible until the ids were counted.
"""
import argparse
import collections
import json
import os
import random
import re

INSTRUCTIONS = (
    "For each row, read the passage and put the speaker's NAME in ANSWER "
    "(uppercase). The line being judged sits between passage_before and "
    "passage_after, which are the surrounding entries in reading order.\n\n"
    "Rules:\n"
    "- Use the name the book uses. If a character is called both a short and a "
    "full form, either is fine.\n"
    "- Use NARRATOR if the line is not speech at all - a sign, a caption, a "
    "heading, or narration the segmenter split out by mistake.\n"
    "- Use AMBIGUOUS if the passage genuinely supports more than one reading. "
    "Do not guess; a marked-ambiguous line is more useful than a coin flip.\n"
    "- Name the speaker if you can identify them, even when the name is not "
    "printed in the window - following 'he', 'she' or first-person continuity "
    "is legitimate inference, not guessing. Reserve AMBIGUOUS for lines whose "
    "speaker is genuinely undetermined, and NARRATOR for lines that are not "
    "speech.\n"
    "- Never invent a name that appears nowhere in the book.\n"
    "- Put a short justification in reasoning.\n\n"
    "These are randomly sampled lines, so many will be easy; that is intended. "
    "Return every row."
)


def normalize(text):
    return re.sub(r"\W+", "", text or "").lower()


def load_run(root, run, book):
    """The whole checkpoint from a completed three-pass run."""
    path = os.path.join(root, run, book, "result.json.threepass_checkpoint.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def eligible_indexes(segmented, min_chars=4):
    """SPOKEN entries whose text occurs exactly once in the book.

    A repeated line cannot be aligned to a single position, so a judgement on it
    cannot be attached to one entry.
    """
    counts = collections.Counter(normalize(e.get("text")) for e in segmented)
    return [i for i, entry in enumerate(segmented)
            if entry.get("type") == "SPOKEN"
            and len((entry.get("text") or "").strip()) >= min_chars
            and counts[normalize(entry.get("text"))] == 1]


def book_names(checkpoint, source_text=None):
    """Character names this book actually uses, from the run's own roster.

    A loose capitalised-word heuristic was tried first and was useless as an
    expansion trigger: it matched "Ahh" and "Army", so almost every window
    already contained two "names" and only 3 of 400 rows ever expanded. The
    pipeline's attested-name gate is the right source - it is the same check
    that keeps invented speakers out of the roster.
    """
    from three_pass_generate import build_roster
    named = [entry for entry in (checkpoint.get("named") or []) if entry]
    # Not .title(): it capitalises after every non-letter, turning "BRI-CHAN"
    # into "Bri-Chan" which never matches the book's "Bri-chan". That mistake
    # has now cost this project a shipped bug in the attestation gate and a
    # silent failure here. Names are matched case-insensitively instead.
    return set(build_roster(named, source_text))


def context(segmented, index, before=9, after=6, names=None,
            min_names=2, max_before=40, max_after=20):
    """Surrounding entries, widened until the speaker is identifiable.

    A fixed window forced a judge to mark 13 answerable lines AMBIGUOUS: the
    speaker was recoverable from "he", "she" or first-person continuity, but no
    name appeared nearby, and naming them would have failed the anti-invention
    check. Policy decision by the repository owner: expand the window until
    identity is supported rather than exclude those lines or accept abstention.

    Expansion is backwards-first, since a speaker is usually established before
    they speak, and stops at a cap so a batch cannot grow without bound.
    """
    def span(start, stop):
        return "\n\n".join(" ".join((segmented[j].get("text") or "").split())
                           for j in range(max(0, start), min(len(segmented), stop)))

    if not names:
        return span(index - before, index), span(index + 1, index + 1 + after)

    while True:
        window = span(index - before, index + 1 + after)
        found = {name for name in names
                 if re.search(r"\b" + re.escape(name) + r"\b", window, re.IGNORECASE)}
        if len(found) >= min_names:
            break
        if before >= max_before and after >= max_after:
            break
        if index - before <= 0 and index + after >= len(segmented) - 1:
            break
        before = min(before + 6, max_before)
        after = min(after + 3, max_after)
    return span(index - before, index), span(index + 1, index + 1 + after)


def build(segmented, book, count, batch_size, seed=11, names=None):
    """Judgement batches over randomly sampled, unambiguous SPOKEN lines."""
    pool = eligible_indexes(segmented)
    chosen = sorted(random.Random(seed).sample(pool, min(count, len(pool))))
    rows = []
    for index in chosen:
        before, after = context(segmented, index, names=names)
        rows.append({"id": f"{book}-{index:05d}", "entry_index": index,
                     "line": " ".join(segmented[index]["text"].split()),
                     "passage_before": before, "passage_after": after,
                     "ANSWER": "", "reasoning": ""})
    batches = []
    total = (len(rows) + batch_size - 1) // batch_size
    for number, start in enumerate(range(0, len(rows), batch_size), 1):
        batches.append({"book": book, "batch": f"{number} of {total}",
                        "instructions": INSTRUCTIONS,
                        "rows": rows[start:start + batch_size]})
    return batches, len(pool)


def read_filled(paths):
    """{id: {answer, reasoning}} from filled batches, ignoring blanks."""
    answers = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        for row in (payload["rows"] if isinstance(payload, dict) else payload):
            value = (row.get("ANSWER") or "").strip().upper()
            if value:
                answers[row["id"]] = {"answer": value,
                                      "reasoning": (row.get("reasoning") or "").strip()}
    return answers


def support_for(answer, row, source_text=None):
    """Where the evidence for this answer lives: window, book, or nowhere.

    The distinction matters and the first version of this tool missed it. A name
    printed in the supplied window is directly supported. A name absent from the
    window but present in the book is *inferred from wider context* - a reader
    following "he", "she" or first-person continuity can be right without the
    name appearing nearby. A name absent from the book entirely is invented,
    which is the only case worth refusing.

    Collapsing the middle case into the last one forced a judge to mark 13
    answerable lines AMBIGUOUS, which would have rewarded abstention on lines
    whose speaker is recoverable - silently changing the task being measured.
    """
    if answer in ("NARRATOR", "AMBIGUOUS", "UNKNOWN"):
        return "not_a_name"
    first = answer.split()[0]
    window = " ".join((row["passage_before"], row["line"], row["passage_after"]))
    if re.search(r"\b" + re.escape(first) + r"\b", window, re.IGNORECASE):
        return "window"
    if source_text and re.search(r"\b" + re.escape(first) + r"\b",
                                 source_text, re.IGNORECASE):
        return "book"
    return "absent"


def validate(answers, batches, source_text=None):
    """Problems that would make a fixture untrustworthy.

    Pass source_text to distinguish an invented name from one inferred from
    wider context. Without it the check falls back to the window, which is
    stricter than intended and was the original defect.
    """
    expected = {row["id"]: row for batch in batches for row in batch["rows"]}
    problems = []
    unknown = sorted(set(answers) - set(expected))
    if unknown:
        problems.append(f"{len(unknown)} answers for unknown ids, e.g. {unknown[:3]}")
    missing = sorted(set(expected) - set(answers))
    if missing:
        problems.append(f"{len(missing)} rows unanswered, e.g. {missing[:3]}")
    for gold_id, value in sorted(answers.items()):
        row = expected.get(gold_id)
        if not row:
            continue
        if support_for(value["answer"], row, source_text) == "absent":
            where = "the book" if source_text else "its passage"
            problems.append(f"{gold_id}: {value['answer']!r} appears nowhere in "
                            f"{where} - invented")
    return problems


def support_summary(answers, batches, source_text=None):
    """How many answers are window-supported, book-only, or neither."""
    expected = {row["id"]: row for batch in batches for row in batch["rows"]}
    counts = collections.Counter()
    book_only = []
    for gold_id, value in sorted(answers.items()):
        row = expected.get(gold_id)
        if not row:
            continue
        kind = support_for(value["answer"], row, source_text)
        counts[kind] += 1
        if kind == "book":
            book_only.append(gold_id)
    return counts, book_only


def merge(answers, batches, book, source_run, judged_by, aliases=None):
    expected = {row["id"]: row for batch in batches for row in batch["rows"]}
    entries = [{"id": gold_id, "book": book,
                "entry_index": expected[gold_id]["entry_index"],
                "line": expected[gold_id]["line"],
                "expected_speaker": value["answer"],
                "judged_by": judged_by, "reasoning": value["reasoning"]}
               for gold_id, value in sorted(answers.items()) if gold_id in expected]
    return {"description":
            ("Hand-judged attribution answers on randomly sampled lines. Lines "
             "whose text repeats in the book are excluded, so every entry "
             "aligns to exactly one position."),
            "book": book, "source_run": source_run,
            "sampling": "random over SPOKEN entries with unique text",
            "entries": entries, "aliases": aliases or []}


def agreement(first, second, aliases=()):
    """(agreed, disagreements) between two judges over the ids they share."""
    groups = [{name.upper() for name in group} for group in aliases]

    def same(a, b):
        a, b = (a or "").upper(), (b or "").upper()
        return a == b or any(a in g and b in g for g in groups)

    shared = sorted(set(first) & set(second))
    disagreements = [(i, first[i]["answer"], second[i]["answer"])
                     for i in shared
                     if not same(first[i]["answer"], second[i]["answer"])]
    return len(shared) - len(disagreements), disagreements


DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "ab_test_runtime", "results", "matrix_20260725-115148")
DEFAULT_RUN = "qwen3.5-9b-uncensored-hauhaucs-aggressive"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("build", help="write judgement batches for a judge")
    make.add_argument("book")
    make.add_argument("--count", type=int, default=400)
    make.add_argument("--batch-size", type=int, default=40)
    make.add_argument("--seed", type=int, default=11)
    make.add_argument("--out", default="judgements")
    make.add_argument("--root", default=DEFAULT_ROOT)
    make.add_argument("--run", default=DEFAULT_RUN)

    join = sub.add_parser("merge", help="validate filled batches into a fixture")
    join.add_argument("book")
    join.add_argument("filled", nargs="+")
    join.add_argument("--batches", nargs="+", required=True,
                      help="the unfilled batches, to check ids and passages")
    join.add_argument("--judged-by", required=True)
    join.add_argument("--out", required=True)
    join.add_argument("--alias", action="append", default=[],
                      help="comma-separated names for one character, repeatable")
    join.add_argument("--force", action="store_true",
                      help="write despite validation problems")

    check = sub.add_parser("agree", help="compare two judges, list disagreements")
    check.add_argument("first", nargs="+")
    check.add_argument("--second", nargs="+", required=True)
    check.add_argument("--alias", action="append", default=[])

    args = parser.parse_args(argv)

    if args.command == "build":
        checkpoint = load_run(args.root, args.run, args.book)
        segmented = checkpoint["segmented"]
        source = os.path.join(args.root, "inputs", f"{args.book}.txt")
        source_text = (open(source, encoding="utf-8").read()
                       if os.path.exists(source) else None)
        names = book_names(checkpoint, source_text)
        print(f"cast: {len(names)} attested names")
        batches, pool = build(segmented, args.book, args.count,
                              args.batch_size, args.seed, names)
        os.makedirs(args.out, exist_ok=True)
        for number, batch in enumerate(batches, 1):
            path = os.path.join(args.out, f"{args.book}_batch{number:02d}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(batch, handle, indent=1, ensure_ascii=False)
            print(f"{path}  {len(batch['rows'])} rows")
        judged = sum(len(b["rows"]) for b in batches)
        print(f"\n{judged} lines sampled from {pool} eligible "
              f"({len(segmented)} segmented entries)")
        return 0

    if args.command == "merge":
        batches = [json.load(open(p, encoding="utf-8")) for p in args.batches]
        answers = read_filled(args.filled)
        problems = validate(answers, batches)
        for problem in problems:
            print(f"  ! {problem}")
        if problems and not args.force:
            print(f"\n{len(problems)} problems; not writing. Use --force to override.")
            return 1
        aliases = [group.split(",") for group in args.alias]
        fixture = merge(answers, batches, args.book, args.run if hasattr(args, "run")
                        else DEFAULT_RUN, args.judged_by, aliases)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(fixture, handle, indent=1, ensure_ascii=False)
        print(f"wrote {args.out}: {len(fixture['entries'])} entries")
        return 0

    first, second = read_filled(args.first), read_filled(args.second)
    aliases = [group.split(",") for group in args.alias]
    agreed, disagreements = agreement(first, second, aliases)
    shared = agreed + len(disagreements)
    print(f"{shared} lines judged by both: {agreed} agree "
          f"({agreed/max(shared,1)*100:.1f}%), {len(disagreements)} differ\n")
    for gold_id, a, b in disagreements:
        print(f"  {gold_id}  judge1={a:20} judge2={b}")
    print("\nOnly these need a human. Everything else two judges already agree on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
