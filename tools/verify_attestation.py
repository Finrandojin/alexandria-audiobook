"""Offline dry-run analyzer for speaker-label attestation.

Speaker labels are the one field in the annotation schema the LLM still
*generates* rather than selects, and it transcribes names imperfectly:
systematic misreadings, invented names for unnamed speakers, and spelling
drift that fragments one character across several roster entries and therefore
across several voices.

`speaker_canon.attest_speaker()` can tell the three cases apart by checking a
label's core name tokens against the source text near that label's own
entries. Turning that check into a GATE at annotation time -- rejecting a
label the book does not support, so the line is narrated instead of claiming a
fabricated voice -- is only safe if the real-world rejection rate is known.
That number cannot be obtained from an offline property test, because it
depends entirely on the book and the model. This script measures it.

The gate does not simply reject every UNATTESTED label: a refuted spelling the
book never uses, one edit from an established name that is present in the
label's own window, is REPAIRED onto that name (speaker_canon.repair_speaker).
This report mirrors that, so the "would be rejected" number stays the number a
real run would produce -- and it itemizes the repairs separately, because a
repair silently changes who a line is attributed to and deserves an audit.

Read-only: it opens annotated_script.json and the source text, writes nothing,
calls no LLM and touches no network. Run it before enabling any gate, and
again after, to see what changed.

Usage:
    python tools/verify_attestation.py
        Analyze the current project (annotated_script.json plus the source
        pointed at by state.json) and print the would-be rejection rate.

    python tools/verify_attestation.py --script PATH --source PATH
        Analyze a specific pair, e.g. an archived script.

    python tools/verify_attestation.py --list unattested
        Also list every speaker with that verdict, worst (most entries)
        first. Accepts: unattested, unverifiable, attested, all.

Exit codes:
    0  analysis completed (whatever it found -- this is a report, not a test)
    2  inputs unavailable (no script, no source), with an explanation
"""

import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(ROOT_DIR, "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from speaker_canon import (  # noqa: E402  (must follow the sys.path tweak)
    ATTESTED,
    UNATTESTED,
    UNVERIFIABLE,
    attest_speaker,
    remember_in_roster,
    repair_speaker,
    source_word_index,
)

# Deliberately mirrors app.py's _compute_label_flags so this dry run measures
# what the endpoint (and a future gate) would actually see, rather than a
# more-generous or more-punishing approximation. Overridable on the command
# line to test the sensitivity of the result to the window size.
DEFAULT_MAX_SAMPLED_ENTRIES = 20
DEFAULT_WINDOW_RADIUS = 400

NARRATOR = "NARRATOR"


def load_source_text(explicit_path=None):
    """Return (text, path, note). Applies the SAME mojibake fix generation
    applied, because the entry texts in annotated_script.json were produced
    from the fixed text -- comparing them against the raw file makes every
    str.find() miss and reports a book with perfect labels as unattestable.
    """
    path = explicit_path
    if path is None:
        state_path = os.path.join(ROOT_DIR, "state.json")
        if not os.path.exists(state_path):
            return None, None, "no state.json; pass --source explicitly"
        try:
            with open(state_path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            return None, None, f"state.json unreadable ({exc}); pass --source"
        path = state.get("input_file_path")
        if not path:
            return None, None, "state.json has no input_file_path; pass --source"

    if not os.path.exists(path):
        return None, path, f"source file not found: {path}"

    if path.lower().endswith(".epub"):
        try:
            from app import extract_epub_text  # noqa: PLC0415  (optional, heavy)
        except Exception as exc:  # pragma: no cover - depends on env
            return None, path, (
                f"cannot import extract_epub_text from app.py ({exc}). "
                "Install the server deps, or export the epub to .txt and pass "
                "--source."
            )
        raw = extract_epub_text(path)
    else:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()

    try:
        from generate_script import fix_mojibake
    except Exception as exc:  # pragma: no cover - depends on env
        return raw, path, (
            f"WARNING: could not import fix_mojibake ({exc}); comparing "
            "against RAW source. Entry lookups may miss."
        )
    return fix_mojibake(raw), path, None


def load_script(explicit_path=None):
    path = explicit_path or os.path.join(ROOT_DIR, "annotated_script.json")
    if not os.path.exists(path):
        return None, path, f"no script at {path}"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle), path, None
    except (json.JSONDecodeError, ValueError) as exc:
        return None, path, f"script is not valid JSON ({exc})"


def analyze(script_data, source_text, max_sampled, radius):
    """Group entries by canonical speaker, build local windows, and return a
    per-speaker report plus totals. Pure apart from reading its arguments.
    """
    roster_index = {}
    by_speaker = {}
    for entry in script_data:
        raw_speaker = entry.get("speaker") or entry.get("type") or ""
        canonical = remember_in_roster(roster_index, raw_speaker)
        if not canonical or canonical == NARRATOR:
            continue
        by_speaker.setdefault(canonical, []).append(entry.get("text") or "")

    # Book-wide word index and full roster, so the dry run can mirror the
    # generation-time gate INCLUDING its bounded repair step. Without these two
    # the dry run would over-report rejections relative to what a real run does.
    source_words = source_word_index(source_text)

    report = []
    for name, texts in by_speaker.items():
        windows = []
        located = 0
        for text in texts[:max_sampled]:
            if not text:
                continue
            found = source_text.find(text)
            if found == -1:
                continue
            located += 1
            start = max(0, found - radius)
            end = min(len(source_text), found + len(text) + radius)
            windows.append(source_text[start:end])

        # The roster this label would face at generation time is not knowable
        # after the fact, so the dry run uses the FULL roster of the finished
        # script -- an upper bound on what was established, and therefore an
        # upper bound on both partial attestation and repair.
        # The label itself is excluded from that roster: at generation time an
        # established name skips the gate entirely, so leaving it in would make
        # every label trivially attested and the report useless.
        others = {key: value for key, value in roster_index.items()
                  if value != name}
        verdict = attest_speaker(name, windows, roster_index=others)
        repaired = None
        if verdict == UNATTESTED:
            repaired = repair_speaker(name, windows, others, source_words)

        report.append({
            "name": name,
            "entry_count": len(texts),
            "sampled": min(len(texts), max_sampled),
            "located": located,
            "verdict": verdict,
            "repaired_to": repaired,
        })

    report.sort(key=lambda row: (-row["entry_count"], row["name"]))
    return report


def summarize(report, script_data):
    total_entries = len(script_data)
    dialogue_entries = sum(row["entry_count"] for row in report)
    by_verdict = {ATTESTED: [], UNATTESTED: [], UNVERIFIABLE: []}
    for row in report:
        by_verdict[row["verdict"]].append(row)

    def pct(part, whole):
        return f"{(100.0 * part / whole):5.1f}%" if whole else "    --"

    print("=" * 72)
    print("Speaker-label attestation dry run")
    print("=" * 72)
    print(f"Script entries:              {total_entries}")
    print(f"Non-narrator entries:        {dialogue_entries}")
    print(f"Distinct speakers:           {len(report)}")

    unlocated = [row for row in report if row["located"] == 0]
    if unlocated:
        print()
        print(f"WARNING: {len(unlocated)} speaker(s) had NO entry text found in the")
        print("  source at all. Their verdicts are meaningless -- this points at a")
        print("  script/source mismatch (wrong book, or text transformed after")
        print("  generation), not at bad labels. Fix that before trusting anything")
        print("  below.")

    print()
    print("Verdict           speakers          entries")
    print("-" * 72)
    for verdict in (ATTESTED, UNVERIFIABLE, UNATTESTED):
        rows = by_verdict[verdict]
        entries = sum(row["entry_count"] for row in rows)
        print(f"{verdict:<16}{len(rows):>6} {pct(len(rows), len(report))}"
              f"{entries:>10} {pct(entries, dialogue_entries)}")
    print("-" * 72)

    repairable = [row for row in by_verdict[UNATTESTED] if row["repaired_to"]]
    if repairable:
        repaired_entries = sum(row["entry_count"] for row in repairable)
        print()
        print("Of the UNATTESTED labels, these would NOT be rejected: bounded")
        print("repair folds them onto an established spelling the book actually")
        print("uses (speaker_canon.repair_speaker), so they keep a voice -- but")
        print("under a name the model did not emit. Audit them:")
        print(f"    {len(repairable)} speakers, {repaired_entries} entries")
        for row in sorted(repairable, key=lambda r: -r["entry_count"]):
            print(f"      {row['entry_count']:>6}  \"{row['name']}\" -> "
                  f"\"{row['repaired_to']}\"")

    rejected = [row for row in by_verdict[UNATTESTED] if not row["repaired_to"]]
    rejected_entries = sum(row["entry_count"] for row in rejected)
    print()
    print("If a gate rejected only UNATTESTED labels (the only verdict that is")
    print("positive evidence a label is wrong), these lines would be narrated")
    print("instead of getting a character voice:")
    print(f"    {len(rejected)} of {len(report)} speakers "
          f"({pct(len(rejected), len(report)).strip()})")
    print(f"    {rejected_entries} of {dialogue_entries} non-narrator entries "
          f"({pct(rejected_entries, dialogue_entries).strip()})")
    print()
    print("UNVERIFIABLE is never rejected: it means the check does not apply to")
    print("this text (a title-only label, or a name present but not at a word")
    print("boundary, as in unsegmented scripts). Those keep their voices.")
    return by_verdict


def list_rows(by_verdict, which):
    wanted = ([ATTESTED, UNVERIFIABLE, UNATTESTED] if which == "all" else [which])
    for verdict in wanted:
        rows = by_verdict.get(verdict) or []
        if not rows:
            continue
        print()
        print(f"--- {verdict} ({len(rows)} speakers, most entries first) ---")
        print(f"{'entries':>8}  {'located':>8}  name")
        for row in rows:
            suffix = (f"  -> repaired to \"{row['repaired_to']}\""
                      if row.get("repaired_to") else "")
            print(f"{row['entry_count']:>8}  "
                  f"{row['located']:>3}/{row['sampled']:<4}  {row['name']}{suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline dry run: how many speaker labels would an "
                    "attestation gate reject on this book?")
    parser.add_argument("--script", default=None,
                        help="path to annotated_script.json "
                             "(default: repo root)")
    parser.add_argument("--source", default=None,
                        help="path to the source .txt/.md/.epub "
                             "(default: state.json's input_file_path)")
    parser.add_argument("--list", dest="list_verdict", default=None,
                        choices=[ATTESTED, UNVERIFIABLE, UNATTESTED, "all"],
                        help="also list the speakers with this verdict")
    parser.add_argument("--max-sampled", type=int,
                        default=DEFAULT_MAX_SAMPLED_ENTRIES,
                        help="entries sampled per speaker "
                             f"(default {DEFAULT_MAX_SAMPLED_ENTRIES}, "
                             "mirrors the label_flags endpoint)")
    parser.add_argument("--radius", type=int, default=DEFAULT_WINDOW_RADIUS,
                        help="source characters each side of a sampled entry "
                             f"(default {DEFAULT_WINDOW_RADIUS})")
    args = parser.parse_args()

    script_data, script_path, script_note = load_script(args.script)
    if script_data is None:
        print(f"Cannot analyze: {script_note}")
        print("Generate a script first, or pass --script.")
        return 2
    if not isinstance(script_data, list) or not script_data:
        print(f"Cannot analyze: {script_path} contains no entries.")
        return 2

    source_text, source_path, source_note = load_source_text(args.source)
    if source_text is None:
        print(f"Cannot analyze: {source_note}")
        return 2
    if source_note:
        print(source_note)
        print()

    print(f"Script: {script_path}")
    print(f"Source: {source_path}")
    print()

    report = analyze(script_data, source_text, args.max_sampled, args.radius)
    if not report:
        print("No non-narrator speakers in this script; nothing to attest.")
        return 0

    by_verdict = summarize(report, script_data)
    if args.list_verdict:
        list_rows(by_verdict, args.list_verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
