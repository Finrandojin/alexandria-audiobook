"""Goal 5.1: what actually reaches the TTS engine with no spoken form.

THE MISSING MEASUREMENT. 5.1 has a source-level baseline - counts over the
input `.txt` files - and records "At the TTS boundary: still NO BASELINE." The
source gate proves the app can refuse bad input; it says nothing about what
survives the journey to the speaker, which is a different question with a
different answer, because `get_speech_normalization` sits in between and
rewrites some of it.

WHERE THE BOUNDARY ACTUALLY IS. Every voice path in `tts.py` - clone, lora,
design, ensemble, custom - calls `normalize_for_speech(text)` as its first
statement inside `generate_voice`. That single call is the boundary, so this
audits its OUTPUT rather than the script text. Auditing the input would
re-measure the source-level baseline that already exists and would blame the
normaliser for characters it removes.

WHAT COUNTS AS UNSPEAKABLE. Not "non-ASCII" - that would flag every accented
name and all the CJK in a Japanese book, which are speakable and are the point.
The test is Unicode general category: symbols (So/Sm/Sk), private use (Co),
unassigned (Cn), and the replacement character U+FFFD, which is not a character
at all but the residue of a decoding failure. Letters, marks, digits,
whitespace and punctuation are left alone in every script.

The corpus is the 82 saved books in `scripts/`, which is what this app has
actually been asked to speak, rather than a constructed sample.
"""
import argparse
import collections
import glob
import json
import os
import sys
import unicodedata

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if not os.path.isdir(os.path.join(REPO, "scripts")):
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "app"))

from speech_text import get_speech_normalization  # noqa: E402

# Categories with no spoken form. Sc (currency) is deliberately absent: "$"
# and "£" have well-defined spoken forms and belong in a verbalization table,
# not in a defect count.
UNSPEAKABLE_CATEGORIES = {"So", "Sm", "Sk", "Co", "Cn", "Cs"}
REPLACEMENT_CHAR = "�"


def is_unspeakable(ch):
    if ch == REPLACEMENT_CHAR:
        return True
    if ch.isspace():
        return False
    return unicodedata.category(ch) in UNSPEAKABLE_CATEGORIES


def audit_text(text):
    return collections.Counter(ch for ch in text if is_unspeakable(ch))


def script_entries(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("entries") or payload.get("lines") or []
    return payload if isinstance(payload, list) else []


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scripts", default=os.path.join(REPO, "scripts"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "tts_boundary_audit.json"))
    args = ap.parse_args()

    before = collections.Counter()
    after = collections.Counter()
    per_book = {}
    lines_total = lines_affected = 0

    paths = sorted(p for p in glob.glob(os.path.join(args.scripts, "*.json"))
                   if not p.endswith(".generation_quality.json"))
    for path in paths:
        book = os.path.basename(path)[:-len(".json")]
        book_after = collections.Counter()
        book_lines = book_hits = 0
        for entry in script_entries(path):
            text = (entry or {}).get("text")
            if not text:
                continue
            book_lines += 1
            before.update(audit_text(text))
            spoken = get_speech_normalization(text)["text"] or ""
            hits = audit_text(spoken)
            if hits:
                book_hits += 1
                book_after.update(hits)
        lines_total += book_lines
        lines_affected += book_hits
        after.update(book_after)
        if book_after:
            per_book[book] = {
                "lines": book_lines, "lines_affected": book_hits,
                "characters": sum(book_after.values()),
                "distinct": len(book_after),
                "top": [[c, n] for c, n in book_after.most_common(8)],
            }

    def describe(counter):
        return [{"char": c, "codepoint": f"U+{ord(c):04X}",
                 "name": unicodedata.name(c, "<unnamed>"),
                 "category": unicodedata.category(c), "count": n}
                for c, n in counter.most_common(40)]

    result = {
        "scope": "output of normalize_for_speech, the single call every "
                 "tts.py voice path makes before synthesis",
        "books": len(paths), "lines": lines_total,
        "lines_reaching_tts_with_unspeakable": lines_affected,
        "characters_before_normalization": sum(before.values()),
        "characters_at_tts_boundary": sum(after.values()),
        "removed_by_normalization": sum(before.values()) - sum(after.values()),
        "distinct_at_boundary": len(after),
        "at_boundary": describe(after),
        "per_book": per_book,
    }

    from utils import atomic_json_write
    atomic_json_write(result, args.out)

    print("=== goal 5.1: the TTS boundary ===")
    print(f"  {len(paths)} books, {lines_total} lines")
    print(f"  unspeakable characters in raw script text : "
          f"{sum(before.values())}")
    print(f"  removed by normalize_for_speech           : "
          f"{result['removed_by_normalization']}")
    print(f"  STILL PRESENT at the TTS boundary         : "
          f"{sum(after.values())}  ({len(after)} distinct)")
    print(f"  lines affected                            : {lines_affected} "
          f"({100.0 * lines_affected / max(1, lines_total):.2f}%)")
    if after:
        print("\n  what survives:")
        for row in describe(after)[:15]:
            print(f"    {row['codepoint']:8} {row['category']}  "
                  f"{row['count']:6}  {row['name'][:44]}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
