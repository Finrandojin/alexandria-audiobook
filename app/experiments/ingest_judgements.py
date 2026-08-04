"""Read the judge's JSON replies back into the labelling bundle.

Tolerates the usual frontier-model output habits - a markdown fence, a
preamble, a trailing comment - because re-prompting for clean JSON costs more
of the user's attention than parsing loosely costs us. What it does NOT
tolerate is a missing or extra id: a reply that silently drops items would
produce a fixture shorter than the sample, and a fixture that quietly differs
from its own sampling record cannot be reproduced.

Reports the abstention and NOT_DIALOGUE rates, which are measurements in their
own right: NOT_DIALOGUE is the segmenter's error rate on spoken segments, and
UNKNOWN plus not-confident is the share of the book that is underdetermined
for a strong reader - an upper bound on what any model could get right.
"""
import collections
import glob, json, os, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
BOOK = os.environ.get("EXPERIMENT_BOOK", "index18")
BUNDLE = os.environ.get("EXPERIMENT_BUNDLE", os.path.join(
    REPO, "ab_test_runtime", "fixtures_draft", f"labelling_bundle__{BOOK}.json"))
REPLIES = os.environ.get("EXPERIMENT_REPLIES", os.path.join(
    REPO, "ab_test_runtime", "fixtures_draft", f"judge_prompts__{BOOK}"))

bundle = json.load(open(BUNDLE, encoding="utf-8"))
by_id = {e["id"]: e for e in bundle["entries"]}


def parse(path):
    """Return a list of judgement dicts, whatever shape the judge replied in.

    Observed shapes so far: a bare array of {id, speaker, ...}; the same
    fenced in markdown; and an object {book, batch, rows: [...]} whose rows
    carry `ANSWER` rather than `speaker`. Re-prompting for one canonical shape
    costs more of the user's attention than accepting several costs us, so
    long as nothing is guessed - ids and labels are read, never inferred.
    """
    raw = open(path, encoding="utf-8").read().strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if fence:
        raw = fence.group(1).strip()
    try:
        data = json.loads(raw)
    except ValueError:
        start, end = raw.find("["), raw.rfind("]")
        if start < 0 or end < 0:
            raise SystemExit(f"{path}: no JSON array found")
        data = json.loads(raw[start:end + 1])
    if isinstance(data, dict):
        rows = next((v for v in data.values() if isinstance(v, list)), None)
        if rows is None:
            raise SystemExit(f"{path}: object reply with no list of rows")
        data = rows
    out = []
    for item in data:
        if not isinstance(item, dict) or "id" not in item:
            raise SystemExit(f"{path}: row without an id: {str(item)[:120]}")
        speaker = item.get("speaker")
        if speaker is None:
            speaker = item.get("ANSWER") or item.get("answer")
        if speaker is None:
            raise SystemExit(f"{path}: row {item['id']} has no speaker/ANSWER")
        out.append({**item, "speaker": speaker})
    return out


seen, duplicates = {}, []
files = sorted(glob.glob(os.path.join(REPLIES, "reply_*.json")))
if not files:
    raise SystemExit(f"no reply_*.json under {REPLIES}")
for path in files:
    for item in parse(path):
        ident = item.get("id")
        if ident not in by_id:
            raise SystemExit(f"{path}: id {ident!r} is not in the bundle")
        if ident in seen:
            # A prompt re-run duplicates ids. That is fine when the two
            # judgements agree - and it is a free reproducibility check, which
            # is worth reporting rather than rejecting. A CONFLICTING duplicate
            # is a different matter and still stops the ingest.
            before = (seen[ident].get("speaker") or "").strip().upper()
            now = (item.get("speaker") or "").strip().upper()
            if before != now:
                raise SystemExit(f"{path}: id {ident!r} judged twice and the "
                                 f"answers disagree: {before!r} vs {now!r}")
            duplicates.append(ident)
            continue
        seen[ident] = item

missing = [i for i in by_id if i not in seen]
if missing:
    raise SystemExit(f"{len(missing)} items unjudged, first {missing[0]}. "
                     f"A short fixture cannot be reproduced from its sampling "
                     f"record; re-run the prompt covering them.")

if duplicates:
    print(f"  {len(duplicates)} ids were judged twice and agreed every time - "
          f"a re-run of a prompt, and an unplanned reproducibility check")
counts = collections.Counter()
for ident, item in seen.items():
    speaker = str(item.get("speaker") or "").strip().upper() or "UNKNOWN"
    entry = by_id[ident]
    entry["expected_speaker"] = speaker
    entry["confident"] = bool(item.get("confident"))
    # Accept either key: `reasoning` matches the gold schema, `why` is what
    # the first batch of prompts asked for.
    entry["note"] = (item.get("reasoning") or item.get("why") or "").strip() or None
    if item.get("alias"):
        entry["alias"] = str(item["alias"]).strip().upper()
    counts[speaker if speaker in ("UNKNOWN", "NOT_DIALOGUE") else "named"] += 1
    if not entry["confident"]:
        counts["not_confident"] += 1

total = len(seen)
# Collect the flagged pairs into the bundle's alias list. finalise_fixture
# copies that list into the fixture, and an alias that stays only on the
# individual entry is an alias the scorer never sees - which silently marks a
# correct answer wrong, exactly the ZODIAC/ZODIAC-KUN failure that took blind
# adjudication to find in grimgar03.
pairs = sorted({(e["expected_speaker"], e["alias"]) for e in bundle["entries"]
                if e.get("alias") and e["alias"] != e["expected_speaker"]})
existing = [set(group) for group in bundle.get("aliases", [])]
for canonical, other in pairs:
    for group in existing:
        if canonical in group or other in group:
            group.update({canonical, other})
            break
    else:
        existing.append({canonical, other})
bundle["aliases"] = [sorted(group) for group in existing]

with open(BUNDLE, "w", encoding="utf-8") as fh:
    json.dump(bundle, fh, indent=1, ensure_ascii=False)
print(f"{BOOK}: {total} judgements merged into {BUNDLE}")
print(f"  named speaker  {counts['named']:4} ({counts['named']/total*100:.1f}%)")
print(f"  UNKNOWN        {counts['UNKNOWN']:4} ({counts['UNKNOWN']/total*100:.1f}%)"
      "   <- underdetermined for a strong reader")
print(f"  NOT_DIALOGUE   {counts['NOT_DIALOGUE']:4} "
      f"({counts['NOT_DIALOGUE']/total*100:.1f}%)   <- segmenter error rate")
print(f"  not confident  {counts['not_confident']:4} "
      f"({counts['not_confident']/total*100:.1f}%)")

aliases = pairs
if aliases:
    print("\n  aliases the judge flagged - add to the fixture before trusting it,")
    print("  since an undeclared alias scores a correct answer as wrong:")
    for canonical, other in aliases:
        print(f"    {canonical} = {other}")
print("\nNext: finalise_fixture.py. Rows marked NOT_DIALOGUE should be dropped")
print("rather than scored - they are a segmentation measurement, not attribution")
print("gold, and leaving them in would score every model against a line that has")
print("no speaker.")
