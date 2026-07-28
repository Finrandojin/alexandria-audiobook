"""Convert a filled labelling bundle into a gold fixture the harnesses can read.

Separate from `make_fixture` so that sampling and conversion cannot share a
random seed by accident, and so re-running the sampler can never overwrite
labels that have already been done by hand.

Refuses to write a fixture with blank labels: a half-filled bundle silently
converted into gold would produce a fixture whose missing rows read as errors
in every run scored against it.
"""
import collections
import json, os, re, sys

REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"
BOOK = os.environ.get("EXPERIMENT_BOOK", "owarimonogatari3")
JUDGE = os.environ.get("EXPERIMENT_JUDGE", "frontier-model (single judge, provisional)")
BUNDLE = os.environ.get("EXPERIMENT_BUNDLE", os.path.join(
    REPO, "ab_test_runtime", "fixtures_draft", f"labelling_bundle__{BOOK}.json"))
OUT = os.environ.get("EXPERIMENT_OUT", os.path.join(
    REPO, "app", "fixtures", f"attribution_gold_{BOOK}_provisional.json"))

bundle = json.load(open(BUNDLE, encoding="utf-8"))
blank = [e["id"] for e in bundle["entries"] if not (e.get("expected_speaker") or "").strip()]
if blank:
    raise SystemExit(f"{len(blank)} of {len(bundle['entries'])} entries are "
                     f"unlabelled, first is {blank[0]}. Fill them or drop them "
                     f"from the bundle; a fixture with blanks scores every run "
                     f"against nothing.")

# NOT_DIALOGUE rows are a segmentation measurement, not attribution gold:
# scoring a model against a line that has no speaker punishes it for the
# segmenter's error. They are dropped here and their rate reported.
segmenter_errors = [e for e in bundle["entries"]
                    if (e.get("expected_speaker") or "").upper() == "NOT_DIALOGUE"]
entries = []
for n, e in enumerate([x for x in bundle["entries"] if x not in segmenter_errors], 1):
    entries.append({"id": e["id"], "book": BOOK, "entry_index": n,
                    "line": e["line"],
                    "expected_speaker": e["expected_speaker"].strip().upper(),
                    "judged_by": JUDGE,
                    "reasoning": (e.get("note") or "").strip() or None})

counts = collections.Counter(x["expected_speaker"] for x in entries)
fixture = {
    "description": (f"Hand-labelled attribution gold for {BOOK}. Sampled "
                    f"uniformly from spoken, non-deterministic, textually "
                    f"unique segments; labelled without seeing any model "
                    f"output, so it can be used to measure model accuracy."),
    "book": BOOK,
    "source_run": bundle.get("book") and
                  "matrix_20260725-115148 / qwen3.5-9b-uncensored-hauhaucs-aggressive",
    "sampling": {"seed": bundle.get("seed"), "count": len(entries),
                 "context_segments": bundle.get("context_segments")},
    "entries": entries,
    # Declared by hand after labelling: two names for one character make a
    # correct answer score as wrong. grimgar03 needed BRI-CHAN/BRITNEY and
    # ZODIAC/ZODIAC-KUN, and the second was only found by adjudication.
    "aliases": bundle.get("aliases", []),
    "provisional": True,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(fixture, fh, indent=1, ensure_ascii=False)
if segmenter_errors:
    print(f"dropped {len(segmenter_errors)}/{len(bundle['entries'])} rows marked "
          f"NOT_DIALOGUE = {len(segmenter_errors)/len(bundle['entries'])*100:.1f}% "
          f"segmenter error rate on spoken segments")
print(f"wrote {OUT}: {len(entries)} entries, {len(counts)} distinct speakers")
print(f"  most common: {counts.most_common(5)}")
# UNKNOWN and UNNAMED are both dropped from scoring but mean different things,
# and reporting them together hid that on index18: 9 rows read as 7.5%
# "underdetermined" when 8 of them were determinate lines spoken by characters
# with no name, and one was a named character the roster had simply omitted.
# Ambiguity bounds what any model could achieve; missing names are a roster
# problem with a different fix.
unknown = counts.get("UNKNOWN", 0) + counts.get("AMBIGUOUS", 0)
unnamed = counts.get("UNNAMED", 0)
if unknown:
    print(f"  {unknown} UNKNOWN - genuinely ambiguous, an upper bound on any "
          f"model's accuracy")
if unnamed:
    print(f"  {unnamed} UNNAMED - determinate but the speaker has no name; a "
          f"roster-coverage limit, not ambiguity")
if unknown or unnamed:
    print(f"  scored set will be {len(entries)-unknown-unnamed}")
print("\n  Before trusting any number from this fixture, check the aliases "
      "list.\n  Undeclared aliases show up as unanimous model failures.")
