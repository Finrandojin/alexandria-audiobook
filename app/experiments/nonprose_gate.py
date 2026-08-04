"""If non-prose fails 40% of the time, can we simply detect it?

`prose_vs_nonprose` measured, on a corrected sample, that non-prose segments
fail the output gate about 40% of the time while length-matched prose fails 0%
(WER 47.28% vs 0.67%). The remedy is not better TTS. Real audiobooks do not
narrate an ISBN, and splitting front matter into items was measured to be only
a partial mitigation - 7 of 8 segments still failed.

So the useful question is whether the failing text can be RECOGNISED before
generation, cheaply and without a model. `nonprose_score` already exists and
already separated the two classes well enough to build that experiment.

WHAT DECIDES WHETHER THIS IS USABLE is not accuracy in the abstract but the
shape of the errors, because the two mistakes cost completely different things:

    false positive   a real narration line flagged as non-prose. If the gate
                     skips it, the listener loses a sentence of the book. This
                     is the expensive error and must be near zero.
    false negative   an ISBN line that slips through and gets narrated. The
                     listener hears garbage for a few seconds. Annoying, not
                     destructive - and it is the status quo, so a gate that
                     merely reduces these is already a gain.

A gate flagging ~1% of a book and catching the copyright page is worth having.
One flagging 15% of narration is worse than nothing, however good its headline
number looks, and the per-book share below is what says which it is.

NOT A CLASSIFIER EVALUATION. There is no hand-labelled non-prose set; the only
labels are `prose_vs_nonprose`'s own thresholds, so scoring the gate against
them would be circular. This reports COVERAGE - what fraction of real books it
would flag, and which lines - so the false-positive risk can be inspected by
reading them rather than inferred from a number.
"""
import argparse, collections, glob, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

from experiments.prose_vs_nonprose import nonprose_score, is_machine_output


def survey(texts, threshold):
    flagged = [t for t in texts if nonprose_score(t) >= threshold]
    return flagged


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--script", default=os.path.join(REPO, "chunks.json"))
    ap.add_argument("--library", action="store_true",
                    help="survey every saved book instead of the live one")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.35, 0.45, 0.55])
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "nonprose_gate.json"))
    args = ap.parse_args()

    books = {}
    if args.library:
        for path in sorted(glob.glob(os.path.join(REPO, "scripts", "*.json"))):
            if "voice_config" in path or "generation_quality" in path:
                continue
            try:
                doc = json.load(open(path, encoding="utf-8"))
            except Exception:
                continue
            entries = doc if isinstance(doc, list) else (
                doc.get("entries") or doc.get("chunks") or [])
            texts = [e["text"] for e in entries
                     if isinstance(e, dict) and e.get("text")
                     and not is_machine_output(e["text"])]
            if texts:
                books[os.path.basename(path)[:-5][:30]] = texts
    else:
        doc = json.load(open(args.script, encoding="utf-8"))
        entries = doc if isinstance(doc, list) else (doc.get("entries") or [])
        books["live book"] = [e["text"] for e in entries
                              if isinstance(e, dict) and e.get("text")]

    if not books:
        print("no books found")
        return

    print(f"{len(books)} book(s), "
          f"{sum(len(v) for v in books.values())} lines\n")
    print(f"  {'threshold':>10}{'flagged':>10}{'share':>9}"
          f"{'worst book':>34}{'that book':>11}")
    results = {}
    for th in args.thresholds:
        total = flagged = 0
        per_book = {}
        for name, texts in books.items():
            f = len(survey(texts, th))
            per_book[name] = {"n": len(texts), "flagged": f,
                              "share": f / max(len(texts), 1)}
            total += len(texts)
            flagged += f
        worst = max(per_book.items(), key=lambda kv: kv[1]["share"])
        results[str(th)] = {"flagged": flagged, "total": total,
                            "share": flagged / max(total, 1),
                            "per_book": per_book}
        print(f"  {th:10.2f}{flagged:10}{flagged/max(total,1)*100:8.2f}%"
              f"{worst[0][:32]:>34}{worst[1]['share']*100:10.1f}%")

    # The whole decision rests on what a flag actually lands on, so print them.
    th = args.thresholds[len(args.thresholds) // 2]
    sample = []
    for name, texts in books.items():
        for t in texts:
            if nonprose_score(t) >= th:
                sample.append((name, t))
    print(f"\n  what threshold {th} flags - read these for false positives")
    for name, t in sample[:14]:
        print(f"    [{name[:18]:18}] {t[:74]!r}")
    if len(sample) > 14:
        print(f"    ... and {len(sample) - 14} more")

    json.dump({"thresholds": results,
               "sampled_flags": [t for _, t in sample[:60]],
               "caveat": "coverage, not accuracy - there is no hand-labelled "
                         "non-prose set, so the flags are for reading rather "
                         "than scoring"},
              open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
