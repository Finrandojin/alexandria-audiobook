"""Does splitting front matter into its items fix it? Measure before building.

`prose_vs_nonprose` established the problem: at matched length, with symbols
already normalised away, non-prose fails 21 of 25 (WER 56.65%) while prose
fails 0 of 25 (WER 0.52%). What it did NOT establish is the remedy. The obvious
guess - send the items separately instead of as one blob - is a guess, and the
generation path should not be changed on a guess.

So this generates each failing segment BOTH ways, on the same text with the
same voice, and compares:

    whole    exactly as production does today, one call for the segment
    split    one call per item, concatenated

The comparison is per segment and paired, so a segment that is simply hard
cannot make one arm look better by being sampled into it.

WHY IT MIGHT NOT WORK, stated now rather than after a null result. The
mechanism behind the 84% failure is unknown - fragment length, missing clauses,
digit runs and list layout all move together in real front matter and were
never varied independently. Splitting addresses only the "too many unrelated
fragments in one call" story. If that is not the mechanism, split will fail too,
and the right response is to find the mechanism rather than to try another
guess.

A NULL RESULT IS USEFUL and must not be buried: it would rule out the cheapest
remedy and say the problem is per-item, not per-blob.
"""
import argparse, json, os, statistics, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

MIN_ITEM = 3        # characters; below this an "item" is punctuation debris


def split_items(text):
    """Front matter into the units a narrator would pause between.

    normalize_for_speech has already turned bullets and rules into full stops,
    so sentence-enders are the item boundaries by the time this runs.
    """
    import re
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text or "")]
    return [p for p in parts if len(p.strip(" .")) >= MIN_ITEM]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "prose_vs_nonprose.json"))
    ap.add_argument("--voice-config", default=os.path.join(REPO, "voice_config.json"))
    ap.add_argument("--config", default=os.path.join(APP, "config.json"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "ab_test_runtime", "split_audio"))
    ap.add_argument("--voice", default="NARRATOR")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234,
                    help="fixed generation seed; the first run was unseeded so "
                         "whole and split arms differed by random draw")
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "nonprose_split.json"))
    args = ap.parse_args()

    doc = json.load(open(args.source, encoding="utf-8"))
    failing = [r for r in doc["rows"]
               if r["class"] == "nonprose" and r["failed"]][:args.limit]
    if not failing:
        print("no failing non-prose rows in", args.source)
        return

    # The source artifact stores no text, only the wav path and uid, so the
    # texts come back from the library the same way the experiment drew them.
    import argparse as _a
    from experiments.prose_vs_nonprose import load_chunks
    pool = {c["uid"]: c for c in load_chunks(
        _a.Namespace(pool_library=True, voice=args.voice, script=""))}
    targets = [pool[r["uid"]] for r in failing if r["uid"] in pool]
    print(f"{len(targets)} failing non-prose segments recovered\n")

    raw_vc = json.load(open(args.voice_config, encoding="utf-8"))
    voice_config = (raw_vc.get("characters")
                    if isinstance(raw_vc.get("characters"), dict) else raw_vc)
    voice_data = dict(voice_config.get(args.voice) or {})
    voice_data["seed"] = str(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    import numpy as np, soundfile as sf
    from tts import TTSEngine, normalize_for_speech
    from experiments.generation import render
    from experiments.tts_output_validation import transcribe, validate
    engine = TTSEngine(json.load(open(args.config, encoding="utf-8")))

    rows, skipped = [], []
    for i, chunk in enumerate(targets, 1):
        text = chunk["text"]
        # This comparison is PAIRED, so a failure in either arm must drop the
        # whole segment rather than leave one arm scored against nothing.
        # render() raises, so one bad segment would otherwise kill the run -
        # and silently keeping the surviving arm is the asymmetry that review
        # finding 5 was about.
        try:
            whole_wav = os.path.join(args.out_dir, f"whole_{chunk['uid']}.wav")
            render(engine, text, chunk.get("instruct", ""), args.voice,
                   voice_config, voice_data, whole_wav)
            whole = validate(text, transcribe(whole_wav))

            items = split_items(normalize_for_speech(text))
            pieces, rate = [], None
            for j, item in enumerate(items):
                p = os.path.join(args.out_dir, f"split_{chunk['uid']}_{j}.wav")
                render(engine, item, chunk.get("instruct", ""), args.voice,
                       voice_config, voice_data, p)
                audio, rate = sf.read(p)
                pieces.append(audio)
        except Exception as exc:                        # noqa: BLE001
            skipped.append({"uid": chunk["uid"], "error": str(exc)[:160]})
            print(f"  [{i}/{len(targets)}] SKIPPED (both arms): {str(exc)[:70]}")
            continue
        if not pieces:
            skipped.append({"uid": chunk["uid"], "error": "no split pieces"})
            continue
        gap = np.zeros(int((rate or 24000) * 0.25))
        joined = np.concatenate(
            [x for pair in zip(pieces, [gap] * len(pieces)) for x in pair])
        split_wav = os.path.join(args.out_dir, f"joined_{chunk['uid']}.wav")
        sf.write(split_wav, joined, rate or 24000)
        split = validate(text, transcribe(split_wav))

        rows.append({"uid": chunk["uid"], "chars": len(text),
                     "items": len(items),
                     "whole_errors": whole["errors"], "whole_failed": whole["failed"],
                     "whole_wer": whole["errors"] / max(whole["words"], 1),
                     "split_errors": split["errors"], "split_failed": split["failed"],
                     "split_wer": split["errors"] / max(split["words"], 1)})
        print(f"  [{i}/{len(targets)}] {len(text):4}ch -> {len(items):2} items | "
              f"whole {whole['errors']:4} err {'FAIL' if whole['failed'] else 'ok  '} | "
              f"split {split['errors']:4} err {'FAIL' if split['failed'] else 'ok  '}")

    if not rows:
        print("nothing generated")
        return
    wf = sum(r["whole_failed"] for r in rows)
    sf_ = sum(r["split_failed"] for r in rows)
    print(f"\n  whole  {wf}/{len(rows)} failed  "
          f"mean WER {statistics.mean(r['whole_wer'] for r in rows)*100:6.2f}%")
    print(f"  split  {sf_}/{len(rows)} failed  "
          f"mean WER {statistics.mean(r['split_wer'] for r in rows)*100:6.2f}%")
    fixed = sum(1 for r in rows if r["whole_failed"] and not r["split_failed"])
    broke = sum(1 for r in rows if r["split_failed"] and not r["whole_failed"])
    print(f"\n  split fixes {fixed}, breaks {broke}, of {len(rows)} segments")
    # "fixed one, broke none" is NOT sufficient to call something a remedy, and
    # saying so was the first draft's mistake: 7 of 8 still failed. A remedy
    # has to clear most of the failures, not shave the error count.
    if sf_ == 0:
        print("  Splitting clears every failure. It is the remedy.")
    elif sf_ <= len(rows) * 0.4:
        print("  Splitting clears most failures. Worth building, with the "
              "residual\n  measured separately.")
    elif fixed >= broke and statistics.mean(r["split_wer"] for r in rows) < \
            statistics.mean(r["whole_wer"] for r in rows) * 0.75:
        print("  PARTIAL MITIGATION ONLY. Error rate drops materially but most\n"
              "  segments still fail, so the failure is also per ITEM, not just\n"
              "  per blob. Splitting alone does not make this text safe to ship.")
    else:
        print("  Splitting is NOT the remedy - the failure is per ITEM, not per\n"
              "  blob. Do not change the generation path; find the mechanism.")
    if skipped:
        print(f"\n  {len(skipped)} segments dropped from BOTH arms")
    json.dump({"rows": rows, "whole_failed": wf, "split_failed": sf_,
               "skipped": skipped},
              open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
