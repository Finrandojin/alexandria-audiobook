"""What is it about a non-prose line that breaks generation?

The question is now narrow enough to be worth asking, because two cheaper
answers have been eliminated on seeded audio:

    symbols       ruled out. Both classes are generated with the symbol
                  normaliser active and non-prose still loses 28.5 WER points.
    blob length   ruled out. `nonprose_split_v2` split eight failing segments
                  into their items and got 8/8 failing either way, 59.46% WER
                  whole against 62.58% split. Two of those segments contained
                  ONE item and could not be split at all, and failed anyway.

A 64-character single item that fails is not a blob problem and not a symbol
problem. Something about the item's own text is doing it, and up to now
fragment length, digit runs, capitalisation and the absence of a verb have all
moved together in real front matter, so no one of them has ever been tested.

THIS SEPARATES THEM BY ABLATION. Each failing item is rewritten four ways, one
property removed at a time, and each variant is generated at the same seed with
the same voice:

    digits        "9780316315302" -> spelled out as words
    caps          "LCCN" -> "Lccn", so capitalisation stops signalling acronym
    punctuation   bracket and slash runs reduced to commas
    syntax        the fragment wrapped into a sentence with a verb

WER is measured against the SPOKEN form of each variant, not against the
original, because a variant that legitimately says something different must not
be scored as a mistake. The comparison is which ablation moves the failure
rate, not whether the text changed.

WHAT A RESULT LOOKS LIKE. If one ablation clears most failures, that property
is the mechanism and the gate can target it precisely. If none does, the
failure is joint - the model is out of distribution on this register as a
whole - and the honest conclusion is to route non-prose away from TTS rather
than to keep trying to repair it. THAT IS A LEGITIMATE OUTCOME and the reason
the non-prose gate already exists; it would make the gate the answer rather
than a stopgap.
"""
import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

_DIGITS = re.compile(r"\d+")
_PUNCT_RUN = re.compile(r"[\[\](){}/|;:]+")


def ablate_digits(text):
    """Digit runs spoken as words, so no arithmetic is left for the model."""
    from experiments.tts_output_validation import say_number
    # say_number returns a LIST of words, not a string.
    return _DIGITS.sub(
        lambda m: " " + " ".join(say_number(m.group(0))) + " ", text)


def ablate_caps(text):
    """Acronyms lose their all-caps shape but keep their letters."""
    return re.sub(r"\b[A-Z]{2,}\b", lambda m: m.group(0).capitalize(), text)


def ablate_punct(text):
    """Bracket and slash runs become ordinary commas."""
    return re.sub(r"\s*,\s*,+", ", ", _PUNCT_RUN.sub(", ", text)).strip(" ,")


def ablate_syntax(text):
    """A bare fragment becomes a sentence with a subject and a verb."""
    body = text.strip().rstrip(".")
    return f"The record states that {body}."


ABLATIONS = {"none": lambda t: t, "digits": ablate_digits,
             "caps": ablate_caps, "punct": ablate_punct,
             "syntax": ablate_syntax}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "prose_vs_nonprose_v3.json"))
    ap.add_argument("--voice-config", default=os.path.join(REPO, "voice_config.json"))
    ap.add_argument("--config", default=os.path.join(APP, "config.json"))
    ap.add_argument("--voice", default="NARRATOR")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "ab_test_runtime", "mechanism_audio"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "nonprose_mechanism.json"))
    args = ap.parse_args()

    doc = json.load(open(args.source, encoding="utf-8"))
    failing = [r for r in doc["rows"]
               if r["class"] == "nonprose" and r["failed"]][:args.limit]
    if not failing:
        sys.exit(f"no failing non-prose rows in {args.source}")

    import argparse as _a
    from experiments.prose_vs_nonprose import load_chunks
    pool = {c["uid"]: c for c in load_chunks(
        _a.Namespace(pool_library=True, voice=args.voice, script=""))}
    targets = [pool[r["uid"]] for r in failing if r["uid"] in pool]
    print(f"{len(targets)} failing non-prose items, "
          f"{len(ABLATIONS)} variants each\n")

    raw_vc = json.load(open(args.voice_config, encoding="utf-8"))
    vc = (raw_vc.get("characters")
          if isinstance(raw_vc.get("characters"), dict) else raw_vc)
    voice = dict(vc.get(args.voice) or {})
    voice["seed"] = str(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    from tts import TTSEngine
    from experiments.generation import render, GenerationFailed
    from experiments.tts_output_validation import transcribe, validate
    engine = TTSEngine(json.load(open(args.config, encoding="utf-8")))

    rows, skipped = [], []
    for i, chunk in enumerate(targets, 1):
        # Every variant of a segment must render, or the segment is dropped
        # entirely. A partial row would compare ablations against different
        # subsets, which is the asymmetry review finding 5 was about.
        variants, failed_here = {}, None
        for label, fn in ABLATIONS.items():
            try:
                text = fn(chunk["text"])
                wav = os.path.join(args.out_dir, f"{chunk['uid']}_{label}.wav")
                render(engine, text, "", args.voice, vc, voice, wav)
                # Scored against the variant's OWN text: an ablation changes
                # what should be said, and must not be charged for that.
                variants[label] = validate(text, transcribe(wav))
            except (GenerationFailed, Exception) as exc:   # noqa: BLE001
                failed_here = f"{label}: {str(exc)[:100]}"
                break
        if failed_here:
            skipped.append({"uid": chunk["uid"], "error": failed_here})
            print(f"  [{i}/{len(targets)}] DROPPED (all variants): {failed_here[:60]}")
            continue
        row = {"uid": chunk["uid"], "chars": len(chunk["text"])}
        for label, v in variants.items():
            row[label] = {"errors": v["errors"], "failed": v["failed"],
                          "wer": v["errors"] / max(v["words"], 1)}
        rows.append(row)
        print(f"  [{i}/{len(targets)}] {len(chunk['text']):4}ch  " +
              "  ".join(f"{k}:{'F' if row[k]['failed'] else '.'}"
                        f"{row[k]['wer']*100:5.1f}%" for k in ABLATIONS))

    if not rows:
        sys.exit("nothing generated")

    import statistics
    print()
    base_failed = sum(r["none"]["failed"] for r in rows)
    summary = {}
    for label in ABLATIONS:
        failed = sum(r[label]["failed"] for r in rows)
        wer = statistics.mean(r[label]["wer"] for r in rows)
        summary[label] = {"failed": failed, "n": len(rows), "wer": wer}
        print(f"  {label:8} {failed}/{len(rows)} failed   mean WER {wer*100:6.2f}%")

    # The decision. A mechanism has to clear most failures, not shave WER -
    # calling a one-segment improvement a remedy was the split test's first
    # mistake and is not repeated here.
    ranked = sorted(((v["failed"], k) for k, v in summary.items() if k != "none"))
    best_failed, best = ranked[0]
    print()
    if base_failed and best_failed <= max(1, base_failed * 0.4):
        verdict = (f"'{best}' clears most failures ({base_failed} -> "
                   f"{best_failed} of {len(rows)}). That property is the "
                   f"mechanism and the gate can target it directly.")
    elif base_failed and best_failed < base_failed:
        verdict = (f"'{best}' helps but does not fix ({base_failed} -> "
                   f"{best_failed} of {len(rows)}). The failure is JOINT "
                   f"across properties, so no single repair makes this text "
                   f"safe; the non-prose gate stays the answer.")
    else:
        verdict = (f"No ablation reduces failures ({base_failed}/{len(rows)} "
                   f"baseline). The model is out of distribution on this "
                   f"register as a whole. Route non-prose away from TTS "
                   f"rather than repairing it - the gate is the answer, not "
                   f"a stopgap.")
    print(f"  {verdict}")

    json.dump({"seed": args.seed, "summary": summary, "rows": rows,
               "skipped": skipped, "verdict": verdict},
              open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
