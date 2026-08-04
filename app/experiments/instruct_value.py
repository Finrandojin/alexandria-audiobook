"""Is the per-line delivery instruction worth what it costs?

Every chunk carries its own `instruct` string - "Measured, academic narration;
slightly cautionary tone." On the live book that is 2,356 DISTINCT values
across 2,606 chunks, with only 37 reused at all. NATSUKI SUBARU gets 168 lines
and 168 different instructions; it never repeats.

That costs twice. The LLM generates one per line during annotation, and because
consecutive chunks never share an instruct, no two segments can be merged for
generation - measured at 13% fewer TTS calls at a 200-character cap and 35% at
500. It has never been tested against the obvious cheaper alternative.

THREE ARMS, same text, same voice, same seed:

    per_line    the instruct the annotator produced          (production)
    per_char    ONE constant instruction for the character   (the cheap option)
    none        no instruction at all                        (the floor)

WHAT THIS CAN AND CANNOT SETTLE. The output gate measures CONTENT - dropped,
repeated, hallucinated or truncated words. If per-line instructions make the
audio worse in that sense, that is decisive and cheap to see. If word error is
flat across all three, it does NOT prove the instructions are worthless: they
are meant to shape delivery - warmth, pace, emphasis - and WER is blind to
that. A flat result moves the question to a listening test rather than
answering it.

So the honest readings, fixed before running:

  per_line clearly worse   the feature is actively harmful and both its costs
                           are pure loss
  per_line clearly better  it is buying real fidelity, and the throughput cost
                           is justified
  all three flat           WER cannot see the difference. The 13-35% throughput
                           and the per-line LLM tokens are being spent on
                           something only ears can judge, which is worth
                           knowing before spending more.
"""
import argparse, collections, json, os, statistics, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

# One neutral instruction per speaking role. Deliberately generic: the point is
# to test whether per-LINE tailoring beats a per-CHARACTER constant, not to
# hand-tune a better constant.
CONSTANT = {
    "NARRATOR": "Clear, measured narration.",
    "_default": "Natural conversational delivery.",
}


def constant_for(speaker):
    return CONSTANT.get((speaker or "").upper(), CONSTANT["_default"])


def pick_segments(chunks, per_speaker=4, min_chars=60, seed=7):
    """A spread across speakers, so one voice cannot carry the result.

    Seeded, because an unseeded draw would make two runs differ for reasons
    unrelated to the arm.
    """
    import random
    rng = random.Random(seed)
    by_speaker = collections.defaultdict(list)
    for c in chunks:
        if len(c.get("text", "")) >= min_chars and c.get("instruct"):
            by_speaker[c.get("speaker")].append(c)
    picked = []
    for speaker, items in sorted(by_speaker.items(),
                                 key=lambda kv: -len(kv[1])):
        picked.extend(rng.sample(items, min(per_speaker, len(items))))
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--script", default=os.path.join(REPO, "chunks.json"))
    ap.add_argument("--voice-config", default=os.path.join(REPO, "voice_config.json"))
    ap.add_argument("--config", default=os.path.join(APP, "config.json"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "ab_test_runtime", "instruct_audio"))
    ap.add_argument("--per-speaker", type=int, default=3)
    ap.add_argument("--speakers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234,
                    help="fixed generation seed. WITHOUT THIS THE COMPARISON "
                         "IS UNCONTROLLED: generate_lora_voice ignored the "
                         "seed field entirely until it was fixed, so the same "
                         "input produced an 18%% swing in clip length and the "
                         "arms differed for reasons unrelated to the "
                         "instruction. Pass -1 to reproduce the old behaviour.")
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "instruct_value.json"))
    args = ap.parse_args()

    chunks = [c for c in json.load(open(args.script, encoding="utf-8"))
              if c.get("text") and c.get("uid")]
    picked = pick_segments(chunks, per_speaker=args.per_speaker)
    # Keep the most-spoken few characters so the sample is what a listener
    # actually hears most of.
    keep = [s for s, _ in collections.Counter(
        c["speaker"] for c in picked).most_common(args.speakers)]
    picked = [c for c in picked if c["speaker"] in keep]
    print(f"{len(picked)} segments across {len(keep)} speakers: "
          f"{', '.join(str(k) for k in keep)}\n")

    raw_vc = json.load(open(args.voice_config, encoding="utf-8"))
    voice_config = (raw_vc.get("characters")
                    if isinstance(raw_vc.get("characters"), dict) else raw_vc)
    os.makedirs(args.out_dir, exist_ok=True)

    from tts import TTSEngine, voice_category
    from experiments.tts_output_validation import transcribe, validate
    from experiments.generation import render
    engine = TTSEngine(json.load(open(args.config, encoding="utf-8")))

    rows = []
    for i, chunk in enumerate(picked, 1):
        speaker = chunk.get("speaker")
        voice_data = voice_config.get(speaker) or {}
        category = voice_category(voice_data)
        arms = {"per_line": chunk.get("instruct") or "",
                "per_char": constant_for(speaker),
                "none": ""}
        # Same seed in every arm, so the only difference is the instruction.
        voice_data = dict(voice_data)
        voice_data["seed"] = str(args.seed)
        row = {"uid": chunk["uid"], "speaker": speaker,
               "chars": len(chunk["text"])}
        for arm, instruct in arms.items():
            wav = os.path.join(args.out_dir, f"{arm}_{chunk['uid']}.wav")
            try:
                render(engine, chunk["text"], instruct, speaker, voice_config,
                       voice_data, wav)
                r = validate(chunk["text"], transcribe(wav))
            except Exception as exc:                    # noqa: BLE001
                print(f"  [{i}] {arm} FAILED: {str(exc)[:70]}")
                continue
            row[arm] = {"errors": r["errors"], "words": r["words"],
                        "failed": r["failed"], "non_speech": r["non_speech"]}
        rows.append(row)
        got = " | ".join(f"{a} {row[a]['errors']:3}" for a in arms if a in row)
        print(f"  [{i}/{len(picked)}] {speaker[:14]:14} {row['chars']:4}ch  {got}")

    print()
    summary = {}
    for arm in ("per_line", "per_char", "none"):
        sel = [r[arm] for r in rows if arm in r]
        if not sel:
            continue
        wer = sum(x["errors"] for x in sel) / max(sum(x["words"] for x in sel), 1)
        summary[arm] = {"n": len(sel), "wer": wer,
                        "failed": sum(x["failed"] for x in sel),
                        "non_speech": sum(x["non_speech"] for x in sel)}
        print(f"  {arm:9} n={len(sel):3}  WER {wer*100:6.2f}%  "
              f"failed {summary[arm]['failed']:2}  "
              f"non-speech {summary[arm]['non_speech']:2}")

    if len(summary) == 3:
        spread = max(s["wer"] for s in summary.values()) - \
                 min(s["wer"] for s in summary.values())
        print(f"\n  spread between best and worst arm: {spread*100:.2f} points")
        if spread < 0.02:
            print("  FLAT. Word error cannot separate these arms, so the per-line\n"
                  "  instruction is not buying CONTENT fidelity. It may still be\n"
                  "  buying delivery, which this cannot see - that needs ears.\n"
                  "  What is certain is the cost: 13-35% of TTS throughput and one\n"
                  "  LLM generation per line.")
        else:
            best = min(summary, key=lambda a: summary[a]["wer"])
            print(f"  NOT flat - {best} has the lowest word error. Worth a paired\n"
                  "  test on more segments before acting on it.")

    json.dump({"summary": summary, "rows": rows,
               "caveat": "WER measures content, not delivery. A flat result "
                         "moves the question to a listening test rather than "
                         "settling it."},
              open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
