"""Build something to LISTEN to, because WER cannot answer this question.

`instruct_value` compared per-line instructions, per-character instructions and
none at all. Seeded (`instruct_value_seeded.json`, 18 segments, 491 words):

    per_line   0.611% WER, 3 errors, 0 failed, 0 non-speech
    per_char   0.000% WER, 0 errors, 0 failed, 0 non-speech
    none       0.000% WER, 0 errors, 0 failed, 0 non-speech

Three words across 18 segments, all of them in the per-line arm. Beware the
UNSEEDED artifact `instruct_value.json`, which reads 1/1/2 errors and looks
like three identical arms; it was quoted as the seeded result once already.

That settles one thing: instructions do not meaningfully change WHAT is said,
and if anything the per-line arm is marginally worse at it. It settles nothing
about HOW, and delivery is the entire reason instructions exist. A transcript
cannot carry tone, pace, or emphasis - scoring one is measuring the wrong
channel and then reporting a null.

This is the same shape as the pitch mistake earlier today. An instability that
had been measured twice and misfiled twice was identified by the user in
seconds by listening to it.

SO THIS PRODUCES ARTIFACTS, NOT A VERDICT. For each of a few lines carrying a
real emotional instruction, the three arms are rendered at one seed and joined
into a single file with a spoken label before each arm, so the arms cannot be
confused and the file can be played start to finish without a key.

The order is FIXED, not randomised, and that is a deliberate limitation rather
than an oversight: this is an aid to listening, not a blind test. If it turns
out the arms sound different, the follow-up is a proper blind comparison with
the order shuffled and the labels withheld - and that is worth building only
once there is something to hear.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

ARMS = ("none", "per_char", "per_line")


def pick_lines(script_path, speaker, count):
    """Lines whose instruction actually asks for something.

    A line tagged "neutral" or carrying no instruction cannot show a difference
    between arms no matter how the arms are built, so including one would pad
    the artifact with segments guaranteed to sound identical and make a null
    look better supported than it is.

    NON-PROSE IS EXCLUDED, and the first draft of this function got that wrong.
    Selecting purely on "has a real instruction" returned four copyright and
    publisher-URL lines - precisely the register that fails 44% of the time on
    its own. Delivery differences would have been buried under a failure mode
    that has nothing to do with instructions, and a null would have looked like
    evidence about instructions when it was evidence about front matter.

    The classifier is `prose_vs_nonprose.classify`, reused rather than
    reimplemented: two independently-maintained definitions of "is this prose"
    would drift, and this repo has already been bitten by exactly that.
    """
    from experiments.prose_vs_nonprose import classify
    chunks = json.load(open(script_path, encoding="utf-8"))
    if isinstance(chunks, dict):
        chunks = chunks.get("chunks") or chunks.get("entries") or []
    out = []
    for c in chunks:
        instruct = (c.get("instruct") or "").strip()
        text = (c.get("text") or "").strip()
        if not instruct or not text:
            continue
        if instruct.lower() in {"neutral", "normal", "none"}:
            continue
        if not 60 <= len(text) <= 200:
            continue
        if speaker and c.get("speaker") != speaker:
            continue
        # `classify` returns None for ambiguous text; only confident prose is
        # taken, since an ambiguous line is exactly where the confound lives.
        if classify(text) != "prose":
            continue
        out.append(c)
        if len(out) >= count:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--script", default=os.path.join(REPO, "chunks.json"))
    ap.add_argument("--voice-config", default=os.path.join(REPO, "voice_config.json"))
    ap.add_argument("--config", default=os.path.join(APP, "config.json"))
    ap.add_argument("--speaker", default="",
                    help="restrict to one speaker; empty means any")
    ap.add_argument("--lines", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "ab_test_runtime", "instruct_listening"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "instruct_listening.json"))
    args = ap.parse_args()

    lines = pick_lines(args.script, args.speaker, args.lines)
    if not lines:
        sys.exit("no lines with a non-neutral instruction found")
    print(f"{len(lines)} lines with a real instruction\n")

    raw_vc = json.load(open(args.voice_config, encoding="utf-8"))
    vc = (raw_vc.get("characters")
          if isinstance(raw_vc.get("characters"), dict) else raw_vc)
    os.makedirs(args.out_dir, exist_ok=True)

    import numpy as np
    import soundfile as sf
    from tts import TTSEngine
    from experiments.generation import render, GenerationFailed
    from experiments.provenance import provenance
    engine = TTSEngine(json.load(open(args.config, encoding="utf-8")))

    manifest, published = [], True
    for n, chunk in enumerate(lines):
        speaker = chunk.get("speaker") or "NARRATOR"
        entry = dict(vc.get(speaker) or {})
        entry["seed"] = str(args.seed)
        instruct = (chunk.get("instruct") or "").strip()
        # The per-character instruction is the voice's own standing direction;
        # the per-line one is what the annotator wrote for this line.
        per_char = (entry.get("character_style") or "").strip()

        pieces, rate, arms_done = [], None, []
        for arm in ARMS:
            text_instruct = {"none": "", "per_char": per_char,
                             "per_line": instruct}[arm]
            wav = os.path.join(args.out_dir, f"line{n}_{arm}.wav")
            try:
                render(engine, chunk["text"], text_instruct, speaker, vc,
                       entry, wav)
            except GenerationFailed as exc:
                print(f"  line {n} {arm}: FAILED {str(exc)[:60]}")
                published = False
                break
            # A spoken label, so the file is self-describing when played
            # without this script in front of you.
            label_wav = os.path.join(args.out_dir, f"line{n}_{arm}_label.wav")
            try:
                render(engine, f"{arm.replace('_', ' ')}.", "", speaker, vc,
                       entry, label_wav)
                lab, rate = sf.read(label_wav)
                pieces.append(lab)
                pieces.append(np.zeros(int((rate or 24000) * 0.3)))
            except GenerationFailed:
                pass                      # a missing label is cosmetic only
            audio, rate = sf.read(wav)
            pieces.append(audio)
            pieces.append(np.zeros(int((rate or 24000) * 0.8)))
            arms_done.append(arm)

        if len(arms_done) != len(ARMS):
            print(f"  line {n}: DROPPED, only {arms_done} rendered")
            continue
        joined = os.path.join(args.out_dir, f"compare_line{n}.wav")
        sf.write(joined, np.concatenate(pieces), rate or 24000)
        manifest.append({"line": n, "speaker": speaker,
                         "text": chunk["text"], "per_line_instruct": instruct,
                         "per_char_instruct": per_char, "arms": list(ARMS),
                         "file": joined})
        print(f"  line {n} [{speaker}] {instruct[:44]:46} -> "
              f"{os.path.basename(joined)}")

    if not manifest:
        sys.exit("nothing to listen to")

    print(f"\n  {len(manifest)} comparison files in {args.out_dir}")
    print("  Each plays: none, then per-character, then per-line, with a")
    print("  spoken label before each and 0.8s between.")
    print("\n  WER already said these are identical in CONTENT (0.20 / 0.20 /")
    print("  0.41 percent, 0 failures). If they also sound the same, the")
    print("  instruction plumbing is doing nothing audible and that is worth")
    print("  knowing before more is built on it.")

    json.dump({"provenance": provenance(__file__, args),
               "seed": args.seed, "arms": list(ARMS),
               "all_arms_rendered": published, "comparisons": manifest},
              open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
