"""Is a voice's pitch stable once generation is seeded?

This exists to re-open a question that was closed on contaminated evidence.

`scene_aware_casting` proposed casting on `mean_f0`, and that was WITHDRAWN
after measuring the same NARRATOR adapter producing clips at 97, 120, 108, 123,
154, 126, 125, 159, 125 and 200 Hz. A ~100 Hz within-voice spread swamps the
17 Hz between-voice separation the allocator was optimising, so pitch looked
useless as a contrast metric.

But that measurement predates the discovery that `generate_lora_voice` never
read the seed - every clip was an independent draw of the voice, which the user
identified by ear as "multiple narrators". The spread may have been the bug
rather than the model.

    if pitch is stable when seeded    the withdrawal was premature, mean_f0 is
                                      a usable casting metric, and the
                                      scene-aware allocator comes back
    if it still varies widely         pitch genuinely cannot separate voices in
                                      this stack, the withdrawal stands, and
                                      contrast has to come from timbre features
                                      or from listening

TWO THINGS ARE MEASURED SEPARATELY, because they answer different questions:

    same text, same seed       pure reproducibility. Should be ~0 Hz spread.
    different text, same seed  what a listener actually meets - the voice
                               reading a book, not one line repeated. Some
                               spread here is natural prosody, not instability.

The second is the one that matters for casting. A voice whose pitch moves 100
Hz across sentences cannot be separated from another voice by 17 Hz however
reproducible each individual clip is.
"""
import argparse, json, os, statistics, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)


def median_pitch(path):
    """Median f0 over voiced frames, or None if there is too little voicing."""
    import numpy as np, librosa
    y, sr = librosa.load(path, sr=16000)
    f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=350, sr=sr)
    vals = f0[voiced & np.isfinite(f0)]
    return float(np.median(vals)) if len(vals) > 20 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--script", default=os.path.join(REPO, "chunks.json"))
    ap.add_argument("--voice-config", default=os.path.join(REPO, "voice_config.json"))
    ap.add_argument("--config", default=os.path.join(APP, "config.json"))
    ap.add_argument("--speaker", default="NARRATOR")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--lines", type=int, default=8)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "ab_test_runtime", "pitch_stability"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "pitch_stability.json"))
    args = ap.parse_args()

    chunks = [c for c in json.load(open(args.script, encoding="utf-8"))
              if c.get("text") and c.get("speaker") == args.speaker
              and 80 <= len(c["text"]) <= 220]
    if len(chunks) < args.lines:
        sys.exit(f"only {len(chunks)} usable lines for {args.speaker}")
    sample = chunks[:args.lines]

    raw_vc = json.load(open(args.voice_config, encoding="utf-8"))
    vc = (raw_vc.get("characters")
          if isinstance(raw_vc.get("characters"), dict) else raw_vc)
    voice = dict(vc.get(args.speaker) or {})
    voice["seed"] = str(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    from tts import TTSEngine
    engine = TTSEngine(json.load(open(args.config, encoding="utf-8")))
    print(f"{args.speaker}, adapter {voice.get('adapter_id')}, seed {args.seed}\n")

    # A: same text three times. Pure reproducibility.
    repeats = []
    for i in range(3):
        wav = os.path.join(args.out_dir, f"same_{i}.wav")
        engine.generate_lora_voice(sample[0]["text"], "", voice, wav)
        p = median_pitch(wav)
        if p:
            repeats.append(p)
    print(f"  SAME text x{len(repeats)}: "
          f"{[round(x) for x in repeats]}  "
          f"spread {max(repeats) - min(repeats):.0f} Hz" if repeats else "  none")

    # B: different lines, same seed. What a listener meets.
    across = []
    for i, chunk in enumerate(sample):
        wav = os.path.join(args.out_dir, f"line_{i}.wav")
        engine.generate_lora_voice(chunk["text"], "", voice, wav)
        p = median_pitch(wav)
        if p:
            across.append(p)
        print(f"  [{i + 1}/{len(sample)}] {len(chunk['text']):4}ch  "
              f"{f'{p:.0f} Hz' if p else 'unvoiced'}")

    manifest = json.load(open(os.path.join(REPO, "lora_models", "manifest.json"),
                              encoding="utf-8"))
    items = manifest if isinstance(manifest, list) else list(manifest.values())
    declared = next(((i.get("voice_features") or {}).get("mean_f0")
                     for i in items if isinstance(i, dict)
                     and i.get("id") == voice.get("adapter_id")), None)

    spread = max(across) - min(across) if len(across) > 1 else None
    print(f"\n  ACROSS {len(across)} different lines: median "
          f"{statistics.median(across):.0f} Hz, spread {spread:.0f} Hz")
    print(f"  manifest declares {declared} Hz")
    if spread is not None:
        print(f"\n  The scene-aware allocator wanted 17 Hz between voices.")
        if spread < 17:
            print("  Spread is BELOW that, so pitch separates voices and the\n"
                  "  withdrawal of mean_f0 as a casting metric was premature.")
        else:
            print(f"  Spread is {spread / 17:.1f}x that gap, so two adapters 17 Hz\n"
                  "  apart still overlap across a book. The withdrawal stands and\n"
                  "  contrast must come from timbre or from listening.")

    json.dump({"speaker": args.speaker, "adapter": voice.get("adapter_id"),
               "seed": args.seed, "declared_mean_f0": declared,
               "same_text": repeats, "across_lines": across,
               "across_spread": spread},
              open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
