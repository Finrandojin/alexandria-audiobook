"""Can pitch separate two voices, once the seed bug is out of the way?

`pitch_stability` answered half of this on 2026-08-04. Seeded, the same adapter
reading the same line three times is EXACTLY reproducible (150.048 Hz, three
times, to twelve decimal places). Reading eight DIFFERENT lines it moves across
a 48 Hz band - down from the ~103 Hz measured before the seed fix. Half the
instability was the bug; half is prosody, and prosody does not go away.

That 48 Hz is the number that matters, because a casting metric has to separate
voices by more than each voice moves on its own. The scene-aware allocator was
built around a 17 Hz target, which is a third of the band one voice covers by
itself. On one adapter. Whether 48 Hz is typical or whether that adapter is
unusually variable is NOT known, and the whole question turns on it.

WHAT THIS MEASURES

    1. within-voice spread for adapters spanning the pool's declared range, so
       the 48 Hz figure rests on more than a single voice
    2. whether the manifest's declared mean_f0 survives seeding - it was
       profiled on unseeded audio, and if the declared numbers are wrong then
       every pairwise distance computed from them is wrong too
    3. the pool's real pairwise separation against the measured within-voice
       spread: what FRACTION of adapter pairs are actually distinguishable

The third is the decision. If most pairs are separated by less than one voice's
own range, pitch cannot carry casting in this pool and the withdrawal of
mean_f0 becomes final rather than provisional. If a usable subset separates
cleanly, the allocator returns with a threshold derived from measurement rather
than from the 17 Hz that was assumed.

A NULL RESULT IS THE USEFUL ONE HERE and must not be softened. "Pitch does not
work for casting in this pool" closes a line of work that has already been
reopened once.
"""
import argparse
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

from experiments.pitch_stability import median_pitch      # noqa: E402


def spanning_adapters(manifest_path, count):
    """`count` adapters spread evenly across the pool's declared pitch range.

    Evenly across the RANGE rather than randomly, because the question is
    whether within-voice spread depends on where a voice sits - a low male
    voice and a high female one may not behave the same way, and sampling the
    dense middle would never show it.
    """
    raw = json.load(open(manifest_path, encoding="utf-8"))
    items = raw if isinstance(raw, list) else list(raw.values())
    pool = sorted(((i["id"], (i.get("voice_features") or {})["mean_f0"])
                   for i in items if isinstance(i, dict) and i.get("id")
                   and (i.get("voice_features") or {}).get("mean_f0")),
                  key=lambda x: x[1])
    if len(pool) <= count:
        return pool, pool
    step = (len(pool) - 1) / (count - 1)
    picked = [pool[round(i * step)] for i in range(count)]
    return picked, pool


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--script", default=os.path.join(REPO, "chunks.json"))
    ap.add_argument("--voice-config", default=os.path.join(REPO, "voice_config.json"))
    ap.add_argument("--config", default=os.path.join(APP, "config.json"))
    ap.add_argument("--manifest", default=os.path.join(REPO, "lora_models", "manifest.json"))
    ap.add_argument("--speaker", default="NARRATOR",
                    help="whose lines to read; the TEXT is held constant across "
                         "adapters so the comparison is voice, not content")
    ap.add_argument("--adapters", type=int, default=6)
    ap.add_argument("--lines", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "ab_test_runtime", "pitch_separation"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "pitch_separation.json"))
    args = ap.parse_args()

    picked, pool = spanning_adapters(args.manifest, args.adapters)
    print(f"{len(pool)} adapters in pool, {len(picked)} sampled across "
          f"{pool[0][1]:.0f}-{pool[-1][1]:.0f} Hz\n")

    chunks = [c for c in json.load(open(args.script, encoding="utf-8"))
              if c.get("text") and c.get("speaker") == args.speaker
              and 80 <= len(c["text"]) <= 220]
    if len(chunks) < args.lines:
        sys.exit(f"only {len(chunks)} usable lines for {args.speaker}")
    # The SAME lines for every adapter. Different text per adapter would let
    # sentence content masquerade as voice difference, which is the confound
    # this whole question is about.
    sample = chunks[:args.lines]

    raw_vc = json.load(open(args.voice_config, encoding="utf-8"))
    vc = (raw_vc.get("characters")
          if isinstance(raw_vc.get("characters"), dict) else raw_vc)
    base = dict(vc.get(args.speaker) or {})
    os.makedirs(args.out_dir, exist_ok=True)

    from tts import TTSEngine
    from experiments.generation import render, GenerationFailed
    engine = TTSEngine(json.load(open(args.config, encoding="utf-8")))

    measured, skipped = [], []
    for adapter_id, declared in picked:
        entry = dict(base)
        entry["type"] = "lora"
        entry["adapter_id"] = adapter_id
        entry["adapter_path"] = os.path.join("lora_models", adapter_id)
        entry["seed"] = str(args.seed)
        vals = []
        for i, chunk in enumerate(sample):
            wav = os.path.join(args.out_dir, f"{adapter_id}_{i}.wav")
            try:
                render(engine, chunk["text"], "", args.speaker, vc, entry, wav)
            except GenerationFailed as exc:
                skipped.append({"adapter": adapter_id, "line": i,
                                "error": str(exc)[:120]})
                continue
            p = median_pitch(wav)
            if p:
                vals.append(p)
        if len(vals) < 3:
            skipped.append({"adapter": adapter_id, "error": "too few voiced"})
            print(f"  {adapter_id:44} SKIPPED, {len(vals)} voiced clips")
            continue
        spread = max(vals) - min(vals)
        med = statistics.median(vals)
        measured.append({"adapter": adapter_id, "declared": declared,
                         "measured_median": med, "spread": spread,
                         "n": len(vals), "values": vals})
        print(f"  {adapter_id:44} declared {declared:6.1f}  "
              f"measured {med:6.1f}  spread {spread:5.1f} Hz  (n={len(vals)})")

    if not measured:
        sys.exit("nothing measured")

    # (2) does the manifest survive seeding?
    errs = [abs(m["measured_median"] - m["declared"]) for m in measured]
    print(f"\n  DECLARED vs MEASURED: mean absolute error "
          f"{statistics.mean(errs):.1f} Hz, worst {max(errs):.1f} Hz")
    print("  The manifest was profiled on UNSEEDED audio. A mean over many "
          "clips\n  converges even when single clips scatter, so a small error "
          "here means\n  the declared numbers are reusable and a large one "
          "means every pairwise\n  distance in the pool is built on sand.")

    # (1) is 48 Hz typical?
    spreads = [m["spread"] for m in measured]
    typical = statistics.median(spreads)
    print(f"\n  WITHIN-VOICE spread across {len(measured)} adapters: "
          f"median {typical:.1f} Hz, range {min(spreads):.1f}-{max(spreads):.1f}")

    # (3) the decision.
    dist = sorted(p[1] for p in pool)
    pairs = total = 0
    for i in range(len(dist)):
        for j in range(i + 1, len(dist)):
            total += 1
            if dist[j] - dist[i] > typical:
                pairs += 1
    frac = pairs / total if total else 0.0
    print(f"\n  POOL SEPARATION: {pairs} of {total} adapter pairs "
          f"({frac*100:.1f}%) are further\n  apart than one voice's own "
          f"{typical:.0f} Hz range.")
    print(f"  The allocator assumed 17 Hz was enough; the measured requirement "
          f"is {typical:.0f}.")
    if frac >= 0.5:
        verdict = ("Most pairs clear the bar. Pitch CAN separate voices here, "
                   "and the allocator returns with a measured threshold.")
    elif frac >= 0.2:
        verdict = ("A minority of pairs clear the bar. Pitch works only for "
                   "deliberately distant voices, not as a general contrast "
                   "metric - useful as a CONSTRAINT (never cast two voices "
                   "closer than this) rather than as an objective.")
    else:
        verdict = ("Almost no pair clears the bar. Pitch cannot carry casting "
                   "in this pool; the withdrawal of mean_f0 is final and "
                   "contrast must come from timbre or from listening.")
    print(f"\n  {verdict}")

    json.dump({"seed": args.seed, "speaker": args.speaker,
               "lines": len(sample), "pool_size": len(pool),
               "measured": measured, "skipped": skipped,
               "declared_vs_measured_mae": statistics.mean(errs),
               "typical_within_voice_spread": typical,
               "separable_pairs": pairs, "total_pairs": total,
               "separable_fraction": frac, "verdict": verdict},
              open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
