"""Does a trained LoRA beat simply cloning from the same reference?

The Voice Lab pipeline is the most expensive thing in this repo - preparer,
dedup, LoRA training, profiling, naming - and there are 78 trained adapters on
disk. tts.py also implements zero-shot cloning, which needs only a reference
sample, and exactly ONE voice in voice_config uses it.

That asymmetry may be entirely right: a LoRA plausibly holds a voice steadier
over a whole book than a few seconds of reference can, and a zero-shot demo
never shows drift. But nothing in the repo measures it, and
`voice_adapter_health` found two of the 78 adapters are effectively no-ops
(|B| at 6% and 22% of the population mean), which for those two makes plain
cloning almost certainly better right now.

THE COMPARISON IS FAIR BECAUSE BOTH ARMS USE THE SAME REFERENCE. Each voice
directory carries the ref_sample.wav the adapter was trained around, so:

    clone   zero-shot from that reference
    lora    the trained adapter, which also uses that reference as its prompt

Same text, same reference audio, same model. The only difference is whether the
fine-tuned weights are applied.

SIMILARITY METRIC. A speaker embedding is the right measure and lives in the
sibling repo's environment; where it is unavailable this falls back to a
distance over the acoustic features voice_profiler computes. Which one ran is
printed, because the two are not comparable and quoting the weaker as the
stronger would be worse than reporting nothing.

WHAT WOULD CHANGE THE PIPELINE. If the adapter's advantage over cloning is
small, much of the Voice Lab chain is optional for most characters and the
sample-count question becomes "when is a LoRA worth training at all" rather
than "how many samples does one need". If it is large, the cost is justified
and that is worth knowing with a number rather than by assumption.
"""
import argparse, glob, json, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = REPO + "/app"
sys.path.insert(0, APP)
MODELS = REPO + "/lora_models"
SENTENCE = ("The road bent away between the hills, and neither of them spoke "
            "again until the lights of the town came into view.")


def acoustic_vector(path):
    import librosa, warnings
    warnings.filterwarnings("ignore")
    y, sr = librosa.load(path, sr=22050, mono=True)
    f0, vf, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
    v = f0[vf & ~np.isnan(f0)]
    return np.array([
        float(np.mean(v)) if len(v) else 0.0,
        float(np.std(v)) if len(v) else 0.0,
        float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
        float(np.mean(librosa.feature.rms(y=y))) * 1000,
    ])


def similarity(a_path, b_path):
    a, b = acoustic_vector(a_path), acoustic_vector(b_path)
    scale = np.maximum(np.abs(a), np.abs(b)) + 1e-9
    return float(1.0 - np.mean(np.abs(a - b) / scale))


def pick(limit_full):
    rows = []
    for meta in sorted(glob.glob(MODELS + "/*/training_meta.json")):
        folder = os.path.dirname(meta)
        ref = os.path.join(folder, "ref_sample.wav")
        if not os.path.exists(ref):
            continue
        try:
            d = json.load(open(meta))
        except Exception:
            continue
        rows.append({"voice": os.path.basename(folder), "dir": folder,
                     "ref": ref, "ref_text": d.get("ref_sample_text", ""),
                     "samples": d.get("num_samples")})
    weak = [r for r in rows if (r["samples"] or 0) < 100]
    full = [r for r in rows if (r["samples"] or 0) >= 100][:limit_full]
    return weak + full


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out_dir", default=REPO + "/ab_test_runtime/clone_vs_lora")
    ap.add_argument("--full", type=int, default=5)
    ap.add_argument("--json", default=REPO + "/ab_test_runtime/experiments/clone_vs_lora.json")
    ap.add_argument("--seed", type=int, default=1234,
                    help="both arms at one seed. The clone arm previously hard "
                         "-coded seed -1 and the lora arm passed none, so the "
                         "two arms differed by random draw as well as by "
                         "method - the difference this experiment exists to "
                         "measure.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    voices = pick(args.full)
    print(f"{len(voices)} voices "
          f"({sum(1 for v in voices if (v['samples'] or 0) < 100)} trained on "
          f"<100 samples)\n")

    from tts import TTSEngine
    from experiments.generation import render, GenerationFailed
    config = json.load(open(REPO + "/config.json")) if os.path.exists(REPO + "/config.json") else {}
    engine = TTSEngine(config)

    print(f"  {'voice':34}{'samples':>8}{'clone':>9}{'lora':>9}{'lora-clone':>12}")
    results = []
    for v in voices:
        outs = {}
        ok = True
        for arm in ("clone", "lora"):
            path = os.path.join(args.out_dir, f"{v['voice']}__{arm}.wav")
            # Always regenerate. The old `if not os.path.exists(path)` reused
            # whatever WAV happened to be at that path from an earlier run,
            # which on an UNSEEDED path is a different draw of the voice - so a
            # cached arm and a fresh arm could differ by chance alone and the
            # difference would be read as clone-vs-lora.
            #
            # This file did check its return value, unlike the six routed
            # through render() earlier; its defect was reuse, not a swallowed
            # False. It goes through render() anyway so there is one way to
            # render a segment rather than two that drift.
            if arm == "lora":
                entry = {"type": "lora", "adapter_path": v["dir"],
                         "seed": str(args.seed)}
                cfg = {"X": entry}
            else:
                entry = {"type": "clone", "ref_audio": v["ref"],
                         "ref_text": v["ref_text"], "seed": str(args.seed)}
                cfg = {"X": entry}
            try:
                render(engine, SENTENCE, "", "X", cfg, entry, path)
            except GenerationFailed:
                ok = False
                break
            outs[arm] = path
        if not ok:
            print(f"  {v['voice'][:32]:34}{str(v['samples']):>8}   generation FAILED")
            continue
        try:
            sc = similarity(outs["clone"], v["ref"])
            sl = similarity(outs["lora"], v["ref"])
        except Exception as exc:
            print(f"  {v['voice'][:32]:34} scoring failed: {type(exc).__name__}")
            continue
        results.append({"voice": v["voice"], "samples": v["samples"],
                        "clone": sc, "lora": sl})
        print(f"  {v['voice'][:32]:34}{str(v['samples']):>8}{sc:9.3f}{sl:9.3f}"
              f"{sl-sc:+12.3f}")

    if results:
        d = np.mean([r["lora"] - r["clone"] for r in results])
        wins = sum(1 for r in results if r["lora"] > r["clone"])
        print(f"\n  mean lora - clone {d:+.3f}, adapter ahead on "
              f"{wins}/{len(results)} voices")
        weak = [r for r in results if (r["samples"] or 0) < 100]
        if weak:
            dw = np.mean([r["lora"] - r["clone"] for r in weak])
            print(f"  on the {len(weak)} under-trained voices: {dw:+.3f}")
        print("\n  metric: acoustic feature distance, which is crude and sensitive")
        print("  to the sentence read. It can say a voice is far off; it should")
        print("  not be quoted as speaker similarity.")
        json.dump({"metric": "acoustic feature distance", "sentence": SENTENCE,
                   "results": results}, open(args.json, "w"), indent=1)
        print("\nwrote", args.json)


if __name__ == "__main__":
    main()
