"""How much audio does a voice LoRA actually need?

`voice_adapter_health` found the adapters' |B| norms plateau near 120 samples
against a 200-sample cap, and flagged two voices - trained on 2 and 24 samples -
as barely moving the base model. But norm is MAGNITUDE, not fidelity: a large
norm can be overfitting as easily as learning, so it cannot say whether a voice
sounds like the character it was cloned from.

This measures fidelity. Every voice already carries the reference audio it was
cloned from (`ref_sample.wav`), so for each voice we:

  1. generate one fixed sentence with the LoRA
  2. compare the generated audio to that voice's OWN reference
  3. plot similarity against the number of training samples

The training-set sizes already vary from 2 to 200 across the library, so this
needs no new training - only generation.

SIMILARITY METRIC. A speaker embedding (speechbrain ECAPA) is the right measure
and lives in the sibling repo's environment, not this one. Where it is
unavailable this falls back to a distance over the acoustic features
`voice_profiler` already computes - pitch, brightness, rolloff, energy, rate -
which is cruder and sensitive to the sentence being read. Whichever is used is
printed, because the two are not comparable and quoting one as the other would
be worse than having no number.

READING IT. Similarity rising then flattening near 100-120 samples would mean
the 200 cap wastes audio, and characters currently skipped for insufficient
material could be voiced. Similarity still climbing at 200 would mean the cap is
right and the low-sample voices in the library are unreliable.
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


def pick_voices(limit_full=6):
    """Every below-cap voice, plus a few at the cap as the reference group."""
    rows = []
    for meta in sorted(glob.glob(MODELS + "/*/training_meta.json")):
        folder = os.path.dirname(meta)
        try:
            d = json.load(open(meta))
        except Exception:
            continue
        n = d.get("num_samples")
        if n is None or not os.path.exists(os.path.join(folder, "ref_sample.wav")):
            continue
        rows.append({"voice": os.path.basename(folder), "dir": folder,
                     "samples": int(n), "ref_text": d.get("ref_sample_text", "")})
    below = [r for r in rows if r["samples"] < 200]
    at_cap = [r for r in rows if r["samples"] == 200][:limit_full]
    return sorted(below + at_cap, key=lambda r: r["samples"])


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


def embedder():
    """speechbrain ECAPA if importable; otherwise None."""
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        return EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(REPO, "ab_test_runtime", "ecapa"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out_dir", default=REPO + "/ab_test_runtime/voice_saturation")
    ap.add_argument("--json", default=REPO + "/ab_test_runtime/experiments/voice_data_saturation.json")
    ap.add_argument("--seed", type=int, default=1234,
                    help="fixed generation seed; unseeded, two voices could\n"
                         "differ by draw rather than by sample count")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    voices = pick_voices()
    print(f"{len(voices)} voices, {sum(1 for v in voices if v['samples'] < 200)} below the cap\n")

    from tts import TTSEngine
    from experiments.generation import render, GenerationFailed
    config = json.load(open(REPO + "/config.json")) if os.path.exists(REPO + "/config.json") else {}
    engine = TTSEngine(config)

    enc = embedder()
    metric = "ecapa speaker embedding" if enc else "acoustic feature distance"
    print(f"similarity metric: {metric}\n")

    results = []
    for v in voices:
        out = os.path.join(args.out_dir, v["voice"] + ".wav")
        # Always regenerate, at a fixed seed. The old `if not os.path.exists`
        # reused any WAV left at that path by an earlier run; on an unseeded
        # path that is a different draw of the voice, so a cached voice and a
        # fresh one could differ by chance and the difference would be charged
        # to the sample count this experiment is measuring.
        entry = {"type": "lora", "adapter_path": v["dir"],
                 "seed": str(args.seed)}
        try:
            render(engine, SENTENCE, "", "X", {"X": entry}, entry, out)
        except GenerationFailed as exc:
            print(f"  {v['voice'][:38]:40} generation FAILED: {str(exc)[:40]}")
            continue
        ref = os.path.join(v["dir"], "ref_sample.wav")
        try:
            if enc:
                import torchaudio, torch
                def emb(p):
                    w, sr = torchaudio.load(p)
                    if sr != 16000:
                        w = torchaudio.functional.resample(w, sr, 16000)
                    e = enc.encode_batch(w).squeeze().detach().cpu().numpy()
                    return e / (np.linalg.norm(e) + 1e-9)
                sim = float(np.dot(emb(out), emb(ref)))
            else:
                a, b = acoustic_vector(out), acoustic_vector(ref)
                scale = np.maximum(np.abs(a), np.abs(b)) + 1e-9
                sim = float(1.0 - np.mean(np.abs(a - b) / scale))
        except Exception as exc:
            print(f"  {v['voice'][:38]:40} scoring failed: {type(exc).__name__}")
            continue
        results.append({**{k: v[k] for k in ("voice", "samples")}, "similarity": sim})
        print(f"  {v['voice'][:38]:40}{v['samples']:6} samples   similarity {sim:6.3f}")

    if len(results) >= 4:
        xs = np.array([r["samples"] for r in results], dtype=float)
        ys = np.array([r["similarity"] for r in results])
        low = ys[xs < 100]
        high = ys[xs >= 100]
        print(f"\n  below 100 samples: mean similarity "
              f"{low.mean():.3f} (n={len(low)})" if len(low) else "")
        print(f"  100 and above:     mean similarity "
              f"{high.mean():.3f} (n={len(high)})" if len(high) else "")
        if len(low) and len(high):
            print(f"  difference {high.mean()-low.mean():+.3f}")
        print(f"\n  metric was {metric}. An acoustic-feature distance is crude and")
        print("  sensitive to the sentence read; only an embedding score should be")
        print("  quoted as speaker similarity.")

    json.dump({"metric": metric, "sentence": SENTENCE, "results": results},
              open(args.json, "w"), indent=1)
    print("\nwrote", args.json)


if __name__ == "__main__":
    main()
