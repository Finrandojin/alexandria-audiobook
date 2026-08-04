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

SIMILARITY METRIC. SpeechBrain ECAPA lives in the sibling repo's environment,
not this one. Generation and scoring are separate required phases; scoring
fails if ECAPA is unavailable and never substitutes acoustic-feature distance.

READING IT. Similarity rising then flattening near 100-120 samples would mean
the 200 cap wastes audio, and characters currently skipped for insufficient
material could be voiced. Similarity still climbing at 200 would mean the cap is
right and the low-sample voices in the library are unreliable.
"""
import argparse, glob, json, os, sys

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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", required=True, choices=("generate", "score"),
                    help="generate under app/env; score under the sibling env")
    ap.add_argument("--out_dir", default=REPO + "/ab_test_runtime/voice_saturation_seeded")
    ap.add_argument("--json", default=REPO + "/ab_test_runtime/experiments/voice_data_saturation_seeded.json")
    ap.add_argument("--config", default=os.path.join(APP, "config.json"))
    ap.add_argument("--ecapa-cache", default=os.path.join(
        REPO, "ab_test_runtime", "ecapa"))
    ap.add_argument("--seed", type=int, default=1234,
                    help="fixed generation seed; unseeded, two voices could\n"
                         "differ by draw rather than by sample count")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.phase == "score":
        if not os.path.exists(args.json):
            raise SystemExit(f"generation manifest does not exist: {args.json}")
        doc = json.load(open(args.json, encoding="utf-8"))
        rows = doc.get("results")
        if doc.get("status") != "generated" or not isinstance(rows, list) or not rows:
            raise SystemExit("refusing to score an incomplete generation manifest")
        from experiments.speaker_similarity import get_ecapa_encoder, score_ecapa_pair
        encoder = get_ecapa_encoder(args.ecapa_cache)
        for row in rows:
            paths = {}
            for field in ("reference", "generated_file"):
                paths[field] = os.path.join(REPO, row.get(field, ""))
                if not os.path.isfile(paths[field]):
                    raise SystemExit(f"missing scoring input {field}: {row.get(field)}")
            row["similarity"] = score_ecapa_pair(
                encoder, paths["generated_file"], paths["reference"])
        from experiments.provenance import provenance
        doc["metric"] = "speechbrain/spkrec-ecapa-voxceleb cosine similarity"
        doc["status"] = "scored"
        doc["scoring_provenance"] = provenance(__file__, args)
        json.dump(doc, open(args.json, "w", encoding="utf-8"), indent=1)
        print(f"scored {len(rows)} voices with ECAPA\nwrote {args.json}")
        return

    voices = pick_voices()
    if not voices:
        raise SystemExit("no voices with reference audio and sample counts")
    print(f"{len(voices)} voices, {sum(1 for v in voices if v['samples'] < 200)} below the cap\n")

    from tts import TTSEngine
    from experiments.generation import render, GenerationFailed
    if not os.path.isfile(args.config):
        raise SystemExit(f"TTS config does not exist: {args.config}")
    config = json.load(open(args.config, encoding="utf-8"))
    engine = TTSEngine(config)

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
        results.append({"voice": v["voice"], "samples": v["samples"],
                        "reference": os.path.relpath(ref, REPO),
                        "generated_file": os.path.relpath(out, REPO)})
        print(f"  generated {v['voice']} ({v['samples']} samples)")

    if len(results) != len(voices):
        raise SystemExit(f"generation incomplete: {len(results)}/{len(voices)} voices")
    from experiments.provenance import provenance
    json.dump({"status": "generated", "metric": None, "sentence": SENTENCE,
               "provenance": provenance(__file__, args),
               "results": results}, open(args.json, "w", encoding="utf-8"), indent=1)
    print("\nwrote generation manifest", args.json)


if __name__ == "__main__":
    main()
