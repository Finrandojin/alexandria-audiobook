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

SIMILARITY METRIC. ECAPA speaker embeddings live in the sibling repo's
environment. Generation and scoring are separate required phases; scoring
fails if ECAPA is unavailable and never substitutes an acoustic metric.

WHAT WOULD CHANGE THE PIPELINE. If the adapter's advantage over cloning is
small, much of the Voice Lab chain is optional for most characters and the
sample-count question becomes "when is a LoRA worth training at all" rather
than "how many samples does one need". If it is large, the cost is justified
and that is worth knowing with a number rather than by assumption.
"""
import argparse, glob, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = REPO + "/app"
sys.path.insert(0, APP)
MODELS = REPO + "/lora_models"
SENTENCE = ("The road bent away between the hills, and neither of them spoke "
            "again until the lights of the town came into view.")


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
    ap.add_argument("--phase", required=True, choices=("generate", "score"),
                    help="generate under app/env; score under the sibling env")
    ap.add_argument("--out_dir", default=REPO + "/ab_test_runtime/clone_vs_lora_seeded")
    ap.add_argument("--full", type=int, default=5)
    ap.add_argument("--json", default=REPO + "/ab_test_runtime/experiments/clone_vs_lora_seeded.json")
    ap.add_argument("--config", default=os.path.join(APP, "config.json"))
    ap.add_argument("--ecapa-cache", default=os.path.join(
        REPO, "ab_test_runtime", "ecapa"))
    ap.add_argument("--seed", type=int, default=1234,
                    help="both arms at one seed. The clone arm previously hard "
                         "-coded seed -1 and the lora arm passed none, so the "
                         "two arms differed by random draw as well as by "
                         "method - the difference this experiment exists to "
                         "measure.")
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
            for field in ("reference", "clone_file", "lora_file"):
                paths[field] = os.path.join(REPO, row.get(field, ""))
                if not os.path.isfile(paths[field]):
                    raise SystemExit(f"missing scoring input {field}: {row.get(field)}")
            row["clone_similarity"] = score_ecapa_pair(
                encoder, paths["clone_file"], paths["reference"])
            row["lora_similarity"] = score_ecapa_pair(
                encoder, paths["lora_file"], paths["reference"])
            row["lora_minus_clone"] = (row["lora_similarity"] -
                                        row["clone_similarity"])
        from experiments.provenance import provenance
        doc["metric"] = "speechbrain/spkrec-ecapa-voxceleb cosine similarity"
        doc["status"] = "scored"
        doc["scoring_provenance"] = provenance(__file__, args)
        json.dump(doc, open(args.json, "w", encoding="utf-8"), indent=1)
        print(f"scored {len(rows)} voices with ECAPA\nwrote {args.json}")
        return

    voices = pick(args.full)
    if not voices:
        raise SystemExit("no voices with reference audio and training metadata")
    print(f"{len(voices)} voices "
          f"({sum(1 for v in voices if (v['samples'] or 0) < 100)} trained on "
          f"<100 samples)\n")

    from tts import TTSEngine
    from experiments.generation import render, GenerationFailed
    if not os.path.isfile(args.config):
        raise SystemExit(f"TTS config does not exist: {args.config}")
    config = json.load(open(args.config, encoding="utf-8"))
    engine = TTSEngine(config)

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
        results.append({"voice": v["voice"], "samples": v["samples"],
                        "reference": os.path.relpath(v["ref"], REPO),
                        "clone_file": os.path.relpath(outs["clone"], REPO),
                        "lora_file": os.path.relpath(outs["lora"], REPO)})
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
