"""Is LoRA training a lottery, or was the improvement the settings that changed?

WHAT THIS SEPARATES. Five adapters that resembled nobody (0.027-0.404) all
improved on a retrain, two of them to ~0.67. That was reported as "training is
stochastically unreliable" - but the retrain was NOT a clean rerun. Two things
changed at once:

    seed        None on the originals, 1234 on the retrains
    samples     200 (train+val) -> 180 (train only)

Three explanations predict exactly the same improvement:

    a genuine training lottery, resolved differently on the second draw
    the fixed seed happening to land better than whatever None gave
    180 clean samples beating 200 with the val clips mixed in

They imply different fixes - seed control, a retry loop, or neither - so
choosing between them matters before anything is built.

THE DESIGN. Train the same adapter N times with EVERYTHING held constant,
including the seed. Then:

    identical outputs      training is deterministic. The improvement came from
                           the seed or the data change, not from luck, and a
                           retry loop would be useless.
    outputs vary widely    it IS a lottery, and no seed pins it - which means a
                           post-training gate is mandatory, not optional.
    outputs vary slightly  deterministic in effect; ordinary numerical noise.

Compared by BOTH weight hash and generated-audio similarity: identical weights
prove determinism outright, while similar-but-not-identical weights still leave
the question of whether the audio differs audibly.

CHEAP. Training is ~170s, so three runs is under fifteen minutes against a
question that decides what gets built.
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(APP, "experiments"))


def weight_hash(adapter_dir):
    p = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", default="husky_baritone_20s_m_anime",
                    help="the adapter whose dataset is reused; chosen by "
                         "default as the most extreme recovery (0.027 -> 0.685)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--eval-lines", type=int, default=6)
    ap.add_argument("--models", default=os.path.join(REPO, "lora_models"))
    ap.add_argument("--zips", default=os.environ.get(
        "ALEXANDRIA_VOICE_ZIPS",
        os.path.join(os.path.expanduser("~"), "Desktop", "zips2",
                     "_deduped_labeled")))
    ap.add_argument("--work", default=os.path.join(
        REPO, "ab_test_runtime", "training_determinism"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "training_determinism.json"))
    args = ap.parse_args()

    from retrain_honest import dataset_of, find_zip, extract
    from library_voice_fidelity import extract_val, ecapa_pairs

    dataset, _ = dataset_of(args.adapter, args.models)
    zp = find_zip(dataset, args.zips) if dataset else None
    if not zp:
        sys.exit(f"no source zip for {args.adapter} ({dataset})")
    ddir = os.path.join(args.work, "data")
    if not os.path.exists(os.path.join(ddir, "metadata.jsonl")):
        extract(zp, ddir)
    py = os.path.join(APP, "env", "bin", "python")
    print(f"{args.adapter}: {args.runs} runs, seed {args.seed} held constant\n")

    runs = []
    for i in range(args.runs):
        odir = os.path.join(args.work, f"run{i}")
        log = os.path.join(REPO, "ab_test_runtime", "logs",
                           f"determinism_run{i}.log")
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.run(
                [py, "-u", os.path.join(APP, "train_lora.py"),
                 "--data_dir", ddir, "--output_dir", odir,
                 "--epochs", str(args.epochs), "--lora_r", str(args.lora_r),
                 "--lora_alpha", str(args.lora_alpha), "--seed", str(args.seed)],
                stdout=fh, stderr=subprocess.STDOUT, timeout=7200).returncode
        loss = None
        with open(log, encoding="utf-8") as fh:
            for line in fh:
                if "avg_loss=" in line:
                    loss = line.strip().split("avg_loss=")[-1][:8]
        runs.append({"run": i, "rc": rc, "dir": odir,
                     "weight_sha256": weight_hash(odir), "final_loss": loss})
        print(f"  run {i}: rc={rc} loss={loss} sha={str(runs[-1]['weight_sha256'])[:12]}")

    ok = [r for r in runs if r["weight_sha256"]]
    hashes = {r["weight_sha256"] for r in ok}
    identical = len(hashes) == 1 and len(ok) > 1
    print(f"\n  distinct weight hashes: {len(hashes)} across {len(ok)} runs")

    # Audio comparison regardless: identical weights make it a formality, and
    # differing weights leave open whether the difference is audible.
    clips = extract_val(zp, os.path.join(args.work, "val"), args.eval_lines)
    from tts import TTSEngine
    from experiments.generation import render, GenerationFailed
    engine = TTSEngine(json.load(open(os.path.join(APP, "config.json"),
                                      encoding="utf-8")))
    sib = os.environ.get(
        "ALEXANDRIA_SIBLING_PYTHON",
        os.path.join(os.path.dirname(REPO), "alexandria-audiobook.git",
                     "app", "env", "bin", "python"))
    for r in ok:
        entry = {"type": "lora", "adapter_path": os.path.relpath(r["dir"], REPO),
                 "seed": str(args.seed)}
        pairs = []
        for i, (human_wav, text) in enumerate(clips):
            gen = os.path.join(r["dir"], f"eval_{i}.wav")
            try:
                render(engine, text, "", "SPEAKER", {"SPEAKER": entry}, entry, gen)
            except GenerationFailed:
                continue
            pairs.append([human_wav, gen])
        cos, err = ecapa_pairs(pairs, sib)
        vals = [c for c in (cos or []) if c is not None]
        r["ecapa"] = round(statistics.median(vals), 4) if vals else None
        r["ecapa_error"] = err
        print(f"  run {r['run']}: held-out ecapa {r['ecapa']}")

    scores = [r["ecapa"] for r in ok if r.get("ecapa") is not None]
    spread = (max(scores) - min(scores)) if len(scores) > 1 else None
    verdict = ("deterministic - the improvement came from the seed or the data "
               "change, not from luck; a retry loop would be useless"
               if identical else
               "NOT deterministic at a fixed seed - a post-training gate is "
               "mandatory, because nothing pins the outcome"
               if spread is not None and spread > 0.15 else
               "varies slightly - deterministic in effect; ordinary numerical "
               "noise" if spread is not None else "inconclusive")
    print(f"\n  ecapa spread across runs: "
          f"{spread if spread is not None else float('nan'):.4f}")
    print(f"  VERDICT: {verdict}")

    doc = {"adapter": args.adapter, "dataset": dataset, "runs": args.runs,
           "seed": args.seed, "identical_weights": identical,
           "distinct_hashes": len(hashes), "ecapa_spread": spread,
           "verdict": verdict, "results": runs}
    try:
        from experiments.provenance import provenance
        doc["provenance"] = provenance(__file__, args)
    except Exception as exc:                                # noqa: BLE001
        doc["provenance"] = {"error": str(exc)[:120]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
    print(f"\nwrote {args.out}")
    if not ok:
        sys.exit(3)


if __name__ == "__main__":
    main()
