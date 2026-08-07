"""Retrain with a genuinely held-out split, and see what changes.

TWO QUESTIONS AT ONCE, both unanswerable before the train/val fix landed.

1. DO THE CLEAN-DATA FAILURES RECOVER? Five adapters scored 0.027-0.404 while
   their training data was verifiably one speaker (0.69-0.81 internal
   consistency, as good as the working ones). Same learning rate, same 200
   samples, same 6 epochs, same final loss ~4.1 as the adapters that worked.
   Nothing in the recorded settings distinguishes them.

   If a retrain with identical settings produces a working adapter, training is
   stochastically unreliable and the library needs a post-training gate. If it
   reproduces the failure, something in that dataset is wrong in a way speaker
   consistency does not capture, and the recipe is not the problem.

2. HOW BIG IS THE CONTAMINATION BOUND? Every existing library score is measured
   on material the adapter trained on - `train_lora.py` read the root
   metadata.jsonl (all 200) rather than train/ (180). Retraining on 180 leaves
   the 20 val clips genuinely unseen, so these are the FIRST honest per-adapter
   numbers in this project.

   Controls are included for exactly this: working adapters retrained the same
   way. The gap between their contaminated and honest scores is the size of the
   bound, and it applies to every number in the library.

WRITES TO A SEPARATE DIRECTORY. The existing lora_models/ library is not
touched - a retrain that turns out worse must not destroy the adapter that
shipped.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

DEFAULT_ZIPS = os.environ.get(
    "ALEXANDRIA_VOICE_ZIPS",
    os.path.join(os.path.expanduser("~"), "Desktop", "zips2",
                 "_deduped_labeled"))


def find_zip(dataset, zip_dir):
    key = "".join(c for c in dataset.lower() if c.isalnum())
    for f in sorted(os.listdir(zip_dir)):
        if not f.endswith(".zip"):
            continue
        if "".join(c for c in f.lower() if c.isalnum()).startswith(key):
            return os.path.join(zip_dir, f)
    return None


def extract(zip_path, dest):
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    return dest


def dataset_of(adapter, models_dir):
    p = os.path.join(models_dir, adapter, "training_meta.json")
    if not os.path.exists(p):
        return None, {}
    meta = json.load(open(p, encoding="utf-8"))
    return os.path.basename(os.path.dirname(
        str(meta.get("ref_sample_audio") or ""))), meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--controls", nargs="*", default=[],
                    help="working adapters, retrained to size the "
                         "contamination bound")
    ap.add_argument("--models", default=os.path.join(REPO, "lora_models"))
    ap.add_argument("--zips", default=DEFAULT_ZIPS)
    ap.add_argument("--work", default=os.path.join(
        REPO, "ab_test_runtime", "retrain_honest"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--use-medoid", action="store_true",
                    help="write a medoid ref.wav before training instead of "
                         "letting train_lora fall back to the first training "
                         "clip. The fallback is a LOTTERY - it recovered "
                         "husky_baritone_20s_m_anime (0.004 -> 0.691) and did "
                         "nothing for husky_baritone_40s_m_military "
                         "(0.141 -> 0.149), which is the difference this flag "
                         "removes.")
    ap.add_argument("--medoid-clips", type=int, default=14)
    ap.add_argument("--eval-lines", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "retrain_honest.json"))
    args = ap.parse_args()

    py = os.path.join(APP, "env", "bin", "python")
    os.makedirs(args.work, exist_ok=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vcv", os.path.join(APP, "experiments", "voice_compare_view.py"))
    vcv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vcv)
    sys.path.insert(0, os.path.join(APP, "experiments"))
    from library_voice_fidelity import ecapa_pairs, extract_val

    jobs = [(a, "failure") for a in args.adapters] + \
           [(a, "control") for a in args.controls]
    results = []
    print(f"{len(jobs)} retrains ({len(args.adapters)} failures, "
          f"{len(args.controls)} controls)\n")

    for adapter, role in jobs:
        dataset, old_meta = dataset_of(adapter, args.models)
        zp = find_zip(dataset, args.zips) if dataset else None
        if not zp:
            results.append({"adapter": adapter, "role": role,
                            "error": "source zip not found"})
            print(f"  {adapter[:34]:36} NO ZIP")
            continue
        ddir = os.path.join(args.work, adapter, "data")
        odir = os.path.join(args.work, adapter, "adapter")
        if not os.path.exists(os.path.join(ddir, "metadata.jsonl")):
            extract(zp, ddir)

        # CHOOSE THE REFERENCE DELIBERATELY when asked. Without this,
        # train_lora falls back to the first training clip - which is how the
        # earlier retrains "recovered": the fallback happened to be a better
        # reference than the shipped one. That makes recovery a property of
        # clip ordering rather than of anything chosen, and it explains why one
        # adapter jumped 0.687 and another moved 0.009 under identical
        # treatment.
        ref_note = None
        if args.use_medoid:
            from voice_reference import select_reference_sample
            meta_rel = os.path.join(ddir, "train", "metadata.jsonl")
            if not os.path.exists(meta_rel):
                meta_rel = os.path.join(ddir, "metadata.jsonl")
            drows = [json.loads(l) for l in open(meta_rel, encoding="utf-8")
                     if l.strip()][:args.medoid_clips]
            cand = [os.path.join(ddir, r["audio_filepath"]) for r in drows
                    if os.path.exists(os.path.join(ddir, r["audio_filepath"]))]
            pick, score = select_reference_sample(cand, max_clips=args.medoid_clips)
            if pick is not None:
                import shutil as _sh
                _sh.copy2(cand[pick], os.path.join(ddir, "ref.wav"))
                ref_note = {"clip": os.path.basename(cand[pick]),
                            "similarity": score}
                print(f"    reference: {ref_note['clip']} "
                      f"(similarity {score})")
            else:
                print("    reference: could not choose a medoid; "
                      "train_lora will fall back to the first clip")

        # The trainer now prefers train/metadata.jsonl when it exists, so the
        # 20 val clips stay unseen. That is the whole point of this run.
        cmd = [py, "-u", os.path.join(APP, "train_lora.py"),
               "--data_dir", ddir, "--output_dir", odir,
               "--epochs", str(args.epochs), "--lora_r", str(args.lora_r),
               "--lora_alpha", str(args.lora_alpha), "--seed", str(args.seed)]
        log = os.path.join(REPO, "ab_test_runtime", "logs",
                           f"retrain_{adapter}.log")
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=7200).returncode
        if rc != 0 or not os.path.exists(
                os.path.join(odir, "adapter_model.safetensors")):
            results.append({"adapter": adapter, "role": role,
                            "error": f"train rc={rc}"})
            print(f"  {adapter[:34]:36} TRAIN FAILED rc={rc}")
            continue
        trained_on = None
        with open(log, encoding="utf-8") as fh:
            for line in fh:
                if "[DATA] Found" in line:
                    trained_on = line.strip()[:90]
                    break

        # Score on the held-out val clips - unseen for the first time.
        clips = extract_val(zp, os.path.join(args.work, adapter, "val"),
                            args.eval_lines)
        from tts import TTSEngine
        from experiments.generation import render, GenerationFailed
        engine = TTSEngine(json.load(open(os.path.join(APP, "config.json"),
                                          encoding="utf-8")))
        entry = {"type": "lora",
                 "adapter_path": os.path.relpath(odir, REPO),
                 "seed": str(args.seed)}
        pairs, durs = [], []
        for i, (human_wav, text) in enumerate(clips):
            gen = os.path.join(args.work, adapter, f"gen_{i}.wav")
            try:
                render(engine, text, "", "SPEAKER", {"SPEAKER": entry},
                       entry, gen)
            except GenerationFailed:
                continue
            import soundfile as sf
            durs.append((sf.info(gen).frames / sf.info(gen).samplerate) /
                        max(sf.info(human_wav).frames /
                            sf.info(human_wav).samplerate, 1e-6))
            pairs.append([human_wav, gen])
        cos, err = ecapa_pairs(pairs, os.environ.get(
            "ALEXANDRIA_SIBLING_PYTHON",
            os.path.join(os.path.dirname(REPO), "alexandria-audiobook.git",
                         "app", "env", "bin", "python")))
        vals = [c for c in (cos or []) if c is not None]
        rec = {"adapter": adapter, "role": role, "dataset": dataset,
               "trained_on": trained_on, "medoid_reference": ref_note,
               "old_ecapa_contaminated": None,
               "new_ecapa_heldout": round(statistics.median(vals), 4)
               if vals else None,
               "dur_ratio": round(statistics.median(durs), 3) if durs else None,
               "n": len(vals), "ecapa_error": err}
        results.append(rec)
        print(f"  {adapter[:34]:36} {role:8} held-out ecapa "
              f"{rec['new_ecapa_heldout']}  dur {rec['dur_ratio']}")

    # Attach the old contaminated score for comparison.
    fid = os.path.join(REPO, "ab_test_runtime", "experiments",
                       "library_voice_fidelity.json")
    if os.path.exists(fid):
        old = {r["adapter"]: r.get("ecapa")
               for r in json.load(open(fid, encoding="utf-8"))["results"]}
        for r in results:
            r["old_ecapa_contaminated"] = old.get(r["adapter"])

    print(f"\n  {'adapter':34}{'role':9}{'was':>8}{'now':>8}{'delta':>8}")
    for r in results:
        if r.get("new_ecapa_heldout") is None:
            continue
        o = r.get("old_ecapa_contaminated")
        d = (r["new_ecapa_heldout"] - o) if o is not None else float("nan")
        print(f"  {r['adapter'][:33]:34}{r['role']:9}"
              f"{o if o is not None else float('nan'):8.3f}"
              f"{r['new_ecapa_heldout']:8.3f}{d:+8.3f}")
    print("\n  'was' is contaminated (trained on its own eval clips);")
    print("  'now' is honest. Controls show how much of any drop is the")
    print("  contamination rather than the retrain.")

    doc = {"epochs": args.epochs, "lora_r": args.lora_r, "seed": args.seed,
           "eval_lines": args.eval_lines, "results": results}
    try:
        from experiments.provenance import provenance
        doc["provenance"] = provenance(__file__, args)
    except Exception as exc:                                # noqa: BLE001
        doc["provenance"] = {"error": str(exc)[:120]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
    print(f"\nwrote {args.out}")
    if not any(r.get("new_ecapa_heldout") is not None for r in results):
        sys.exit(3)


if __name__ == "__main__":
    main()
