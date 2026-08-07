"""Does the reference clip CAUSE adapter quality, or merely correlate with it?

WHAT IS ESTABLISHED AND WHAT IS NOT. Across 75 shipped adapters, reference
quality correlates +0.76 with adapter quality, and a mismatched reference makes
an adapter 6.4x more likely to fail (86% vs 13%). On that basis the dataset
builder now picks the medoid instead of sample 0.

That is a correlation. Both could follow from a third cause - a dataset with
many misdiarized clips is both more likely to hand you a bad reference AND
harder to learn from - and the fix would then be worthless.

THE INTERVENTION. Train the same dataset twice, changing ONLY the reference:

    medoid arm   reference = the clip most similar to all the others
    worst arm    reference = the clip least similar to all the others

Same 180 training clips, same seed, same epochs, same rank, same everything
else. Any difference is caused by the reference, because nothing else differs.

    medoid >> worst   the reference is causal and the builder fix is real
    medoid ~= worst   it is a symptom of dataset quality, and the fix does
                      nothing. Revert it and look elsewhere.

WHY THIS IS WORTH THE GPU TIME. The builder change is already merged on the
strength of a correlation. If it does nothing, that should be known now rather
than after a hundred datasets are built on it.

A NOTE ON WHY THE EARLIER RETRAINS RECOVERED. Those extracted zips carry no
ref.wav, so train_lora fell through to its "first training sample" fallback -
a DIFFERENT reference from the one the original adapter used. The recovery from
0.027 to 0.685 was therefore already a reference change, unplanned. This makes
it deliberate and controlled.
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(APP, "experiments"))


def rank_clips(paths):
    """-> (best_index, worst_index, scores) by median similarity to the rest."""
    from voice_reference import _speaker_similarities
    pairs, index = [], []
    for a in range(len(paths)):
        for b in range(a + 1, len(paths)):
            pairs.append((paths[a], paths[b]))
            index.append((a, b))
    sims = _speaker_similarities(pairs, timeout=1800)
    if not sims or len(sims) != len(pairs):
        return None, None, None
    scores = {a: [] for a in range(len(paths))}
    for (a, b), v in zip(index, sims):
        if v is None:
            continue
        scores[a].append(v)
        scores[b].append(v)
    med = {a: statistics.median(v) for a, v in scores.items() if v}
    if len(med) < 2:
        return None, None, None
    return max(med, key=med.get), min(med, key=med.get), med


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", default="husky_baritone_20s_m_anime",
                    help="whose dataset to use; default is the most extreme "
                         "case, reference -0.026 and adapter 0.004")
    ap.add_argument("--models", default=os.path.join(REPO, "lora_models"))
    ap.add_argument("--zips", default=os.environ.get(
        "ALEXANDRIA_VOICE_ZIPS",
        os.path.join(os.path.expanduser("~"), "Desktop", "zips2",
                     "_deduped_labeled")))
    ap.add_argument("--foreign-from", default=None,
                    help="adapter whose dataset supplies a DIFFERENT narrator's "
                         "clip as a third arm. Without it the 'worst' arm is "
                         "only the least-typical clip of the same speaker, "
                         "which on a clean dataset is barely worse than the "
                         "medoid - 0.873 vs 0.815 on the first run, too small "
                         "a contrast to detect anything. A foreign clip "
                         "reproduces the ACTUAL failure: a reference that is "
                         "not the narrator at all, which is what -0.026 meant.")
    ap.add_argument("--rank-clips", type=int, default=14)
    ap.add_argument("--eval-lines", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--work", default=os.path.join(
        REPO, "ab_test_runtime", "reference_intervention"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "reference_intervention.json"))
    args = ap.parse_args()

    from retrain_honest import dataset_of, find_zip, extract
    from library_voice_fidelity import extract_val, ecapa_pairs

    dataset, _ = dataset_of(args.adapter, args.models)
    zp = find_zip(dataset, args.zips) if dataset else None
    if not zp:
        sys.exit(f"no source zip for {args.adapter}")
    base = os.path.join(args.work, args.adapter)
    ddir = os.path.join(base, "data")
    if not os.path.exists(os.path.join(ddir, "metadata.jsonl")):
        extract(zp, ddir)

    meta_p = os.path.join(ddir, "train", "metadata.jsonl")
    if not os.path.exists(meta_p):
        meta_p = os.path.join(ddir, "metadata.jsonl")
    rows = [json.loads(l) for l in open(meta_p, encoding="utf-8") if l.strip()]
    candidates = [os.path.join(ddir, r["audio_filepath"])
                  for r in rows[:args.rank_clips]
                  if os.path.exists(os.path.join(ddir, r["audio_filepath"]))]
    best, worst, med = rank_clips(candidates)
    if best is None:
        sys.exit("could not rank clips; is the speaker model available?")
    print(f"{args.adapter}\n  dataset {dataset}")
    print(f"  medoid clip {os.path.basename(candidates[best])} "
          f"(similarity {med[best]:.3f})")
    print(f"  worst clip  {os.path.basename(candidates[worst])} "
          f"(similarity {med[worst]:.3f})\n")

    py = os.path.join(APP, "env", "bin", "python")
    sib = os.environ.get(
        "ALEXANDRIA_SIBLING_PYTHON",
        os.path.join(os.path.dirname(REPO), "alexandria-audiobook.git",
                     "app", "env", "bin", "python"))
    clips = extract_val(zp, os.path.join(base, "val"), args.eval_lines)
    results = []

    # A third arm from ANOTHER narrator, when asked for. This is the condition
    # the library actually exhibited: a reference scoring -0.026 against its own
    # dataset is not an atypical clip of the right speaker, it is the wrong
    # person. Ranking within one clean dataset cannot produce that.
    arms = [("medoid", candidates[best]), ("worst", candidates[worst])]
    if args.foreign_from:
        fds, _ = dataset_of(args.foreign_from, args.models)
        fzp = find_zip(fds, args.zips) if fds else None
        if fzp:
            fdir = os.path.join(base, "foreign")
            os.makedirs(fdir, exist_ok=True)
            with zipfile.ZipFile(fzp) as z:
                names = set(z.namelist())
                fmeta = ("train/metadata.jsonl" if "train/metadata.jsonl"
                         in names else "metadata.jsonl")
                frows = [json.loads(l) for l in
                         z.read(fmeta).decode("utf-8").splitlines() if l.strip()]
                for e in frows[:1]:
                    fp = e.get("audio_filepath")
                    if fp and fp in names:
                        dest = os.path.join(fdir, "foreign_ref.wav")
                        with open(dest, "wb") as fh:
                            fh.write(z.read(fp))
                        arms.append(("foreign", dest))
                        print(f"  foreign clip from {args.foreign_from}")
        else:
            print(f"  foreign arm skipped: no zip for {args.foreign_from}")

    for arm, ref_path in arms:
        # A separate data directory per arm, differing ONLY in ref.wav. Sharing
        # one directory and swapping the file would make the two runs depend on
        # execution order.
        adir = os.path.join(base, f"data_{arm}")
        if not os.path.exists(os.path.join(adir, "metadata.jsonl")):
            shutil.copytree(ddir, adir)
        shutil.copy2(ref_path, os.path.join(adir, "ref.wav"))
        odir = os.path.join(base, f"adapter_{arm}")
        log = os.path.join(REPO, "ab_test_runtime", "logs",
                           f"refintervene_{args.adapter}_{arm}.log")
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.run(
                [py, "-u", os.path.join(APP, "train_lora.py"),
                 "--data_dir", adir, "--output_dir", odir,
                 "--epochs", str(args.epochs), "--lora_r", "64",
                 "--lora_alpha", "128", "--seed", str(args.seed)],
                stdout=fh, stderr=subprocess.STDOUT, timeout=7200).returncode
        sim = (round(med[best], 4) if arm == "medoid" else
               round(med[worst], 4) if arm == "worst" else None)
        rec = {"arm": arm, "ref_clip": os.path.basename(ref_path),
               "ref_similarity": sim, "rc": rc}
        if rc != 0:
            rec["error"] = f"train rc={rc}"
            results.append(rec)
            print(f"  {arm}: TRAIN FAILED rc={rc}")
            continue

        from tts import TTSEngine
        from experiments.generation import render, GenerationFailed
        engine = TTSEngine(json.load(open(os.path.join(APP, "config.json"),
                                          encoding="utf-8")))
        entry = {"type": "lora", "adapter_path": os.path.relpath(odir, REPO),
                 "seed": str(args.seed)}
        pairs = []
        for i, (human_wav, text) in enumerate(clips):
            gen = os.path.join(odir, f"eval_{i}.wav")
            try:
                render(engine, text, "", "SPEAKER", {"SPEAKER": entry}, entry, gen)
            except GenerationFailed:
                continue
            pairs.append([human_wav, gen])
        cos, err = ecapa_pairs(pairs, sib)
        vals = [c for c in (cos or []) if c is not None]
        rec["ecapa"] = round(statistics.median(vals), 4) if vals else None
        rec["n"] = len(vals)
        rec["ecapa_error"] = err
        results.append(rec)
        shown = (f"{rec['ref_similarity']:.3f}" if rec['ref_similarity']
                 is not None else "foreign")
        print(f"  {arm}: ref {shown} -> adapter {rec['ecapa']}")

    scored = {r["arm"]: r.get("ecapa") for r in results
              if r.get("ecapa") is not None}
    # JUDGE THE EFFECT AGAINST THE CONTRAST THAT WAS AVAILABLE, not against a
    # fixed number. The first version required delta > 0.10 and reported "NOT
    # CAUSAL" for a run whose two arms differed by only 0.058 in the reference
    # and 0.046 in the adapter - near 1:1 transfer, which is strong evidence
    # FOR causation. A fixed threshold against a variable input measures the
    # threshold.
    verdict = "inconclusive - an arm failed to train or score"
    ratio = None
    if "medoid" in scored and "worst" in scored:
        delta = scored["medoid"] - scored["worst"]
        contrast = med[best] - med[worst]
        ratio = (delta / contrast) if contrast > 0.01 else None
        print(f"\n  medoid {scored['medoid']:.3f} vs worst "
              f"{scored['worst']:.3f}   adapter delta {delta:+.3f}")
        print(f"  reference contrast {contrast:.3f}  -> transfer ratio "
              f"{ratio if ratio is not None else float('nan'):.2f}")
        if ratio is None:
            verdict = (f"INCONCLUSIVE: the two references differ by only "
                       f"{contrast:.3f}, too little contrast to detect "
                       f"anything. Use --foreign-from for a real one.")
        elif ratio > 0.5:
            verdict = (f"CAUSAL: {delta:+.3f} adapter change from a "
                       f"{contrast:.3f} reference change - transfer ratio "
                       f"{ratio:.2f}. The reference moves the adapter roughly "
                       f"proportionally, so the builder fix is real. Note the "
                       f"absolute gap is small only because the contrast was.")
        elif ratio > 0.15:
            verdict = (f"WEAKLY CAUSAL: transfer ratio {ratio:.2f}. The "
                       f"reference matters but explains less than the "
                       f"correlation suggests.")
        else:
            verdict = (f"NOT CAUSAL: transfer ratio {ratio:.2f} - a "
                       f"{contrast:.3f} reference change moved the adapter "
                       f"{delta:+.3f}. The +0.76 correlation is a symptom of "
                       f"dataset quality, not a lever. Revert the builder "
                       f"change.")
    if "foreign" in scored and "medoid" in scored:
        fd = scored["medoid"] - scored["foreign"]
        print(f"  medoid {scored['medoid']:.3f} vs FOREIGN narrator "
              f"{scored['foreign']:.3f}   delta {fd:+.3f}")
        verdict += (f" | foreign-reference arm: {fd:+.3f}, the condition the "
                    f"library actually exhibited")
    print(f"  VERDICT: {verdict}")

    doc = {"adapter": args.adapter, "dataset": dataset, "seed": args.seed,
           "epochs": args.epochs, "verdict": verdict, "results": results}
    try:
        from experiments.provenance import provenance
        doc["provenance"] = provenance(__file__, args)
    except Exception as exc:                                # noqa: BLE001
        doc["provenance"] = {"error": str(exc)[:120]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
    print(f"\nwrote {args.out}")
    if len(scored) < 2:
        sys.exit(3)


if __name__ == "__main__":
    main()
