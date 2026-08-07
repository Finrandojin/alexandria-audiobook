"""Is each voice dataset actually ONE speaker?

WHY THIS EXISTS. 74 shipped adapters were scored against the narrators they
imitate and the spread was 0.027 to 0.737 - a 27x range - while their training
settings were identical: same learning rate, same 200 samples, same 6 epochs,
and final loss flat at ~4.1 across the whole range. Hyperparameters cannot
explain it, and loss does not see it.

So the cause is upstream, and this measures the most likely one. The preparer
diarizes an audiobook into per-character datasets. When diarization is wrong,
a "character" is several people blended together, and an adapter trained on
three voices learns an average that resembles nobody.

WHAT IT MEASURES. Clips WITHIN one dataset compared against EACH OTHER by
speaker embedding. A clean dataset scores high because every clip is the same
person. A contaminated one scores low.

WHY THAT MATTERS MORE THAN THE ADAPTER SCORE. It splits the failures into two
groups with different remedies, and confuses them otherwise:

    consistency LOW,  adapter LOW   the dataset is wrong. Retraining reproduces
                                    the same average of several people. Rebuild
                                    the dataset.
    consistency HIGH, adapter LOW   the data was fine and training failed.
                                    Retraining is worth the GPU time.
    consistency HIGH, adapter HIGH  working.
    consistency LOW,  adapter HIGH  should not happen; investigate the metric
                                    before believing the adapter.

Reference points measured 2026-08-06: working datasets sit at 0.74-0.82,
`warm_alto_50s_f_gothic` at 0.152 and `husky_baritone_20s_m_supernatural` at
0.098 - clips that do not resemble each other at all.
"""
import argparse
import glob
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

SIBLING_PY = os.environ.get(
    "ALEXANDRIA_SIBLING_PYTHON",
    os.path.join(os.path.dirname(REPO), "alexandria-audiobook.git",
                 "app", "env", "bin", "python"))
DEFAULT_ZIPS = os.environ.get(
    "ALEXANDRIA_VOICE_ZIPS",
    os.path.join(os.path.expanduser("~"), "Desktop", "zips2",
                 "_deduped_labeled"))


def ecapa(pairs):
    """Speaker cosine in the interpreter that has speechbrain. Never falls back
    to an acoustic distance - a silent substitution of a weaker metric is the
    failure this project keeps rediscovering."""
    if not pairs:
        return None, "no pairs"
    if not os.path.exists(SIBLING_PY):
        return None, "sibling interpreter missing"
    script = os.path.join(APP, "experiments", "_ecapa_batch.py")
    try:
        out = subprocess.run([SIBLING_PY, script],
                             input=json.dumps([[a, b] for a, b in pairs]),
                             capture_output=True, text=True, timeout=3600,
                             cwd=APP)
    except subprocess.SubprocessError as exc:
        return None, str(exc)[:140]
    if out.returncode != 0:
        return None, f"rc={out.returncode} {out.stderr[-160:]}"
    try:
        return json.loads(out.stdout.strip().splitlines()[-1]), None
    except Exception as exc:                                # noqa: BLE001
        return None, f"unparsable: {exc}"


def zip_index(zip_dir):
    idx = {}
    for p in glob.glob(os.path.join(zip_dir, "*.zip")):
        key = "".join(c for c in os.path.basename(p).lower() if c.isalnum())
        idx[key] = p
    return idx


def find_zip(dataset, idx):
    key = "".join(c for c in dataset.lower() if c.isalnum())
    for zkey, path in idx.items():
        if zkey.startswith(key):
            return path
    return None


def consistency(zip_path, clips, rng, workdir):
    """Median pairwise similarity among clips of one dataset."""
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        meta = "train/metadata.jsonl" if "train/metadata.jsonl" in names \
            else "metadata.jsonl"
        if meta not in names:
            return None, 0, "no metadata"
        rows = [json.loads(l) for l in z.read(meta).decode("utf-8").splitlines()
                if l.strip()]
        rng.shuffle(rows)
        paths = []
        for e in rows[:clips]:
            fp = e.get("audio_filepath")
            if fp and fp in names:
                dest = os.path.join(workdir, os.path.basename(fp))
                with open(dest, "wb") as fh:
                    fh.write(z.read(fp))
                paths.append(dest)
    # Disjoint pairs rather than all-pairs: the median is what is reported and
    # O(n) sampling reaches it without the quadratic cost.
    pairs = [(paths[i], paths[i + 1]) for i in range(0, len(paths) - 1, 2)]
    cos, err = ecapa(pairs)
    if err:
        return None, len(pairs), err
    vals = [c for c in (cos or []) if c is not None]
    return (statistics.median(vals) if vals else None), len(vals), None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fidelity", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "library_voice_fidelity.json"))
    ap.add_argument("--zips", default=DEFAULT_ZIPS)
    ap.add_argument("--clips", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--clean-threshold", type=float, default=0.60)
    ap.add_argument("--adapter-threshold", type=float, default=0.45)
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments",
        "dataset_speaker_consistency.json"))
    args = ap.parse_args()

    fid = json.load(open(args.fidelity, encoding="utf-8"))
    scored = {r["adapter"]: r for r in fid["results"]}
    idx = zip_index(args.zips)
    rng = random.Random(args.seed)
    rows = []
    print(f"  {len(scored)} datasets, {args.clips} clips each\n")
    print(f"  {'adapter':36}{'adapter':>9}{'dataset':>10}  verdict")
    for name, rec in sorted(scored.items(),
                            key=lambda kv: kv[1].get("ecapa") if
                            kv[1].get("ecapa") is not None else 9):
        ds = rec.get("dataset") or ""
        zp = find_zip(ds, idx)
        if not zp:
            rows.append({"adapter": name, "dataset": ds, "error": "no zip"})
            continue
        work = tempfile.mkdtemp(prefix="cons_")
        cons, n, err = consistency(zp, args.clips, rng, work)
        a = rec.get("ecapa")
        verdict = "unknown"
        if cons is not None and a is not None:
            clean = cons >= args.clean_threshold
            good = a >= args.adapter_threshold
            verdict = ("working" if clean and good else
                       "RETRAIN - data is clean, training failed" if clean else
                       "REBUILD DATASET - clips are not one speaker" if not good
                       else "check - mixed data but adapter scores")
        rows.append({"adapter": name, "dataset": ds, "adapter_ecapa": a,
                     "dataset_consistency": round(cons, 4) if cons else None,
                     "pairs": n, "verdict": verdict, "error": err})
        print(f"  {name[:35]:36}{a if a is not None else float('nan'):9.3f}"
              f"{cons if cons is not None else float('nan'):10.3f}  {verdict}")

    counts = {}
    for r in rows:
        counts[r.get("verdict", "error")] = counts.get(r.get("verdict", "error"), 0) + 1
    print("\n  SUMMARY")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {v:3}  {k}")

    doc = {"clean_threshold": args.clean_threshold,
           "adapter_threshold": args.adapter_threshold,
           "clips_per_dataset": args.clips, "seed": args.seed,
           "summary": counts, "results": rows}
    try:
        from experiments.provenance import provenance
        doc["provenance"] = provenance(__file__, args)
    except Exception as exc:                                # noqa: BLE001
        doc["provenance"] = {"error": str(exc)[:120]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
    print(f"\nwrote {args.out}")
    if not any(r.get("dataset_consistency") for r in rows):
        sys.exit(3)


if __name__ == "__main__":
    main()
