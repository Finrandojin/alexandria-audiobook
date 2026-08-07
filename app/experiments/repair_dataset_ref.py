"""Pick a reference clip that actually represents the dataset.

THE DEFECT. `dataset_builder.py` takes `ref_index: int = 0` - the reference is
whatever sample happens to be first. `train_lora.py` then extracts the speaker
embedding from that ONE file and uses it for every training sample, so a single
unrepresentative clip at position 0 anchors all 200 samples to the wrong voice.

Nothing checked it, and it shows. Measured across all 75 shipped adapters on
2026-08-07:

    correlation(reference matches its dataset, adapter quality) = +0.76
    reference mismatched (<0.3):  7 adapters, 6 of them poor  (86%)
    reference matching:          67 adapters, 9 of them poor  (13%)

A mismatched reference makes an adapter 6.4x more likely to fail. It is the
single largest identified cause of bad voices in the library.

THE FIX IS THE MEDOID, not the first clip. The medoid is the clip most similar
to all the others, so it is representative by construction and robust to a
minority of bad clips - which is exactly the failure mode, since a dataset with
a few misdiarized clips still has a clear majority speaker.

WHY THIS IS A TOOL AND NOT A BUILDER CHANGE. Choosing the medoid needs a
speaker model, which lives in the sibling interpreter, not in app/env. Running
that inside a FastAPI request path would make dataset building depend on a
second environment. So this audits and repairs datasets out of band, and
`verify_adapter_identity` catches the consequence at training time. Wiring
medoid selection into the builder is worth doing once the embedding is
available in-process.
"""
import argparse
import glob
import json
import os
import random
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

SIBLING_PY = os.environ.get(
    "ALEXANDRIA_SIBLING_PYTHON",
    os.path.join(os.path.dirname(REPO), "alexandria-audiobook.git",
                 "app", "env", "bin", "python"))


def ecapa(pairs):
    if not pairs:
        return None, "no pairs"
    if not os.path.exists(SIBLING_PY):
        return None, "sibling interpreter missing"
    script = os.path.join(APP, "experiments", "_ecapa_batch.py")
    try:
        out = subprocess.run(
            [SIBLING_PY, script],
            input=json.dumps([[os.path.abspath(a), os.path.abspath(b)]
                              for a, b in pairs]),
            capture_output=True, text=True, timeout=3600, cwd=APP)
    except subprocess.SubprocessError as exc:
        return None, str(exc)[:140]
    if out.returncode != 0:
        return None, f"rc={out.returncode} {out.stderr[-160:]}"
    try:
        return json.loads(out.stdout.strip().splitlines()[-1]), None
    except Exception as exc:                                # noqa: BLE001
        return None, f"unparsable: {exc}"


def choose_medoid(paths):
    """The clip most similar to the others, with its score.

    All-pairs over a sample rather than the full dataset: the medoid of 16
    clips is a good enough representative, and 16 clips is 120 comparisons
    against 20k for 200.
    """
    if len(paths) < 3:
        return (paths[0] if paths else None), None, "too few clips"
    pairs, index = [], []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            pairs.append((paths[i], paths[j]))
            index.append((i, j))
    cos, err = ecapa(pairs)
    if err:
        return None, None, err
    scores = {i: [] for i in range(len(paths))}
    for (i, j), c in zip(index, cos or []):
        if c is None:
            continue
        scores[i].append(c)
        scores[j].append(c)
    medians = {i: statistics.median(v) for i, v in scores.items() if v}
    if not medians:
        return None, None, "no comparable pairs"
    best = max(medians, key=medians.get)
    return paths[best], round(medians[best], 4), None


def audit(zip_path, clips, rng, workdir):
    """-> (current_ref_score, best_ref_score, best_name, error)"""
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        meta = ("train/metadata.jsonl" if "train/metadata.jsonl" in names
                else "metadata.jsonl")
        if meta not in names:
            return None, None, None, "no metadata"
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
        ref_local = None
        if "ref.wav" in names:
            ref_local = os.path.join(workdir, "ref.wav")
            with open(ref_local, "wb") as fh:
                fh.write(z.read("ref.wav"))
    if not paths:
        return None, None, None, "no clips"
    best_path, best_score, err = choose_medoid(paths)
    if err:
        return None, None, None, err
    cur = None
    if ref_local:
        cos, e2 = ecapa([(ref_local, p) for p in paths])
        vals = [c for c in (cos or []) if c is not None]
        cur = round(statistics.median(vals), 4) if vals else None
    return cur, best_score, os.path.basename(best_path or ""), None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zips", default=os.environ.get(
        "ALEXANDRIA_VOICE_ZIPS",
        os.path.join(os.path.expanduser("~"), "Desktop", "zips2",
                     "_deduped_labeled")))
    ap.add_argument("--clips", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "dataset_ref_audit.json"))
    args = ap.parse_args()

    zips = sorted(glob.glob(os.path.join(args.zips, "*.zip")))
    if args.limit:
        zips = zips[:args.limit]
    rng = random.Random(args.seed)
    rows, improved = [], 0
    print(f"  {len(zips)} datasets, {args.clips} clips each\n")
    print(f"  {'dataset':52}{'current':>9}{'medoid':>9}{'gain':>8}")
    for zp in zips:
        work = tempfile.mkdtemp(prefix="refaudit_")
        cur, best, best_name, err = audit(zp, args.clips, rng, work)
        shutil.rmtree(work, ignore_errors=True)
        name = os.path.basename(zp)[:-4]
        rec = {"dataset": name, "current_ref_score": cur,
               "medoid_score": best, "medoid_clip": best_name, "error": err}
        rows.append(rec)
        if cur is not None and best is not None:
            gain = best - cur
            if gain > 0.15:
                improved += 1
            print(f"  {name[:51]:52}{cur:9.3f}{best:9.3f}{gain:+8.3f}")
    print(f"\n  datasets where the medoid beats the current ref by >0.15: "
          f"{improved}/{len([r for r in rows if r.get('current_ref_score') is not None])}")
    print("  Those are the datasets whose adapter is anchored to an "
          "unrepresentative clip.")

    doc = {"clips": args.clips, "seed": args.seed,
           "improved_over_threshold": improved, "results": rows}
    try:
        from experiments.provenance import provenance
        doc["provenance"] = provenance(__file__, args)
    except Exception as exc:                                # noqa: BLE001
        doc["provenance"] = {"error": str(exc)[:120]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
    print(f"\nwrote {args.out}")
    if not any(r.get("medoid_score") for r in rows):
        sys.exit(3)


if __name__ == "__main__":
    main()
