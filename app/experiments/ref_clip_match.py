"""Is each adapter's reference clip actually its own narrator?

WHY THIS COULD EXPLAIN THE FAILURES. `train_lora.py` extracts the speaker
embedding from ONE reference clip and uses it for every training sample - the
official Qwen3-TTS fine-tuning approach. So a single wrong file sets the voice
identity for all 200 samples, and no amount of clean training audio overrides
it.

That is the exact signature of the unexplained group: adapters whose training
data is verifiably one speaker (0.79-0.81 internal consistency, as good as the
working ones) that nevertheless produced a voice resembling nobody (0.027,
0.061). Same learning rate, same sample count, same epochs, same final loss as
the adapters that worked.

A first look at eight adapters was suggestive and not clean: three of the five
worst had a reference that did not match their own training data (-0.012,
0.036, 0.043), but `warm_tenor_20s_m` scored 0.725 with a mismatched reference
of 0.091. One counter-example is why this runs over the whole library instead
of being written up from eight.

`ref_sample.wav` is the INPUT reference, not a generated demo - train_lora.py
copies it with the comment "Copy reference audio as ref_sample.wav for
inference". That was checked, because measuring adapter output and calling it a
cause would invert the whole argument.

NO GENERATION. This compares files already on disk.
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
    if not pairs:
        return None, "no pairs"
    if not os.path.exists(SIBLING_PY):
        return None, "sibling interpreter missing"
    script = os.path.join(APP, "experiments", "_ecapa_batch.py")
    try:
        # ABSOLUTE paths: the subprocess runs with cwd=APP, and passing
        # repo-relative paths silently resolved to app/lora_models/... which
        # produced a table of nulls that looked like an ECAPA failure.
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


def find_zip(dataset, idx):
    key = "".join(c for c in dataset.lower() if c.isalnum())
    return next((p for zkey, p in idx.items() if zkey.startswith(key)), None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=os.path.join(REPO, "lora_models"))
    ap.add_argument("--zips", default=DEFAULT_ZIPS)
    ap.add_argument("--fidelity", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "library_voice_fidelity.json"))
    ap.add_argument("--clips", type=int, default=10)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "ref_clip_match.json"))
    args = ap.parse_args()

    scores = {}
    if os.path.exists(args.fidelity):
        scores = {r["adapter"]: r.get("ecapa")
                  for r in json.load(open(args.fidelity,
                                          encoding="utf-8"))["results"]}
    idx = {"".join(c for c in os.path.basename(p).lower() if c.isalnum()): p
           for p in glob.glob(os.path.join(args.zips, "*.zip"))}
    rng = random.Random(args.seed)

    rows = []
    print(f"  {'adapter':36}{'adapter':>9}{'ref vs own':>12}")
    print(f"  {'':36}{'ecapa':>9}{'training':>12}\n")
    for d in sorted(glob.glob(os.path.join(args.models, "*/"))):
        name = os.path.basename(d.rstrip("/"))
        ref = os.path.join(d, "ref_sample.wav")
        meta_p = os.path.join(d, "training_meta.json")
        if not (os.path.exists(ref) and os.path.exists(meta_p)):
            rows.append({"adapter": name, "error": "no ref or meta"})
            continue
        meta = json.load(open(meta_p, encoding="utf-8"))
        dataset = os.path.basename(os.path.dirname(
            str(meta.get("ref_sample_audio") or "")))
        zp = find_zip(dataset, idx)
        if not zp:
            rows.append({"adapter": name, "dataset": dataset,
                         "error": "no zip"})
            continue
        work = tempfile.mkdtemp(prefix="refmatch_")
        with zipfile.ZipFile(zp) as z:
            names = set(z.namelist())
            meta_name = ("train/metadata.jsonl" if "train/metadata.jsonl"
                         in names else "metadata.jsonl")
            if meta_name not in names:
                rows.append({"adapter": name, "error": "no metadata"})
                continue
            entries = [json.loads(l) for l in
                       z.read(meta_name).decode("utf-8").splitlines()
                       if l.strip()]
            rng.shuffle(entries)
            paths = []
            for e in entries[:args.clips]:
                fp = e.get("audio_filepath")
                if fp and fp in names:
                    dest = os.path.join(work, os.path.basename(fp))
                    with open(dest, "wb") as fh:
                        fh.write(z.read(fp))
                    paths.append(dest)
        cos, err = ecapa([(ref, p) for p in paths])
        vals = [c for c in (cos or []) if c is not None]
        match = round(statistics.median(vals), 4) if vals else None
        a = scores.get(name)
        rows.append({"adapter": name, "dataset": dataset,
                     "ref_vs_training": match, "adapter_ecapa": a,
                     "clips": len(vals), "error": err})
        print(f"  {name[:35]:36}"
              f"{a if a is not None else float('nan'):9.3f}"
              f"{match if match is not None else float('nan'):12.3f}")

    paired = [r for r in rows if r.get("ref_vs_training") is not None
              and r.get("adapter_ecapa") is not None]
    doc = {"clips_per_adapter": args.clips, "seed": args.seed, "results": rows}
    if len(paired) > 4:
        import numpy as np
        x = np.array([r["ref_vs_training"] for r in paired])
        y = np.array([r["adapter_ecapa"] for r in paired])
        r = float(np.corrcoef(x, y)[0, 1])
        bad_ref = [p for p in paired if p["ref_vs_training"] < 0.3]
        bad_both = [p for p in bad_ref if p["adapter_ecapa"] < 0.3]
        doc["correlation"] = round(r, 4)
        doc["mismatched_refs"] = len(bad_ref)
        doc["mismatched_ref_and_failed_adapter"] = len(bad_both)
        print(f"\n  n={len(paired)}  correlation(ref match, adapter quality) "
              f"= {r:+.3f}")
        print(f"  adapters with a mismatched reference (<0.3): {len(bad_ref)}")
        print(f"  of those, adapter also failed (<0.3): {len(bad_both)}")
        print("\n  A strong positive correlation means the reference clip is a")
        print("  real cause and those adapters need a REF fix, not a rerun.")
        print("  A weak one means the reference is not what is going wrong.")

    try:
        from experiments.provenance import provenance
        doc["provenance"] = provenance(__file__, args)
    except Exception as exc:                                # noqa: BLE001
        doc["provenance"] = {"error": str(exc)[:120]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
    print(f"\nwrote {args.out}")
    if not paired:
        sys.exit(3)


if __name__ == "__main__":
    main()
