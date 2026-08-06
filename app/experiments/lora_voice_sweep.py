"""Which training choices actually make a cloned voice closer to the human?

The baseline runs MEASURE. This one tries to IMPROVE, which only became
possible once there was a target: the human reading the same line, and a
ceiling saying how close anything can get.

The current settings - 200 samples, lora_r 32, alpha 128, 6 epochs - are what
the library does, not what anything measured. This varies them ONE AT A TIME
from that baseline and scores each against ground truth.

    samples   50 / 100 / 200      is the 200 cap earning its cost?
    lora_r    16 / 32 / 64        capacity
    epochs    3 / 6 / 12          under- or over-fitting

One-at-a-time, not a full grid: 3x3x3 is 27 configurations and roughly six GPU
hours for a first look at which lever even moves. If two levers turn out to
matter, their interaction is worth a grid; assuming it in advance is not.

CHEAP METRIC ON PURPOSE. Scoring uses ECAPA and duration only, on a subset.
The full pass - DTW-aligned F0 contour and MCD - runs ~900 pyin extractions and
takes most of an hour per arm, which is right for a result and wrong for a
sweep. The winner gets the full treatment; the sweep only needs to rank.

WHAT A NULL RESULT MEANS HERE, said now rather than after. If no lever moves
ECAPA beyond the run-to-run spread, that is evidence the adapter is not the
thing limiting voice similarity - and given `clone_vs_lora` already found the
adapter behind plain zero-shot cloning, it would be consistent rather than
surprising. It would point at the reference clip, the base model, or the
premise.
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

BASELINE = {"samples": 200, "lora_r": 32, "lora_alpha": 128, "epochs": 6}


def configs(levers):
    """Baseline plus one-at-a-time variations. Deduplicated."""
    out, seen = [], set()

    def add(name, **over):
        cfg = dict(BASELINE)
        cfg.update(over)
        key = tuple(sorted(cfg.items()))
        if key in seen:
            return
        seen.add(key)
        cfg["name"] = name
        out.append(cfg)

    add("baseline")
    if "samples" in levers:
        for n in (50, 100):
            add(f"samples{n}", samples=n)
    if "rank" in levers:
        for r in (16, 64):
            # alpha tracks rank at the library's 4x ratio, so rank is varied
            # rather than rank-and-scaling-together being confounded.
            add(f"rank{r}", lora_r=r, lora_alpha=r * 4)
    if "epochs" in levers:
        for e in (3, 12):
            add(f"epochs{e}", epochs=e)
    return out


def run(cmd, log_path, timeout):
    with open(log_path, "w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                           timeout=timeout)
    return p.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", default=os.path.join(
        REPO, "ab_test_runtime", "ljspeech_eval", "build.json"))
    ap.add_argument("--work", default=os.path.join(
        REPO, "ab_test_runtime", "lora_sweep"))
    ap.add_argument("--levers", nargs="+",
                    default=["samples", "rank", "epochs"])
    ap.add_argument("--score-lines", type=int, default=50,
                    help="subset for ranking; the winner is re-scored in full")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "lora_voice_sweep.json"))
    args = ap.parse_args()

    build = json.load(open(args.build, encoding="utf-8"))
    train_dir = os.path.join(REPO, build["train_dir"])
    meta_src = os.path.join(train_dir, "metadata.jsonl")
    with open(meta_src, encoding="utf-8") as fh:
        all_meta = [l for l in fh if l.strip()]
    py = os.path.join(APP, "env", "bin", "python")
    os.makedirs(args.work, exist_ok=True)
    logs = os.path.join(REPO, "ab_test_runtime", "logs")
    os.makedirs(logs, exist_ok=True)

    plan = configs(args.levers)
    print(f"{len(plan)} configurations, one lever at a time from baseline")
    for c in plan:
        print(f"  {c['name']:12} samples={c['samples']:3} r={c['lora_r']:2} "
              f"alpha={c['lora_alpha']:3} epochs={c['epochs']:2}")
    print()

    results = []
    for c in plan:
        t0 = time.time()
        cdir = os.path.join(args.work, c["name"])
        adapter = os.path.join(cdir, "adapter")
        data = os.path.join(cdir, "train")
        os.makedirs(data, exist_ok=True)

        # A smaller sample count takes a PREFIX of the same ordered set, so
        # configurations differ by size alone and not by which clips.
        import shutil
        for line in all_meta[:c["samples"]]:
            name = json.loads(line)["audio_filepath"]
            src = os.path.join(train_dir, name)
            if os.path.exists(src) and not os.path.exists(os.path.join(data, name)):
                shutil.copy2(src, os.path.join(data, name))
        with open(os.path.join(data, "metadata.jsonl"), "w",
                  encoding="utf-8") as fh:
            fh.writelines(all_meta[:c["samples"]])
        for extra in ("ref.wav", "ref_text.txt"):
            s = os.path.join(train_dir, extra)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(data, extra))

        rc = run([py, "-u", os.path.join(APP, "train_lora.py"),
                  "--data_dir", data, "--output_dir", adapter,
                  "--epochs", str(c["epochs"]), "--lora_r", str(c["lora_r"]),
                  "--lora_alpha", str(c["lora_alpha"]),
                  "--seed", str(args.seed)],
                 os.path.join(logs, f"sweep_{c['name']}_train.log"), 14400)
        if rc != 0:
            results.append({**c, "error": f"train rc={rc}"})
            print(f"  {c['name']:12} TRAIN FAILED rc={rc}")
            continue

        gen_json = os.path.join(cdir, "generate.json")
        rc = run([py, "-u", os.path.join(APP, "experiments",
                                         "ljspeech_generate.py"),
                  "--build", args.build, "--adapter", adapter,
                  "--out-dir", os.path.join(cdir, "generated"),
                  "--arms", "lora", "--limit", str(args.score_lines),
                  "--seed", str(args.seed), "--out", gen_json],
                 os.path.join(logs, f"sweep_{c['name']}_gen.log"), 10800)
        if rc != 0 or not os.path.exists(gen_json):
            results.append({**c, "error": f"generate rc={rc}"})
            print(f"  {c['name']:12} GENERATE FAILED rc={rc}")
            continue

        # ECAPA only. The full metric pass is for the winner, not the ranking.
        doc = json.load(open(gen_json, encoding="utf-8"))
        pairs = [[os.path.join(REPO, r["human_wav"]),
                  os.path.join(REPO, r["lora_wav"])] for r in doc["rows"]]
        from experiments.ljspeech_score import ecapa_scores
        cos, err = ecapa_scores(pairs)
        if err:
            results.append({**c, "error": err})
            print(f"  {c['name']:12} SCORING FAILED: {err[:50]}")
            continue
        vals = [v for v in cos if v is not None]
        import statistics
        rec = {**c, "n": len(vals),
               "ecapa_mean": statistics.mean(vals) if vals else None,
               "ecapa_median": statistics.median(vals) if vals else None,
               "minutes": round((time.time() - t0) / 60, 1)}
        results.append(rec)
        print(f"  {c['name']:12} ecapa {rec['ecapa_mean']:.4f}  "
              f"n={rec['n']}  {rec['minutes']}m")

    ok = [r for r in results if r.get("ecapa_mean") is not None]
    print()
    if ok:
        base = next((r for r in ok if r["name"] == "baseline"), None)
        ok.sort(key=lambda r: -r["ecapa_mean"])
        print(f"  {'config':12}{'ecapa':>9}{'vs baseline':>13}")
        for r in ok:
            d = (f"{r['ecapa_mean'] - base['ecapa_mean']:+.4f}"
                 if base else "-")
            print(f"  {r['name']:12}{r['ecapa_mean']:9.4f}{d:>13}")
        print("\n  Ceiling for reference: the same narrator on a different "
              "line\n  scored 0.805. A configuration is only worth adopting if "
              "it moves\n  meaningfully toward that, not merely up.")

    out = {"baseline": BASELINE, "levers": args.levers,
           "score_lines": args.score_lines, "seed": args.seed,
           "results": results}
    try:
        from experiments.provenance import provenance
        out["provenance"] = provenance(__file__, args)
    except Exception as exc:                            # noqa: BLE001
        out["provenance"] = {"error": str(exc)[:120]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
