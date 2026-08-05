"""Run isolated baseline/native SIMD arms sequentially and publish hard evidence."""
import argparse
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(APP)
WORKER = os.path.join(APP, "experiments", "simd_benchmark_worker.py")
if APP not in sys.path:
    sys.path.insert(0, APP)
DISABLED = "AVX2,FMA3,AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL"


def _run_worker(arm, profile):
    env = os.environ.copy()
    env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1"})
    if arm == "baseline":
        env["NPY_DISABLE_CPU_FEATURES"] = DISABLED
    else:
        env.pop("NPY_DISABLE_CPU_FEATURES", None)
    completed = subprocess.run(
        [sys.executable, WORKER, "--arm", arm, "--profile", profile],
        env=env, cwd=APP, capture_output=True, text=True, check=False,
        timeout=5400 if profile == "full" else 300)
    if completed.returncode:
        raise RuntimeError(f"{arm} worker failed: {completed.stderr[-2000:]}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    try:
        return json.loads(lines[-1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"{arm} worker returned no JSON: {completed.stdout[-1000:]}") from exc


def _bootstrap_ratio(baseline, native, seed=20260805, draws=10000):
    rng = random.Random(seed)
    ratios = []
    for _ in range(draws):
        left = [rng.choice(baseline) for _ in baseline]
        right = [rng.choice(native) for _ in native]
        ratios.append(statistics.median(left) / statistics.median(right))
    ratios.sort()
    return ratios[int(draws * .025)], ratios[int(draws * .975)]


def summarize(runs):
    by_arm = {arm: [run for run in runs if run["arm"] == arm]
              for arm in ("baseline", "native")}
    for run in by_arm["baseline"]:
        if run["cpu_features"].get("AVX2") or run["cpu_features"].get("AVX512F"):
            raise RuntimeError("baseline arm did not disable advanced SIMD")
    for run in by_arm["native"]:
        if not run["cpu_features"].get("AVX2"):
            raise RuntimeError("native arm does not report AVX2")
    names = [case["name"] for case in runs[0]["cases"]]
    results = []
    for index, name in enumerate(names):
        signatures = [run["cases"][index]["result_signature"] for run in runs]
        if any(signature["shape"] != signatures[0]["shape"]
               for signature in signatures[1:]):
            raise RuntimeError(f"output shape mismatch across SIMD arms: {name}")
        fields = ("mean", "rms", "minimum", "maximum")
        reference = [signatures[0][field] for field in fields]
        for signature in signatures[1:]:
            values = [signature[field] for field in fields]
            if any(not math.isclose(left, right, rel_tol=2e-5, abs_tol=1e-7)
                   for left, right in zip(reference, values)):
                raise RuntimeError(f"output mismatch across SIMD arms: {name}")
        baseline = [timing for run in by_arm["baseline"]
                    for timing in run["cases"][index]["timings_ns"]]
        native = [timing for run in by_arm["native"]
                  for timing in run["cases"][index]["timings_ns"]]
        low, high = _bootstrap_ratio(baseline, native, seed=20260805 + index)
        speedup = statistics.median(baseline) / statistics.median(native)
        verdict = "native_faster" if low > 1 else \
            "native_slower" if high < 1 else "not_proven"
        results.append({"name": name, "baseline_median_ns": statistics.median(baseline),
                        "native_median_ns": statistics.median(native),
                        "native_speedup": speedup,
                        "speedup_bootstrap_95pct": [low, high], "verdict": verdict})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "simd_benchmark.json"))
    args = parser.parse_args()
    if os.path.exists(args.out):
        raise RuntimeError(f"refusing to overwrite existing {args.out}")
    order = ("baseline", "native", "native", "baseline")
    runs = []
    for position, arm in enumerate(order, 1):
        print(f"[{position}/{len(order)}] {arm}", flush=True)
        runs.append(_run_worker(arm, args.profile))
    from experiments.provenance import provenance
    from utils import atomic_json_write
    artifact = {"status": "complete", "profile": args.profile,
                "arm_order": list(order), "host": platform.node(),
                "interpretation": {
                    "native_faster": "95% bootstrap interval for baseline/native median ratio is entirely above 1.",
                    "native_slower": "95% bootstrap interval is entirely below 1.",
                    "not_proven": "interval overlaps 1; this run does not prove a SIMD benefit.",
                    "scope": "Results apply to this CPU, NumPy/SciPy build, and these measured workloads only."},
                "results": summarize(runs), "runs": runs,
                "provenance": provenance(__file__, args)}
    atomic_json_write(artifact, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
