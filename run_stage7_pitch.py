#!/usr/bin/env python3
"""Run, resume, and strictly validate the Stage 7 pitch matrix."""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

from experiments.pitch_profile_matrix import (  # noqa: E402
    load_adapters, validate_artifact)
from local_gpu_job import run_gpu_job  # noqa: E402

PYTHON = os.path.join(APP, "env", "bin", "python")
MANIFEST = os.path.join(REPO, "lora_models", "manifest.json")
PILOT = os.path.join(
    REPO, "ab_test_runtime", "experiments", "pitch_profile_matrix_pilot.json")
PILOT_DIR = os.path.join(REPO, "ab_test_runtime", "pitch_profile_matrix_pilot")
FULL = os.path.join(
    REPO, "ab_test_runtime", "experiments", "pitch_profile_matrix.json")
FULL_DIR = os.path.join(REPO, "ab_test_runtime", "pitch_profile_matrix")


def run(command):
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def require_index_entries(*filenames):
    for index_name in ("RESULTS_INDEX.md", "results_index.csv"):
        with open(os.path.join(REPO, index_name), encoding="utf-8") as handle:
            contents = handle.read()
        missing = [name for name in filenames if name not in contents]
        if missing:
            raise RuntimeError(
                f"{index_name} is missing {', '.join(missing)}")


def require_pilot_gate(doc):
    measured = sum(row.get("pitch_status") == "measured" for row in doc["rows"])
    if measured < 3:
        raise RuntimeError(
            f"Stage 7 pilot gate failed: only {measured}/4 pitch tracks measured")


def ensure_pilot(adapter):
    if not os.path.exists(PILOT):
        run_gpu_job(
            REPO, APP, PYTHON, "stage7_pitch_pilot", 3600,
            "experiments/pitch_profile_matrix.py",
            ["--adapters", adapter, "--seeds", "1234", "5678",
             "--categories", "narration", "plain_dialogue",
             "--out-dir", PILOT_DIR, "--out", PILOT],
            "stage7_pitch.log")
    doc = validate_artifact(PILOT, expected_count=4)
    require_pilot_gate(doc)
    print("Stage 7 pilot validated strictly and cleared (4/4 rows).", flush=True)


def ensure_full(adapter_count):
    expected = adapter_count * 3 * 6
    if not os.path.exists(FULL):
        run_gpu_job(
            REPO, APP, PYTHON, "stage7_pitch_full", 129600,
            "experiments/pitch_profile_matrix.py",
            ["--out-dir", FULL_DIR, "--out", FULL], "stage7_pitch.log")
    validate_artifact(FULL, expected_count=expected)
    print(f"Stage 7 full matrix validated strictly ({expected}/{expected} rows).",
          flush=True)


def main():
    adapters = load_adapters(MANIFEST)
    if len(adapters) != 75:
        raise RuntimeError(
            f"Stage 7 expected 75 usable adapters; found {len(adapters)}")
    ensure_pilot(adapters[0]["adapter"])
    ensure_full(len(adapters))
    run([PYTHON, "audit_experiment_artifacts.py"])
    run([PYTHON, "collect_results.py"])
    require_index_entries("pitch_profile_matrix_pilot.json",
                          "pitch_profile_matrix.json")
    run([PYTHON, "-m", "unittest", "discover", "-s", "app",
         "-p", "test_*.py"])
    print("Stage 7 pitch checkpoint complete.", flush=True)


if __name__ == "__main__":
    main()
