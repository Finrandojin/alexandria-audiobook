#!/usr/bin/env python3
"""Generate, resume, and strictly validate Stage 6 listening materials."""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

from experiments.blinded_listening import (  # noqa: E402
    _load_document, _resolve_source, validate_package)
from experiments.provenance import input_sha256, provenance  # noqa: E402

PYTHON = os.path.join(APP, "env", "bin", "python")
GPU_JOB = os.path.join(REPO, "gpu_job.sh")
SCENE = os.path.join(
    REPO, "ab_test_runtime", "experiments", "stage6_scene_aware_casting.json")
INSTRUCTION = os.path.join(
    REPO, "ab_test_runtime", "experiments", "stage6_instruction_source.json")
INSTRUCTION_DIR = os.path.join(REPO, "ab_test_runtime", "stage6_instruction")
CASTING_DIR = os.path.join(REPO, "ab_test_runtime", "stage6_casting")
CASTING = os.path.join(CASTING_DIR, "manifest.json")
PUBLIC = os.path.join(
    REPO, "ab_test_runtime", "experiments", "blinded_listening.json")
PACKAGE_DIR = os.path.join(REPO, "ab_test_runtime", "blinded_listening")
KEY = os.path.join(
    REPO, "ab_test_runtime", "blinded_listening_concealed_key.json")
CONTROLS = os.path.join(
    REPO, "ab_test_runtime", "experiments", "seed_instruction_controls.json")
CHUNKS = os.path.join(REPO, "chunks.json")
VOICE_CONFIG = os.path.join(REPO, "voice_config.json")
CONFIG = os.path.join(APP, "config.json")
ALIASES = os.path.join(REPO, "character_aliases.json")
LORA_MANIFEST = os.path.join(REPO, "lora_models", "manifest.json")
CURRENT_HARNESS = provenance(__file__)["git"]["harness_sha256"]


def run(command, cwd=REPO, stdout=None):
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True, stdout=stdout,
                   stderr=subprocess.STDOUT if stdout else None)


def _relative(path):
    return os.path.relpath(path, REPO)


def require_identity(path, script, expected_args, expected_inputs):
    """Accept a checkpoint only when code, options, and inputs are identical."""
    doc = _load_document(path, script)
    if doc.get("status") != "complete":
        raise RuntimeError(f"checkpoint is not complete: {path}")
    recorded = doc["provenance"]
    if recorded.get("git", {}).get("harness_sha256") != CURRENT_HARNESS:
        raise RuntimeError(f"checkpoint harness changed: {path}")
    if recorded.get("args") != expected_args:
        raise RuntimeError(
            f"checkpoint arguments changed: {path}\n"
            f"recorded={recorded.get('args')!r}\nexpected={expected_args!r}")
    if recorded.get("input_sha256") != expected_inputs:
        raise RuntimeError(f"checkpoint inputs changed: {path}")
    return doc


def validate_instruction(path):
    args = {"script": _relative(CHUNKS), "voice_config": _relative(VOICE_CONFIG),
            "config": _relative(CONFIG), "speaker": "", "lines": 4,
            "seed": 1234, "out_dir": _relative(INSTRUCTION_DIR),
            "out": _relative(INSTRUCTION)}
    doc = require_identity(
        path, "instruct_listening.py", args,
        input_sha256((CHUNKS, VOICE_CONFIG, CONFIG)))
    if not doc.get("all_arms_rendered"):
        raise RuntimeError("instruction checkpoint did not render every arm")
    comparisons = doc.get("comparisons") or []
    if len(comparisons) != 4:
        raise RuntimeError("instruction checkpoint must have four comparisons")
    for index, comparison in enumerate(comparisons):
        files = comparison.get("arm_files") or {}
        if set(files) != {"none", "per_char", "per_line"}:
            raise RuntimeError(f"instruction comparison {index} has wrong arms")
        for path_value in files.values():
            _resolve_source(path_value)
    return doc


def validate_casting(path):
    args = {"script": _relative(CHUNKS), "voice_config": _relative(VOICE_CONFIG),
            "config": _relative(CONFIG), "casting": _relative(SCENE),
            "characters": ["FELT", "REINHARD"], "size": 14,
            "out_dir": _relative(CASTING_DIR), "seed": 1234}
    doc = require_identity(
        path, "casting_ab_audio.py", args,
        input_sha256((CHUNKS, VOICE_CONFIG, CONFIG, SCENE, ALIASES,
                      LORA_MANIFEST)))
    if not doc.get("published") or doc.get("lines") != 14:
        raise RuntimeError("casting checkpoint is incomplete")
    arms = doc.get("arms") or {}
    if set(arms) != {"current", "scene_aware"}:
        raise RuntimeError("casting checkpoint has wrong arms")
    expected = list(range(14))
    for name, arm in arms.items():
        if arm.get("lines") != expected:
            raise RuntimeError(f"casting arm {name} has unequal line coverage")
        _resolve_source(arm.get("path", ""))
    return doc


def validate_scene(path):
    args = {"script": _relative(CHUNKS), "voice_config": _relative(VOICE_CONFIG),
            "manifest": _relative(LORA_MANIFEST), "window": 20,
            "aliases": _relative(ALIASES), "out": _relative(SCENE)}
    doc = require_identity(
        path, "scene_aware_casting.py", args,
        input_sha256((CHUNKS, VOICE_CONFIG, LORA_MANIFEST, ALIASES)))
    if not isinstance(doc.get("scene_aware", {}).get("assignment"), dict):
        raise RuntimeError("scene-aware checkpoint lacks an assignment")
    return doc


def run_gpu(name, timeout, script, arguments):
    log_path = os.path.join(REPO, "ab_test_runtime", "logs", "stage6_listening.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        run([GPU_JOB, name, "timeout", str(timeout), PYTHON, "-u", script,
             *arguments], cwd=APP, stdout=log)


def ensure_scene():
    if os.path.exists(SCENE):
        validate_scene(SCENE)
        print("RESUME: strictly validated scene assignment", flush=True)
        return
    run([PYTHON, "experiments/scene_aware_casting.py", "--out", SCENE], cwd=APP)
    validate_scene(SCENE)


def ensure_instruction():
    if os.path.exists(INSTRUCTION):
        validate_instruction(INSTRUCTION)
        print("RESUME: strictly validated instruction audio", flush=True)
        return
    if os.path.exists(INSTRUCTION_DIR):
        raise RuntimeError(
            f"instruction audio exists without a valid manifest: {INSTRUCTION_DIR}")
    run_gpu("stage6_instruction_source", 7200,
            "experiments/instruct_listening.py",
            ["--out-dir", INSTRUCTION_DIR, "--out", INSTRUCTION])
    validate_instruction(INSTRUCTION)


def ensure_casting():
    if os.path.exists(CASTING):
        validate_casting(CASTING)
        print("RESUME: strictly validated casting audio", flush=True)
        return
    if os.path.exists(CASTING_DIR):
        raise RuntimeError(
            f"casting audio exists without a valid manifest: {CASTING_DIR}")
    run_gpu("stage6_casting_source", 7200,
            "experiments/casting_ab_audio.py",
            ["--casting", SCENE, "--out-dir", CASTING_DIR])
    validate_casting(CASTING)


def ensure_package():
    existing = [path for path in (PUBLIC, KEY, PACKAGE_DIR) if os.path.exists(path)]
    if existing:
        if len(existing) != 3:
            raise RuntimeError("partial blind package exists: " + ", ".join(existing))
        public, key = validate_package(PUBLIC, KEY, PACKAGE_DIR)
        if public["provenance"]["git"]["harness_sha256"] != CURRENT_HARNESS:
            raise RuntimeError("blind package harness changed")
        if key.get("randomization_seed") != 20260804:
            raise RuntimeError("blind package randomization seed changed")
        print("RESUME: strictly validated complete blind package", flush=True)
        return
    run([PYTHON, "experiments/blinded_listening.py",
         "--instruction", INSTRUCTION, "--casting", CASTING,
         "--package-dir", PACKAGE_DIR, "--out", PUBLIC, "--key", KEY], cwd=APP)
    public, key = validate_package(PUBLIC, KEY, PACKAGE_DIR)
    if public["provenance"]["git"]["harness_sha256"] != CURRENT_HARNESS or \
            key.get("randomization_seed") != 20260804:
        raise RuntimeError("new blind package identity is wrong")
    print("Stage 6 blind package validated strictly (8/8 sets).", flush=True)


def main():
    _load_document(CONTROLS, "seed_instruction_controls.py")
    ensure_scene()
    ensure_instruction()
    ensure_casting()
    ensure_package()
    run([PYTHON, "collect_results.py"])
    run([PYTHON, "-m", "unittest", "discover", "-s", "app", "-p", "test_*.py"])
    print("Stage 6 listening-material checkpoint complete.", flush=True)


if __name__ == "__main__":
    main()
