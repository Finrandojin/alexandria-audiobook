#!/usr/bin/env python3
"""Validate and finish the Stage 4 non-prose replication checkpoint."""
import hashlib
import json
import os
import subprocess
import sys

from local_gpu_job import run_gpu_job


REPO = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)
PILOT = os.path.join(
    REPO, "ab_test_runtime", "experiments", "nonprose_replication_pilot.json")
FULL = os.path.join(
    REPO, "ab_test_runtime", "experiments", "nonprose_replication.json")
EXPANSION_PILOT = os.path.join(
    REPO, "ab_test_runtime", "experiments",
    "nonprose_category_expansion_pilot.json")
EXPANSION_FULL = os.path.join(
    REPO, "ab_test_runtime", "experiments",
    "nonprose_category_expansion.json")
DEFAULT_ADAPTERS = (
    "husky_tenor_30s_m_fantasy",
    "husky_soprano_20s_f",
    "warm_baritone_50s_m_gothic",
)
DEFAULT_SEEDS = (1234, 5678, 9012)


class ArtifactValidationError(RuntimeError):
    """A Stage 4 artifact cannot support a completed checkpoint."""


def _provenance_harness_matches(provenance):
    from experiments.provenance import get_reproducible_harness_source
    return get_reproducible_harness_source(provenance, REPO) is not None


def _read_wav_fully(path):
    import soundfile as sf
    from experiments.generation import _check_riff_completeness
    info = sf.info(path)
    if not info.frames or not info.samplerate or not info.channels:
        raise ArtifactValidationError(f"WAV has no usable audio: {path}")
    with sf.SoundFile(path) as handle:
        while handle.read(65536).size:
            pass
    _check_riff_completeness(path, "lora", "Stage 4 validation")


def validate_stage4_artifact(
        path, expected_rows, required_matrix=None,
        expected_script="nonprose_replication.py",
        class_prefixes=None, extra_arg_fields=("limit",),
        expected_categories=None, summary_function=None):
    """Return a complete artifact or raise with the first concrete defect."""
    try:
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ArtifactValidationError(f"cannot read {path}: {exc}") from exc
    rows = doc.get("rows")
    if doc.get("status") != "complete":
        raise ArtifactValidationError(f"artifact status is {doc.get('status')!r}")
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise ArtifactValidationError(
            f"artifact has {len(rows) if isinstance(rows, list) else 'non-list'} "
            f"rows; expected {expected_rows}")

    provenance = doc.get("provenance")
    if not isinstance(provenance, dict):
        raise ArtifactValidationError("artifact has no provenance object")
    for field in ("script", "written", "host", "git", "args"):
        if field not in provenance:
            raise ArtifactValidationError(f"provenance is missing {field}")
    if provenance["script"] != expected_script:
        raise ArtifactValidationError(
            f"unexpected provenance script {provenance['script']!r}")
    if not _provenance_harness_matches(provenance):
        raise ArtifactValidationError(
            "provenance harness hash cannot be reproduced from its commit "
            "or the current harness")

    args = provenance.get("args") or {}
    for field in (("source", "config", "adapters", "seeds", "out_dir", "out")
                  + tuple(extra_arg_fields)):
        if field not in args:
            raise ArtifactValidationError(
                f"provenance arguments are missing {field}")
    adapters = tuple(args["adapters"])
    seeds = tuple(args["seeds"])
    class_prefixes = class_prefixes or {"nonprose": "nonprose",
                                        "prose": "prose"}
    classes = tuple(class_prefixes)
    pair_manifest = ((doc.get("selection") or {}).get("pairs") or [])
    pair_count = len(pair_manifest)
    if "limit" in extra_arg_fields and args["limit"] != pair_count:
        raise ArtifactValidationError(
            "provenance limit does not match the selection pair count")
    if expected_categories is not None:
        selected_categories = tuple((doc.get("selection") or {}).get(
            "categories") or ())
        if selected_categories != tuple(expected_categories):
            raise ArtifactValidationError("selection categories are wrong")
        wanted_pairs = args["limit_per_category"] * len(expected_categories)
        if pair_count != wanted_pairs:
            raise ArtifactValidationError(
                f"selection has {pair_count} pairs; expected {wanted_pairs}")
    inferred = {(adapter, seed, pair, label)
                for adapter in adapters for seed in seeds
                for pair in range(pair_count)
                for label in classes}
    if required_matrix is not None and inferred != required_matrix:
        raise ArtifactValidationError("provenance arguments do not match the fixed matrix")

    keys = []
    pair_inputs = {}
    for index, pair in enumerate(pair_manifest):
        for label, prefix in class_prefixes.items():
            try:
                pair_inputs[(index, label)] = (
                    pair[f"{prefix}_uid"], pair[f"{prefix}_sha256"])
            except KeyError as exc:
                raise ArtifactValidationError(
                    f"selection pair {index} is missing {exc.args[0]}") from exc
            text = pair.get(f"{prefix}_text")
            if text is not None and _sha256_text(text) != pair[f"{prefix}_sha256"]:
                raise ArtifactValidationError(
                    f"selection pair {index} {prefix} text hash is wrong")
        feature_prefix = next(prefix for label, prefix in class_prefixes.items()
                              if label != "prose")
        for field in (f"{feature_prefix}_features", "prose_features",
                      "absolute_feature_gap"):
            if field not in pair:
                raise ArtifactValidationError(
                    f"selection pair {index} is missing {field}")

    required = {"words", "errors", "failed", "substitutions", "deletions",
                "insertions", "adapter", "seed", "pair", "class", "uid",
                "source_sha256", "wav", "transcript"}
    repo_real = os.path.realpath(REPO)
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ArtifactValidationError(
                f"row {index} is missing {', '.join(sorted(missing))}")
        key = (row["adapter"], row["seed"], row["pair"], row["class"])
        keys.append(key)
        if (row["uid"], row["source_sha256"]) != pair_inputs.get(
                (row["pair"], row["class"])):
            raise ArtifactValidationError(f"row {index} input identity is wrong")
        if expected_categories is not None:
            wanted_category = pair_manifest[row["pair"]].get("category")
            if row.get("category") != wanted_category:
                raise ArtifactValidationError(
                    f"row {index} category does not match its pair")
        total = row["substitutions"] + row["deletions"] + row["insertions"]
        if row["errors"] != total:
            raise ArtifactValidationError(f"row {index} error breakdown is wrong")
        wav = os.path.realpath(os.path.join(REPO, row["wav"]))
        if os.path.commonpath((repo_real, wav)) != repo_real:
            raise ArtifactValidationError(f"row {index} WAV escapes the repository")
        if not os.path.isfile(wav) or os.path.getsize(wav) <= 44:
            raise ArtifactValidationError(f"row {index} WAV is missing or empty")
        try:
            _read_wav_fully(wav)
        except Exception as exc:  # noqa: BLE001
            raise ArtifactValidationError(
                f"row {index} WAV is not fully decodable: {exc}") from exc
    if len(set(keys)) != len(keys) or set(keys) != inferred:
        raise ArtifactValidationError("matrix keys are duplicated, missing, or foreign")

    if summary_function is None:
        from experiments.nonprose_replication import summarize as summary_function
    if doc.get("summary") != summary_function(rows):
        raise ArtifactValidationError("summary does not exactly recompute from rows")
    return doc


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(command, cwd=REPO, env=None, stdout=None):
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, stdout=stdout,
                   stderr=subprocess.STDOUT if stdout else None, check=True)


def run_gpu_experiment(name, timeout_seconds, script, arguments, log_name):
    run_gpu_job(REPO, APP, sys.executable, name, timeout_seconds, script,
                arguments, log_name)


def require_index_entries(*filenames):
    for index_name in ("RESULTS_INDEX.md", "results_index.csv"):
        path = os.path.join(REPO, index_name)
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        missing = [name for name in filenames if name not in text]
        if missing:
            raise ArtifactValidationError(
                f"{index_name} is missing {', '.join(missing)}")


def main():
    validate_stage4_artifact(PILOT, 4)
    print("Stage 4 pilot validated strictly (4/4 rows).", flush=True)
    matrix = {(adapter, seed, pair, label)
              for adapter in DEFAULT_ADAPTERS for seed in DEFAULT_SEEDS
              for pair in range(8) for label in ("nonprose", "prose")}
    if not os.path.exists(FULL):
        run_gpu_experiment(
            "nonprose_replication_full", 43200,
            "experiments/nonprose_replication.py", [],
            "nonprose_replication.log")
    validate_stage4_artifact(FULL, 144, matrix)
    print("Stage 4 full matrix validated strictly (144/144 rows).", flush=True)

    from experiments.nonprose_category_expansion import (
        CATEGORIES, summarize as category_summarize)
    expansion_classes = {"probe": "probe", "prose": "prose"}
    pilot_matrix = {(DEFAULT_ADAPTERS[0], 1234, pair, label)
                    for pair in range(len(CATEGORIES))
                    for label in expansion_classes}
    if not os.path.exists(EXPANSION_PILOT):
        run_gpu_experiment(
            "nonprose_category_expansion_pilot", 3600,
            "experiments/nonprose_category_expansion.py",
            ["--adapters", DEFAULT_ADAPTERS[0], "--seeds", "1234",
             "--limit-per-category", "1",
             "--out-dir", os.path.join(
                 REPO, "ab_test_runtime",
                 "nonprose_category_expansion_pilot"),
             "--out", EXPANSION_PILOT],
            "nonprose_category_expansion.log")
    validate_stage4_artifact(
        EXPANSION_PILOT, 12, pilot_matrix,
        expected_script="nonprose_category_expansion.py",
        class_prefixes=expansion_classes,
        extra_arg_fields=("limit_per_category",),
        expected_categories=CATEGORIES, summary_function=category_summarize)
    print("Stage 4 category pilot validated strictly (12/12 rows).", flush=True)

    expansion_matrix = {(adapter, seed, pair, label)
                        for adapter in DEFAULT_ADAPTERS
                        for seed in DEFAULT_SEEDS for pair in range(24)
                        for label in expansion_classes}
    if not os.path.exists(EXPANSION_FULL):
        run_gpu_experiment(
            "nonprose_category_expansion_full", 43200,
            "experiments/nonprose_category_expansion.py", [],
            "nonprose_category_expansion.log")
    validate_stage4_artifact(
        EXPANSION_FULL, 432, expansion_matrix,
        expected_script="nonprose_category_expansion.py",
        class_prefixes=expansion_classes,
        extra_arg_fields=("limit_per_category",),
        expected_categories=CATEGORIES, summary_function=category_summarize)
    print("Stage 4 category expansion validated strictly (432/432 rows).",
          flush=True)

    run([sys.executable, "collect_results.py"])
    require_index_entries("nonprose_replication.json",
                          "nonprose_replication_pilot.json",
                          "nonprose_category_expansion.json",
                          "nonprose_category_expansion_pilot.json")
    print("Both results indexes explicitly contain the Stage 4 artifacts.",
          flush=True)
    run([sys.executable, "-m", "unittest",
         "discover", "-s", "app", "-p", "test_*.py"])
    print("Stage 4 local-test checkpoint complete.", flush=True)


if __name__ == "__main__":
    main()
