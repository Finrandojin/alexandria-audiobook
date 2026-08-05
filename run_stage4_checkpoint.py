#!/usr/bin/env python3
"""Validate and finish the Stage 4 non-prose replication checkpoint."""
import hashlib
import json
import os
import subprocess
import sys


REPO = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)
PILOT = os.path.join(
    REPO, "ab_test_runtime", "experiments", "nonprose_replication_pilot.json")
FULL = os.path.join(
    REPO, "ab_test_runtime", "experiments", "nonprose_replication.json")
DEFAULT_ADAPTERS = (
    "husky_tenor_30s_m_fantasy",
    "husky_soprano_20s_f",
    "warm_baritone_50s_m_gothic",
)
DEFAULT_SEEDS = (1234, 5678, 9012)


class ArtifactValidationError(RuntimeError):
    """A Stage 4 artifact cannot support a completed checkpoint."""


def _git_harness_fingerprint(commit):
    """Reconstruct manifest._source_fingerprint for a recorded commit."""
    names = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", commit, "--",
         "app/experiments"], cwd=REPO, capture_output=True, check=True,
        text=True).stdout.splitlines()
    names = sorted(name for name in names
                   if name.startswith("app/experiments/")
                   and "/" not in name[len("app/experiments/"):]
                   and name.endswith(".py"))
    digest = hashlib.sha256()
    for path in names:
        content = subprocess.run(
            ["git", "show", f"{commit}:{path}"], cwd=REPO,
            capture_output=True, check=True).stdout
        digest.update(os.path.basename(path).encode("utf-8"))
        digest.update(content)
    return digest.hexdigest()


def _provenance_harness_matches(provenance):
    git = provenance.get("git") or {}
    recorded = git.get("harness_sha256")
    commit = git.get("commit")
    if not recorded or len(recorded) != 64:
        return False
    try:
        if commit and _git_harness_fingerprint(commit) == recorded:
            return True
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        from experiments.manifest import _source_fingerprint
        return _source_fingerprint(os.path.join(APP, "experiments")) == recorded
    except (OSError, ImportError):
        return False


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


def validate_stage4_artifact(path, expected_rows, required_matrix=None):
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
    if provenance["script"] != "nonprose_replication.py":
        raise ArtifactValidationError(
            f"unexpected provenance script {provenance['script']!r}")
    if not _provenance_harness_matches(provenance):
        raise ArtifactValidationError(
            "provenance harness hash cannot be reproduced from its commit "
            "or the current harness")

    args = provenance.get("args") or {}
    for field in ("source", "config", "adapters", "seeds", "limit",
                  "out_dir", "out"):
        if field not in args:
            raise ArtifactValidationError(
                f"provenance arguments are missing {field}")
    adapters = tuple(args["adapters"])
    seeds = tuple(args["seeds"])
    pair_count = args["limit"]
    inferred = {(adapter, seed, pair, label)
                for adapter in adapters for seed in seeds
                for pair in range(pair_count)
                for label in ("nonprose", "prose")}
    if required_matrix is not None and inferred != required_matrix:
        raise ArtifactValidationError("provenance arguments do not match the fixed matrix")

    keys = []
    pair_manifest = ((doc.get("selection") or {}).get("pairs") or [])
    if len(pair_manifest) != pair_count:
        raise ArtifactValidationError(
            f"selection has {len(pair_manifest)} pairs; expected {pair_count}")
    pair_inputs = {}
    for index, pair in enumerate(pair_manifest):
        for label in ("nonprose", "prose"):
            try:
                pair_inputs[(index, label)] = (
                    pair[f"{label}_uid"], pair[f"{label}_sha256"])
            except KeyError as exc:
                raise ArtifactValidationError(
                    f"selection pair {index} is missing {exc.args[0]}") from exc
        for field in ("nonprose_features", "prose_features",
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

    from experiments.nonprose_replication import summarize
    if doc.get("summary") != summarize(rows):
        raise ArtifactValidationError("summary does not exactly recompute from rows")
    return doc


def run(command, cwd=REPO, env=None, stdout=None):
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, stdout=stdout,
                   stderr=subprocess.STDOUT if stdout else None, check=True)


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
        env = os.environ.copy()
        env.setdefault("GPU_LOCK", os.path.expanduser("~/.alexandria_gpu.lock"))
        env.setdefault("GPU_QLOG", os.path.join(
            REPO, "ab_test_runtime", "logs", "gpu_jobq.log"))
        log_path = os.path.join(
            REPO, "ab_test_runtime", "logs", "nonprose_replication.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        command = [
            os.path.join(REPO, "gpu_job.sh"),
            "nonprose_replication_full", "timeout", "43200",
            sys.executable, "-u",
            "experiments/nonprose_replication.py",
        ]
        with open(log_path, "a", encoding="utf-8") as log:
            run(command, cwd=APP, env=env, stdout=log)
    validate_stage4_artifact(FULL, 144, matrix)
    print("Stage 4 full matrix validated strictly (144/144 rows).", flush=True)

    run([sys.executable, "collect_results.py"])
    require_index_entries("nonprose_replication.json",
                          "nonprose_replication_pilot.json")
    print("Both results indexes explicitly contain the Stage 4 artifacts.",
          flush=True)
    run([sys.executable, "-m", "unittest",
         "discover", "-s", "app", "-p", "test_*.py"])
    print("Stage 4 local-test checkpoint complete.", flush=True)


if __name__ == "__main__":
    main()
