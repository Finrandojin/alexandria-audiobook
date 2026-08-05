"""Build an unlabeled listening package and a separate concealed key."""
import argparse
import json
import os
import random
import shutil
import string
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

from experiments.provenance import file_sha256

RATING_FIELDS = (
    "delivery", "emotional_fit", "voice_distinction", "intelligibility",
    "defects", "preference",
)


class ListeningPackageError(RuntimeError):
    """The source arms cannot support a blinded listening package."""


def _load_document(path, expected_script=None):
    try:
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ListeningPackageError(f"cannot read {path}: {exc}") from exc
    provenance = doc.get("provenance")
    if not isinstance(provenance, dict):
        raise ListeningPackageError(f"{path} has no provenance")
    if expected_script and provenance.get("script") != expected_script:
        raise ListeningPackageError(
            f"{path} came from {provenance.get('script')!r}, expected "
            f"{expected_script!r}")
    git = provenance.get("git") or {}
    if not isinstance(git.get("harness_sha256"), str) or \
            len(git["harness_sha256"]) != 64:
        raise ListeningPackageError(f"{path} has no harness fingerprint")
    return doc


def _validate_wav(path):
    from experiments.generation import _check_riff_completeness
    try:
        import soundfile as sf
        info = sf.info(path)
        if not info.frames or not info.samplerate or not info.channels:
            raise ListeningPackageError(f"WAV has no usable audio: {path}")
        with sf.SoundFile(path) as handle:
            while handle.read(65536).size:
                pass
        _check_riff_completeness(path, "listening", "blind package")
    except ListeningPackageError:
        raise
    except Exception as exc:
        raise ListeningPackageError(
            f"cannot fully decode WAV {path}: {exc}") from exc


def _resolve_source(path):
    resolved = os.path.realpath(
        path if os.path.isabs(path) else os.path.join(REPO, path))
    repo = os.path.realpath(REPO)
    if os.path.commonpath((repo, resolved)) != repo:
        raise ListeningPackageError(f"source WAV escapes repository: {path}")
    if not os.path.isfile(resolved):
        raise ListeningPackageError(f"source WAV does not exist: {path}")
    _validate_wav(resolved)
    return resolved


def _source_groups(instruction_doc, casting_doc, control_doc):
    if instruction_doc.get("status") != "complete" or not \
            instruction_doc.get("all_arms_rendered"):
        raise ListeningPackageError("instruction source is incomplete")
    comparisons = instruction_doc.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ListeningPackageError("instruction source has no comparisons")
    groups = []
    for index, comparison in enumerate(comparisons):
        files = comparison.get("arm_files")
        if set(files or {}) != {"none", "per_char", "per_line"}:
            raise ListeningPackageError(
                f"instruction comparison {index} has wrong arms")
        groups.append({"kind": "instruction_delivery",
                       "source_id": f"instruction_{index:02d}",
                       "arms": dict(files)})

    if casting_doc.get("status") != "complete" or not \
            casting_doc.get("published"):
        raise ListeningPackageError("casting source is incomplete")
    casting_arms = casting_doc.get("arms") or {}
    if set(casting_arms) != {"current", "scene_aware"}:
        raise ListeningPackageError("casting source has wrong arms")
    expected_lines = list(range(casting_doc.get("lines", -1)))
    if any(arm.get("lines") != expected_lines for arm in casting_arms.values()):
        raise ListeningPackageError("casting arms do not contain identical lines")
    groups.append({"kind": "casting", "source_id": "casting_00",
                   "arms": {name: value.get("path")
                            for name, value in casting_arms.items()}})

    rows = control_doc.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ListeningPackageError("positive-control source has no rows")
    for index, row in enumerate(rows):
        controls = row.get("instruction_controls") or {}
        if not row.get("duration_order_control_passes") or \
                not {"very_slow", "very_fast"} <= set(controls):
            raise ListeningPackageError(
                f"positive control {index} did not pass")
        arms = {}
        for name in ("very_slow", "very_fast"):
            item = controls[name]
            source = _resolve_source(item.get("file", ""))
            if file_sha256(source) != item.get("sha256"):
                raise ListeningPackageError(
                    f"positive control {index} {name} hash changed")
            arms[name] = item["file"]
        groups.append({"kind": "positive_control",
                       "source_id": f"control_{index:02d}", "arms": arms})
    return groups


def build_package(instruction_path, casting_path, control_path, package_dir,
                  public_path, key_path, seed):
    """Create the package once; refuse overwrite and return both manifests."""
    for path in (package_dir, public_path, key_path):
        if os.path.exists(path):
            raise ListeningPackageError(f"refusing to overwrite existing {path}")
    instruction = _load_document(
        instruction_path, "instruct_listening.py")
    casting = _load_document(casting_path, "casting_ab_audio.py")
    controls = _load_document(control_path, "seed_instruction_controls.py")
    groups = _source_groups(instruction, casting, controls)

    temporary = package_dir + ".building"
    if os.path.exists(temporary):
        raise ListeningPackageError(
            f"stale temporary package exists: {temporary}")
    os.makedirs(temporary)
    rng = random.Random(seed)
    public_sets, key_sets = [], []
    try:
        for index, group in enumerate(groups):
            ordered = list(group["arms"].items())
            rng.shuffle(ordered)
            samples, mapping = [], {}
            for sample_index, (arm, source_path) in enumerate(ordered):
                source = _resolve_source(source_path)
                letter = string.ascii_uppercase[sample_index]
                filename = f"set_{index:02d}_{letter}.wav"
                destination = os.path.join(temporary, filename)
                shutil.copy2(source, destination)
                _validate_wav(destination)
                samples.append({"file": filename,
                                "sha256": file_sha256(destination)})
                mapping[filename] = {"arm": arm,
                                     "source_sha256": file_sha256(source)}
            public_sets.append({"id": f"set_{index:02d}",
                                "kind": group["kind"], "samples": samples,
                                "rating_fields": list(RATING_FIELDS)})
            key_sets.append({"id": f"set_{index:02d}",
                             "source_id": group["source_id"],
                             "mapping": mapping})
        os.rename(temporary, package_dir)
    except Exception:
        if os.path.isdir(temporary):
            shutil.rmtree(temporary)
        raise

    from experiments.provenance import provenance
    from utils import atomic_json_write
    key = {"status": "complete", "randomization_seed": seed,
           "sets": key_sets}
    try:
        atomic_json_write(key, key_path)
        public = {
            "status": "complete",
            "provenance": provenance(
                __file__, None,
                source_artifacts={
                    os.path.relpath(path, REPO): file_sha256(path)
                    for path in (instruction_path, casting_path, control_path)},
                randomization_seed=seed),
            "package_dir": os.path.relpath(package_dir, REPO),
            "sets": public_sets,
            "concealed_key_sha256": file_sha256(key_path),
            "limitations": [
                "No human ratings are included; this artifact only prepares it.",
                "Non-prose remedies are absent because Stage 5 stopped at its gate.",
            ],
        }
        atomic_json_write(public, public_path)
    except Exception:
        for path in (public_path, key_path):
            if os.path.isfile(path):
                os.unlink(path)
        if os.path.isdir(package_dir):
            shutil.rmtree(package_dir)
        raise
    return public, key


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise ListeningPackageError(f"cannot read {path}: {exc}") from exc


def validate_package(public_path, key_path, package_dir):
    """Fully verify a previously built package without exposing its key."""
    public = _load_document(public_path, "blinded_listening.py")
    key = _load_json(key_path)
    if public.get("status") != "complete" or key.get("status") != "complete":
        raise ListeningPackageError("blind package is not complete")
    if file_sha256(key_path) != public.get("concealed_key_sha256"):
        raise ListeningPackageError("concealed key hash does not match")
    if os.path.realpath(os.path.join(REPO, public.get("package_dir", ""))) != \
            os.path.realpath(package_dir):
        raise ListeningPackageError("public manifest names a different package")
    source_artifacts = public["provenance"].get("source_artifacts") or {}
    if not source_artifacts:
        raise ListeningPackageError("public manifest has no source identities")
    for relative, expected_hash in source_artifacts.items():
        source_path = os.path.realpath(os.path.join(REPO, relative))
        if os.path.commonpath((os.path.realpath(REPO), source_path)) != \
                os.path.realpath(REPO) or not os.path.isfile(source_path):
            raise ListeningPackageError(f"source artifact is unavailable: {relative}")
        if file_sha256(source_path) != expected_hash:
            raise ListeningPackageError(f"source artifact changed: {relative}")
    public_sets = public.get("sets") or []
    key_sets = key.get("sets") or []
    if len(public_sets) != 8 or len(key_sets) != 8:
        raise ListeningPackageError("blind package must contain eight sets")
    keyed = {item.get("id"): item for item in key_sets}
    if len(keyed) != len(key_sets):
        raise ListeningPackageError("concealed key has duplicate set IDs")
    forbidden = ("very_slow", "very_fast", "per_char", "per_line",
                 '\"current\"', '\"scene_aware\"')
    leaked = [label for label in forbidden
              if label in json.dumps(public, sort_keys=True)]
    if leaked:
        raise ListeningPackageError(
            f"public manifest leaks arm labels: {leaked}")
    for item in public_sets:
        set_id = item.get("id")
        key_item = keyed.get(set_id)
        if not key_item:
            raise ListeningPackageError(f"concealed key lacks {set_id}")
        samples = item.get("samples") or []
        mapping = key_item.get("mapping") or {}
        names = [sample.get("file") for sample in samples]
        if len(samples) < 2 or set(names) != set(mapping):
            raise ListeningPackageError(f"sample/key mismatch in {set_id}")
        for sample in samples:
            path = os.path.join(package_dir, sample["file"])
            _validate_wav(path)
            if file_sha256(path) != sample.get("sha256"):
                raise ListeningPackageError(
                    f"packaged WAV hash changed: {sample['file']}")
            if mapping[sample["file"]].get("source_sha256") != \
                    sample.get("sha256"):
                raise ListeningPackageError(
                    f"source/package hash mismatch: {sample['file']}")
    expected_files = {sample["file"] for item in public_sets
                      for sample in item["samples"]}
    actual_files = set(os.listdir(package_dir))
    if actual_files != expected_files:
        raise ListeningPackageError(
            f"package file inventory changed: expected {len(expected_files)}, "
            f"found {len(actual_files)}")
    return public, key


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--instruction", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "stage6_instruction_source.json"))
    ap.add_argument("--casting", default=os.path.join(
        REPO, "ab_test_runtime", "stage6_casting", "manifest.json"))
    ap.add_argument("--controls", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "seed_instruction_controls.json"))
    ap.add_argument("--package-dir", default=os.path.join(
        REPO, "ab_test_runtime", "blinded_listening"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "blinded_listening.json"))
    ap.add_argument("--key", default=os.path.join(
        REPO, "ab_test_runtime", "blinded_listening_concealed_key.json"))
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()
    public, _ = build_package(
        args.instruction, args.casting, args.controls, args.package_dir,
        args.out, args.key, args.seed)
    print(f"wrote {len(public['sets'])} blinded sets to {args.package_dir}")
    print(f"public manifest: {args.out}")
    print(f"concealed key: {args.key}")


if __name__ == "__main__":
    main()
