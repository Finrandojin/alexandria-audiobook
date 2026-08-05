"""Seeded pitch profiling for every usable adapter across locked text types."""
import argparse
import collections
import hashlib
import json
import math
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

DEFAULT_SOURCE = os.path.join(APP, "experiments", "pitch_profile_passages.json")
DEFAULT_SEEDS = (1234, 5678, 9012)
MIN_VOICED_FRAMES = 20
FMIN = 50.0
FMAX = 500.0


class PitchProfileError(RuntimeError):
    """The matrix or one of its checkpoints cannot support a result."""


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_text(text):
    return " ".join((text or "").split())


def load_passages(source, categories=None):
    with open(source, encoding="utf-8") as handle:
        fixture = json.load(handle)
    rows = fixture.get("passages")
    if not isinstance(rows, list) or not rows:
        raise PitchProfileError("pitch fixture has no passages")
    wanted = set(categories or [row.get("category") for row in rows])
    chunks_path = os.path.join(REPO, fixture.get("source_artifact", ""))
    with open(chunks_path, encoding="utf-8") as handle:
        chunks = json.load(handle)
    if isinstance(chunks, dict):
        chunks = chunks.get("chunks") or chunks.get("entries") or []
    selected, seen = [], set()
    for row in rows:
        category = row.get("category")
        if category not in wanted:
            continue
        if category in seen:
            raise PitchProfileError(f"duplicate passage category: {category}")
        text = normalized_text(row.get("text"))
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != row.get("source_sha256"):
            raise PitchProfileError(f"locked text hash changed for {category}")
        index = row.get("source_index")
        if not isinstance(index, int) or not 0 <= index < len(chunks):
            raise PitchProfileError(f"invalid source index for {category}")
        if normalized_text(chunks[index].get("text")) != text:
            raise PitchProfileError(f"source text changed for {category}")
        selected.append(dict(row, text=text, uid=digest[:12]))
        seen.add(category)
    missing = wanted - seen
    if missing:
        raise PitchProfileError(
            "pitch fixture lacks categories: " + ", ".join(sorted(missing)))
    return selected, fixture.get("selection_rule"), chunks_path


def load_adapters(manifest_path, selected=None):
    with open(manifest_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    items = raw if isinstance(raw, list) else list(raw.values())
    requested = set(selected or [])
    adapters = []
    for item in items:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        adapter = item["id"]
        if requested and adapter not in requested:
            continue
        path = os.path.join(REPO, "lora_models", adapter)
        missing = [name for name in ("adapter_config.json",
                                     "adapter_model.safetensors")
                   if not os.path.isfile(os.path.join(path, name))]
        declared = (item.get("voice_features") or {}).get("mean_f0")
        if missing or not isinstance(declared, (int, float)) or declared <= 0:
            raise PitchProfileError(
                f"adapter is not usable: {adapter}; missing={missing}; "
                f"mean_f0={declared!r}")
        adapters.append({"adapter": adapter, "path": path,
                         "declared_mean_f0": float(declared)})
    found = {item["adapter"] for item in adapters}
    if requested - found:
        raise PitchProfileError(
            "unknown requested adapters: " + ", ".join(sorted(requested - found)))
    if not adapters:
        raise PitchProfileError("no usable adapters")
    return adapters


def measure_pitch(path):
    """Return pYIN measurements including failures rather than hiding them."""
    import librosa
    import numpy as np
    try:
        audio, rate = librosa.load(path, sr=16000, mono=True)
        f0, voiced, _ = librosa.pyin(
            audio, fmin=FMIN, fmax=FMAX, sr=rate)
    except Exception as exc:
        return {"pitch_status": "tracker_failure",
                "pitch_error": f"{type(exc).__name__}: {str(exc)[:120]}"}
    total = int(len(f0))
    valid = f0[voiced & np.isfinite(f0)]
    coverage = float(len(valid) / total) if total else 0.0
    if len(valid) < MIN_VOICED_FRAMES:
        return {"pitch_status": "tracker_failure",
                "pitch_error": f"only {len(valid)} voiced frames",
                "pitch_frames": total, "voiced_frames": int(len(valid)),
                "voiced_coverage": coverage}
    median = float(np.median(valid))
    q10, q25, q75, q90 = (float(value) for value in
                           np.percentile(valid, (10, 25, 75, 90)))
    ratios = np.abs(np.log2(valid / median))
    frame_octave_fraction = float(
        np.mean((ratios >= 0.80) & (ratios <= 1.20)))
    return {
        "pitch_status": "measured", "pitch_error": None,
        "pitch_frames": total, "voiced_frames": int(len(valid)),
        "voiced_coverage": coverage, "median_pitch_hz": median,
        "pitch_q10_hz": q10, "pitch_q25_hz": q25,
        "pitch_q75_hz": q75, "pitch_q90_hz": q90,
        "pitch_iqr_hz": q75 - q25,
        "frame_octave_fraction": frame_octave_fraction,
    }


def is_likely_octave_ratio(value, reference):
    if not value or not reference or value <= 0 or reference <= 0:
        return False
    distance = abs(math.log2(value / reference))
    return 0.80 <= distance <= 1.20


def add_octave_flags(rows):
    by_adapter = collections.defaultdict(list)
    for row in rows:
        if row.get("pitch_status") == "measured":
            by_adapter[row["adapter"]].append(row["median_pitch_hz"])
    baselines = {adapter: statistics.median(values)
                 for adapter, values in by_adapter.items()}
    output = []
    for original in rows:
        row = dict(original)
        measured = row.get("median_pitch_hz")
        reasons = []
        if is_likely_octave_ratio(measured, row.get("declared_mean_f0")):
            reasons.append("declared_mean")
        if is_likely_octave_ratio(measured, baselines.get(row["adapter"])):
            reasons.append("adapter_matrix_median")
        if row.get("frame_octave_fraction", 0) >= 0.05:
            reasons.append("within_clip_frame_modes")
        row["likely_octave_error"] = bool(reasons)
        row["likely_octave_reasons"] = reasons
        output.append(row)
    return output


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] * (high - position)
                 + ordered[high] * (position - low))


def summarize(rows, adapters):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["adapter"]].append(row)
    adapter_summary = []
    for adapter in adapters:
        selected = grouped[adapter["adapter"]]
        measured = [row for row in selected
                    if row.get("pitch_status") == "measured"]
        values = [row["median_pitch_hz"] for row in measured]
        adapter_summary.append({
            "adapter": adapter["adapter"],
            "declared_mean_f0": adapter["declared_mean_f0"],
            "rows": len(selected), "measured_rows": len(measured),
            "tracker_failures": len(selected) - len(measured),
            "median_pitch_hz": statistics.median(values) if values else None,
            "across_clip_p10_hz": percentile(values, 10),
            "across_clip_p90_hz": percentile(values, 90),
            "across_clip_p90_p10_hz": (
                percentile(values, 90) - percentile(values, 10)
                if values else None),
            "across_clip_range_hz": max(values) - min(values) if values else None,
            "median_clip_iqr_hz": statistics.median(
                row["pitch_iqr_hz"] for row in measured) if measured else None,
            "median_voiced_coverage": statistics.median(
                row["voiced_coverage"] for row in measured) if measured else None,
            "minimum_voiced_coverage": min(
                (row["voiced_coverage"] for row in measured), default=None),
            "likely_octave_rows": sum(
                bool(row.get("likely_octave_error")) for row in selected),
        })
    usable = [row for row in adapter_summary if row["median_pitch_hz"] is not None]
    dispersions = [row["across_clip_p90_p10_hz"] for row in usable]
    typical = statistics.median(dispersions) if dispersions else None
    pairs, separated = 0, 0
    for index, first in enumerate(usable):
        for second in usable[index + 1:]:
            pairs += 1
            if typical is not None and abs(
                    first["median_pitch_hz"] - second["median_pitch_hz"]) > typical:
                separated += 1
    threshold_disagreements = sum(
        (row["declared_mean_f0"] >= 165) != (row["median_pitch_hz"] >= 165)
        for row in usable)
    return adapter_summary, {
        "adapters_requested": len(adapters), "adapters_measured": len(usable),
        "tracker_failure_rows": sum(
            row["tracker_failures"] for row in adapter_summary),
        "likely_octave_rows": sum(
            row["likely_octave_rows"] for row in adapter_summary),
        "typical_within_adapter_p90_p10_hz": typical,
        "voice_pairs": pairs, "pairs_beyond_typical_dispersion": separated,
        "separable_pair_fraction": separated / pairs if pairs else None,
        "declared_vs_measured_165hz_side_disagreements": threshold_disagreements,
        "production_threshold_hz": 165,
    }


def get_fingerprint(args, passages, adapters, chunks_path):
    code = (__file__, os.path.join(APP, "experiments", "generation.py"),
            os.path.join(APP, "tts.py"))
    return {
        "schema": 1,
        "code_sha256": {os.path.relpath(path, REPO): file_sha256(path)
                        for path in code},
        "input_sha256": {os.path.relpath(path, REPO): file_sha256(path)
                         for path in (args.source, chunks_path, args.config,
                                      args.manifest)},
        "adapters": {item["adapter"]: {
            name: file_sha256(os.path.join(item["path"], name))
            for name in ("adapter_config.json", "adapter_model.safetensors")}
            for item in adapters},
        "seeds": list(args.seeds), "categories": list(args.categories),
        "passages": passages,
    }


def get_adapter_sha256(adapters):
    return {item["adapter"]: {
        name: file_sha256(os.path.join(item["path"], name))
        for name in ("adapter_config.json", "adapter_model.safetensors")}
        for item in adapters}


def get_public_adapters(adapters):
    return [{"adapter": item["adapter"],
             "declared_mean_f0": item["declared_mean_f0"]}
            for item in adapters]


def validate_rows(rows, expected, recompute=True):
    from experiments.generation import _check_riff_completeness
    import soundfile as sf
    expected = {key: identity for key, identity in expected}
    seen = set()
    required = {"adapter", "seed", "passage", "category", "uid",
                "source_sha256", "wav", "declared_mean_f0", "pitch_status"}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise PitchProfileError(
                f"row {index} missing {', '.join(sorted(missing))}")
        key = (row["adapter"], row["seed"], row["passage"])
        if key in seen or key not in expected:
            raise PitchProfileError(f"duplicate or foreign row: {key}")
        seen.add(key)
        if (row["uid"], row["source_sha256"], row["category"], row["wav"],
                row["declared_mean_f0"]) != expected[key]:
            raise PitchProfileError(f"row input identity changed: {key}")
        path = os.path.realpath(os.path.join(REPO, row["wav"]))
        if os.path.commonpath((os.path.realpath(REPO), path)) != os.path.realpath(REPO):
            raise PitchProfileError(f"row WAV escapes repository: {key}")
        try:
            info = sf.info(path)
            if not info.frames or not info.samplerate or not info.channels:
                raise ValueError("no usable audio")
            with sf.SoundFile(path) as handle:
                while handle.read(65536).size:
                    pass
            _check_riff_completeness(path, "pitch", str(key))
        except Exception as exc:
            raise PitchProfileError(f"row WAV is unusable {key}: {exc}") from exc
        if recompute:
            measured = measure_pitch(path)
            for field, value in measured.items():
                saved = row.get(field)
                if isinstance(value, float):
                    if not isinstance(saved, (int, float)) or not math.isclose(
                            saved, value, rel_tol=1e-10, abs_tol=1e-10):
                        raise PitchProfileError(
                            f"row pitch measurement changed {key}: {field}")
                elif saved != value:
                    raise PitchProfileError(
                        f"row pitch measurement changed {key}: {field}")
    return seen


def expected_rows(adapters, seeds, passages, out_dir):
    expected = []
    for adapter in adapters:
        for seed in seeds:
            for passage, item in enumerate(passages):
                wav = os.path.join(
                    out_dir, f"{adapter['adapter']}__s{seed}__p{passage}.wav")
                expected.append(((adapter["adapter"], seed, passage),
                                 (item["uid"], item["source_sha256"],
                                  item["category"], os.path.relpath(wav, REPO),
                                  adapter["declared_mean_f0"])))
    return expected


def validate_artifact(path, expected_count=None, recompute=True):
    with open(path, encoding="utf-8") as handle:
        doc = json.load(handle)
    if doc.get("status") != "complete" or not isinstance(doc.get("rows"), list):
        raise PitchProfileError("pitch artifact is incomplete")
    provenance = doc.get("provenance") or {}
    if provenance.get("script") != "pitch_profile_matrix.py":
        raise PitchProfileError("pitch artifact has wrong provenance")
    from experiments.provenance import (
        get_reproducible_harness_source, input_sha256)
    if get_reproducible_harness_source(provenance, REPO) is None:
        raise PitchProfileError("pitch artifact harness cannot be reproduced")
    args = provenance.get("args") or {}
    required_args = {"source", "config", "manifest", "adapters", "seeds",
                     "categories", "out_dir", "out", "checkpoint"}
    if required_args - set(args):
        raise PitchProfileError("pitch artifact lacks complete arguments")
    if os.path.realpath(path) != os.path.realpath(os.path.join(REPO, args["out"])):
        raise PitchProfileError("pitch artifact path differs from its provenance")
    input_paths = [os.path.join(REPO, args[name])
                   for name in ("source", "config", "manifest")]
    passages = (doc.get("selection") or {}).get("passages") or []
    if not passages:
        raise PitchProfileError("pitch artifact has no locked passages")
    locked_passages, selection_rule, chunks_path = load_passages(
        input_paths[0], args["categories"])
    if passages != locked_passages or \
            (doc.get("selection") or {}).get("rule") != selection_rule:
        raise PitchProfileError("pitch artifact passage selection changed")
    expected_inputs = input_sha256((*input_paths, chunks_path))
    if provenance.get("input_sha256") != expected_inputs:
        raise PitchProfileError("pitch artifact inputs changed")
    adapters = doc.get("adapters") or []
    if any(set(item) != {"adapter", "declared_mean_f0"} for item in adapters):
        raise PitchProfileError("pitch artifact has invalid adapter records")
    if args.get("adapters") != [item["adapter"] for item in adapters]:
        raise PitchProfileError("pitch artifact adapter arguments changed")
    loaded_adapters = load_adapters(input_paths[2], args["adapters"])
    if get_public_adapters(loaded_adapters) != adapters:
        raise PitchProfileError("pitch adapter declarations changed")
    if provenance.get("adapter_sha256") != get_adapter_sha256(loaded_adapters):
        raise PitchProfileError("pitch adapter weights changed")
    if args.get("categories") != [item.get("category") for item in passages]:
        raise PitchProfileError("pitch passage categories changed")
    expected = expected_rows(adapters, args.get("seeds") or [], passages,
                             os.path.join(REPO, args.get("out_dir", "")))
    if len(expected) != len(doc["rows"]):
        raise PitchProfileError(
            f"pitch artifact matrix is incomplete: {len(doc['rows'])}/{len(expected)}")
    if expected_count is not None and len(doc["rows"]) != expected_count:
        raise PitchProfileError(
            f"pitch artifact has {len(doc['rows'])}/{expected_count} rows")
    validate_rows(doc["rows"], expected, recompute=recompute)
    flagged = add_octave_flags([
        {key: value for key, value in row.items()
         if key not in ("likely_octave_error", "likely_octave_reasons")}
        for row in doc["rows"]])
    if flagged != doc["rows"]:
        raise PitchProfileError("pitch octave flags do not recompute")
    adapter_summary, summary = summarize(doc["rows"], adapters)
    if adapter_summary != doc.get("adapter_summary") or summary != doc.get("summary"):
        raise PitchProfileError("pitch summary does not recompute")
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--config", default=os.path.join(APP, "config.json"))
    parser.add_argument("--manifest", default=os.path.join(
        REPO, "lora_models", "manifest.json"))
    parser.add_argument("--adapters", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(DEFAULT_SEEDS))
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--out-dir", default=os.path.join(
        REPO, "ab_test_runtime", "pitch_profile_matrix"))
    parser.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "pitch_profile_matrix.json"))
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    fixture = json.load(open(args.source, encoding="utf-8"))
    all_categories = [row["category"] for row in fixture.get("passages") or []]
    args.categories = args.categories or all_categories
    passages, selection_rule, chunks_path = load_passages(
        args.source, args.categories)
    adapters = load_adapters(args.manifest, args.adapters)
    args.adapters = [item["adapter"] for item in adapters]
    if not args.seeds:
        parser.error("at least one seed is required")
    checkpoint = args.checkpoint or args.out + ".ckpt"
    fingerprint = get_fingerprint(args, passages, adapters, chunks_path)
    expected = expected_rows(adapters, args.seeds, passages, args.out_dir)
    from experiments.nonprose_replication import (
        archive_checkpoint, get_resumable_rows, save_checkpoint)
    try:
        rows, mismatch = get_resumable_rows(checkpoint, fingerprint)
        if mismatch:
            archive_checkpoint(checkpoint, mismatch)
            rows = []
        completed = validate_rows(rows, expected) if rows else set()
    except (PitchProfileError, ValueError) as exc:
        raise SystemExit(f"invalid pitch checkpoint: {exc}") from exc
    if rows:
        print(f"resumed {len(rows)} fully validated rows", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    total = len(expected)
    if len(completed) < total:
        from tts import TTSEngine
        from experiments.generation import render
        engine = TTSEngine(json.load(open(args.config, encoding="utf-8")))
    done = len(rows)
    for adapter in adapters:
        entry = {"type": "lora", "adapter_path": adapter["path"]}
        for seed in args.seeds:
            entry["seed"] = str(seed)
            for passage_index, passage in enumerate(passages):
                key = (adapter["adapter"], seed, passage_index)
                if key in completed:
                    continue
                wav = os.path.join(
                    args.out_dir,
                    f"{adapter['adapter']}__s{seed}__p{passage_index}.wav")
                render(engine, passage["text"], "", "X", {"X": entry},
                       entry, wav)
                row = {
                    "adapter": adapter["adapter"], "seed": seed,
                    "passage": passage_index, "category": passage["category"],
                    "uid": passage["uid"],
                    "source_sha256": passage["source_sha256"],
                    "wav": os.path.relpath(wav, REPO),
                    "declared_mean_f0": adapter["declared_mean_f0"],
                }
                row.update(measure_pitch(wav))
                rows.append(row)
                completed.add(key)
                save_checkpoint(checkpoint, fingerprint, rows)
                done += 1
                print(f"[{done}/{total}] {adapter['adapter']} seed={seed} "
                      f"{passage['category']} {row['pitch_status']}", flush=True)
    if len(rows) != total:
        raise SystemExit(f"incomplete pitch matrix: {len(rows)}/{total}")
    rows = add_octave_flags(rows)
    public_adapters = get_public_adapters(adapters)
    adapter_summary, summary = summarize(rows, public_adapters)
    from experiments.provenance import input_sha256, provenance
    doc = {
        "status": "complete",
        "provenance": provenance(
            __file__, args,
            input_sha256=input_sha256(
                (args.source, chunks_path, args.config, args.manifest)),
            adapter_sha256=get_adapter_sha256(adapters)),
        "selection": {"rule": selection_rule, "passages": passages},
        "adapters": public_adapters, "adapter_summary": adapter_summary,
        "summary": summary, "rows": rows,
        "limitations": [
            "pYIN flags likely octave errors; it does not establish ground truth pitch.",
            "The 165 Hz comparison reports threshold instability, not gender truth.",
            "No numerical pitch threshold is adopted without blinded listening.",
        ],
    }
    from utils import atomic_json_write
    atomic_json_write(doc, args.out)
    validate_artifact(args.out, total)
    if os.path.exists(checkpoint):
        os.remove(checkpoint)
    print(f"wrote {total} strictly validated rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
