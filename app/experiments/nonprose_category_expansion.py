"""Expand the non-prose test across six categories selected before rendering."""
import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

from experiments.nonprose_replication import (
    DEFAULT_ADAPTERS, DEFAULT_SEEDS, archive_checkpoint, feature_gap,
    get_resumable_rows, get_run_fingerprint, save_checkpoint, surface_features,
    validate_resumed_rows)


CATEGORIES = (
    "identifiers", "urls", "copyright", "lists_tables", "dates_numbers",
    "headings_fragments",
)
DEFAULT_FIXTURE = os.path.join(
    APP, "experiments", "nonprose_category_probes.json")


def normalized_text(text):
    return " ".join((text or "").split())


def get_category_run_fingerprint(args, pair_manifest, adapter_paths):
    fingerprint_args = argparse.Namespace(
        **vars(args), limit=args.limit_per_category)
    return get_run_fingerprint(
        fingerprint_args, pair_manifest, adapter_paths, harness_file=__file__)


def load_locked_probes(source=DEFAULT_FIXTURE, limit_per_category=4):
    if not 1 <= limit_per_category <= 4:
        raise ValueError("limit_per_category must be between 1 and 4")
    with open(source, encoding="utf-8") as handle:
        fixture = json.load(handle)
    locked = fixture.get("probes")
    if not isinstance(locked, list):
        raise ValueError("probe fixture has no probes list")
    chosen = collections.Counter()
    probes = []
    seen = set()
    for item in locked:
        category = item.get("category")
        if chosen[category] >= limit_per_category:
            continue
        if category not in CATEGORIES:
            raise ValueError(f"unknown probe category: {category!r}")
        text = normalized_text(item.get("text"))
        expected_sha = item.get("source_sha256")
        actual_sha = hashlib.sha256(text.encode()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(
                f"locked fixture text expected {expected_sha}, got {actual_sha}")
        if actual_sha in seen:
            raise ValueError(f"duplicate locked probe: {actual_sha}")
        seen.add(actual_sha)
        probes.append({"category": category, "text": text,
                       "uid": expected_sha[:12], "source_sha256": expected_sha,
                       "source_path": item.get("source_path"),
                       "source_index": item.get("source_index")})
        chosen[category] += 1
    missing = [category for category in CATEGORIES
               if chosen[category] != limit_per_category]
    if missing:
        raise ValueError(f"locked fixture is incomplete for: {', '.join(missing)}")
    return probes


def load_locked_pairs(source=DEFAULT_FIXTURE, limit_per_category=4):
    """Load predeclared probes and controls without depending on ignored data."""
    probes = load_locked_probes(source, limit_per_category)
    with open(source, encoding="utf-8") as handle:
        controls = json.load(handle).get("controls")
    if not isinstance(controls, list):
        raise ValueError("probe fixture has no controls list")
    by_probe = {}
    seen_controls = set()
    for item in controls:
        text = normalized_text(item.get("text"))
        sha = hashlib.sha256(text.encode()).hexdigest()
        if sha != item.get("source_sha256"):
            raise ValueError(f"locked control text hash is wrong: {sha}")
        if sha in seen_controls:
            raise ValueError(f"duplicate locked control: {sha}")
        if not is_ordinary_prose(text):
            raise ValueError(f"locked control is not ordinary prose: {text!r}")
        probe_sha = item.get("probe_sha256")
        if probe_sha in by_probe:
            raise ValueError(f"probe has multiple locked controls: {probe_sha}")
        seen_controls.add(sha)
        by_probe[probe_sha] = {
            "text": text, "uid": sha[:12], "source_sha256": sha,
            "source_path": item.get("source_path"),
            "source_index": item.get("source_index"),
        }
    pairs = []
    for probe in probes:
        control = by_probe.get(probe["source_sha256"])
        if control is None:
            raise ValueError(
                f"probe has no locked control: {probe['source_sha256']}")
        pairs.append((probe, control))
    return pairs


def load_prose_pool(excluded_hashes):
    from experiments.prose_vs_nonprose import classify, is_machine_output
    seen = set()
    candidates = []
    for path in sorted(glob.glob(os.path.join(REPO, "scripts", "*.json"))):
        if any(marker in path for marker in
               ("voice_config", "generation_quality", "generation_checkpoint",
                "review_checkpoint")):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                doc = json.load(handle)
        except (OSError, ValueError):
            continue
        entries = doc if isinstance(doc, list) else (
            doc.get("entries") or doc.get("chunks") or [])
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            text = normalized_text(entry.get("text"))
            sha = hashlib.sha256(text.encode()).hexdigest()
            if (not 15 <= len(text) <= 500 or sha in seen
                    or sha in excluded_hashes or is_machine_output(text)
                    or not is_ordinary_prose(text, classify)):
                continue
            seen.add(sha)
            candidates.append({
                "text": text, "uid": sha[:12], "source_sha256": sha,
                "source_path": os.path.relpath(path, REPO),
                "source_index": index,
                "_features": surface_features(text),
            })
    if not candidates:
        raise ValueError("no eligible prose controls found")
    return candidates


def is_ordinary_prose(text, classifier=None):
    """Exclude metadata that the broad legacy classifier calls prose."""
    if classifier is None:
        from experiments.prose_vs_nonprose import classify as classifier
    words = re.findall(r"[^\W\d_]+", text or "")
    low = (text or "").lower()
    if len(words) < 4 or classifier(text) != "prose":
        return False
    first_alpha = re.search(r"[^\W\d_]", text or "")
    if not first_alpha or not first_alpha.group(0).isupper():
        return False
    if not re.search(r"[.!?…][\"'”’]?$", (text or "").rstrip()):
        return False
    if "•" in text or re.search(
            r"(?i)(?:https?://|www\.|\b(?:isbn|lccn|copyright)\b)", text):
        return False
    if re.match(r"(?i)^(?:chapter|part|prologue|epilogue|interlude|contents|"
                r"navigation|table of contents|first published)\b", text):
        return False
    alpha_words = [word for word in words if word.isalpha()]
    if alpha_words and all(word.isupper() for word in alpha_words):
        return False
    return not any(marker in low for marker in
                   ("all rights reserved", "translation rights", "yen press"))


def feature_matching_cost(a, b):
    return (
        3 * abs(a["chars"] - b["chars"]) / max(a["chars"], 1)
        + abs(a["words"] - b["words"]) / max(a["words"], 1)
        + 2 * abs(a["digit_fraction"] - b["digit_fraction"])
        + abs(a["uppercase_word_fraction"] - b["uppercase_word_fraction"])
        + abs(a["punctuation_fraction"] - b["punctuation_fraction"])
    )


def matching_cost(probe, control):
    return feature_matching_cost(surface_features(probe),
                                 surface_features(control))


def match_controls(probes, candidates):
    """Greedily choose distinct controls using one declared surface-feature cost."""
    used = set()
    pairs = []
    # Hardest-to-match first: short and symbol-dense probes have fewer controls.
    ordered = sorted(enumerate(probes), key=lambda item: (
        -surface_features(item[1]["text"])["punctuation_fraction"],
        len(item[1]["text"]), item[1]["source_sha256"]))
    selected = {}
    for original_index, probe in ordered:
        available = [control for control in candidates
                     if control["source_sha256"] not in used]
        if not available:
            raise ValueError("not enough distinct prose controls")
        probe_features = surface_features(probe["text"])
        control = min(available, key=lambda item: (
            feature_matching_cost(probe_features, item["_features"]),
            item["source_path"], item["source_index"]))
        used.add(control["source_sha256"])
        selected[original_index] = (probe, control)
    for index in range(len(probes)):
        probe, control = selected[index]
        control = {key: value for key, value in control.items()
                   if not key.startswith("_")}
        pairs.append((probe, control))
    return pairs


def summarize(rows):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row["category"], row["adapter"], row["seed"],
                 row["class"])].append(row)
    result = []
    for (category, adapter, seed, label), selected in sorted(grouped.items()):
        words = sum(row["words"] for row in selected)
        result.append({
            "category": category, "adapter": adapter, "seed": seed,
            "class": label, "n": len(selected),
            "wer": sum(row["errors"] for row in selected) / max(words, 1),
            "failed": sum(bool(row["failed"]) for row in selected),
            "substitutions": sum(row["substitutions"] for row in selected),
            "deletions": sum(row["deletions"] for row in selected),
            "insertions": sum(row["insertions"] for row in selected),
        })
    return result


def build_pair_manifest(pairs):
    manifest = []
    for probe, prose in pairs:
        manifest.append({
            "category": probe["category"],
            "probe_uid": probe["uid"], "probe_sha256": probe["source_sha256"],
            "probe_text": probe["text"],
            "probe_source_path": probe["source_path"],
            "probe_source_index": probe["source_index"],
            "prose_uid": prose["uid"], "prose_sha256": prose["source_sha256"],
            "prose_text": prose["text"],
            "prose_source_path": prose["source_path"],
            "prose_source_index": prose["source_index"],
            "probe_features": surface_features(probe["text"]),
            "prose_features": surface_features(prose["text"]),
            "absolute_feature_gap": feature_gap(probe["text"], prose["text"]),
            "matching_cost": matching_cost(probe["text"], prose["text"]),
        })
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default=DEFAULT_FIXTURE,
                    help="identity-bearing locked probe source")
    ap.add_argument("--config", default=os.path.join(APP, "config.json"))
    ap.add_argument("--adapters", nargs="+", default=list(DEFAULT_ADAPTERS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--limit-per-category", type=int, default=4)
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "ab_test_runtime", "nonprose_category_expansion"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments",
        "nonprose_category_expansion.json"))
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    if not args.adapters or not args.seeds:
        ap.error("at least one adapter and seed are required")
    if not os.path.isfile(args.config):
        raise SystemExit(f"TTS config does not exist: {args.config}")

    pairs = load_locked_pairs(args.source, args.limit_per_category)
    probes = [probe for probe, _ in pairs]
    pair_manifest = build_pair_manifest(pairs)

    adapter_paths = {}
    for adapter in args.adapters:
        path = os.path.join(REPO, "lora_models", adapter)
        if not (os.path.isfile(os.path.join(path, "adapter_config.json")) and
                os.path.isfile(os.path.join(path,
                                            "adapter_model.safetensors"))):
            raise SystemExit(f"adapter is incomplete: {adapter}")
        adapter_paths[adapter] = path

    checkpoint = args.checkpoint or args.out + ".checkpoint.json"
    fingerprint = get_category_run_fingerprint(
        args, pair_manifest, adapter_paths)
    expected_rows = []
    for adapter in args.adapters:
        for seed in args.seeds:
            for pair_index, (probe, prose) in enumerate(pairs):
                for label, chunk in (("probe", probe), ("prose", prose)):
                    wav = os.path.join(
                        args.out_dir,
                        f"{adapter}__s{seed}__p{pair_index}__{label}.wav")
                    expected_rows.append((
                        (adapter, seed, pair_index, label),
                        (chunk["uid"], chunk["source_sha256"],
                         os.path.relpath(wav, REPO))))
    try:
        rows, mismatch = get_resumable_rows(checkpoint, fingerprint)
        if mismatch:
            archive_checkpoint(checkpoint, mismatch)
            rows = []
        completed = validate_resumed_rows(rows, expected_rows)
    except ValueError as exc:
        raise SystemExit(f"invalid checkpoint: {exc}") from exc
    if rows:
        print(f"resumed {len(rows)} validated rows from {checkpoint}",
              flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    from experiments.provenance import provenance
    total = len(args.adapters) * len(args.seeds) * len(pairs) * 2
    if len(completed) < total:
        from tts import TTSEngine
        from experiments.generation import render
        from experiments.tts_output_validation import transcribe, validate
        with open(args.config, encoding="utf-8") as handle:
            engine = TTSEngine(json.load(handle))
    done = len(rows)
    for adapter in args.adapters:
        for seed in args.seeds:
            entry = {"type": "lora", "adapter_path": adapter_paths[adapter],
                     "seed": str(seed)}
            for pair_index, (probe, prose) in enumerate(pairs):
                for label, chunk in (("probe", probe), ("prose", prose)):
                    key = (adapter, seed, pair_index, label)
                    if key in completed:
                        continue
                    wav = os.path.join(
                        args.out_dir,
                        f"{adapter}__s{seed}__p{pair_index}__{label}.wav")
                    render(engine, chunk["text"], "", "X", {"X": entry},
                           entry, wav)
                    heard = transcribe(wav)
                    result = validate(chunk["text"], heard)
                    result.pop("detail", None)
                    result.update({
                        "category": probe["category"], "adapter": adapter,
                        "seed": seed, "pair": pair_index, "class": label,
                        "uid": chunk["uid"],
                        "source_sha256": chunk["source_sha256"],
                        "wav": os.path.relpath(wav, REPO),
                        "transcript": heard,
                    })
                    rows.append(result)
                    completed.add(key)
                    save_checkpoint(checkpoint, fingerprint, rows)
                    done += 1
                    print(f"[{done}/{total}] {adapter} seed={seed} "
                          f"{probe['category']} pair={pair_index} {label}: "
                          f"{result['errors']} errors "
                          f"({result['insertions']} insertions) "
                          f"{'FAIL' if result['failed'] else 'ok'}", flush=True)

    if len(rows) != total:
        raise SystemExit(f"incomplete matrix: {len(rows)}/{total}")
    doc = {
        "status": "complete",
        "provenance": provenance(__file__, args),
        "selection": {
            "rule": ("four locked, hand-reviewed saved-script entries per "
                     "predeclared category; no TTS outcome used; distinct prose "
                     "controls greedily minimize the recorded surface-feature "
                     "cost, hardest-to-match probes first"),
            "categories": list(CATEGORIES),
            "pairs": pair_manifest,
        },
        "summary": summarize(rows),
        "rows": rows,
        "limits": [
            "All probes come from one saved Re:Zero script library.",
            "Several publisher-front-matter forms are related rather than independent draws.",
            "Surface matching is optimized and reported, not assumed exact.",
            "WER measures content, not delivery quality.",
        ],
    }
    from utils import atomic_json_write
    atomic_json_write(doc, args.out)
    if os.path.exists(checkpoint):
        os.remove(checkpoint)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
