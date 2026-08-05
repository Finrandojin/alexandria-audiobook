"""Replicate the non-prose gap across adapters, seeds, and matched controls."""
import argparse
import collections
import hashlib
import json
import os
import re
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

DEFAULT_ADAPTERS = (
    "husky_tenor_30s_m_fantasy",
    "husky_soprano_20s_f",
    "warm_baritone_50s_m_gothic",
)
DEFAULT_SEEDS = (1234, 5678, 9012)
CHECKPOINT_SCHEMA = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_run_fingerprint(args, pair_manifest, adapter_paths,
                        harness_file=__file__):
    """Return everything that must match before rows may be resumed."""
    sources = [
        harness_file,
        os.path.join(APP, "experiments", "generation.py"),
        os.path.join(APP, "experiments", "tts_output_validation.py"),
        os.path.join(APP, "tts.py"),
    ]
    return {
        "schema": CHECKPOINT_SCHEMA,
        "sources": {os.path.relpath(path, REPO): file_sha256(path)
                    for path in sources},
        "source_artifact_sha256": file_sha256(args.source),
        "config_sha256": file_sha256(args.config),
        "adapters": {
            name: {
                filename: file_sha256(os.path.join(path, filename))
                for filename in ("adapter_config.json",
                                 "adapter_model.safetensors")
            }
            for name, path in adapter_paths.items()
        },
        "seeds": list(args.seeds),
        "limit": args.limit,
        "pairs": pair_manifest,
    }


def get_resumable_rows(path, fingerprint):
    """Purely inspect a checkpoint; return (rows, mismatch reason)."""
    if not os.path.exists(path):
        return [], None
    try:
        with open(path, encoding="utf-8") as handle:
            saved = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError(f"checkpoint is unreadable: {exc}") from exc
    if saved.get("status") != "in_progress":
        raise ValueError(
            f"checkpoint status is not in_progress: {saved.get('status')!r}")
    if saved.get("fingerprint") != fingerprint:
        return [], "checkpoint fingerprint does not match this run"
    rows = saved.get("rows")
    if not isinstance(rows, list):
        raise ValueError("checkpoint rows is not a list")
    return rows, None


def archive_checkpoint(path, reason):
    """Move an incompatible checkpoint aside, loudly, before starting fresh."""
    base = f"{path}.stale-{time.strftime('%Y%m%d-%H%M%S')}"
    archived = base
    suffix = 1
    while os.path.exists(archived):
        archived = f"{base}-{suffix}"
        suffix += 1
    os.replace(path, archived)
    print(f"REFUSING to resume: {reason}; moved checkpoint to {archived}",
          flush=True)
    return archived


def save_checkpoint(path, fingerprint, rows):
    """Atomically preserve every completed row for an exact-identity resume."""
    from utils import atomic_json_write
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    atomic_json_write({"status": "in_progress", "fingerprint": fingerprint,
                       "rows": rows}, path)


def validate_resumed_rows(rows, expected_rows):
    """Reject duplicate, foreign, or damaged rows instead of silently rerunning."""
    from experiments.generation import _check_riff_completeness
    expected = {key: value for key, value in expected_rows}
    seen = set()
    keyed_rows = []
    for row in rows:
        key = (row.get("adapter"), row.get("seed"), row.get("pair"),
               row.get("class"))
        if key in seen:
            raise ValueError(f"checkpoint has duplicate row key {key!r}")
        if key not in expected:
            raise ValueError(f"checkpoint has foreign row key {key!r}")
        seen.add(key)
        keyed_rows.append((key, row))
    for key, row in keyed_rows:
        wanted = expected[key]
        if (row.get("uid"), row.get("source_sha256"), row.get("wav")) != wanted:
            raise ValueError(f"checkpoint row {key!r} does not match its input")
        for field in ("words", "errors", "failed", "substitutions",
                      "deletions", "insertions", "transcript"):
            if field not in row:
                raise ValueError(f"checkpoint row {key!r} is missing {field}")
        if row["errors"] != (row["substitutions"] + row["deletions"]
                             + row["insertions"]):
            raise ValueError(
                f"checkpoint row {key!r} has inconsistent error counts")
        wav = os.path.realpath(os.path.join(REPO, row["wav"]))
        if os.path.commonpath((os.path.realpath(REPO), wav)) != os.path.realpath(REPO):
            raise ValueError(f"checkpoint row {key!r} WAV escapes the repository")
        try:
            import soundfile as sf
            info = sf.info(wav)
            if not info.frames or not info.samplerate:
                raise ValueError("WAV has no audio frames")
            with sf.SoundFile(wav) as handle:
                while handle.read(65536).size:
                    pass
            _check_riff_completeness(wav, "lora", str(key))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"checkpoint row {key!r} has unusable WAV: {exc}") from exc
    return seen


def surface_features(text):
    words = re.findall(r"[^\W\d_]+", text or "")
    chars = max(len(text or ""), 1)
    return {
        "chars": len(text or ""),
        "words": len(words),
        "digit_fraction": sum(c.isdigit() for c in text or "") / chars,
        "uppercase_word_fraction": (
            sum(len(w) > 1 and w.isupper() for w in words) / max(len(words), 1)),
        "punctuation_fraction": (
            sum(not c.isalnum() and not c.isspace() for c in text or "") / chars),
    }


def feature_gap(first, second):
    a, b = surface_features(first), surface_features(second)
    return {key: abs(a[key] - b[key]) for key in a}


def load_selected_pairs(source, limit):
    """Recover the mechanism experiment's passages and their prose pairs."""
    doc = json.load(open(source, encoding="utf-8"))
    target_uids = [r["uid"] for r in doc.get("rows", [])
                   if r.get("class") == "nonprose" and r.get("failed")][:limit]
    if len(target_uids) != limit:
        raise ValueError(f"source has {len(target_uids)} failing non-prose rows; "
                         f"{limit} required")
    from experiments.prose_vs_nonprose import load_chunks, match_pairs
    pool = load_chunks(argparse.Namespace(pool_library=True, voice="NARRATOR",
                                          script=""))
    paired = {a["uid"]: (a, b) for a, b in match_pairs(pool, limit=10000)}
    missing = [uid for uid in target_uids if uid not in paired]
    if missing:
        raise ValueError(f"selected non-prose rows have no recoverable pair: {missing}")
    return [paired[uid] for uid in target_uids]


def summarize(rows):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row["adapter"], row["seed"], row["class"])].append(row)
    result = []
    for (adapter, seed, label), selected in sorted(grouped.items()):
        words = sum(r["words"] for r in selected)
        result.append({
            "adapter": adapter, "seed": seed, "class": label,
            "n": len(selected),
            "wer": sum(r["errors"] for r in selected) / max(words, 1),
            "failed": sum(bool(r["failed"]) for r in selected),
            "substitutions": sum(r["substitutions"] for r in selected),
            "deletions": sum(r["deletions"] for r in selected),
            "insertions": sum(r["insertions"] for r in selected),
        })
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "prose_vs_nonprose_v3.json"))
    ap.add_argument("--config", default=os.path.join(APP, "config.json"))
    ap.add_argument("--adapters", nargs="+", default=list(DEFAULT_ADAPTERS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "ab_test_runtime", "nonprose_replication"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "nonprose_replication.json"))
    ap.add_argument("--checkpoint", default=None,
                    help="row-level resume file (defaults beside --out)")
    args = ap.parse_args()
    if not args.adapters or not args.seeds or args.limit < 1:
        ap.error("at least one adapter, seed, and passage are required")
    if not os.path.isfile(args.config):
        raise SystemExit(f"TTS config does not exist: {args.config}")

    pairs = load_selected_pairs(args.source, args.limit)
    pair_manifest = []
    for nonprose, prose in pairs:
        pair_manifest.append({
            "nonprose_uid": nonprose["uid"], "prose_uid": prose["uid"],
            "nonprose_sha256": hashlib.sha256(
                nonprose["text"].encode()).hexdigest(),
            "prose_sha256": hashlib.sha256(prose["text"].encode()).hexdigest(),
            "nonprose_features": surface_features(nonprose["text"]),
            "prose_features": surface_features(prose["text"]),
            "absolute_feature_gap": feature_gap(nonprose["text"], prose["text"]),
        })

    adapter_paths = {}
    for adapter in args.adapters:
        path = os.path.join(REPO, "lora_models", adapter)
        if not (os.path.isfile(os.path.join(path, "adapter_config.json")) and
                os.path.isfile(os.path.join(path, "adapter_model.safetensors"))):
            raise SystemExit(f"adapter is incomplete: {adapter}")
        adapter_paths[adapter] = path

    checkpoint = args.checkpoint or args.out + ".checkpoint.json"
    fingerprint = get_run_fingerprint(args, pair_manifest, adapter_paths)
    expected_rows = []
    for adapter in args.adapters:
        for seed in args.seeds:
            for pair_index, (nonprose, prose) in enumerate(pairs):
                for label, chunk in (("nonprose", nonprose), ("prose", prose)):
                    wav = os.path.join(
                        args.out_dir,
                        f"{adapter}__s{seed}__p{pair_index}__{label}.wav")
                    key = (adapter, seed, pair_index, label)
                    value = (chunk["uid"], hashlib.sha256(
                        chunk["text"].encode()).hexdigest(),
                        os.path.relpath(wav, REPO))
                    expected_rows.append((key, value))

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
            for pair_index, (nonprose, prose) in enumerate(pairs):
                for label, chunk in (("nonprose", nonprose), ("prose", prose)):
                    key = (adapter, seed, pair_index, label)
                    wav = os.path.join(
                        args.out_dir,
                        f"{adapter}__s{seed}__p{pair_index}__{label}.wav")
                    if key in completed:
                        continue
                    render(engine, chunk["text"], "", "X", {"X": entry},
                           entry, wav)
                    heard = transcribe(wav)
                    result = validate(chunk["text"], heard)
                    result.pop("detail", None)
                    result.update({
                        "adapter": adapter, "seed": seed,
                        "pair": pair_index, "class": label,
                        "uid": chunk["uid"],
                        "source_sha256": hashlib.sha256(
                            chunk["text"].encode()).hexdigest(),
                        "wav": os.path.relpath(wav, REPO),
                        "transcript": heard,
                    })
                    rows.append(result)
                    completed.add(key)
                    save_checkpoint(checkpoint, fingerprint, rows)
                    done += 1
                    print(f"[{done}/{total}] {adapter} seed={seed} pair={pair_index} "
                          f"{label}: {result['errors']} errors "
                          f"({result['insertions']} insertions) "
                          f"{'FAIL' if result['failed'] else 'ok'}")

    if len(rows) != total:
        raise SystemExit(f"incomplete matrix: {len(rows)}/{total}")
    doc = {
        "status": "complete",
        "provenance": provenance(__file__, args),
        "selection": {
            "rule": "first failing non-prose rows used by the mechanism source; paired through prose_vs_nonprose.match_pairs",
            "pairs": pair_manifest,
        },
        "summary": summarize(rows),
        "rows": rows,
        "limits": [
            "Selected passages begin from earlier observed failures.",
            "Feature matching is measured and reported, not assumed perfect.",
            "WER measures content, not delivery quality.",
        ],
    }
    from utils import atomic_json_write
    atomic_json_write(doc, args.out)
    if os.path.exists(checkpoint):
        os.remove(checkpoint)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
