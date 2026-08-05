"""Test deterministic identifier/date pronunciation across independent books.

This is an offline experiment. It does not alter production verbalization.
Each raw metadata passage is paired with one deterministic spoken rendering,
using identical adapters and seeds. Results are retained per book/category;
pooled improvement alone is not a production gate.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

from experiments.nonprose_replication import (  # noqa: E402
    DEFAULT_ADAPTERS, DEFAULT_SEEDS, archive_checkpoint, get_resumable_rows,
    get_run_fingerprint, save_checkpoint, validate_resumed_rows)

DEFAULT_FIXTURE = os.path.join(APP, "experiments", "crossbook_normalization_fixture.json")
MONTHS = {"Jan": "January", "Feb": "February", "Mar": "March",
          "Apr": "April", "Jun": "June", "Jul": "July", "Aug": "August",
          "Sep": "September", "Oct": "October", "Nov": "November",
          "Dec": "December"}
ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
            6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
            11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
            15: "fifteenth", 16: "sixteenth", 17: "seventeenth",
            18: "eighteenth", 19: "nineteenth", 20: "twentieth",
            21: "twenty first", 22: "twenty second", 23: "twenty third",
            24: "twenty fourth", 25: "twenty fifth", 26: "twenty sixth",
            27: "twenty seventh", 28: "twenty eighth", 29: "twenty ninth",
            30: "thirtieth", 31: "thirty first"}


def get_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_identifier(text):
    match = re.fullmatch(r"eBook-No\. (\d+)", text)
    if not match:
        raise ValueError(f"unsupported identifier form: {text!r}")
    from experiments.tts_output_validation import ONES
    return "e book number " + " ".join(ONES[int(char)] for char in match.group(1))


def normalize_date(text):
    match = re.fullmatch(r"Release Date: ([A-Z][a-z]{2}) (\d{1,2}), (\d{4})", text)
    if not match or match.group(1) not in MONTHS:
        raise ValueError(f"unsupported date form: {text!r}")
    from experiments.tts_output_validation import say_number
    day = int(match.group(2))
    return (f"release date {MONTHS[match.group(1)]} {ORDINALS[day]}, "
            + " ".join(say_number(match.group(3))))


def load_locked_samples(path=DEFAULT_FIXTURE):
    with open(path, encoding="utf-8") as handle:
        doc = json.load(handle)
    sources = doc.get("sources")
    if not isinstance(sources, list) or len(sources) < 3:
        raise ValueError("fixture requires at least three independent books")
    samples = []
    books = set()
    for source in sources:
        book = source.get("book")
        if not book or book in books or not str(source.get("url", "")).startswith(
                "https://www.gutenberg.org/ebooks/"):
            raise ValueError("fixture books and Project Gutenberg URLs must be unique")
        books.add(book)
        for category, normalizer in (("identifier", normalize_identifier),
                                     ("date", normalize_date)):
            raw = source.get(category)
            normalized = normalizer(raw)
            samples.append({"book": book, "title": source.get("title"),
                            "url": source["url"], "category": category,
                            "raw": raw, "normalized": normalized,
                            "source_sha256": get_sha256(raw)})
    return samples, doc["selection_rule"]


def summarize(rows):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row["book"], row["category"], row["arm"])].append(row)
    summary = []
    for (book, category, arm), selected in sorted(grouped.items()):
        words = sum(row["words"] for row in selected)
        summary.append({"book": book, "category": category, "arm": arm,
                        "n": len(selected),
                        "wer": sum(row["errors"] for row in selected) / max(words, 1),
                        "failed": sum(bool(row["failed"]) for row in selected)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default=DEFAULT_FIXTURE)
    parser.add_argument("--config", default=os.path.join(APP, "config.json"))
    parser.add_argument("--adapters", nargs="+", default=list(DEFAULT_ADAPTERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--out-dir", default=os.path.join(REPO, "ab_test_runtime", "crossbook_normalization"))
    parser.add_argument("--out", default=os.path.join(REPO, "ab_test_runtime", "experiments", "crossbook_normalization.json"))
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    if not args.adapters or not args.seeds:
        parser.error("at least one adapter and seed are required")
    if not os.path.isfile(args.config):
        raise SystemExit(f"TTS config does not exist: {args.config}")
    samples, selection_rule = load_locked_samples(args.source)
    adapter_paths = {}
    for adapter in args.adapters:
        path = os.path.join(REPO, "lora_models", adapter)
        if not all(os.path.isfile(os.path.join(path, name)) for name in
                   ("adapter_config.json", "adapter_model.safetensors")):
            raise SystemExit(f"adapter is incomplete: {adapter}")
        adapter_paths[adapter] = path
    manifest = [{key: sample[key] for key in
                 ("book", "title", "url", "category", "raw", "normalized",
                  "source_sha256")} for sample in samples]
    fingerprint_args = argparse.Namespace(**vars(args), limit=len(samples))
    fingerprint = get_run_fingerprint(
        fingerprint_args, manifest, adapter_paths, __file__)
    checkpoint = args.checkpoint or args.out + ".checkpoint.json"
    expected = []
    for adapter in args.adapters:
        for seed in args.seeds:
            for index, sample in enumerate(samples):
                for arm in ("raw", "normalized"):
                    wav = os.path.join(args.out_dir,
                                       f"{adapter}__s{seed}__p{index}__{arm}.wav")
                    uid = f"{sample['book']}:{sample['category']}:{arm}"
                    expected.append(((adapter, seed, index, arm),
                                     (uid, sample["source_sha256"],
                                      os.path.relpath(wav, REPO))))
    try:
        rows, mismatch = get_resumable_rows(checkpoint, fingerprint)
        if mismatch:
            archive_checkpoint(checkpoint, mismatch)
            rows = []
        completed = validate_resumed_rows(rows, expected)
    except ValueError as exc:
        raise SystemExit(f"invalid checkpoint: {exc}") from exc
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    total = len(expected)
    if len(completed) < total:
        from experiments.generation import render
        from experiments.tts_output_validation import transcribe, validate
        from tts import TTSEngine
        with open(args.config, encoding="utf-8") as handle:
            engine = TTSEngine(json.load(handle))
    for adapter in args.adapters:
        for seed in args.seeds:
            entry = {"type": "lora", "adapter_path": adapter_paths[adapter],
                     "seed": str(seed)}
            for index, sample in enumerate(samples):
                for arm in ("raw", "normalized"):
                    key = (adapter, seed, index, arm)
                    if key in completed:
                        continue
                    text = sample[arm]
                    wav = os.path.join(args.out_dir,
                                       f"{adapter}__s{seed}__p{index}__{arm}.wav")
                    render(engine, text, "", "X", {"X": entry}, entry, wav)
                    heard = transcribe(wav)
                    result = validate(text, heard)
                    result.pop("detail", None)
                    result.update({"book": sample["book"],
                                   "category": sample["category"], "arm": arm,
                                   "class": arm,
                                   "adapter": adapter, "seed": seed,
                                   "pair": index,
                                   "uid": f"{sample['book']}:{sample['category']}:{arm}",
                                   "source_sha256": sample["source_sha256"],
                                   "wav": os.path.relpath(wav, REPO),
                                   "transcript": heard})
                    rows.append(result)
                    completed.add(key)
                    save_checkpoint(checkpoint, fingerprint, rows)
                    print(f"[{len(rows)}/{total}] {adapter} seed={seed} "
                          f"{sample['book']} {sample['category']} {arm}: "
                          f"{result['errors']} errors", flush=True)
    if len(rows) != total:
        raise SystemExit(f"incomplete matrix: {len(rows)}/{total}")
    from experiments.provenance import provenance
    from utils import atomic_json_write
    atomic_json_write({"status": "complete", "provenance": provenance(__file__, args),
                       "selection": {"rule": selection_rule, "samples": manifest},
                       "summary": summarize(rows), "rows": rows,
                       "limits": ["WER measures content adherence, not naturalness.",
                                  "Each category has three books; pooled-only claims are invalid.",
                                  "No production normalization behavior changed."]}, args.out)
    if os.path.exists(checkpoint):
        os.remove(checkpoint)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
