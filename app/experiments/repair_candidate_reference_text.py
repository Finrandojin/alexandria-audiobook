"""Repair medoid-reference transcripts on existing clean retrain candidates.

The decontamination run copied each chosen medoid audio to ``ref.wav`` but did
not write its transcript. ``train_lora.py`` therefore stored the first sample's
text beside different reference audio. Training weights are unaffected (they
use the audio embedding), but every inference and identity gate is invalid
until the matching text is restored.
"""
import argparse
import glob
import json
import os
import re
import sys


REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)

from experiments.provenance import provenance  # noqa: E402
from utils import atomic_json_write  # noqa: E402


def get_reference_text(data_dir, clip):
    metadata = os.path.join(data_dir, "train", "metadata.jsonl")
    matches = []
    with open(metadata, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if os.path.basename(row["audio_filepath"]) == clip:
                matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"expected one metadata row for {clip}, got {len(matches)}")
    text = str(matches[0].get("text") or "").strip()
    if not text:
        raise ValueError(f"empty transcript for {clip}")
    return text


def get_repaired_metadata(metadata, reference_text):
    repaired = dict(metadata)
    repaired["ref_sample_text"] = reference_text
    return repaired


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--experiments", default=os.path.join(
        REPO, "ab_test_runtime", "experiments"))
    parser.add_argument("--work", default=os.path.join(
        REPO, "ab_test_runtime", "decontaminate"))
    parser.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "experiments",
        "repair_candidate_reference_text.json"))
    args = parser.parse_args()

    repaired, skipped, errors = [], [], []
    pattern = os.path.join(args.experiments, "decontaminate_batch*.json")
    for artifact in sorted(glob.glob(pattern)):
        match = re.search(r"batch(\d+)\.json$", artifact)
        if not match:
            continue
        batch = "batch" + match.group(1)
        with open(artifact, encoding="utf-8") as handle:
            rows = json.load(handle).get("results", [])
        for row in rows:
            name = row.get("adapter")
            clip = (row.get("medoid_reference") or {}).get("clip")
            base = os.path.join(args.work, batch, str(name))
            meta_path = os.path.join(base, "adapter", "training_meta.json")
            try:
                if not name or not clip or not os.path.exists(meta_path):
                    skipped.append({"adapter": name, "batch": batch,
                                    "reason": "training produced no candidate"})
                    continue
                text = get_reference_text(os.path.join(base, "data"), clip)
                with open(meta_path, encoding="utf-8") as handle:
                    metadata = json.load(handle)
                changed = metadata.get("ref_sample_text") != text
                if changed:
                    atomic_json_write(get_repaired_metadata(metadata, text),
                                      meta_path)
                ref_text_path = os.path.join(base, "data", "ref_text.txt")
                if changed or not os.path.exists(ref_text_path):
                    with open(ref_text_path, "w", encoding="utf-8") as handle:
                        handle.write(text)
                repaired.append({"adapter": name, "batch": batch,
                                 "clip": clip, "changed": changed})
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                errors.append({"adapter": name, "batch": batch,
                               "error": str(exc)[:160]})

    result = {"repaired": repaired, "skipped": skipped, "errors": errors,
              "provenance": provenance(__file__, args)}
    atomic_json_write(result, args.out)
    print(f"repaired {sum(row['changed'] for row in repaired)}/{len(repaired)}; "
          f"skipped {len(skipped)}; errors {len(errors)}")
    print(f"wrote {args.out}")
    if errors or not repaired:
        sys.exit(3)


if __name__ == "__main__":
    main()
