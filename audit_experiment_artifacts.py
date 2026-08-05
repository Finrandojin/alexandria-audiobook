#!/usr/bin/env python3
"""Regenerate the structural experiment-artifact audit."""
import argparse
import collections
import glob
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)
EXPERIMENT_DIR = os.path.join(REPO, "ab_test_runtime", "experiments")
DEFAULT_OUT = os.path.join(
    REPO, "ab_test_runtime", "audit", "artifact_structural_audit.json")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_artifact(path):
    row = {"artifact": os.path.basename(path), "bytes": os.path.getsize(path),
           "sha256": file_sha256(path)}
    try:
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, ValueError) as exc:
        row.update({"identity_contract": None, "seed": None, "status": None,
                    "has_rows": False, "classification": "exploratory",
                    "reason": f"unreadable JSON: {type(exc).__name__}"})
        return row
    if not isinstance(doc, dict):
        row.update({"classification": "exploratory",
                    "reason": "top-level list, no embedded identity"})
        return row

    provenance = doc.get("provenance")
    meta = doc.get("meta")
    row.update({"seed": None, "status": doc.get("status"),
                "has_rows": isinstance(doc.get("rows"), list)})
    if isinstance(provenance, dict):
        git = provenance.get("git") or {}
        args = provenance.get("args") or {}
        row.update({"identity_contract": "tts_provenance",
                    "seed": provenance.get("seed", args.get("seed")),
                    "commit": git.get("commit"), "dirty": git.get("dirty")})
        if git.get("commit") and len(git.get("harness_sha256", "")) == 64:
            row.update({"classification": "supported_structure",
                        "reason": "embedded provenance and identifiable harness"})
        else:
            row.update({"classification": "exploratory",
                        "reason": "embedded provenance is incomplete"})
    elif isinstance(meta, dict):
        git = meta.get("git") or {}
        row.update({"identity_contract": "experiment_meta",
                    "seed": meta.get("seed"), "commit": git.get("commit"),
                    "dirty": git.get("dirty")})
        if git.get("commit"):
            row.update({"classification": "provisional",
                        "reason": "older metadata contract; semantic/manual review required"})
        else:
            row.update({"classification": "exploratory",
                        "reason": "metadata present without commit identity"})
    else:
        row.update({"identity_contract": "none",
                    "classification": "exploratory",
                    "reason": "no embedded provenance or experiment metadata"})
    return row


def build_audit(experiment_dir=EXPERIMENT_DIR):
    artifacts = [classify_artifact(path) for path in sorted(
        glob.glob(os.path.join(experiment_dir, "*.json")))]
    summary = dict(collections.Counter(
        row["classification"] for row in artifacts))
    return {
        "scope": "structural audit only; classifications do not validate scientific conclusions",
        "summary": summary, "artifacts": artifacts}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    audit = build_audit()
    if args.check:
        try:
            with open(args.out, encoding="utf-8") as handle:
                current = json.load(handle)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"structural audit is unreadable: {exc}") from exc
        if current != audit:
            raise SystemExit("structural audit is stale; regenerate it")
        print(f"structural audit is current ({len(audit['artifacts'])} artifacts)")
        return
    from utils import atomic_json_write
    atomic_json_write(audit, args.out)
    print(f"wrote {len(audit['artifacts'])} artifacts to {args.out}")


if __name__ == "__main__":
    main()
