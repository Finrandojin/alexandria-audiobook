"""Durable record for an attribution experiment.

Aggregate tables cannot support an architecture decision: a later reader cannot
tell a real result from a prompt, roster, alias, indexing or scoring difference.
Every run writes its environment, its exact inputs, and one record per scored
line, so any number in a report can be recomputed from the artifact.

Process idleness is recorded from LM Studio and the app's own state, not
inferred from a process search - `pgrep -f` matched its own command line three
times during the 2026-07-26 experiments and gave the wrong answer each time.
"""
import hashlib
import json
import os
import platform
import subprocess
import time


def _sha(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _git_state(repo):
    def run(*args):
        try:
            out = subprocess.run(args, cwd=repo, capture_output=True, timeout=10)
            return out.stdout.decode("utf-8").strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    return {"commit": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            # A dirty tree means the commit alone does not identify the code.
            "dirty": bool(run("git", "status", "--porcelain"))}


def lmstudio_state(base_url, model_name):
    """What the server actually has loaded, and how it is configured."""
    try:
        from lmstudio_settings import get_lmstudio_status
        status = get_lmstudio_status(base_url, model_name)
        return {k: status.get(k) for k in
                ("loaded", "context_length", "parallel", "quantization")
                if k in status}
    except Exception as error:            # never let bookkeeping fail a run
        return {"error": f"{type(error).__name__}: {error}"}


class ExperimentRecord:
    """Collect per-line records, then write one self-describing artifact."""

    def __init__(self, name, repo, model_name, base_url, gold_path,
                 decoding, notes=""):
        self.name = name
        self.started = time.time()
        with open(gold_path, "rb") as handle:
            gold_bytes = handle.read()
        self.meta = {
            "experiment": name,
            "notes": notes,
            "git": _git_state(repo),
            "host": platform.node(),
            "model": model_name,
            "endpoint": base_url,
            "lmstudio": lmstudio_state(base_url, model_name),
            "decoding": dict(decoding),
            "gold_path": os.path.relpath(gold_path, repo),
            "gold_sha256": hashlib.sha256(gold_bytes).hexdigest(),
            "gold_lines": len(json.loads(gold_bytes)["entries"]),
        }
        self.rows = []

    def add(self, arm, gold_id, line, expected, predicted, correct,
            candidates=None, provenance=None, prompt=None, raw=None,
            retries=None):
        """One scored line. Prompts are hashed; raw responses kept verbatim."""
        self.rows.append({
            "arm": arm,
            "id": gold_id,
            "line": line,
            "expected": expected,
            "predicted": predicted,
            "correct": bool(correct),
            "candidates": candidates,
            "candidate_provenance": provenance,
            "in_candidates": (None if candidates is None
                              else expected in (candidates or [])),
            "prompt_sha256": _sha(prompt) if prompt is not None else None,
            "prompt_chars": len(prompt) if prompt is not None else None,
            "raw_response": raw,
            "retries": retries,
        })

    def summary(self):
        arms = {}
        for row in self.rows:
            bucket = arms.setdefault(row["arm"], {"n": 0, "correct": 0,
                                                  "available": 0, "cond": 0})
            bucket["n"] += 1
            bucket["correct"] += row["correct"]
            if row["in_candidates"]:
                bucket["available"] += 1
                bucket["cond"] += row["correct"]
        for bucket in arms.values():
            bucket["accuracy"] = bucket["correct"] / max(bucket["n"], 1)
            bucket["conditional"] = bucket["cond"] / max(bucket["available"], 1)
        return arms

    def write(self, path):
        self.meta["finished"] = time.time()
        self.meta["elapsed_s"] = round(self.meta["finished"] - self.started, 1)
        payload = {"meta": self.meta, "summary": self.summary(), "rows": self.rows}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, ensure_ascii=False)
        return path
