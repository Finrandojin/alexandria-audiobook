"""Durable record for an attribution experiment.

Aggregate tables cannot support an architecture decision: a later reader cannot
tell a real result from a prompt, roster, alias, indexing or scoring difference.
Every run writes its environment, its exact inputs, and one record per scored
line, so any number in a report can be recomputed from the artifact.

Process idleness is recorded from LM Studio and the app's own state, not
inferred from a process search - `pgrep -f` matched its own command line three
times during the 2026-07-26 experiments and gave the wrong answer each time.
"""
import collections
import hashlib
import json
import os
import platform
import subprocess
import time


def _sha(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _source_fingerprint(directory):
    """Hash of every harness source file, in name order."""
    digest = hashlib.sha256()
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(directory, name), "rb") as handle:
            digest.update(name.encode("utf-8"))
            digest.update(handle.read())
    return digest.hexdigest()


def _git_state(repo):
    def run(*args):
        try:
            out = subprocess.run(args, cwd=repo, capture_output=True, timeout=10)
            return out.stdout.decode("utf-8").strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    # Untracked notes and scratch files do not change behaviour; modified
    # tracked files do. Reporting the former as "dirty" made the flag useless -
    # it was true on every run because three markdown drafts sat in the tree.
    modified = run("git", "status", "--porcelain", "--untracked-files=no")
    # An untracked harness is the dangerous case, and the first version missed
    # it: a new experiment script is untracked while it runs, so the tree
    # reported clean and the artifact claimed a commit that did not contain the
    # code that produced it. Untracked .py inside the harness directory is dirt.
    harness_dir = os.path.dirname(os.path.abspath(__file__))
    untracked = [n for n in (run("git", "ls-files", "--others",
                                 "--exclude-standard", "--", harness_dir)
                             or "").splitlines() if n.endswith(".py")]
    return {"commit": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(modified) or bool(untracked),
            "modified_tracked_files": (modified or "").splitlines() or None,
            "untracked_harness_files": untracked or None,
            # The commit identifies the repository; this identifies the code
            # that actually ran, which is what a later reader needs to trust a
            # number produced from an edited working tree.
            "harness_sha256": _source_fingerprint(os.path.dirname(__file__))}


class EnvironmentCaptureError(RuntimeError):
    """The run's environment could not be recorded, so it is not comparable."""


def lmstudio_state(model_name):
    """What the server actually has loaded, and how it is configured.

    Raises rather than returning an error string. A GPU result whose context
    length and parallel setting are unknown cannot be compared against another
    run, and this project's determinism claim depends on both. The first
    version swallowed a TypeError from calling the helper with the wrong
    signature, and three artifacts shipped with no environment at all.
    """
    from lmstudio_settings import get_lmstudio_status
    status = get_lmstudio_status(model_name)
    if not isinstance(status, dict) or not status.get("available"):
        raise EnvironmentCaptureError(
            f"LM Studio status unavailable for {model_name!r}: {status!r}")
    if not status.get("loaded"):
        raise EnvironmentCaptureError(
            f"{model_name!r} is not loaded; refusing to record a run whose "
            "model state is unknown")
    state = {key: status.get(key) for key in
             ("loaded", "context_length", "parallel", "optimized")}
    # get_lmstudio_status matches on identifier/modelKey, so loaded=True is
    # itself confirmation that *this* model is the one loaded - recorded
    # explicitly rather than re-parsing `lms ps` in a second place.
    state["verified_model"] = model_name
    return state


class ExperimentRecord:
    """Collect per-line records, then write one self-describing artifact."""

    def __init__(self, name, repo, model_name, base_url, gold_path,
                 decoding, notes="", environment=None):
        """environment: pass a captured state to skip the live query. Real runs
        leave it None so a missing environment aborts before any GPU time is
        spent; tests supply one so they need no server."""
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
            "lmstudio": (environment if environment is not None
                         else lmstudio_state(model_name)),
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
        if getattr(self, "_ckpt_path", None):
            self._ckpt_done.add((arm, gold_id))
            self._ckpt_save()


    # ---- TEMPORARY: row-level checkpoint/resume -------------------------
    # Added 2026-07-27 because the Thunder tunnel dropped twice in ninety
    # minutes, and the second drop killed magistral-small's five-arm run three
    # minutes in. Retry handles a blip; it cannot save a run when the endpoint
    # is gone for minutes, and the arms run for hours.
    #
    # REMOVE THIS when experiments no longer run against a remote endpoint that
    # can vanish mid-run, or when a durable queue replaces ad-hoc scripts. It
    # exists to protect GPU hours, not because resumable experiments are a
    # design goal - a resumed artifact is inherently weaker evidence than a
    # single-process one, for the reason below.
    #
    # THE HAZARD: resume can silently merge rows produced under DIFFERENT
    # configurations, which is worse than losing the run because the artifact
    # still looks valid. So a checkpoint is only adopted when the experiment
    # name, model, endpoint, gold fixture hash, harness source hash AND decoding
    # settings all match. Anything else and the stale file is moved aside and
    # the run starts clean, loudly.

    def enable_checkpoint(self, path, save_every=25):
        """Resume from `path` if it matches this run exactly; else start fresh."""
        self._ckpt_path = path
        self._ckpt_every = save_every
        self._ckpt_pending = 0
        self._ckpt_done = set()
        self.meta["checkpoint"] = {"path": os.path.basename(path),
                                   "resumed": False, "rows_restored": 0,
                                   "temporary": True}
        if not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as handle:
                saved = json.load(handle)
        except (OSError, ValueError) as exc:
            print(f"  checkpoint unreadable ({exc}); starting fresh", flush=True)
            return
        mine, theirs = self._ckpt_fingerprint(self.meta), saved.get("fingerprint")
        if mine != theirs:
            stale = path + ".stale"
            os.replace(path, stale)
            differing = [k for k in mine
                         if mine.get(k) != (theirs or {}).get(k)]
            print(f"  REFUSING to resume: checkpoint does not match this run "
                  f"(differs on {', '.join(differing) or 'structure'}). "
                  f"Moved to {os.path.basename(stale)}; starting fresh.",
                  flush=True)
            return
        self.rows = saved.get("rows") or []
        self._ckpt_done = {(r["arm"], r["id"]) for r in self.rows}
        self.meta["checkpoint"].update({"resumed": True,
                                        "rows_restored": len(self.rows)})
        print(f"  resumed {len(self.rows)} rows from checkpoint "
              f"({len(self._ckpt_done)} arm/id pairs already done)", flush=True)

    @staticmethod
    def _ckpt_fingerprint(meta):
        """Everything that must be identical for two runs to share an artifact."""
        return {"experiment": meta.get("experiment"),
                "model": meta.get("model"),
                "endpoint": meta.get("endpoint"),
                "gold_sha256": meta.get("gold_sha256"),
                "harness_sha256": (meta.get("git") or {}).get("harness_sha256"),
                "decoding": meta.get("decoding")}

    def done(self, arm, gold_id):
        """True if this arm/id was already scored (in this run or a resumed one)."""
        return (arm, gold_id) in getattr(self, "_ckpt_done", ())

    def _ckpt_save(self, force=False):
        path = getattr(self, "_ckpt_path", None)
        if not path:
            return
        self._ckpt_pending += 1
        if not force and self._ckpt_pending < self._ckpt_every:
            return
        self._ckpt_pending = 0
        payload = {"fingerprint": self._ckpt_fingerprint(self.meta),
                   "rows": self.rows}
        tmp = path + ".tmp"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Write-then-rename: a crash mid-write must not leave a half file that
        # the next run would either refuse or, worse, partially trust.
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(tmp, path)
    # ---- end TEMPORARY -------------------------------------------------

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

    def validate(self, contract=None):
        """Return problems that make this artifact untrustworthy.

        ``contract`` optionally states what the run was supposed to produce -
        ``expected_arms``, ``expected_ids``, ``require_clean_tree`` - because a
        run that silently drops an arm or half its lines still validates when
        the summary correctly describes the incomplete rows.

        Shared by every harness, because the same two defects have now appeared
        in three separate scripts: a duplicate (arm, gold_id) counts one
        judgement twice, and a summary that does not follow from the rows means
        the reported number cannot be checked. Relying on each new script to
        get identity and aggregation right has produced drift every time.
        """
        problems = []
        seen = collections.Counter((row["arm"], row["id"]) for row in self.rows)
        duplicates = sorted(key for key, count in seen.items() if count > 1)
        if duplicates:
            problems.append(
                f"{len(duplicates)} duplicate (arm, id) identities, "
                f"e.g. {duplicates[:3]}")
        for arm, bucket in self.summary().items():
            rows = [r for r in self.rows if r["arm"] == arm]
            if bucket["n"] != len(rows):
                problems.append(f"{arm}: summary n={bucket['n']} but "
                                f"{len(rows)} rows")
            recomputed = sum(1 for r in rows if r["correct"])
            if bucket["correct"] != recomputed:
                problems.append(f"{arm}: summary correct={bucket['correct']} "
                                f"but rows give {recomputed}")
        contract = contract or {}
        environment = self.meta.get("lmstudio") or {}
        if not environment.get("loaded"):
            problems.append("no LM Studio load state recorded")
        for field in ("context_length", "parallel"):
            if environment.get(field) is None:
                problems.append(f"environment is missing {field}")
        # Deliberately not fatal by default: "optimized" compares the load
        # against an ideal computed from live VRAM at query time, so it moves
        # with whatever else is on the card and read False during a run whose
        # settings were correct. What matters for comparability is the recorded
        # context_length and parallel, which are checked above. A contract may
        # still demand it.
        if contract and contract.get("require_optimized") and \
                environment.get("optimized") is False:
            problems.append("model was loaded with non-ideal settings")
        # A cascade runs two models in one experiment and declares both, as
        # "cheap + expensive". The environment can only ever verify the one
        # currently loaded, so accept a match against any declared component
        # rather than the whole string - the check still catches the case it
        # exists for, which is an artifact naming a model the box was not
        # actually running.
        declared = self.meta.get("model") or ""
        components = {part.strip() for part in declared.split("+") if part.strip()}
        components.add(declared)
        if environment.get("verified_model") not in components | {None}:
            problems.append(
                f"loaded model {environment.get('verified_model')!r} is not the "
                f"declared model {self.meta.get('model')!r}")
        if not self.meta.get("git", {}).get("harness_sha256"):
            problems.append("no harness fingerprint: the code that ran is unidentified")

        arms = set(self.summary())
        expected_arms = contract.get("expected_arms")
        if expected_arms is not None and arms != set(expected_arms):
            problems.append(f"arms {sorted(arms)} != expected {sorted(expected_arms)}")
        expected_ids = contract.get("expected_ids")
        if expected_ids is not None:
            expected_ids = set(expected_ids)
            for arm in sorted(arms):
                got = {r["id"] for r in self.rows if r["arm"] == arm}
                if got != expected_ids:
                    problems.append(
                        f"{arm}: scored {len(got)} ids, expected "
                        f"{len(expected_ids)} (missing {len(expected_ids - got)}, "
                        f"unexpected {len(got - expected_ids)})")
        elif len(arms) > 1:
            # Even without a declared set, every arm must score the same lines
            # or the arms are not comparable.
            per_arm = {arm: {r["id"] for r in self.rows if r["arm"] == arm}
                       for arm in arms}
            reference = per_arm[sorted(arms)[0]]
            for arm, ids in sorted(per_arm.items()):
                if ids != reference:
                    problems.append(f"{arm} scored a different set of ids")
        if contract.get("require_clean_tree") and self.meta.get("git", {}).get("dirty"):
            problems.append("tree had modified tracked files: "
                            f"{self.meta['git'].get('modified_tracked_files')}")
        return problems

    def write(self, path, require_valid=True, contract=None):
        problems = self.validate(contract)
        if problems and require_valid:
            raise EnvironmentCaptureError(
                "refusing to write an unverifiable artifact: " + "; ".join(problems))
        self.meta["validation"] = problems or "ok"
        self.meta["finished"] = time.time()
        self.meta["elapsed_s"] = round(self.meta["finished"] - self.started, 1)
        payload = {"meta": self.meta, "summary": self.summary(), "rows": self.rows}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, ensure_ascii=False)
        ckpt = getattr(self, "_ckpt_path", None)
        if ckpt and os.path.exists(ckpt):
            os.remove(ckpt)
        return path
