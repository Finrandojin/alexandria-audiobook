#!/usr/bin/env python3
"""Audit every legacy ExperimentRecord artifact against current evidence."""
import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "app"))
from experiments.manifest import ExperimentRecord  # noqa: E402
from experiments.scoring import alias_groups, same_speaker  # noqa: E402

STRUCTURAL = os.path.join(
    REPO, "ab_test_runtime", "audit", "artifact_structural_audit.json")
EXPERIMENTS = os.path.join(REPO, "ab_test_runtime", "experiments")
DEFAULT_JSON = os.path.join(
    REPO, "ab_test_runtime", "audit", "legacy_attribution_audit.json")
DEFAULT_MD = os.path.join(REPO, "LEGACY_ATTRIBUTION_AUDIT_2026-08-05.md")

FAMILY_LIMITS = {
    "batch_contiguity": "Isolates companion ordering, not end-to-end production quality.",
    "batch_size": "Accuracy and throughput must be considered together; books differ.",
    "because_production": "A justification field test; explanations are not confidence estimates.",
    "candidate_id": "One model/corpus comparison; opaque IDs do not prove general naming gains.",
    "closed_set": "Oracle candidate arms are invalid for current claims because their lists used superseded labels.",
    "committed_history": "Oracle history is an upper bound and is not shippable state.",
    "context_width": "A harness diagnostic; production-path confirmation is separate.",
    "context_width_production": "Book-specific repeats; report each book/repeat rather than pooling.",
    "grammar_constraint": "Roster-valid output does not establish correct speaker identity.",
    "joint_scene": "Joint and shuffled controls answer ordering only within the tested fixtures.",
    "lora_serving_eval": "Two gold books and one serving stack; not a universal adapter claim.",
    "narrator_prior": "A predeclared book-contrast test, not a general narrator rule.",
    "pdnc_narrator_prior": "Two books and 120 rows per book with an explicitly supplied narrator identity; not a general held-out attribution result.",
    "reasoning_arms": "Reasoning/justification settings are model- and serving-stack-specific.",
    "reasoning_check": "Justification disagreement is a routing signal, not calibrated confidence.",
    "reexamine": "Selected previously negative results; selection prevents broad inference.",
    "roster_quality": "Gold-roster arms are upper bounds and not deployable inputs.",
    "roster_warmup": "Book-quartile diagnostic; oracle roster is not deployable.",
    "scene_cast": "Scene-cast extraction and attribution effects cannot be conflated.",
    "segmentation_crossover": "Factorial diagnostic on one book; retain repeat-level uncertainty.",
    "tag_priority": "Prompt rule effects vary by book/model and require per-book reporting.",
    "two_by_two": "The two factors are not independent; this prices context, not batching.",
    "voting": "Voting cost and routing coverage accompany accuracy; no pooled policy claim.",
}


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _commit_is_in_history(commit):
    if not commit:
        return False
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=REPO,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def _current_gold(meta, rows):
    path = os.path.join(REPO, str(meta.get("gold_path") or ""))
    result = {"available": False, "hash_changed": None, "missing_rows": None,
              "expected_changed_rows": None, "correctness_changed_rows": None}
    if not os.path.isfile(path):
        return result
    with open(path, "rb") as handle:
        raw = handle.read()
    fixture = json.loads(raw)
    entries = {entry["id"]: entry for entry in fixture.get("entries", [])}
    groups = alias_groups(fixture)
    missing = expected_changed = correctness_changed = 0
    for row in rows:
        row_id = str(row.get("id", "")).split(":")[-1]
        entry = entries.get(row_id)
        if entry is None:
            missing += 1
            continue
        expected = entry.get("expected_speaker")
        expected_changed += str(row.get("expected") or "").upper() != \
            str(expected or "").upper()
        rescored = bool(row.get("predicted")) and same_speaker(
            expected, row.get("predicted"), groups)
        correctness_changed += rescored != bool(row.get("correct"))
    result.update({
        "available": True,
        "hash_changed": _sha256(raw) != meta.get("gold_sha256"),
        "line_count_changed": len(entries) != meta.get("gold_lines"),
        "missing_rows": missing,
        "expected_changed_rows": expected_changed,
        "correctness_changed_rows": correctness_changed,
    })
    return result


def inspect_artifact(name):
    path = os.path.join(EXPERIMENTS, name)
    with open(path, encoding="utf-8") as handle:
        doc = json.load(handle)
    meta, rows = doc["meta"], doc["rows"]
    record = ExperimentRecord.__new__(ExperimentRecord)
    record.meta, record.rows = meta, rows
    problems = list(record.validate())
    if record.summary() != doc.get("summary"):
        problems.append("saved summary differs from row recomputation")
    if not _commit_is_in_history((meta.get("git") or {}).get("commit")):
        problems.append("recorded commit is unavailable from current history")
    if meta.get("validation") != "ok":
        problems.append("artifact validation is not ok")
    gold = _current_gold(meta, rows)
    core = bool(problems)
    score_stale = bool(gold.get("correctness_changed_rows") or
                       gold.get("missing_rows"))
    # WHAT MAKES A MEASUREMENT PROVISIONAL IS THE ANSWERS MOVING, NOT THE BYTES.
    # This used to key on hash_changed, a sha256 of the whole gold file, so
    # correcting a description field downgraded every measurement scored against
    # that book. On 2026-08-06 fixing one false sentence - the gold said
    # "Hand-labelled" when its own status field said no human read the passages -
    # took supported_measurement from 39 to 1 without a single label changing.
    #
    # That is an incentive to leave documentation wrong, so the check now asks
    # whether the ANSWERS moved: a changed expected_speaker, or a different
    # number of rows. Alias edits are not missed by this - they change scoring,
    # which correctness_changed_rows already catches above as historical_only.
    # hash_changed is still recorded, because "these bytes differ" is true and
    # worth knowing; it just no longer decides the classification alone.
    answers_changed = bool(gold.get("expected_changed_rows") or
                           gold.get("line_count_changed"))
    if core:
        classification = "exploratory"
    elif score_stale:
        classification = "historical_only"
    elif (meta.get("git") or {}).get("dirty") or answers_changed:
        classification = "provisional"
    else:
        classification = "supported_measurement"
    family = meta.get("experiment")
    if family not in FAMILY_LIMITS:
        raise RuntimeError(f"no declared semantic limit for family {family!r}")
    return {
        "artifact": name, "family": family, "rows": len(rows),
        "arms": sorted({row["arm"] for row in rows}),
        "classification": classification,
        "dirty": bool((meta.get("git") or {}).get("dirty")),
        "problems": sorted(set(problems)), "current_gold": gold,
        "semantic_limit": FAMILY_LIMITS[family],
    }


def build_audit():
    with open(STRUCTURAL, "rb") as handle:
        structural_sha256 = _sha256(handle.read())
    with open(STRUCTURAL, encoding="utf-8") as handle:
        structural = json.load(handle)
    names = [row["artifact"] for row in structural["artifacts"]
             if row["classification"] == "provisional"]
    artifacts = [inspect_artifact(name) for name in sorted(names)]
    if len(artifacts) != len(names) or len({r["artifact"] for r in artifacts}) != len(names):
        raise RuntimeError("legacy audit did not cover every provisional structural artifact")
    return {
        "scope": "legacy ExperimentRecord artifacts; integrity and current-gold compatibility, not perceptual evidence",
        "source_structural_sha256": structural_sha256,
        "summary": dict(sorted(collections.Counter(
            row["classification"] for row in artifacts).items())),
        "artifacts": artifacts,
    }


def render_markdown(audit):
    lines = ["# Legacy attribution audit — 2026-08-05", "",
             f"All {len(audit['artifacts'])} legacy-metadata artifacts are listed exactly once. "
             "Classification describes whether the recorded measurement can be "
             "used with today's fixtures; it does not turn accuracy into a "
             "product or perceptual conclusion.", "",
             "## Counts", ""]
    for key, value in audit["summary"].items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "`historical_only` means current-gold rescoring changes at "
              "least one judgment or cannot map at least one row. Original files "
              "remain preserved; their saved summaries were not rewritten.", "",
              "## Per-artifact audit", "",
              "| artifact | family | class | rows | changed scores | unmapped | dirty | problems |",
              "|---|---|---|---:|---:|---:|---|---|"]
    for row in audit["artifacts"]:
        gold = row["current_gold"]
        lines.append("| `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
            row["artifact"], row["family"], row["classification"], row["rows"],
            gold.get("correctness_changed_rows"), gold.get("missing_rows"),
            row["dirty"], "; ".join(row["problems"])))
    lines += ["", "## Family-level interpretation limits", ""]
    for family, limit in sorted(FAMILY_LIMITS.items()):
        lines.append(f"- `{family}`: {limit}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", default=DEFAULT_JSON)
    parser.add_argument("--markdown", default=DEFAULT_MD)
    args = parser.parse_args()
    audit = build_audit()
    markdown = render_markdown(audit)
    if args.check:
        with open(args.json, encoding="utf-8") as handle:
            existing = json.load(handle)
        with open(args.markdown, encoding="utf-8") as handle:
            existing_md = handle.read()
        if existing != audit or existing_md != markdown:
            raise SystemExit("legacy attribution audit is stale")
        print(f"legacy attribution audit is current ({len(audit['artifacts'])} artifacts)")
        return
    from utils import atomic_json_write
    atomic_json_write(audit, args.json)
    tmp = args.markdown + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    os.replace(tmp, args.markdown)
    print(f"wrote {len(audit['artifacts'])} artifacts; {audit['summary']}")


if __name__ == "__main__":
    main()
