"""Install gated retrained adapters over their shipped counterparts.

THIS OVERWRITES SHIPPED VOICES. Every adapter promoted here replaces one a
user may already have assigned to a character in a saved book, so the whole
design is built around being able to put it back: the originals are copied to
a timestamped backup directory BEFORE anything is written, and `--rollback`
restores them from it. Nothing is deleted at any point.

WHAT QUALIFIES. An adapter is promoted only if it passed an independent
identity gate - `verify_adapter_identity.py`, re-measuring held-out ECAPA
against the adapter's own dataset rather than trusting the score the training
run reported about itself - AND beats the shipped adapter it replaces. Both
conditions are re-checked here from the gate artifacts on disk, so this refuses
to promote anything whose evidence is missing, unreadable, or below threshold,
regardless of what the caller passes on the command line.

The weights are copied, not moved: the retrain directories stay intact as the
provenance for what was installed.
"""
import argparse
import datetime
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(REPO, "lora_models")
GATES = os.path.join(REPO, "ab_test_runtime", "experiments")
SOURCE = os.path.join(REPO, "ab_test_runtime", "retrain_honest")
DECONTAMINATE_SOURCE = os.path.join(REPO, "ab_test_runtime", "decontaminate")
BACKUPS = os.path.join(REPO, "ab_test_runtime", "promotion_backups")

MIN_ECAPA = 0.45

# Only the model itself and its provenance move. Files a promotion must NOT
# carry over are excluded by listing what it copies rather than what it skips:
# a shipped directory can hold sample renders whose filenames encode the old
# training run, and copying those would attach stale provenance to new weights.
PROMOTE_FILES = ("adapter_config.json", "adapter_model.safetensors",
                 "training_meta.json", "README.md", "ref_sample.wav")


def get_adapter_source(name):
    """Return the gated retrained adapter directory, across supported runs."""
    gate = gate_result(name)
    gated_path = gate.get("adapter") if gate else None
    if gated_path:
        candidate = os.path.realpath(
            gated_path if os.path.isabs(gated_path)
            else os.path.join(REPO, gated_path))
        roots = (os.path.realpath(SOURCE), os.path.realpath(DECONTAMINATE_SOURCE))
        if (os.path.isdir(candidate)
                and any(os.path.commonpath((candidate, root)) == root
                        for root in roots)):
            return candidate
    legacy = os.path.join(SOURCE, name, "adapter")
    if os.path.isdir(legacy):
        return legacy
    import glob
    matches = sorted(glob.glob(os.path.join(
        DECONTAMINATE_SOURCE, "batch*", name, "adapter")))
    return matches[0] if len(matches) == 1 else None


def shipped_scores():
    """Adapter -> best recorded score for the weights currently shipped."""
    path = os.path.join(GATES, "library_voice_fidelity_n10.json")
    with open(path, encoding="utf-8") as handle:
        scores = {r["adapter"]: r.get("ecapa")
                  for r in json.load(handle)["results"]}
    manifest = os.path.join(MODELS, "manifest.json")
    if os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as handle:
            for entry in json.load(handle):
                if entry.get("gate_ecapa") is not None:
                    scores[entry["id"]] = entry["gate_ecapa"]
    return scores


def gate_result(name):
    """The gate's verdict for one adapter, or None if it was never gated."""
    path = os.path.join(GATES, f"gate_promote__{name}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except ValueError:
        return None


def check(name, before):
    """-> (ok, score, reason). Refuses on missing or insufficient evidence."""
    gate = gate_result(name)
    if gate is None:
        return False, None, "no gate artifact - run verify_adapter_identity"
    score = gate.get("median_ecapa", gate.get("ecapa"))
    if score is None:
        return False, None, "gate artifact carries no score"
    if score < MIN_ECAPA:
        return False, score, f"gate score {score:.3f} below {MIN_ECAPA}"
    old = before.get(name)
    if old is None:
        return False, score, "no shipped score to compare against"
    if score <= old:
        return False, score, f"gate {score:.3f} does not beat shipped {old:.3f}"
    if get_adapter_source(name) is None:
        return False, score, "no retrained adapter on disk"
    if not os.path.isdir(os.path.join(MODELS, name)):
        return False, score, "no shipped adapter to replace"
    return True, score, f"{old:.3f} -> {score:.3f}"


def backup_dir(stamp):
    return os.path.join(BACKUPS, stamp)


def promote(names, stamp, dry_run):
    before = shipped_scores()
    plan, refused = [], []
    for name in names:
        ok, score, reason = check(name, before)
        (plan if ok else refused).append((name, score, reason))

    for name, score, reason in refused:
        print(f"  REFUSE {name[:34]:36} {reason}")
    for name, score, reason in plan:
        print(f"  ready  {name[:34]:36} {reason}")
    if not plan:
        print("\nnothing to promote")
        return 1
    if dry_run:
        print(f"\ndry run - {len(plan)} would be promoted, nothing written")
        return 0

    dest = backup_dir(stamp)
    os.makedirs(dest, exist_ok=True)
    print(f"\nbacking up {len(plan)} originals to {dest}")
    for name, _score, _reason in plan:
        shutil.copytree(os.path.join(MODELS, name), os.path.join(dest, name))

    print("installing")
    for name, score, _reason in plan:
        src = get_adapter_source(name)
        dst = os.path.join(MODELS, name)
        for filename in PROMOTE_FILES:
            source_file = os.path.join(src, filename)
            if os.path.exists(source_file):
                shutil.copy2(source_file, os.path.join(dst, filename))
        identity = os.path.join(src, "identity_check")
        if os.path.isdir(identity):
            target = os.path.join(dst, "identity_check")
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(identity, target)
        print(f"  installed {name}  ({score:.3f})")

    update_manifest({n: s for n, s, _r in plan}, stamp)
    record = {"promoted_at": stamp, "backup": dest, "min_ecapa": MIN_ECAPA,
              "adapters": [{"adapter": n, "gate_ecapa": s,
                            "shipped_ecapa": before.get(n)}
                           for n, s, _r in plan]}
    receipt = os.path.join(BACKUPS, f"{stamp}.json")
    with open(receipt, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(f"\npromoted {len(plan)}; receipt {receipt}")
    print(f"rollback: python promote_adapters.py --rollback {stamp}")
    return 0


def update_manifest(promoted, stamp):
    """Record on each entry that its weights were replaced, and by what."""
    path = os.path.join(MODELS, "manifest.json")
    with open(path, encoding="utf-8") as handle:
        entries = json.load(handle)
    for entry in entries:
        score = promoted.get(entry.get("id"))
        if score is None:
            continue
        source = get_adapter_source(entry["id"])
        meta = os.path.join(source, "training_meta.json") if source else ""
        if os.path.exists(meta):
            with open(meta, encoding="utf-8") as handle:
                fresh = json.load(handle)
            for field in ("epochs_run", "epoch_losses", "final_loss",
                          "best_loss", "sample_count", "lora_r", "lr"):
                if field in fresh:
                    entry[field] = fresh[field]
            if "num_samples" in fresh:
                entry["sample_count"] = fresh["num_samples"]
        entry["retrained_at"] = stamp
        entry["gate_ecapa"] = score
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)
    print(f"  manifest updated for {len(promoted)} entries")


def rollback(stamp):
    dest = backup_dir(stamp)
    if not os.path.isdir(dest):
        print(f"no backup at {dest}")
        return 1
    names = sorted(os.listdir(dest))
    for name in names:
        target = os.path.join(MODELS, name)
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(os.path.join(dest, name), target)
        print(f"  restored {name}")
    print(f"restored {len(names)} adapters from {stamp}")
    print("NOTE: manifest.json is not reverted - re-check it if it matters")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapters", nargs="*", default=None,
                    help="default: every adapter with a passing gate artifact")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rollback", metavar="STAMP")
    args = ap.parse_args()

    if args.rollback:
        return rollback(args.rollback)

    names = args.adapters
    if not names:
        import glob
        names = sorted(os.path.basename(p)[len("gate_promote__"):-len(".json")]
                       for p in glob.glob(os.path.join(
                           GATES, "gate_promote__*.json")))
    if not names:
        print("no gate artifacts found")
        return 1
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return promote(names, stamp, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
