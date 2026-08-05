"""Durable, fail-loud status records for resumable experiment runners."""
import datetime
import os

from experiments.provenance import file_sha256, provenance
from utils import atomic_json_write


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def start_experiment_status(path, experiment, script_file):
    """Create a new status record; an unfinished prior run must be inspected."""
    if os.path.exists(path):
        raise RuntimeError(f"experiment status already exists: {path}")
    status = {
        "schema_version": 1,
        "experiment": experiment,
        "status": "running",
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "identity": provenance(script_file),
        "stages": [],
        "next_action": "Run the first stage.",
    }
    atomic_json_write(status, path)
    return status


def load_experiment_status(path):
    """Load a status record and reject malformed state instead of guessing."""
    import json
    try:
        with open(path, encoding="utf-8") as handle:
            status = json.load(handle)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot read experiment status {path}: {exc}") from exc
    if status.get("schema_version") != 1 or not isinstance(status.get("stages"), list):
        raise RuntimeError(f"invalid experiment status: {path}")
    return status


def record_experiment_stage(path, name, checkpoint_path, resumed):
    """Record a validated checkpoint and whether this run resumed it."""
    status = load_experiment_status(path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    stage = {
        "name": name,
        "status": "complete",
        "checkpoint": os.path.relpath(checkpoint_path, os.path.dirname(path)),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "resume": "validated_existing" if resumed else "generated",
        "completed_at": _utc_now(),
    }
    status["stages"] = [item for item in status["stages"]
                        if item.get("name") != name] + [stage]
    status["updated_at"] = _utc_now()
    status["next_action"] = "Run the next stage."
    atomic_json_write(status, path)
    print(f"STATUS {name}: {stage['resume']} "
          f"sha256={stage['checkpoint_sha256']}", flush=True)
    return status


def finish_experiment_status(path, status_name, next_action, error=None):
    """Finish a status record with an explicit next action."""
    status = load_experiment_status(path)
    status.update({"status": status_name, "updated_at": _utc_now(),
                   "finished_at": _utc_now(), "next_action": next_action,
                   "error": error})
    atomic_json_write(status, path)
    return status
