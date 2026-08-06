import asyncio
import copy
import gc
import json
import logging
import os
import shutil
import sys
import time
from urllib.parse import quote
import zipfile

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from archive_utils import validate_zip_members

from core import (
    BUILTIN_LORA_DIR,
    EVALUATION_REVIEWS_DIR,
    LORA_DATASETS_DIR,
    LORA_MODELS_DIR,
    LORA_MODELS_MANIFEST,
    ROOT_DIR,
    _cancel_task,
    _load_builtin_lora_manifest,
    _load_manifest,
    _require_safe_filename,
    _safe_subpath,
    _save_manifest,
    _save_upload_limited,
    check_global_gpu_lock,
    claim_gpu_task,
    process_state,
    project_manager,
    run_process,
)
from hf_utils import (
    builtin_hf_name,
    download_builtin_adapter,
    fetch_builtin_manifest,
    is_adapter_downloaded,
)
from lora_evidence import get_evidence_error
from train_lora import DEFAULT_LEARNING_RATE
from utils import atomic_json_write, file_lock, get_unique_id, is_path_inside, secure_filename
from runtime_info import get_runtime_info
import evaluation_reviews


logger = logging.getLogger("AlexandriaUI")
router = APIRouter()


def _safe_extractall(zf: "zipfile.ZipFile", dest_dir: str) -> None:
    """zipfile.extractall, but reject members that would escape dest_dir
    (Zip-Slip path traversal via '../' entries or absolute paths)."""
    try:
        validate_zip_members(zf, dest_dir)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    zf.extractall(dest_dir)


class LoraTrainingRequest(BaseModel):
    name: str
    dataset_id: str
    epochs: int = Field(default=5, ge=1, le=1000)
    lr: float = Field(default=DEFAULT_LEARNING_RATE, gt=0, le=1)
    batch_size: int = Field(default=1, ge=1, le=64)
    lora_r: int = Field(default=32, ge=1, le=1024)
    lora_alpha: int = Field(default=128, ge=1, le=4096)
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=1024)
    language: str = "english"

class LoraTestRequest(BaseModel):
    adapter_id: str
    text: str
    instruct: str = ""


PROMOTION_FILES = ("adapter_config.json", "adapter_model.safetensors", "README.md",
                   "ref_sample.wav", "training_meta.json")
CHECKPOINT_SWAP_JOURNAL = ".checkpoint_swap.json"
LORA_DISK_WARNING_BYTES = 25 * 1024**3


def _copy_promotion_files(source_dir: str, destination_dir: str) -> None:
    os.makedirs(destination_dir, exist_ok=True)
    for filename in PROMOTION_FILES:
        source = os.path.join(source_dir, filename)
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Candidate is incomplete: missing {filename}")
        shutil.copy2(source, os.path.join(destination_dir, filename))


def _get_directory_size(path: str) -> int:
    return sum(os.path.getsize(os.path.join(root, filename))
               for root, _dirs, filenames in os.walk(path)
               for filename in filenames)


def _prune_promotion_backups(adapter_dir: str, keep_backup_id: str) -> list[str]:
    backups_dir = os.path.join(adapter_dir, "promotion_backups")
    removed = []
    if not os.path.isdir(backups_dir):
        return removed
    for backup_id in sorted(os.listdir(backups_dir)):
        backup_dir = os.path.join(backups_dir, backup_id)
        if backup_id != keep_backup_id and os.path.isdir(backup_dir):
            shutil.rmtree(backup_dir)
            removed.append(backup_id)
    return removed


def _get_lora_backup_status(models_dir: str, manifest_path: str) -> dict:
    backups = []
    for entry in _load_manifest(manifest_path):
        promotion = entry.get("promotion") or {}
        backup_id = promotion.get("backup_id")
        if not backup_id:
            continue
        adapter_dir = _safe_subpath(models_dir, entry.get("id", ""))
        backup_dir = _safe_subpath(os.path.join(adapter_dir, "promotion_backups"), backup_id)
        if os.path.isdir(backup_dir):
            backups.append({
                "adapter_id": entry["id"], "backup_id": backup_id,
                "size_bytes": _get_directory_size(backup_dir),
                "created_at": promotion.get("promoted_at"),
                "active": promotion.get("status") == "promoted",
            })
    usage = shutil.disk_usage(models_dir)
    return {
        "backups": backups,
        "total_size_bytes": sum(item["size_bytes"] for item in backups),
        "free_bytes": usage.free,
        "warning_threshold_bytes": LORA_DISK_WARNING_BYTES,
        "low_space_warning": usage.free < LORA_DISK_WARNING_BYTES,
    }


def _delete_rollback_backup(adapter_id: str, models_dir: str,
                            manifest_path: str) -> dict:
    adapter_dir = _safe_subpath(models_dir, adapter_id)
    with file_lock(manifest_path):
        if _get_checkpoint_swap_journal(adapter_dir):
            raise HTTPException(status_code=409, detail="Checkpoint recovery is required first")
        manifest = _load_manifest(manifest_path)
        entry = next((item for item in manifest if item.get("id") == adapter_id), None)
        promotion = (entry or {}).get("promotion") or {}
        backup_id = promotion.get("backup_id")
        if not backup_id:
            raise HTTPException(status_code=404, detail="Rollback backup not found")
        backup_dir = _safe_subpath(os.path.join(adapter_dir, "promotion_backups"), backup_id)
        if not os.path.isdir(backup_dir):
            raise HTTPException(status_code=404, detail="Rollback backup not found")
        shutil.rmtree(backup_dir)
        promotion["backup_deleted_at"] = time.time()
        promotion["backup_id"] = None
        _save_manifest(manifest_path, manifest)
    return {"status": "deleted", "adapter_id": adapter_id, "backup_id": backup_id}


def _load_comparison_evaluation(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=409, detail="Evaluation evidence is unavailable") from error
    if not isinstance(result, dict) or not isinstance(result.get("probes"), list):
        raise HTTPException(status_code=409, detail="Evaluation evidence is incomplete")
    return result


def _get_candidate_summary(entry: dict, adapter_dir: str) -> dict:
    evaluation = entry.get("evaluation") or {}
    promotion = entry.get("promotion") or {}
    retained = entry.get("evaluation_candidates") or []
    skipped = entry.get("evaluation_candidate_skips") or []
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    promotion = promotion if isinstance(promotion, dict) else {}
    retained = retained if isinstance(retained, list) else []
    skipped = skipped if isinstance(skipped, list) else []
    recommendation = {}
    try:
        with open(os.path.join(adapter_dir, "evaluation.json"), encoding="utf-8") as handle:
            stored_evaluation = json.load(handle)
        if isinstance(stored_evaluation, dict):
            recommendation = stored_evaluation.get("candidate_recommendation") or {}
    except (OSError, json.JSONDecodeError):
        pass
    recommendation = recommendation if isinstance(recommendation, dict) else {}
    metrics = recommendation.get("candidate_metrics") or {}
    metrics = metrics if isinstance(metrics, dict) else {}
    evaluated_candidates = [candidate_id for candidate_id, candidate_metrics in metrics.items()
                            if candidate_id != "production"
                            and isinstance(candidate_metrics, dict)
                            and candidate_metrics.get("status") != "skipped_duplicate"]
    duplicate_ids = {item.get("id") for item in skipped
                     if isinstance(item, dict) and item.get("id")}
    stored_duplicates = recommendation.get("duplicate_candidates") or []
    stored_duplicates = stored_duplicates if isinstance(stored_duplicates, list) else []
    cleanup = recommendation.get("cleanup") or {}
    cleanup = cleanup if isinstance(cleanup, dict) else {}
    duplicate_ids.update(item.get("id") for item in stored_duplicates
                         if isinstance(item, dict) and item.get("id"))
    recommended = evaluation.get("recommended_candidate")
    if promotion.get("status") == "promoted":
        state = "promoted"
    elif promotion.get("status") == "rolled_back":
        state = "rolled_back"
    elif recommended and recommended != "production":
        state = "candidate_recommended"
    elif evaluation.get("status") in ("pass", "warning"):
        state = "production_recommended"
    elif retained or skipped:
        state = "awaiting_evaluation"
    else:
        state = "no_candidates"
    return {
        "state": state,
        "evaluated_count": len(evaluated_candidates),
        "retained_count": len(retained),
        "duplicate_count": len(duplicate_ids),
        "recommended_candidate": recommended,
        "production_unchanged": state in (
            "candidate_recommended", "production_recommended", "awaiting_evaluation"),
        "promotion_status": promotion.get("status"),
        "cleanup_status": cleanup.get("status"),
    }


def _get_probe_comparison(result: dict, checkpoint_dir: str, url_prefix: str) -> dict:
    probes = {}
    for probe in result["probes"]:
        probe_id = probe.get("id")
        audio_file = probe.get("audio_file", "")
        if not probe_id or probe_id in probes:
            raise HTTPException(status_code=409, detail="Evaluation probe identities are invalid")
        audio_path = _safe_subpath(checkpoint_dir, audio_file)
        if not audio_file or not os.path.isfile(audio_path):
            raise HTTPException(status_code=409, detail=f"Evaluation audio is missing: {probe_id}")
        probes[probe_id] = {
            "id": probe_id,
            "text": probe.get("text", ""),
            "seed": probe.get("seed"),
            "audio_url": f"{url_prefix}/{quote(audio_file, safe='')}",
            "metrics": probe.get("metrics", {}),
            "warnings": probe.get("warnings", []),
        }
    return probes


def _load_candidate_comparison_full(adapter_id: str, models_dir: str,
                                    manifest_path: str) -> tuple:
    """Comparison payload plus the two integrity-checked evaluation results.

    Single loader shared by the advisory comparison endpoint and the human-review
    session builder so the evidence is loaded and validated in exactly one place.
    """
    adapter_dir = _safe_subpath(models_dir, adapter_id)
    if _get_checkpoint_swap_journal(adapter_dir):
        raise HTTPException(status_code=409, detail="Checkpoint recovery is required first")
    entry = next((item for item in _load_manifest(manifest_path)
                  if item.get("id") == adapter_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Adapter not found")
    candidate_id = (entry.get("evaluation") or {}).get("recommended_candidate")
    retained = entry.get("evaluation_candidates") or []
    if (not candidate_id or candidate_id == "production"
            or not any(item.get("id") == candidate_id for item in retained)):
        raise HTTPException(status_code=409, detail="No retained candidate is available to compare")
    candidate_dir = _safe_subpath(os.path.join(adapter_dir, "candidates"), candidate_id)
    production_result = _load_comparison_evaluation(
        os.path.join(adapter_dir, "evaluation.json"))
    candidate_result = _load_comparison_evaluation(
        os.path.join(candidate_dir, "evaluation.json"))
    for result, checkpoint_dir in ((production_result, adapter_dir),
                                   (candidate_result, candidate_dir)):
        try:
            evidence_error = get_evidence_error(result, checkpoint_dir)
        except (OSError, ValueError):
            evidence_error = "evaluation integrity evidence could not be verified"
        if evidence_error:
            raise HTTPException(status_code=409, detail=evidence_error)
    adapter_url = quote(adapter_id, safe="")
    candidate_url = quote(candidate_id, safe="")
    production_probes = _get_probe_comparison(
        production_result, adapter_dir, f"/lora_models/{adapter_url}")
    candidate_probes = _get_probe_comparison(
        candidate_result, candidate_dir,
        f"/lora_models/{adapter_url}/candidates/{candidate_url}")
    if production_probes.keys() != candidate_probes.keys():
        raise HTTPException(status_code=409, detail="Production and candidate probes do not match")
    pairs = []
    for probe_id, production_probe in production_probes.items():
        candidate_probe = candidate_probes[probe_id]
        if (production_probe["text"], production_probe["seed"]) != (
                candidate_probe["text"], candidate_probe["seed"]):
            raise HTTPException(status_code=409, detail=f"Probe evidence is not comparable: {probe_id}")
        pairs.append({"id": probe_id, "text": production_probe["text"],
                      "seed": production_probe["seed"],
                      "production": production_probe, "candidate": candidate_probe})
    recommendation = production_result.get("candidate_recommendation") or {}
    comparison = {
        "adapter_id": adapter_id,
        "candidate_id": candidate_id,
        "advisory_only": True,
        "reason": recommendation.get("reason", ""),
        "ranking": recommendation.get("ranking", []),
        "probe_pairs": pairs,
    }
    return comparison, production_result, candidate_result


def _get_lora_candidate_comparison(adapter_id: str, models_dir: str,
                                   manifest_path: str) -> dict:
    comparison, _production, _candidate = _load_candidate_comparison_full(
        adapter_id, models_dir, manifest_path)
    return comparison


def _get_checkpoint_swap_journal(adapter_dir: str) -> dict | None:
    path = os.path.join(adapter_dir, CHECKPOINT_SWAP_JOURNAL)
    try:
        with open(path, encoding="utf-8") as handle:
            journal = json.load(handle)
        return journal if isinstance(journal, dict) and journal else {"operation": "unknown"}
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return {"operation": "unknown"}


def list_adapters_needing_recovery(models_dir: str, manifest_path: str) -> list[dict]:
    """Return user adapters with a pending checkpoint-swap journal.

    Single source of truth for "recovery required" over user adapters so the
    Voice Lab health dashboard and any other caller don't re-derive the check.
    """
    pending = []
    for entry in _load_manifest(manifest_path):
        if entry.get("builtin"):
            continue
        adapter_id = entry.get("id")
        if not adapter_id:
            continue
        journal = _get_checkpoint_swap_journal(os.path.join(models_dir, adapter_id))
        if journal:
            pending.append({"adapter_id": adapter_id,
                            "operation": journal.get("operation", "unknown")})
    return pending


def _replace_checkpoint_files(adapter_dir: str, source_dir: str,
                              recovery_dir: str, operation: str,
                              keep_recovery: bool, manifest_entry: dict) -> None:
    journal_path = os.path.join(adapter_dir, CHECKPOINT_SWAP_JOURNAL)
    if os.path.exists(journal_path):
        raise HTTPException(status_code=409, detail="Checkpoint recovery is required first")
    staging_dir = os.path.join(adapter_dir, ".checkpoint_swap_staging")
    shutil.rmtree(staging_dir, ignore_errors=True)
    _copy_promotion_files(adapter_dir, recovery_dir)
    _copy_promotion_files(source_dir, staging_dir)
    atomic_json_write({
        "version": 1,
        "operation": operation,
        "recovery_dir": os.path.relpath(recovery_dir, adapter_dir),
        "keep_recovery": keep_recovery,
        "manifest_entry": copy.deepcopy(manifest_entry),
        "created_at": time.time(),
    }, journal_path)
    try:
        for filename in PROMOTION_FILES:
            os.replace(os.path.join(staging_dir, filename), os.path.join(adapter_dir, filename))
    except Exception:
        try:
            _copy_promotion_files(recovery_dir, adapter_dir)
            os.remove(journal_path)
        except Exception:
            logger.exception("Checkpoint swap recovery failed for %s", adapter_dir)
        raise
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _complete_checkpoint_swap(adapter_dir: str, recovery_dir: str,
                              keep_recovery: bool) -> None:
    os.remove(os.path.join(adapter_dir, CHECKPOINT_SWAP_JOURNAL))
    if not keep_recovery:
        shutil.rmtree(recovery_dir, ignore_errors=True)


def _recover_checkpoint_swap(adapter_id: str, models_dir: str,
                             manifest_path: str) -> dict:
    adapter_dir = _safe_subpath(models_dir, adapter_id)
    with file_lock(manifest_path):
        journal = _get_checkpoint_swap_journal(adapter_dir)
        if not journal:
            raise HTTPException(status_code=409, detail="No checkpoint recovery is pending")
        relative_recovery = journal.get("recovery_dir", "")
        recovery_dir = _safe_subpath(adapter_dir, relative_recovery)
        previous_entry = journal.get("manifest_entry")
        if not isinstance(previous_entry, dict) or previous_entry.get("id") != adapter_id:
            raise HTTPException(status_code=409, detail="Checkpoint recovery journal is invalid")
        manifest = _load_manifest(manifest_path)
        entry_index = next((index for index, item in enumerate(manifest)
                            if item.get("id") == adapter_id), None)
        if entry_index is None:
            raise HTTPException(status_code=409, detail="Adapter manifest entry is missing")
        _copy_promotion_files(recovery_dir, adapter_dir)
        manifest[entry_index] = previous_entry
        _save_manifest(manifest_path, manifest)
        shutil.rmtree(os.path.join(adapter_dir, ".checkpoint_swap_staging"), ignore_errors=True)
        os.remove(os.path.join(adapter_dir, CHECKPOINT_SWAP_JOURNAL))
        if not journal.get("keep_recovery", True):
            shutil.rmtree(recovery_dir, ignore_errors=True)
    return {"status": "recovered", "operation": journal.get("operation", "unknown")}


def _promote_lora_candidate(adapter_id: str, models_dir: str,
                            manifest_path: str) -> dict:
    adapter_dir = _safe_subpath(models_dir, adapter_id)
    with file_lock(manifest_path):
        manifest = _load_manifest(manifest_path)
        entry = next((item for item in manifest if item.get("id") == adapter_id), None)
        if not entry:
            raise HTTPException(status_code=404, detail="Adapter not found")
        recommended = (entry.get("evaluation") or {}).get("recommended_candidate")
        if not recommended or recommended == "production":
            raise HTTPException(status_code=409, detail="No retained candidate is recommended")
        candidate_dir = _safe_subpath(os.path.join(adapter_dir, "candidates"), recommended)
        if not os.path.isdir(candidate_dir):
            raise HTTPException(status_code=409, detail="Recommended candidate is no longer available")

        promotion_id = f"{int(time.time())}-{recommended}"
        backup_dir = os.path.join(adapter_dir, "promotion_backups", promotion_id)
        _replace_checkpoint_files(adapter_dir, candidate_dir, backup_dir,
                                  "promotion", keep_recovery=True, manifest_entry=entry)

        entry["evaluation_candidates"] = []
        entry["evaluation"]["recommended_candidate"] = "production"
        entry["promotion"] = {
            "status": "promoted", "candidate": recommended,
            "backup_id": promotion_id, "promoted_at": time.time(),
        }
        _save_manifest(manifest_path, manifest)
        _complete_checkpoint_swap(adapter_dir, backup_dir, keep_recovery=True)
        shutil.rmtree(candidate_dir)
        _prune_promotion_backups(adapter_dir, promotion_id)
    return entry["promotion"]


def _rollback_lora_promotion(adapter_id: str, models_dir: str,
                             manifest_path: str) -> dict:
    adapter_dir = _safe_subpath(models_dir, adapter_id)
    with file_lock(manifest_path):
        manifest = _load_manifest(manifest_path)
        entry = next((item for item in manifest if item.get("id") == adapter_id), None)
        promotion = (entry or {}).get("promotion") or {}
        if promotion.get("status") != "promoted":
            raise HTTPException(status_code=409, detail="No promotion is available to roll back")
        backup_dir = _safe_subpath(os.path.join(adapter_dir, "promotion_backups"),
                                   promotion.get("backup_id", ""))
        recovery_dir = os.path.join(adapter_dir, ".rollback_recovery")
        _replace_checkpoint_files(adapter_dir, backup_dir, recovery_dir,
                                  "rollback", keep_recovery=False, manifest_entry=entry)
        promotion["status"] = "rolled_back"
        promotion["rolled_back_at"] = time.time()
        promotion["backup_id"] = None
        promotion["backup_deleted_at"] = time.time()
        _save_manifest(manifest_path, manifest)
        _complete_checkpoint_swap(adapter_dir, recovery_dir, keep_recovery=False)
        shutil.rmtree(backup_dir, ignore_errors=True)
    return promotion


## ── LoRA Training ──────────────────────────────────────────────


def _extract_lora_dataset_archive(archive_path, dataset_dir):
    """Perform potentially multi-gigabyte extraction outside the event loop."""
    os.makedirs(dataset_dir, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        _safe_extractall(archive, dataset_dir)

@router.post("/api/lora/upload_dataset")
async def lora_upload_dataset(file: UploadFile = File(...)):
    """Upload a ZIP containing WAV files and metadata.jsonl."""
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a .zip archive")

    # Derive dataset name from ZIP filename
    base_name = os.path.splitext(file.filename)[0]
    dataset_name = secure_filename(base_name)
    if not dataset_name:
        raise HTTPException(status_code=400, detail="Invalid dataset name from filename")

    dataset_dir = os.path.join(LORA_DATASETS_DIR, dataset_name)
    if os.path.exists(dataset_dir):
        raise HTTPException(status_code=400, detail=f"Dataset '{dataset_name}' already exists")

    # Save ZIP temporarily, then extract
    tmp_path = os.path.join(LORA_DATASETS_DIR, f"_tmp_{dataset_name}.zip")
    try:
        await _save_upload_limited(file, tmp_path, 4 * 1024**3)

        await asyncio.to_thread(_extract_lora_dataset_archive, tmp_path, dataset_dir)

        # Check for metadata.jsonl (may be inside a subdirectory)
        metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
        if not os.path.exists(metadata_path):
            # Check one level deep
            for entry in os.listdir(dataset_dir):
                candidate = os.path.join(dataset_dir, entry, "metadata.jsonl")
                if os.path.isdir(os.path.join(dataset_dir, entry)) and os.path.exists(candidate):
                    # Move contents up
                    nested = os.path.join(dataset_dir, entry)
                    for item in os.listdir(nested):
                        shutil.move(os.path.join(nested, item), os.path.join(dataset_dir, item))
                    os.rmdir(nested)
                    metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
                    break

        if not os.path.exists(metadata_path):
            shutil.rmtree(dataset_dir)
            raise HTTPException(status_code=400, detail="ZIP must contain metadata.jsonl")

        # Count samples and validate audio file presence
        sample_count = 0
        valid_sample_count = 0
        missing_audio = []
        malformed_lines = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if not isinstance(entry, dict):
                        raise ValueError(f"line is valid JSON but not an object (got {type(entry).__name__})")
                    audio_rel = entry.get("audio_filepath") or entry.get("audio", "")
                    audio_path = os.path.realpath(os.path.join(dataset_dir, audio_rel)) if audio_rel else ""
                    if (not audio_rel or not is_path_inside(audio_path, dataset_dir)
                            or not os.path.isfile(audio_path)):
                        missing_audio.append(audio_rel)
                    else:
                        valid_sample_count += 1
                    sample_count += 1
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    malformed_lines.append((line_num, str(e)))

        wav_count = sum(1 for f in os.listdir(dataset_dir) if f.lower().endswith(".wav"))
        ref_wav = os.path.exists(os.path.join(dataset_dir, "ref.wav"))
        ref_text = os.path.exists(os.path.join(dataset_dir, "ref_text.txt"))

        logger.info(
            f"LoRA dataset '{dataset_name}': {sample_count} metadata entries, "
            f"{wav_count} WAV files, ref.wav={'yes' if ref_wav else 'MISSING'}, "
            f"ref_text.txt={'yes' if ref_text else 'missing'}"
        )
        if missing_audio:
            logger.warning(
                f"LoRA dataset '{dataset_name}': {len(missing_audio)} audio file(s) in "
                f"metadata.jsonl not found in ZIP: {missing_audio[:5]}"
                f"{'  (+more)' if len(missing_audio) > 5 else ''}"
            )
        else:
            logger.info(f"LoRA dataset '{dataset_name}': all {sample_count} audio files present in ZIP")
        if malformed_lines:
            logger.warning(
                f"LoRA dataset '{dataset_name}': {len(malformed_lines)} malformed "
                f"metadata.jsonl line(s) skipped: {malformed_lines[:5]}"
                f"{'  (+more)' if len(malformed_lines) > 5 else ''}"
            )

        if valid_sample_count == 0:
            raise HTTPException(status_code=400, detail="Dataset contains no usable training audio.")

        return {"status": "uploaded", "dataset_id": dataset_name,
                "sample_count": valid_sample_count, "metadata_count": sample_count}
    except Exception:
        if os.path.isdir(dataset_dir):
            shutil.rmtree(dataset_dir)
        raise
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@router.get("/api/lora/datasets")
async def lora_list_datasets():
    """List uploaded LoRA training datasets."""
    datasets = []
    if not os.path.exists(LORA_DATASETS_DIR):
        return datasets

    for name in sorted(os.listdir(LORA_DATASETS_DIR)):
        dataset_dir = os.path.join(LORA_DATASETS_DIR, name)
        if not os.path.isdir(dataset_dir):
            continue
        metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
        sample_count = 0
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        sample_count += 1
        except FileNotFoundError:
            # No metadata yet, or it vanished mid-listing (concurrent delete).
            pass
        datasets.append({"dataset_id": name, "sample_count": sample_count})
    return datasets

@router.delete("/api/lora/datasets/{dataset_id}")
async def lora_delete_dataset(dataset_id: str):
    """Delete an uploaded dataset."""
    dataset_dir = _safe_subpath(LORA_DATASETS_DIR, dataset_id)
    if not os.path.isdir(dataset_dir):
        raise HTTPException(status_code=404, detail="Dataset not found")

    shutil.rmtree(dataset_dir)
    logger.info(f"LoRA dataset deleted: {dataset_id}")
    return {"status": "deleted", "dataset_id": dataset_id}

@router.post("/api/lora/train/cancel")
async def lora_cancel_training():
    """Cancel a running LoRA training subprocess (it holds the global GPU lock for
    hours, so without this the only way to stop it was killing the whole server)."""
    return _cancel_task("lora_training",
                        "No LoRA training is currently running.",
                        "LoRA training already exited.")


@router.post("/api/lora/train")
async def lora_start_training(request: LoraTrainingRequest, background_tasks: BackgroundTasks):
    """Start LoRA training as a subprocess."""
    check_global_gpu_lock("lora_training")

    # Validate dataset exists
    dataset_dir = _safe_subpath(LORA_DATASETS_DIR, request.dataset_id)
    if not os.path.isdir(dataset_dir):
        raise HTTPException(status_code=400, detail=f"Dataset '{request.dataset_id}' not found")

    # Build output directory
    safe_name = _require_safe_filename(request.name, "Invalid adapter name")

    adapter_id = get_unique_id(safe_name)
    output_dir = os.path.join(LORA_MODELS_DIR, adapter_id)

    # Log dataset details and effective settings before training
    try:
        meta_path = os.path.join(dataset_dir, "metadata.jsonl")
        with open(meta_path, encoding="utf-8") as metadata_file:
            dataset_sample_count = sum(1 for line in metadata_file if line.strip())
        total_passes = dataset_sample_count * request.epochs
        alpha_r = request.lora_alpha / request.lora_r
        logger.info(
            f"LoRA training '{request.name}': dataset='{request.dataset_id}' "
            f"samples={dataset_sample_count}, epochs={request.epochs}, "
            f"total_passes={total_passes}, lr={request.lr:.2e}, "
            f"r={request.lora_r}, alpha={request.lora_alpha} (scale={alpha_r:.1f}x), "
            f"grad_accum={request.gradient_accumulation_steps}, language={request.language}"
        )
    except (OSError, ValueError, ZeroDivisionError):
        pass
    if project_manager.engine is not None:
        logger.info("Unloading TTS engine for LoRA training...")
        project_manager.engine = None
        gc.collect()

    # Build subprocess command
    command = [
        sys.executable, "-u", "train_lora.py",
        "--data_dir", dataset_dir,
        "--output_dir", output_dir,
        "--epochs", str(request.epochs),
        "--lr", str(request.lr),
        "--batch_size", str(request.batch_size),
        "--lora_r", str(request.lora_r),
        "--lora_alpha", str(request.lora_alpha),
        "--gradient_accumulation_steps", str(request.gradient_accumulation_steps),
        "--language", request.language,
    ]

    def on_training_complete():
        """After training subprocess finishes, update manifest if adapter was saved."""
        run_process(command, "lora_training")

        # Check if training produced an adapter
        if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "training_meta.json")):
            try:
                with open(os.path.join(output_dir, "training_meta.json"), "r") as f:
                    meta = json.load(f)

                manifest = _load_manifest(LORA_MODELS_MANIFEST)
                manifest.append({
                    "id": adapter_id,
                    "name": request.name,
                    "dataset_id": request.dataset_id,
                    "epochs": meta.get("epochs", request.epochs),
                    "final_loss": meta.get("final_loss"),
                    "sample_count": meta.get("num_samples"),
                    "lora_r": meta.get("lora_r"),
                    "lr": meta.get("lr"),
                    "checkpoint_sha256": meta.get("checkpoint_sha256"),
                    "evaluation_candidates": meta.get("evaluation_candidates", []),
                    "evaluation_candidate_skips": meta.get(
                        "evaluation_candidate_skips", []),
                    "created": time.time(),
                })
                _save_manifest(LORA_MODELS_MANIFEST, manifest)
                logger.info(f"LoRA adapter registered: {adapter_id}")
            except Exception as e:
                logger.error(f"Failed to update LoRA manifest: {e}")

    claim_gpu_task("lora_training")
    background_tasks.add_task(on_training_complete)
    return {"status": "started", "adapter_id": adapter_id}

@router.get("/api/lora/models")
async def lora_list_models():
    """List all LoRA adapters (built-in + user-trained)."""
    models = _load_builtin_lora_manifest() + _load_manifest(LORA_MODELS_MANIFEST)
    for m in models:
        is_builtin = m.get("builtin", False)
        is_downloaded = m.get("downloaded", True)  # user-trained are always downloaded

        if not is_downloaded:
            m["preview_audio_url"] = None
            continue

        if is_builtin:
            adapter_dir = os.path.join(BUILTIN_LORA_DIR, m["id"])
            url_prefix = f"/builtin_lora/{m['id']}"
        else:
            adapter_dir = os.path.join(LORA_MODELS_DIR, m["id"])
            url_prefix = f"/lora_models/{m['id']}"
        preview_path = os.path.join(adapter_dir, "preview_sample.wav")
        m["preview_audio_url"] = f"{url_prefix}/preview_sample.wav" if os.path.exists(preview_path) else None
        journal = None if is_builtin else _get_checkpoint_swap_journal(adapter_dir)
        m["checkpoint_swap"] = ({"status": "recovery_required",
                                  "operation": journal.get("operation", "unknown")}
                                if journal else None)
        if not is_builtin:
            m["candidate_summary"] = _get_candidate_summary(m, adapter_dir)
            # Human-review tally, kept separate from the automated recommendation.
            m["review_summary"] = evaluation_reviews.summarize(
                EVALUATION_REVIEWS_DIR, m["id"])
    return models


@router.get("/api/lora/backups")
async def lora_list_backups():
    """Report rollback-backup storage and host free-space pressure."""
    return await asyncio.to_thread(
        _get_lora_backup_status, LORA_MODELS_DIR, LORA_MODELS_MANIFEST)


@router.get("/api/lora/models/{adapter_id}/comparison")
async def lora_get_candidate_comparison(adapter_id: str):
    """Return validated, paired evaluation audio for a retained candidate."""
    return await asyncio.to_thread(
        _get_lora_candidate_comparison, adapter_id,
        LORA_MODELS_DIR, LORA_MODELS_MANIFEST)


class ReviewSubmitRequest(BaseModel):
    choice: str
    rating: int | None = None
    notes: str = ""


def _current_evidence_fingerprint(adapter_id: str) -> dict:
    """Re-load + integrity-check both sides, returning their evidence fingerprint.

    Raises the same 409s as the comparison endpoint when evidence is missing or
    no longer matches on disk (checkpoint retrained/promoted/rolled back).
    """
    _comparison, production_result, candidate_result = _load_candidate_comparison_full(
        adapter_id, LORA_MODELS_DIR, LORA_MODELS_MANIFEST)
    return evaluation_reviews.evidence_fingerprint(production_result, candidate_result)


def _role_audio_paths(result: dict, checkpoint_dir: str) -> dict:
    """probe_id -> real audio path for one side's evaluation result."""
    return {probe.get("id"): os.path.join(checkpoint_dir, probe.get("audio_file", ""))
            for probe in result.get("probes", []) if probe.get("id")}


def _open_review_session(adapter_id: str) -> dict:
    comparison, production_result, candidate_result = _load_candidate_comparison_full(
        adapter_id, LORA_MODELS_DIR, LORA_MODELS_MANIFEST)
    fingerprint = evaluation_reviews.evidence_fingerprint(production_result, candidate_result)
    adapter_dir = _safe_subpath(LORA_MODELS_DIR, adapter_id)
    candidate_dir = _safe_subpath(
        os.path.join(adapter_dir, "candidates"), comparison["candidate_id"])
    audio_paths_by_role = {
        "production": _role_audio_paths(production_result, adapter_dir),
        "candidate": _role_audio_paths(candidate_result, candidate_dir),
    }
    try:
        session = evaluation_reviews.create_session(
            EVALUATION_REVIEWS_DIR, adapter_id, comparison["candidate_id"],
            fingerprint, comparison["probe_pairs"], audio_paths_by_role,
            build={"short_revision": get_runtime_info(ROOT_DIR).get("short_revision")},
            automated_recommended=comparison["candidate_id"])
    except evaluation_reviews.ReviewError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    # Attach identity-neutral proxy URLs. The candidate's real URL contains
    # "/candidates/", so serving raw URLs would break blindness — stream via a
    # session/label-scoped endpoint that resolves the side server-side.
    adapter_q = quote(adapter_id, safe="")
    session_q = quote(session["session_id"], safe="")
    base = f"/api/lora/models/{adapter_q}/review/session/{session_q}/audio"
    for pair in session["pairs"]:
        probe_q = quote(str(pair["id"]), safe="")
        pair["A"] = {"audio_url": f"{base}/A/{probe_q}"}
        pair["B"] = {"audio_url": f"{base}/B/{probe_q}"}
    return session


def _serve_review_audio(adapter_id: str, session_id: str, label: str, probe_id: str):
    if label not in ("A", "B"):
        raise HTTPException(status_code=404, detail="Unknown sample")
    try:
        path = evaluation_reviews.get_session_audio_path(
            EVALUATION_REVIEWS_DIR, session_id, label, probe_id)
    except evaluation_reviews.ReviewError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    if not is_path_inside(path, LORA_MODELS_DIR) or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path)


def _submit_review_decision(adapter_id: str, session_id: str, req: "ReviewSubmitRequest") -> dict:
    current = _current_evidence_fingerprint(adapter_id)
    try:
        return evaluation_reviews.submit(
            EVALUATION_REVIEWS_DIR, adapter_id, session_id, req.choice, current,
            rating=req.rating, notes=req.notes)
    except evaluation_reviews.ReviewError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post("/api/lora/models/{adapter_id}/review/session")
async def lora_open_review_session(adapter_id: str):
    """Open a blind A/B human-review session (identities hidden until submit)."""
    return await asyncio.to_thread(_open_review_session, adapter_id)


@router.get("/api/lora/models/{adapter_id}/review/session/{session_id}/audio/{label}/{probe_id}")
async def lora_review_audio(adapter_id: str, session_id: str, label: str, probe_id: str):
    """Stream one blind sample (A/B) without revealing which side it is."""
    return await asyncio.to_thread(
        _serve_review_audio, adapter_id, session_id, label, probe_id)


@router.post("/api/lora/models/{adapter_id}/review/session/{session_id}")
async def lora_submit_review(adapter_id: str, session_id: str, request: ReviewSubmitRequest):
    """Record a human listening decision; rejects if evidence changed. Never promotes."""
    return await asyncio.to_thread(_submit_review_decision, adapter_id, session_id, request)


@router.get("/api/lora/models/{adapter_id}/reviews")
async def lora_list_reviews(adapter_id: str):
    """Return this adapter's bounded human-review history, newest first."""
    reviews = await asyncio.to_thread(
        evaluation_reviews.list_reviews, EVALUATION_REVIEWS_DIR, adapter_id)
    return {"reviews": reviews}


@router.post("/api/lora/models/{adapter_id}/reviews/cleanup")
async def lora_cleanup_reviews(adapter_id: str):
    """Delete this adapter's human-review history, reporting count and space freed."""
    return await asyncio.to_thread(
        evaluation_reviews.cleanup, EVALUATION_REVIEWS_DIR, adapter_id)


@router.post("/api/lora/models/{adapter_id}/promote")
async def lora_promote_candidate(adapter_id: str):
    """Promote the evaluated recommendation while preserving production for rollback."""
    claim_gpu_task("lora_training")
    try:
        result = await asyncio.to_thread(
            _promote_lora_candidate, adapter_id, LORA_MODELS_DIR, LORA_MODELS_MANIFEST)
    finally:
        process_state["lora_training"]["running"] = False
    project_manager.engine = None
    gc.collect()
    logger.info("LoRA candidate promoted: %s <- %s", adapter_id, result["candidate"])
    return {"status": "promoted", "adapter_id": adapter_id, "promotion": result}


@router.post("/api/lora/models/{adapter_id}/rollback-promotion")
async def lora_rollback_promotion(adapter_id: str):
    """Restore the production checkpoint preserved by the last promotion."""
    claim_gpu_task("lora_training")
    try:
        result = await asyncio.to_thread(
            _rollback_lora_promotion, adapter_id, LORA_MODELS_DIR, LORA_MODELS_MANIFEST)
    finally:
        process_state["lora_training"]["running"] = False
    project_manager.engine = None
    gc.collect()
    logger.info("LoRA promotion rolled back: %s", adapter_id)
    return {"status": "rolled_back", "adapter_id": adapter_id, "promotion": result}


@router.post("/api/lora/models/{adapter_id}/recover-checkpoint-swap")
async def lora_recover_checkpoint_swap(adapter_id: str):
    """Restore production after a process interruption left a swap journal."""
    claim_gpu_task("lora_training")
    try:
        result = await asyncio.to_thread(
            _recover_checkpoint_swap, adapter_id, LORA_MODELS_DIR, LORA_MODELS_MANIFEST)
    finally:
        process_state["lora_training"]["running"] = False
    project_manager.engine = None
    gc.collect()
    logger.warning("Recovered interrupted LoRA checkpoint swap: %s", adapter_id)
    return {"adapter_id": adapter_id, **result}


@router.delete("/api/lora/models/{adapter_id}/rollback-backup")
async def lora_delete_rollback_backup(adapter_id: str):
    """Delete the preserved production checkpoint after explicit confirmation."""
    result = await asyncio.to_thread(
        _delete_rollback_backup, adapter_id, LORA_MODELS_DIR, LORA_MODELS_MANIFEST)
    logger.warning("LoRA rollback backup deleted: %s", adapter_id)
    return result

@router.delete("/api/lora/models/{adapter_id}")
async def lora_delete_model(adapter_id: str):
    """Delete a trained LoRA adapter. Built-in adapters cannot be deleted."""
    builtin = _load_builtin_lora_manifest()
    if any(m["id"] == adapter_id for m in builtin):
        raise HTTPException(status_code=403, detail="Built-in adapters cannot be deleted")
    manifest = _load_manifest(LORA_MODELS_MANIFEST)
    entry = next((m for m in manifest if m["id"] == adapter_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Adapter not found")

    # Delete adapter directory
    adapter_dir = os.path.join(LORA_MODELS_DIR, adapter_id)
    if os.path.isdir(adapter_dir):
        shutil.rmtree(adapter_dir)

    # Remove from manifest
    manifest = [m for m in manifest if m["id"] != adapter_id]
    _save_manifest(LORA_MODELS_MANIFEST, manifest)

    logger.info(f"LoRA adapter deleted: {adapter_id}")
    return {"status": "deleted", "adapter_id": adapter_id}

@router.post("/api/lora/download/{adapter_id}")
async def lora_download_builtin(adapter_id: str):
    """Download a built-in LoRA adapter from HuggingFace."""
    manifest = await asyncio.to_thread(fetch_builtin_manifest, BUILTIN_LORA_DIR)
    hf_name = builtin_hf_name(adapter_id)
    entry = next((e for e in manifest if e["id"] == hf_name or e["id"] == adapter_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Unknown built-in adapter: {adapter_id}")

    if is_adapter_downloaded(adapter_id, BUILTIN_LORA_DIR):
        return {"status": "already_downloaded", "adapter_id": adapter_id}

    try:
        await asyncio.to_thread(download_builtin_adapter, adapter_id, BUILTIN_LORA_DIR)
        logger.info(f"Built-in adapter downloaded: {adapter_id}")
        return {"status": "downloaded", "adapter_id": adapter_id}
    except Exception as e:
        logger.exception("Download failed for %s", adapter_id)
        raise HTTPException(status_code=500, detail="Built-in adapter download failed — see server logs for details.") from e

@router.post("/api/lora/test")
async def lora_test_model(request: LoraTestRequest):
    """Generate test audio using a LoRA adapter (built-in or user-trained)."""
    # Fail fast before the manifest lookup / possible adapter auto-download
    # below. See F-039.
    check_global_gpu_lock("lora_test")
    # Check both manifests
    builtin = _load_builtin_lora_manifest()
    user_trained = _load_manifest(LORA_MODELS_MANIFEST)
    all_adapters = builtin + user_trained
    entry = next((m for m in all_adapters if m["id"] == request.adapter_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Adapter not found")

    is_builtin = entry.get("builtin", False)
    if is_builtin:
        adapter_dir = os.path.join(BUILTIN_LORA_DIR, request.adapter_id)
        audio_url_prefix = f"/builtin_lora/{request.adapter_id}"
    else:
        adapter_dir = os.path.join(LORA_MODELS_DIR, request.adapter_id)
        audio_url_prefix = f"/lora_models/{request.adapter_id}"

    if not os.path.isdir(adapter_dir) and not is_builtin:
        raise HTTPException(status_code=404, detail="Adapter files not found")

    # Claim the GPU slot now, before the possible adapter download and the
    # engine load below - both can take real time and the engine load
    # allocates VRAM. Claiming only after them (the old order) left a window
    # where two concurrent /api/lora/test (or .../preview, which shares this
    # slot) requests could both pass check_global_gpu_lock above and both
    # start that slow/VRAM work before either's claim landed.
    claim_gpu_task("lora_test")
    try:
        if not os.path.isdir(adapter_dir) and is_builtin:
            try:
                download_builtin_adapter(request.adapter_id, BUILTIN_LORA_DIR)
                adapter_dir = os.path.join(BUILTIN_LORA_DIR, request.adapter_id)
            except Exception as e:
                logger.exception("Auto-download failed for %s", request.adapter_id)
                raise HTTPException(status_code=500, detail="Adapter auto-download failed — see server logs for details.") from e

        engine = project_manager.get_engine()
        if not engine:
            raise HTTPException(status_code=500, detail="Failed to initialize TTS engine")

        output_filename = f"test_{request.adapter_id}_{int(time.time())}.wav"
        output_path = os.path.join(adapter_dir, output_filename)

        voice_data = {
            "type": "lora",
            "adapter_id": request.adapter_id,
            "adapter_path": adapter_dir,
        }
        voice_config = {"_lora_test_": voice_data}
        engine.generate_voice(
            text=request.text,
            instruct_text=request.instruct or "",
            speaker="_lora_test_",
            voice_config=voice_config,
            output_path=output_path,
        )

        return {
            "status": "ok",
            "audio_url": f"{audio_url_prefix}/{output_filename}",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("LoRA test generation failed")
        raise HTTPException(status_code=500, detail="LoRA test generation failed — see server logs for details.") from e
    finally:
        process_state["lora_test"]["running"] = False

LORA_PREVIEW_TEXT = "The ancient library stood at the crossroads of two forgotten paths, its weathered stone walls covered in ivy that had been growing for centuries."

@router.post("/api/lora/preview/{adapter_id}")
async def lora_preview(adapter_id: str):
    """Generate or return cached preview audio for a LoRA adapter."""
    builtin = _load_builtin_lora_manifest()
    user_trained = _load_manifest(LORA_MODELS_MANIFEST)
    all_adapters = builtin + user_trained
    entry = next((m for m in all_adapters if m["id"] == adapter_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Adapter not found")

    is_builtin = entry.get("builtin", False)
    if is_builtin:
        adapter_dir = os.path.join(BUILTIN_LORA_DIR, adapter_id)
        url_prefix = f"/builtin_lora/{adapter_id}"
    else:
        adapter_dir = os.path.join(LORA_MODELS_DIR, adapter_id)
        url_prefix = f"/lora_models/{adapter_id}"

    if not os.path.isdir(adapter_dir) and not is_builtin:
        raise HTTPException(status_code=404, detail="Adapter files not found")

    preview_path = os.path.join(adapter_dir, "preview_sample.wav")

    # Return cached if exists. This check intentionally runs BEFORE the lock -
    # no GPU/download work happens on a cache hit, regardless of whether the
    # adapter directory exists yet (a cached preview implies it does).
    if os.path.exists(preview_path):
        return {"status": "cached", "audio_url": f"{url_prefix}/preview_sample.wav"}

    # Cache miss past this point. Shares the "lora_test" slot with
    # /api/lora/test since both are "try out this adapter" operations that
    # shouldn't run concurrently with each other either. See F-040.
    check_global_gpu_lock("lora_test")
    # Claim immediately after the check, before the possible adapter download
    # AND the engine load below - both can take real time and the engine
    # load allocates VRAM, so the claim has to land before either starts, not
    # after, or two concurrent preview/test requests can both pass the check
    # above and both begin downloading/loading the model.
    claim_gpu_task("lora_test")
    try:
        if not os.path.isdir(adapter_dir) and is_builtin:
            try:
                download_builtin_adapter(adapter_id, BUILTIN_LORA_DIR)
                adapter_dir = os.path.join(BUILTIN_LORA_DIR, adapter_id)
            except Exception as e:
                logger.exception("Auto-download failed for %s", adapter_id)
                raise HTTPException(status_code=500, detail="Adapter auto-download failed — see server logs for details.") from e

        engine = project_manager.get_engine()
        if not engine:
            raise HTTPException(status_code=500, detail="Failed to initialize TTS engine")

        voice_data = {
            "type": "lora",
            "adapter_id": adapter_id,
            "adapter_path": adapter_dir,
        }
        voice_config = {"_lora_preview_": voice_data}
        engine.generate_voice(
            text=LORA_PREVIEW_TEXT,
            instruct_text="",
            speaker="_lora_preview_",
            voice_config=voice_config,
            output_path=preview_path,
        )
        return {"status": "generated", "audio_url": f"{url_prefix}/preview_sample.wav"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("LoRA preview generation failed")
        raise HTTPException(status_code=500, detail="LoRA preview generation failed — see server logs for details.") from e
    finally:
        process_state["lora_test"]["running"] = False
