import json
import logging
import os
import shutil
import threading
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core import (
    DATASET_BUILDER_DIR,
    LORA_DATASETS_DIR,
    _require_safe_filename,
    _safe_subpath,
    check_global_gpu_lock,
    claim_gpu_task,
    process_state,
    project_manager,
)
from utils import atomic_json_write


logger = logging.getLogger("AlexandriaUI")
router = APIRouter()


class LoraDatasetSample(BaseModel):
    emotion: str = ""
    text: str

class DatasetSampleGenRequest(BaseModel):
    description: str      # full voice description (root + emotion already combined by frontend)
    text: str
    dataset_name: str     # working directory name
    sample_index: int = Field(ge=0, le=4999)  # row number
    seed: int = -1        # -1 = random, >= 0 = manual seed

class DatasetBatchGenRequest(BaseModel):
    name: str
    description: str      # root voice description
    samples: List[LoraDatasetSample]
    indices: Optional[List[int]] = None  # which rows to generate (None = all)
    global_seed: int = -1 # -1 = random, >= 0 = same seed for all lines
    seeds: Optional[List[int]] = None  # per-line seeds (overrides global_seed)

class DatasetSaveRequest(BaseModel):
    name: str
    ref_index: int = 0    # which sample to use as ref.wav

class DatasetBuilderCreateRequest(BaseModel):
    name: str

class DatasetBuilderUpdateMetaRequest(BaseModel):
    name: str
    description: str = ""
    global_seed: str = ""

class DatasetBuilderUpdateRowsRequest(BaseModel):
    name: str
    rows: List[dict]  # [{emotion, text, seed}]


## ── Dataset Builder ──────────────────────────────────────────

def _load_builder_state(name):
    """Load project state from dataset builder working directory."""
    state_path = os.path.join(DATASET_BUILDER_DIR, name, "state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if not isinstance(state, dict):
                raise ValueError(f"Expected a JSON object, got {type(state).__name__}")
            # Ensure new fields exist for backward compat
            state.setdefault("description", "")
            state.setdefault("global_seed", "")
            state.setdefault("samples", [])
            return state
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to load builder state '{name}': {e}")
    return {"description": "", "global_seed": "", "samples": []}

def _save_builder_state(name, state):
    """Save per-sample state to dataset builder working directory atomically."""
    work_dir = os.path.join(DATASET_BUILDER_DIR, name)
    os.makedirs(work_dir, exist_ok=True)
    atomic_json_write(state, os.path.join(work_dir, "state.json"))

@router.get("/api/dataset_builder/list")
async def dataset_builder_list():
    """List existing dataset builder projects."""
    projects = []
    if os.path.isdir(DATASET_BUILDER_DIR):
        for name in sorted(os.listdir(DATASET_BUILDER_DIR)):
            state_path = os.path.join(DATASET_BUILDER_DIR, name, "state.json")
            if os.path.isfile(state_path):
                state = _load_builder_state(name)
                samples = state.get("samples", [])
                projects.append({
                    "name": name,
                    "description": state.get("description", ""),
                    "sample_count": len(samples),
                    "done_count": sum(1 for s in samples if s.get("status") == "done"),
                })
    return projects

@router.post("/api/dataset_builder/create")
async def dataset_builder_create(request: DatasetBuilderCreateRequest):
    """Create a new dataset builder project."""
    safe_name = _require_safe_filename(request.name, "Invalid dataset name")
    work_dir = os.path.join(DATASET_BUILDER_DIR, safe_name)
    if os.path.exists(work_dir):
        raise HTTPException(status_code=400, detail=f"Project '{safe_name}' already exists")
    _save_builder_state(safe_name, {"description": "", "global_seed": "", "samples": []})
    return {"name": safe_name}

@router.post("/api/dataset_builder/update_meta")
async def dataset_builder_update_meta(request: DatasetBuilderUpdateMetaRequest):
    """Update project description and global seed without touching samples."""
    safe_name = _require_safe_filename(request.name, "Invalid dataset name")
    work_dir = os.path.join(DATASET_BUILDER_DIR, safe_name)
    if not os.path.exists(work_dir):
        raise HTTPException(status_code=404, detail="Project not found")
    state = _load_builder_state(safe_name)
    state["description"] = request.description
    state["global_seed"] = request.global_seed
    _save_builder_state(safe_name, state)
    return {"status": "ok"}

@router.post("/api/dataset_builder/update_rows")
async def dataset_builder_update_rows(request: DatasetBuilderUpdateRowsRequest):
    """Update row definitions, preserving existing generation status/audio."""
    safe_name = _require_safe_filename(request.name, "Invalid dataset name")
    work_dir = os.path.join(DATASET_BUILDER_DIR, safe_name)
    if not os.path.exists(work_dir):
        raise HTTPException(status_code=404, detail="Project not found")
    state = _load_builder_state(safe_name)
    existing = state.get("samples", [])
    # Merge: keep status/audio_url from existing samples where text unchanged
    new_samples = []
    for i, row in enumerate(request.rows):
        sample = {
            "emotion": row.get("emotion", ""),
            "text": row.get("text", "").strip(),
            "seed": row.get("seed", ""),
            "status": "pending",
            "audio_url": None,
        }
        if i < len(existing):
            old = existing[i]
            # Preserve generation state if text unchanged (trimmed comparison)
            if old.get("text", "").strip() == sample["text"]:
                sample["status"] = old.get("status", "pending")
                sample["audio_url"] = old.get("audio_url")
        new_samples.append(sample)
    state["samples"] = new_samples
    _save_builder_state(safe_name, state)
    return {"status": "ok", "sample_count": len(new_samples)}

@router.post("/api/dataset_builder/generate_sample")
async def dataset_builder_generate_sample(request: DatasetSampleGenRequest):
    """Generate a single dataset sample using VoiceDesign."""
    safe_name = _require_safe_filename(request.dataset_name, "Invalid dataset name")

    # Same "dataset_builder" slot as the sibling /generate_batch route -
    # fail fast before any setup work below. See F-043.
    check_global_gpu_lock("dataset_builder")

    work_dir = os.path.join(DATASET_BUILDER_DIR, safe_name)
    os.makedirs(work_dir, exist_ok=True)

    claim_gpu_task("dataset_builder")
    try:
        engine = project_manager.get_engine()
        if not engine:
            raise HTTPException(status_code=500, detail="Failed to initialize TTS engine")

        wav_path, _ = engine.generate_voice_design(
            description=request.description,
            sample_text=request.text,
            seed=request.seed,
        )

        dest_filename = f"sample_{request.sample_index:03d}.wav"
        dest_path = os.path.join(work_dir, dest_filename)
        shutil.copy2(wav_path, dest_path)

        # Update state (cache-bust URL so browser loads fresh audio on regen)
        cache_bust = int(time.time())
        audio_url = f"/dataset_builder/{safe_name}/{dest_filename}?t={cache_bust}"
        state = _load_builder_state(safe_name)
        samples = state.get("samples", [])
        # Ensure list is large enough
        while len(samples) <= request.sample_index:
            samples.append({"status": "pending"})
        existing_sample = samples[request.sample_index] if request.sample_index < len(samples) else {}
        samples[request.sample_index] = {
            **existing_sample,
            "status": "done",
            "audio_url": audio_url,
            "text": request.text.strip(),
            "description": request.description,
        }
        state["samples"] = samples
        _save_builder_state(safe_name, state)

        return {
            "status": "done",
            "sample_index": request.sample_index,
            "audio_url": audio_url,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Dataset builder sample generation failed")
        # Mark as error in state
        state = _load_builder_state(safe_name)
        samples = state.get("samples", [])
        while len(samples) <= request.sample_index:
            samples.append({"status": "pending"})
        samples[request.sample_index] = {"status": "error", "error": str(e)}
        state["samples"] = samples
        _save_builder_state(safe_name, state)
        raise HTTPException(status_code=500, detail="Sample generation failed — see server logs for details.") from e
    finally:
        process_state["dataset_builder"]["running"] = False

@router.post("/api/dataset_builder/generate_batch")
async def dataset_builder_generate_batch(request: DatasetBatchGenRequest):
    """Batch generate dataset samples as a background task."""
    check_global_gpu_lock("dataset_builder")

    if not request.samples or len(request.samples) == 0:
        raise HTTPException(status_code=400, detail="No samples provided")

    safe_name = _require_safe_filename(request.name, "Invalid dataset name")

    work_dir = os.path.join(DATASET_BUILDER_DIR, safe_name)
    os.makedirs(work_dir, exist_ok=True)
    root_desc = request.description.strip()

    # Determine which indices to generate
    if request.indices is not None:
        to_generate = request.indices
    else:
        to_generate = list(range(len(request.samples)))

    # Reject out-of-range indices up front (e.g. a stale frontend selection
    # referencing a row that was since removed) - letting one through used to
    # crash the background thread with an uncaught IndexError before it ever
    # reached the per-sample try/except, leaving claim_gpu_task's "running"
    # flag stuck True forever and permanently deadlocking every other GPU
    # task behind check_global_gpu_lock until the server was restarted.
    bad_indices = [idx for idx in to_generate if not (0 <= idx < len(request.samples))]
    if bad_indices:
        raise HTTPException(status_code=400,
                            detail=f"indices out of range for {len(request.samples)} sample(s): {bad_indices}")

    total = len(to_generate)

    # Snapshot request data for the thread (request object may not survive)
    samples_snapshot = [(s.emotion.strip(), s.text.strip()) for s in request.samples]
    global_seed = request.global_seed
    per_seeds = request.seeds

    def task():
        process_state["dataset_builder"]["running"] = True
        process_state["dataset_builder"]["logs"] = []
        # Wrapped in try/finally so ANY unexpected exception in this thread -
        # not just the ones already anticipated by the per-sample try/except
        # below - still releases the GPU lock. An uncaught exception in a
        # background thread doesn't propagate or crash the process; it just
        # kills the thread silently, which previously left "running" stuck
        # True forever and permanently deadlocked every other GPU task behind
        # check_global_gpu_lock until the server was restarted.
        try:
            engine = project_manager.get_engine()
            if not engine:
                process_state["dataset_builder"]["logs"].append("[ERROR] Failed to initialize TTS engine")
                return

            state = _load_builder_state(safe_name)
            samples_state = state.get("samples", [])
            # Ensure list is large enough for all samples
            while len(samples_state) < len(samples_snapshot):
                samples_state.append({"status": "pending"})

            completed = 0
            for i, idx in enumerate(to_generate):
                if process_state["dataset_builder"]["cancel"]:
                    process_state["dataset_builder"]["logs"].append(f"[CANCEL] Stopped at {completed}/{total}")
                    break

                emotion, text = samples_snapshot[idx]
                description = f"{root_desc}, {emotion}" if emotion else root_desc

                # Mark as generating (preserve existing fields like emotion, seed)
                existing_s = samples_state[idx] if idx < len(samples_state) else {}
                samples_state[idx] = {**existing_s, "status": "generating", "text": text, "emotion": emotion, "description": description}
                state["samples"] = samples_state
                _save_builder_state(safe_name, state)

                process_state["dataset_builder"]["logs"].append(
                    f"[{i+1}/{total}] {('[' + emotion + '] ' if emotion else '')}\"{text[:60]}{'...' if len(text) > 60 else ''}\""
                )

                try:
                    # Resolve seed: per-line > global > random
                    seed = -1
                    if per_seeds and idx < len(per_seeds) and per_seeds[idx] >= 0:
                        seed = per_seeds[idx]
                    elif global_seed >= 0:
                        seed = global_seed

                    wav_path, _ = engine.generate_voice_design(
                        description=description,
                        sample_text=text,
                        seed=seed,
                    )
                    dest_filename = f"sample_{idx:03d}.wav"
                    dest_path = os.path.join(work_dir, dest_filename)
                    shutil.copy2(wav_path, dest_path)

                    samples_state[idx] = {
                        **samples_state[idx],
                        "status": "done",
                        "audio_url": f"/dataset_builder/{safe_name}/{dest_filename}?t={int(time.time())}",
                        "text": text,
                        "emotion": emotion,
                        "description": description,
                    }
                    completed += 1
                except Exception as e:
                    logger.error(f"Dataset builder sample {idx} failed: {e}")
                    process_state["dataset_builder"]["logs"].append(f"  Error: {e}")
                    samples_state[idx] = {**samples_state[idx], "status": "error", "error": str(e), "text": text, "emotion": emotion}

                state["samples"] = samples_state
                _save_builder_state(safe_name, state)

            process_state["dataset_builder"]["logs"].append(
                f"[DONE] Generated {completed}/{total} samples"
            )
        except Exception as e:
            logger.exception("Dataset builder batch generation crashed")
            process_state["dataset_builder"]["logs"].append(f"[ERROR] Batch generation crashed: {e}")
        finally:
            process_state["dataset_builder"]["running"] = False

    claim_gpu_task("dataset_builder")
    threading.Thread(target=task, daemon=True).start()
    return {"status": "started", "dataset_name": safe_name, "total": total}

@router.post("/api/dataset_builder/cancel")
async def dataset_builder_cancel():
    """Cancel ongoing batch dataset generation."""
    if process_state["dataset_builder"]["running"]:
        process_state["dataset_builder"]["cancel"] = True
        return {"status": "cancelling"}
    return {"status": "not_running"}

@router.get("/api/dataset_builder/status/{name}")
async def dataset_builder_status(name: str):
    """Get per-sample generation status for a dataset builder project."""
    safe_name = _require_safe_filename(name, "Invalid dataset name")
    state = _load_builder_state(safe_name)
    return {
        "description": state.get("description", ""),
        "global_seed": state.get("global_seed", ""),
        "samples": state.get("samples", []),
        "running": process_state["dataset_builder"]["running"],
        "logs": process_state["dataset_builder"]["logs"],
    }

@router.post("/api/dataset_builder/save")
async def dataset_builder_save(request: DatasetSaveRequest):
    """Finalize dataset builder project as a training dataset."""
    safe_name = _require_safe_filename(request.name, "Invalid dataset name")

    work_dir = os.path.join(DATASET_BUILDER_DIR, safe_name)
    if not os.path.exists(work_dir):
        raise HTTPException(status_code=404, detail="Dataset builder project not found")

    state = _load_builder_state(safe_name)
    samples = state.get("samples", [])

    # Collect completed samples
    done_samples = [(i, s) for i, s in enumerate(samples) if s.get("status") == "done"]
    if not done_samples:
        raise HTTPException(status_code=400, detail="No completed samples to save")
    missing_samples = [
        f"sample_{i:03d}.wav" for i, _sample in done_samples
        if not os.path.isfile(os.path.join(work_dir, f"sample_{i:03d}.wav"))
    ]
    if missing_samples:
        preview = ", ".join(missing_samples[:5])
        suffix = " and more" if len(missing_samples) > 5 else ""
        raise HTTPException(
            status_code=400,
            detail=f"Completed sample audio is missing: {preview}{suffix}. Regenerate it before saving.",
        )

    # Check ref_index is valid
    ref_idx = request.ref_index
    ref_sample = next((s for i, s in done_samples if i == ref_idx), None)
    if ref_sample is None:
        # Fall back to first completed sample
        ref_idx = done_samples[0][0]
        ref_sample = done_samples[0][1]

    # PICK A REPRESENTATIVE REFERENCE, NOT JUST THE FIRST CLIP.
    #
    # train_lora.py anchors the speaker embedding for ALL samples to this one
    # file, so an unrepresentative clip here poisons the whole adapter. Across
    # the 75 shipped adapters, a mismatched reference made an adapter 6.4x more
    # likely to fail (86% vs 13%), and correlated +0.76 with adapter quality.
    # The worst case was anchored to a clip scoring -0.026 against its own
    # dataset while a 0.882 clip sat unused in the same data.
    #
    # The medoid - the clip most similar to all the others - is representative
    # by construction and robust to a minority of misdiarized clips.
    #
    # Only when the caller did not choose one. An explicit ref_index is a
    # deliberate human decision and is not second-guessed.
    if request.ref_index == 0 and len(done_samples) >= 3:
        from voice_reference import select_reference_sample
        candidates = [os.path.join(work_dir, f"sample_{i:03d}.wav")
                      for i, _ in done_samples]
        pick, score = select_reference_sample(candidates)
        if pick is not None:
            chosen_i, chosen_sample = done_samples[pick]
            logger.info(
                f"Reference: chose sample_{chosen_i:03d} as the medoid "
                f"(similarity {score} to the rest of the dataset) instead of "
                f"the first clip")
            ref_idx, ref_sample = chosen_i, chosen_sample
        else:
            # Not measurable here - the speaker model lives in the sibling
            # interpreter. Say so, rather than leaving the reference silently
            # unverified, which is how this defect survived.
            logger.warning(
                "Reference: could not verify sample_%03d is representative "
                "(speaker model unavailable); using it unchecked", ref_idx)

    # Create training dataset directory
    dataset_dir = os.path.join(LORA_DATASETS_DIR, safe_name)
    if os.path.exists(dataset_dir):
        raise HTTPException(status_code=400, detail=f"Dataset '{safe_name}' already exists in training datasets")

    os.makedirs(dataset_dir, exist_ok=True)

    try:
        metadata_lines = []
        for i, sample in done_samples:
            src_filename = f"sample_{i:03d}.wav"
            src_path = os.path.join(work_dir, src_filename)

            dest_filename = f"sample_{i:03d}.wav"
            shutil.copy2(src_path, os.path.join(dataset_dir, dest_filename))

            metadata_lines.append(json.dumps({
                "audio_filepath": dest_filename,
                "text": sample.get("text", ""),
                "ref_audio": "ref.wav",
            }, ensure_ascii=False))

        # Copy ref sample and save its text for correct clone prompt alignment
        ref_src = os.path.join(work_dir, f"sample_{ref_idx:03d}.wav")
        shutil.copy2(ref_src, os.path.join(dataset_dir, "ref.wav"))
        ref_text = ref_sample.get("text", "")
        with open(os.path.join(dataset_dir, "ref_text.txt"), "w", encoding="utf-8") as f:
            f.write(ref_text)

        # Write metadata
        with open(os.path.join(dataset_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
            f.write("\n".join(metadata_lines) + "\n")

        sample_count = len(metadata_lines)
        logger.info(f"Dataset saved: '{safe_name}' ({sample_count} samples, ref=sample_{ref_idx:03d})")

        return {
            "status": "saved",
            "dataset_id": safe_name,
            "sample_count": sample_count,
        }
    except Exception as e:
        # Clean up on failure
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir, ignore_errors=True)
        logger.exception("Dataset save failed")
        raise HTTPException(status_code=500, detail="Dataset save failed — see server logs for details.") from e

@router.delete("/api/dataset_builder/{name}")
async def dataset_builder_delete(name: str):
    """Discard a dataset builder working project."""
    work_dir = _safe_subpath(DATASET_BUILDER_DIR, name)
    if not os.path.exists(work_dir):
        raise HTTPException(status_code=404, detail="Dataset builder project not found")
    shutil.rmtree(work_dir, ignore_errors=True)
    logger.info(f"Dataset builder project discarded: {name}")
    return {"status": "deleted", "name": name}
