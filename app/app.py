import os
import sys
import json
import shutil
import asyncio
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import re
import time
import threading
import zipfile
import subprocess
import aiofiles

# Import ProjectManager
from project import ProjectManager
from default_prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, load_default_prompts

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlexandriaUI")

app = FastAPI(title="Alexandria Audiobook")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
VOICES_PATH = os.path.join(ROOT_DIR, "voices.json")
VOICE_CONFIG_PATH = os.path.join(ROOT_DIR, "voice_config.json")
SCRIPT_PATH = os.path.join(ROOT_DIR, "annotated_script.json")
AUDIOBOOK_PATH = os.path.join(ROOT_DIR, "cloned_audiobook.mp3")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
CHUNKS_PATH = os.path.join(ROOT_DIR, "chunks.json")
DESIGNED_VOICES_DIR = os.path.join(ROOT_DIR, "designed_voices")
LORA_MODELS_DIR = os.path.join(ROOT_DIR, "lora_models")
LORA_DATASETS_DIR = os.path.join(ROOT_DIR, "lora_datasets")
DATASET_BUILDER_DIR = os.path.join(ROOT_DIR, "dataset_builder")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(SCRIPTS_DIR, exist_ok=True)
os.makedirs(DESIGNED_VOICES_DIR, exist_ok=True)
os.makedirs(LORA_MODELS_DIR, exist_ok=True)
os.makedirs(LORA_DATASETS_DIR, exist_ok=True)
os.makedirs(DATASET_BUILDER_DIR, exist_ok=True)

# Mount static files with absolute path
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Create voicelines directory if it doesn't exist to prevent startup error
VOICELINES_DIR = os.path.join(ROOT_DIR, "voicelines")
os.makedirs(VOICELINES_DIR, exist_ok=True)
app.mount("/voicelines", StaticFiles(directory=VOICELINES_DIR), name="voicelines")

# Designed voices directory for voice designer feature
app.mount("/designed_voices", StaticFiles(directory=DESIGNED_VOICES_DIR), name="designed_voices")

# LoRA models directory for trained adapter test audio
app.mount("/lora_models", StaticFiles(directory=LORA_MODELS_DIR), name="lora_models")

# Dataset builder directory for preview audio
app.mount("/dataset_builder", StaticFiles(directory=DATASET_BUILDER_DIR), name="dataset_builder")

# Initialize Project Manager
project_manager = ProjectManager(ROOT_DIR)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class LLMConfig(BaseModel):
    base_url: str
    api_key: str
    model_name: str

class TTSConfig(BaseModel):
    mode: str = "external"  # "local" or "external"
    url: str = "http://127.0.0.1:7860"  # external mode only
    device: str = "auto"  # local mode: "auto", "cuda:0", "cpu", etc.
    parallel_workers: int = 2  # concurrent TTS workers
    batch_seed: Optional[int] = None  # Single seed for batch mode, None/-1 = random
    compile_codec: bool = False  # torch.compile the codec for ~3-4x batch throughput (slow first run)
    sub_batch_enabled: bool = True  # split batch by text length to reduce padding waste
    sub_batch_min_size: int = 4  # minimum chunks per sub-batch before allowing a split
    sub_batch_ratio: float = 5.0  # max longest/shortest length ratio before splitting

class PromptConfig(BaseModel):
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None

class GenerationConfig(BaseModel):
    chunk_size: int = 3000
    max_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0
    presence_penalty: float = 0.0
    banned_tokens: List[str] = []

class AppConfig(BaseModel):
    llm: LLMConfig
    tts: TTSConfig
    prompts: Optional[PromptConfig] = None
    generation: Optional[GenerationConfig] = None

class VoiceConfigItem(BaseModel):
    type: str = "custom"
    voice: Optional[str] = "Ryan"
    character_style: Optional[str] = ""
    default_style: Optional[str] = ""  # backward compat, prefer character_style
    seed: Optional[str] = "-1"
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    adapter_id: Optional[str] = None
    adapter_path: Optional[str] = None
    description: Optional[str] = ""  # voice description (for design type)

class ProcessStatus(BaseModel):
    running: bool
    logs: List[str]

class ChunkUpdate(BaseModel):
    text: Optional[str] = None
    instruct: Optional[str] = None
    speaker: Optional[str] = None

class BatchGenerateRequest(BaseModel):
    indices: List[int]

class VoiceDesignPreviewRequest(BaseModel):
    description: str
    sample_text: str
    language: Optional[str] = None

class VoiceDesignSaveRequest(BaseModel):
    name: str
    description: str
    sample_text: str
    preview_file: str

class LoraTrainingRequest(BaseModel):
    name: str
    dataset_id: str
    epochs: int = 5
    lr: float = 5e-6
    batch_size: int = 1
    lora_r: int = 64
    lora_alpha: int = 128
    gradient_accumulation_steps: int = 8

class LoraTestRequest(BaseModel):
    adapter_id: str
    text: str
    instruct: str = ""

class LoraDatasetSample(BaseModel):
    emotion: str = ""
    text: str

class LoraGenerateDatasetRequest(BaseModel):
    name: str
    description: str  # root voice description
    samples: Optional[List[LoraDatasetSample]] = None  # emotion+text pairs
    texts: Optional[List[str]] = None  # legacy: flat text list (no emotions)
    language: Optional[str] = None

class DatasetSampleGenRequest(BaseModel):
    description: str      # full voice description (root + emotion already combined by frontend)
    text: str
    dataset_name: str     # working directory name
    sample_index: int     # row number
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

# Global state for process tracking
process_state = {
    "script": {"running": False, "logs": []},
    "voices": {"running": False, "logs": []},
    "audio": {"running": False, "logs": []},
    "audacity_export": {"running": False, "logs": []},
    "review": {"running": False, "logs": []},
    "lora_training": {"running": False, "logs": []},
    "dataset_gen": {"running": False, "logs": []},
    "dataset_builder": {"running": False, "logs": [], "cancel": False}
}

def run_process(command: List[str], task_name: str):
    """Run a subprocess and capture logs."""
    global process_state
    process_state[task_name]["running"] = True
    process_state[task_name]["logs"] = []

    logger.info(f"Starting task {task_name}: {' '.join(command)}")

    try:
        # Use shell=True for Windows compatibility in some cases, but cleaner to pass list
        # For Windows, we might need shell=True if relying on system path resolution for python
        env = os.environ.copy()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=BASE_DIR,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )

        for line in process.stdout:
            log_line = line.strip()
            if log_line:
                process_state[task_name]["logs"].append(log_line)
                # Keep log size manageable
                if len(process_state[task_name]["logs"]) > 1000:
                    process_state[task_name]["logs"].pop(0)

        process.wait()
        return_code = process.returncode

        if return_code == 0:
            process_state[task_name]["logs"].append(f"Task {task_name} completed successfully.")
        else:
            process_state[task_name]["logs"].append(f"Task {task_name} failed with return code {return_code}.")

    except Exception as e:
        logger.error(f"Error running {task_name}: {e}")
        process_state[task_name]["logs"].append(f"Error: {str(e)}")
    finally:
        process_state[task_name]["running"] = False

# Endpoints

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/favicon.ico")
async def read_favicon():
    favicon_path = os.path.join(ROOT_DIR, "icon.png")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Favicon not found")

@app.get("/api/config")
async def get_config():
    default_config = {
        "llm": {
            "base_url": "http://localhost:11434/v1",
            "api_key": "local",
            "model_name": "richardyoung/qwen3-14b-abliterated:Q8_0"
        },
        "tts": {
            "mode": "external",
            "url": "http://127.0.0.1:7860",
            "device": "auto"
        },
        "prompts": {
            "system_prompt": "",
            "user_prompt": ""
        }
    }

    if not os.path.exists(CONFIG_PATH):
        sys_prompt, usr_prompt = load_default_prompts()
        default_config["prompts"]["system_prompt"] = sys_prompt
        default_config["prompts"]["user_prompt"] = usr_prompt
        return default_config

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Ensure prompts section exists with defaults from file
    if "prompts" not in config:
        sys_prompt, usr_prompt = load_default_prompts()
        config["prompts"] = {"system_prompt": sys_prompt, "user_prompt": usr_prompt}
    else:
        if not config["prompts"].get("system_prompt") or not config["prompts"].get("user_prompt"):
            sys_prompt, usr_prompt = load_default_prompts()
            if not config["prompts"].get("system_prompt"):
                config["prompts"]["system_prompt"] = sys_prompt
            if not config["prompts"].get("user_prompt"):
                config["prompts"]["user_prompt"] = usr_prompt

    return config

@app.get("/api/default_prompts")
async def get_default_prompts():
    system_prompt, user_prompt = load_default_prompts()
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }

@app.post("/api/config")
async def save_config(config: AppConfig):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2, ensure_ascii=False)
    # Reset engine so it picks up new TTS settings on next use
    project_manager.engine = None
    return {"status": "saved"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOADS_DIR, file.filename)
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    # Save input path to state.json to be compatible with original scripts if needed
    state_path = os.path.join(ROOT_DIR, "state.json")
    state = {}
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            try:
                state = json.load(f)
            except: pass

    state["input_file_path"] = file_path
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    return {"filename": file.filename, "path": file_path}

@app.post("/api/generate_script")
async def generate_script(background_tasks: BackgroundTasks):
    # Get input file from state.json
    state_path = os.path.join(ROOT_DIR, "state.json")
    if not os.path.exists(state_path):
        raise HTTPException(status_code=400, detail="No input file selected")

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
        input_file = state.get("input_file_path")

    if not input_file:
         raise HTTPException(status_code=400, detail="No input file found in state")

    if process_state["script"]["running"]:
         raise HTTPException(status_code=400, detail="Script generation already running")

    background_tasks.add_task(run_process, [sys.executable, "-u", "generate_script.py", input_file], "script")
    return {"status": "started"}

@app.post("/api/review_script")
async def review_script(background_tasks: BackgroundTasks):
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=400, detail="No annotated script found. Generate a script first.")

    if process_state["review"]["running"]:
        raise HTTPException(status_code=400, detail="Script review already running")

    background_tasks.add_task(run_process, [sys.executable, "-u", "review_script.py"], "review")
    return {"status": "started"}

@app.get("/api/annotated_script")
async def get_annotated_script():
    """Return the current working annotated_script.json."""
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=404, detail="No annotated script found")
    with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/status/{task_name}")
async def get_status(task_name: str):
    if task_name not in process_state:
        raise HTTPException(status_code=404, detail="Task not found")
    return process_state[task_name]

@app.get("/api/voices")
async def get_voices():
    # Run parse_voices first to ensure we have latest
    # But we don't want to block, so we'll just read what's there
    # User can trigger parse manually if needed

    if not os.path.exists(VOICES_PATH):
        return []

    with open(VOICES_PATH, "r", encoding="utf-8") as f:
        voices_list = json.load(f)

    # Combine with config
    voice_config = {}
    if os.path.exists(VOICE_CONFIG_PATH):
        with open(VOICE_CONFIG_PATH, "r", encoding="utf-8") as f:
            voice_config = json.load(f)

    result = []
    for voice_name in voices_list:
        config = voice_config.get(voice_name, {})
        result.append({
            "name": voice_name,
            "config": config
        })
    return result

@app.post("/api/parse_voices")
async def parse_voices(background_tasks: BackgroundTasks):
    if process_state["voices"]["running"]:
         raise HTTPException(status_code=400, detail="Voice parsing already running")

    background_tasks.add_task(run_process, [sys.executable, "-u", "parse_voices.py"], "voices")
    return {"status": "started"}

@app.post("/api/save_voice_config")
async def save_voice_config(config_data: Dict[str, VoiceConfigItem]):
    # Read existing to preserve any fields not sent?
    # For now, we assume frontend sends full config or we just overwrite specific keys

    current_config = {}
    if os.path.exists(VOICE_CONFIG_PATH):
        with open(VOICE_CONFIG_PATH, "r", encoding="utf-8") as f:
            try:
                current_config = json.load(f)
            except: pass

    # Update current config with new data
    for voice_name, config in config_data.items():
        # Convert Pydantic model to dict
        current_config[voice_name] = config.model_dump()

    with open(VOICE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(current_config, f, indent=2, ensure_ascii=False)

    return {"status": "saved"}

@app.get("/api/audiobook")
async def get_audiobook():
    if not os.path.exists(AUDIOBOOK_PATH):
        raise HTTPException(status_code=404, detail="Audiobook not found")
    return FileResponse(AUDIOBOOK_PATH, filename="audiobook.mp3", media_type="audio/mpeg")

# --- Chunk Management Endpoints ---

@app.get("/api/chunks")
async def get_chunks():
    chunks = project_manager.load_chunks()
    return chunks

@app.post("/api/chunks/{index}")
async def update_chunk(index: int, update: ChunkUpdate):
    data = update.model_dump(exclude_unset=True)
    logger.info(f"Updating chunk {index} with data: {data}")
    chunk = project_manager.update_chunk(index, data)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    logger.info(f"Chunk {index} updated, instruct is now: '{chunk.get('instruct', '')}'")
    return chunk

@app.post("/api/chunks/{index}/generate")
async def generate_chunk_endpoint(index: int, background_tasks: BackgroundTasks):
    def task():
        project_manager.generate_chunk_audio(index)

    background_tasks.add_task(task)
    return {"status": "started"}

@app.post("/api/merge")
async def merge_audio_endpoint(background_tasks: BackgroundTasks):
    # Reuse audio process state for merge if possible, or just background it
    # For simplicity, we just background it and frontend will assume it works
    # Or we can link it to process_state["audio"]

    def task():
        process_state["audio"]["running"] = True
        process_state["audio"]["logs"] = ["Starting merge..."]
        try:
            success, msg = project_manager.merge_audio()
            if success:
                process_state["audio"]["logs"].append(f"Merge complete: {msg}")
            else:
                process_state["audio"]["logs"].append(f"Merge failed: {msg}")
        except Exception as e:
            process_state["audio"]["logs"].append(f"Merge error: {e}")
        finally:
            process_state["audio"]["running"] = False

    background_tasks.add_task(task)
    return {"status": "started"}

@app.post("/api/export_audacity")
async def export_audacity_endpoint(background_tasks: BackgroundTasks):
    if process_state["audacity_export"]["running"]:
        raise HTTPException(status_code=400, detail="Audacity export already running")

    def task():
        process_state["audacity_export"]["running"] = True
        process_state["audacity_export"]["logs"] = ["Starting Audacity export..."]
        try:
            success, msg = project_manager.export_audacity()
            if success:
                process_state["audacity_export"]["logs"].append(f"Export complete: {msg}")
            else:
                process_state["audacity_export"]["logs"].append(f"Export failed: {msg}")
        except Exception as e:
            process_state["audacity_export"]["logs"].append(f"Export error: {e}")
        finally:
            process_state["audacity_export"]["running"] = False

    background_tasks.add_task(task)
    return {"status": "started"}

@app.get("/api/export_audacity")
async def get_audacity_export():
    zip_path = os.path.join(ROOT_DIR, "audacity_export.zip")
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="Audacity export not found. Generate it first.")
    return FileResponse(zip_path, filename="audacity_export.zip", media_type="application/zip")

@app.post("/api/generate_batch")
async def generate_batch_endpoint(request: BatchGenerateRequest, background_tasks: BackgroundTasks):
    """Generate multiple chunks in parallel using configured worker count."""
    if process_state["audio"]["running"]:
        raise HTTPException(status_code=400, detail="Audio generation already running")

    # Load worker count from config
    workers = 2
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                workers = max(1, cfg.get("tts", {}).get("parallel_workers", 2))
        except:
            pass

    indices = request.indices
    total = len(indices)

    def progress_callback(completed, failed, total):
        """Update logs with progress."""
        process_state["audio"]["logs"].append(
            f"Progress: {completed + failed}/{total} ({completed} done, {failed} failed)"
        )

    def task():
        process_state["audio"]["running"] = True
        process_state["audio"]["logs"] = [
            f"Starting parallel generation of {total} chunks with {workers} workers..."
        ]
        try:
            results = project_manager.generate_chunks_parallel(
                indices, workers, progress_callback
            )
            completed = len(results["completed"])
            failed = len(results["failed"])
            process_state["audio"]["logs"].append(
                f"Batch generation complete: {completed} succeeded, {failed} failed"
            )
            if results["failed"]:
                for idx, msg in results["failed"]:
                    process_state["audio"]["logs"].append(f"  Chunk {idx} failed: {msg}")
        except Exception as e:
            logger.error(f"Batch generation error: {e}")
            process_state["audio"]["logs"].append(f"Batch generation error: {e}")
        finally:
            process_state["audio"]["running"] = False

    background_tasks.add_task(task)
    return {"status": "started", "workers": workers, "total_chunks": total}

@app.post("/api/generate_batch_fast")
async def generate_batch_fast_endpoint(request: BatchGenerateRequest, background_tasks: BackgroundTasks):
    """Generate multiple chunks using batch TTS API with single seed. Faster but less flexible.
    Requires custom Qwen3-TTS with /generate_batch endpoint."""
    if process_state["audio"]["running"]:
        raise HTTPException(status_code=400, detail="Audio generation already running")

    # Load batch_seed and batch_size from config
    batch_seed = -1
    batch_size = 4
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                tts_cfg = cfg.get("tts", {})
                seed_val = tts_cfg.get("batch_seed")
                if seed_val is not None and seed_val != "":
                    batch_seed = int(seed_val)
                batch_size = max(1, tts_cfg.get("parallel_workers", 4))
        except:
            pass

    indices = request.indices
    total = len(indices)

    def progress_callback(completed, failed, total):
        process_state["audio"]["logs"].append(
            f"Progress: {completed + failed}/{total} ({completed} done, {failed} failed)"
        )

    def task():
        process_state["audio"]["running"] = True
        process_state["audio"]["logs"] = [
            f"Starting batch generation of {total} chunks (batch_size={batch_size}, seed={batch_seed})..."
        ]
        try:
            results = project_manager.generate_chunks_batch(
                indices, batch_seed, batch_size, progress_callback
            )
            completed = len(results["completed"])
            failed = len(results["failed"])
            process_state["audio"]["logs"].append(
                f"Batch generation complete: {completed} succeeded, {failed} failed"
            )
            if results["failed"]:
                for idx, msg in results["failed"]:
                    process_state["audio"]["logs"].append(f"  Chunk {idx} failed: {msg}")
        except Exception as e:
            logger.error(f"Batch generation error: {e}")
            process_state["audio"]["logs"].append(f"Batch generation error: {e}")
        finally:
            process_state["audio"]["running"] = False

    background_tasks.add_task(task)
    return {"status": "started", "batch_seed": batch_seed, "batch_size": batch_size, "total_chunks": total}

## ── Saved Scripts ──────────────────────────────────────────────

def _sanitize_script_name(name: str) -> str:
    """Make a string safe for use as a filename."""
    name = re.sub(r'[^\w\- ]', '', name).strip()
    name = re.sub(r'\s+', '_', name)
    return name.lower()

@app.get("/api/scripts")
async def list_saved_scripts():
    """List all saved scripts in the scripts/ directory."""
    scripts = []
    for f in os.listdir(SCRIPTS_DIR):
        if f.endswith(".json") and not f.endswith(".voice_config.json"):
            name = f[:-5]  # strip .json
            filepath = os.path.join(SCRIPTS_DIR, f)
            companion = os.path.join(SCRIPTS_DIR, f"{name}.voice_config.json")
            scripts.append({
                "name": name,
                "created": os.path.getmtime(filepath),
                "has_voice_config": os.path.exists(companion)
            })
    scripts.sort(key=lambda x: x["created"], reverse=True)
    return scripts

class ScriptSaveRequest(BaseModel):
    name: str

@app.post("/api/scripts/save")
async def save_script(request: ScriptSaveRequest):
    """Save the current annotated_script.json (and voice_config.json) under a name."""
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=404, detail="No annotated script to save. Generate a script first.")

    safe_name = _sanitize_script_name(request.name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid script name.")

    dest = os.path.join(SCRIPTS_DIR, f"{safe_name}.json")
    shutil.copy2(SCRIPT_PATH, dest)

    if os.path.exists(VOICE_CONFIG_PATH):
        shutil.copy2(VOICE_CONFIG_PATH, os.path.join(SCRIPTS_DIR, f"{safe_name}.voice_config.json"))

    logger.info(f"Script saved as '{safe_name}'")
    return {"status": "saved", "name": safe_name}

class ScriptLoadRequest(BaseModel):
    name: str

@app.post("/api/scripts/load")
async def load_script(request: ScriptLoadRequest):
    """Load a saved script, replacing the current annotated_script.json and chunks."""
    if process_state["audio"]["running"]:
        raise HTTPException(status_code=409, detail="Cannot load a script while audio generation is running.")

    src = os.path.join(SCRIPTS_DIR, f"{request.name}.json")
    if not os.path.exists(src):
        raise HTTPException(status_code=404, detail=f"Saved script '{request.name}' not found.")

    shutil.copy2(src, SCRIPT_PATH)

    companion = os.path.join(SCRIPTS_DIR, f"{request.name}.voice_config.json")
    if os.path.exists(companion):
        shutil.copy2(companion, VOICE_CONFIG_PATH)

    # Delete chunks so they regenerate from the loaded script
    if os.path.exists(CHUNKS_PATH):
        os.remove(CHUNKS_PATH)

    logger.info(f"Script '{request.name}' loaded")
    return {"status": "loaded", "name": request.name}

@app.delete("/api/scripts/{name}")
async def delete_script(name: str):
    """Delete a saved script."""
    filepath = os.path.join(SCRIPTS_DIR, f"{name}.json")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Saved script '{name}' not found.")

    os.remove(filepath)
    companion = os.path.join(SCRIPTS_DIR, f"{name}.voice_config.json")
    if os.path.exists(companion):
        os.remove(companion)

    logger.info(f"Script '{name}' deleted")
    return {"status": "deleted", "name": name}

## ── Voice Designer ──────────────────────────────────────────────

DESIGNED_VOICES_MANIFEST = os.path.join(DESIGNED_VOICES_DIR, "manifest.json")

def _load_design_manifest():
    if os.path.exists(DESIGNED_VOICES_MANIFEST):
        try:
            with open(DESIGNED_VOICES_MANIFEST, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass
    return []

def _save_design_manifest(manifest):
    with open(DESIGNED_VOICES_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

@app.post("/api/voice_design/preview")
async def voice_design_preview(request: VoiceDesignPreviewRequest):
    """Generate a preview voice from a text description."""
    engine = project_manager.get_engine()
    if not engine:
        raise HTTPException(status_code=500, detail="Failed to initialize TTS engine")

    try:
        wav_path, sr = engine.generate_voice_design(
            description=request.description,
            sample_text=request.sample_text,
            language=request.language,
        )
        # Return relative URL for the static mount
        filename = os.path.basename(wav_path)
        return {"status": "ok", "audio_url": f"/designed_voices/previews/{filename}"}
    except Exception as e:
        logger.error(f"Voice design preview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice_design/save")
async def voice_design_save(request: VoiceDesignSaveRequest):
    """Save a preview voice as a permanent designed voice."""
    previews_dir = os.path.join(DESIGNED_VOICES_DIR, "previews")
    preview_path = os.path.join(previews_dir, request.preview_file)

    if not os.path.exists(preview_path):
        raise HTTPException(status_code=404, detail="Preview file not found")

    # Sanitize name for filename
    safe_name = re.sub(r'[^\w\- ]', '', request.name).strip()
    safe_name = re.sub(r'\s+', '_', safe_name).lower()
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid voice name")

    # Generate unique ID
    import time
    voice_id = f"{safe_name}_{int(time.time())}"
    dest_filename = f"{voice_id}.wav"
    dest_path = os.path.join(DESIGNED_VOICES_DIR, dest_filename)

    shutil.copy2(preview_path, dest_path)

    # Update manifest
    manifest = _load_design_manifest()
    manifest.append({
        "id": voice_id,
        "name": request.name,
        "description": request.description,
        "sample_text": request.sample_text,
        "filename": dest_filename,
    })
    _save_design_manifest(manifest)

    logger.info(f"Designed voice saved: '{request.name}' as {dest_filename}")
    return {"status": "saved", "voice_id": voice_id}

@app.get("/api/voice_design/list")
async def voice_design_list():
    """List all saved designed voices."""
    return _load_design_manifest()

@app.delete("/api/voice_design/{voice_id}")
async def voice_design_delete(voice_id: str):
    """Delete a saved designed voice."""
    manifest = _load_design_manifest()
    entry = next((v for v in manifest if v["id"] == voice_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Voice not found")

    # Delete WAV file
    wav_path = os.path.join(DESIGNED_VOICES_DIR, entry["filename"])
    if os.path.exists(wav_path):
        os.remove(wav_path)

    # Remove from manifest
    manifest = [v for v in manifest if v["id"] != voice_id]
    _save_design_manifest(manifest)

    logger.info(f"Designed voice deleted: {voice_id}")
    return {"status": "deleted", "voice_id": voice_id}

## ── LoRA Training ──────────────────────────────────────────────

LORA_MODELS_MANIFEST = os.path.join(LORA_MODELS_DIR, "manifest.json")

def _load_lora_manifest():
    if os.path.exists(LORA_MODELS_MANIFEST):
        try:
            with open(LORA_MODELS_MANIFEST, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass
    return []

def _save_lora_manifest(manifest):
    with open(LORA_MODELS_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

@app.post("/api/lora/upload_dataset")
async def lora_upload_dataset(file: UploadFile = File(...)):
    """Upload a ZIP containing WAV files and metadata.jsonl."""
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a .zip archive")

    # Derive dataset name from ZIP filename
    dataset_name = re.sub(r'[^\w\- ]', '', os.path.splitext(file.filename)[0]).strip()
    dataset_name = re.sub(r'\s+', '_', dataset_name).lower()
    if not dataset_name:
        raise HTTPException(status_code=400, detail="Invalid dataset name from filename")

    dataset_dir = os.path.join(LORA_DATASETS_DIR, dataset_name)
    if os.path.exists(dataset_dir):
        raise HTTPException(status_code=400, detail=f"Dataset '{dataset_name}' already exists")

    # Save ZIP temporarily, then extract
    tmp_path = os.path.join(LORA_DATASETS_DIR, f"_tmp_{dataset_name}.zip")
    try:
        async with aiofiles.open(tmp_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        os.makedirs(dataset_dir, exist_ok=True)
        with zipfile.ZipFile(tmp_path, "r") as zf:
            zf.extractall(dataset_dir)

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

        # Count samples
        sample_count = 0
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sample_count += 1

        logger.info(f"LoRA dataset uploaded: '{dataset_name}' ({sample_count} samples)")
        return {"status": "uploaded", "dataset_id": dataset_name, "sample_count": sample_count}

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/api/lora/generate_dataset")
async def lora_generate_dataset(request: LoraGenerateDatasetRequest, background_tasks: BackgroundTasks):
    """Generate a LoRA training dataset using Voice Designer.

    Generates multiple audio samples with the same voice description,
    saving them as a ready-to-train dataset.
    """
    if process_state["dataset_gen"]["running"]:
        raise HTTPException(status_code=400, detail="Dataset generation already running")

    # Build unified sample list from either format
    sample_list = []
    if request.samples:
        for s in request.samples:
            if s.text.strip():
                sample_list.append({"emotion": s.emotion.strip(), "text": s.text.strip()})
    elif request.texts:
        for t in request.texts:
            if t.strip():
                sample_list.append({"emotion": "", "text": t.strip()})

    if not sample_list:
        raise HTTPException(status_code=400, detail="Provide at least one sample text")

    safe_name = re.sub(r'[^\w\- ]', '', request.name).strip()
    safe_name = re.sub(r'\s+', '_', safe_name).lower()
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid dataset name")

    dataset_dir = os.path.join(LORA_DATASETS_DIR, safe_name)
    if os.path.exists(dataset_dir):
        raise HTTPException(status_code=400, detail=f"Dataset '{safe_name}' already exists")

    total = len(sample_list)
    root_description = request.description.strip()

    def task():
        process_state["dataset_gen"]["running"] = True
        process_state["dataset_gen"]["logs"] = [
            f"Generating {total} samples with VoiceDesign..."
        ]
        try:
            engine = project_manager.get_engine()
            if not engine:
                process_state["dataset_gen"]["logs"].append("Error: TTS engine not initialized")
                return

            os.makedirs(dataset_dir, exist_ok=True)
            metadata_lines = []
            completed = 0

            for i, sample in enumerate(sample_list):
                text = sample["text"]
                emotion = sample["emotion"]
                # Build full description: root + emotion if provided
                description = f"{root_description}, {emotion}" if emotion else root_description

                process_state["dataset_gen"]["logs"].append(
                    f"[{i+1}/{total}] {('[' + emotion + '] ' if emotion else '')}\"{ text[:60]}{'...' if len(text) > 60 else ''}\""
                )
                try:
                    wav_path, sr = engine.generate_voice_design(
                        description=description,
                        sample_text=text,
                        language=request.language,
                    )
                    # Copy to dataset dir with sequential name
                    dest_filename = f"sample_{i:03d}.wav"
                    dest_path = os.path.join(dataset_dir, dest_filename)
                    shutil.copy2(wav_path, dest_path)

                    # Save first successful sample as ref.wav for consistent speaker embedding
                    if completed == 0:
                        shutil.copy2(wav_path, os.path.join(dataset_dir, "ref.wav"))

                    metadata_lines.append(json.dumps({
                        "audio_filepath": dest_filename,
                        "text": text,
                        "ref_audio": "ref.wav",
                    }, ensure_ascii=False))
                    completed += 1
                    process_state["dataset_gen"]["logs"].append(
                        f"  Saved {dest_filename}"
                    )
                except Exception as e:
                    process_state["dataset_gen"]["logs"].append(
                        f"  Failed: {e}"
                    )

            # Write metadata.jsonl
            metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
            with open(metadata_path, "w", encoding="utf-8") as f:
                f.write("\n".join(metadata_lines) + "\n")

            process_state["dataset_gen"]["logs"].append(
                f"Dataset '{safe_name}' complete: {completed}/{total} samples generated."
            )
            logger.info(f"LoRA dataset generated: '{safe_name}' ({completed} samples)")

        except Exception as e:
            process_state["dataset_gen"]["logs"].append(f"Error: {e}")
            logger.error(f"Dataset generation error: {e}")
            # Clean up partial dataset on failure
            if os.path.exists(dataset_dir):
                shutil.rmtree(dataset_dir)
        finally:
            process_state["dataset_gen"]["running"] = False

    background_tasks.add_task(task)
    return {"status": "started", "dataset_id": safe_name, "total": total}

@app.get("/api/lora/datasets")
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
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        sample_count += 1
        datasets.append({"dataset_id": name, "sample_count": sample_count})
    return datasets

@app.delete("/api/lora/datasets/{dataset_id}")
async def lora_delete_dataset(dataset_id: str):
    """Delete an uploaded dataset."""
    dataset_dir = os.path.join(LORA_DATASETS_DIR, dataset_id)
    if not os.path.isdir(dataset_dir):
        raise HTTPException(status_code=404, detail="Dataset not found")

    shutil.rmtree(dataset_dir)
    logger.info(f"LoRA dataset deleted: {dataset_id}")
    return {"status": "deleted", "dataset_id": dataset_id}

@app.post("/api/lora/train")
async def lora_start_training(request: LoraTrainingRequest, background_tasks: BackgroundTasks):
    """Start LoRA training as a subprocess."""
    if process_state["lora_training"]["running"]:
        raise HTTPException(status_code=400, detail="LoRA training already running")

    # Validate dataset exists
    dataset_dir = os.path.join(LORA_DATASETS_DIR, request.dataset_id)
    if not os.path.isdir(dataset_dir):
        raise HTTPException(status_code=400, detail=f"Dataset '{request.dataset_id}' not found")

    # Build output directory
    safe_name = re.sub(r'[^\w\- ]', '', request.name).strip()
    safe_name = re.sub(r'\s+', '_', safe_name).lower()
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid adapter name")

    adapter_id = f"{safe_name}_{int(time.time())}"
    output_dir = os.path.join(LORA_MODELS_DIR, adapter_id)

    # Unload TTS engine to free GPU
    if project_manager.engine is not None:
        logger.info("Unloading TTS engine for LoRA training...")
        project_manager.engine = None
        import gc
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
    ]

    def on_training_complete():
        """After training subprocess finishes, update manifest if adapter was saved."""
        run_process(command, "lora_training")

        # Check if training produced an adapter
        if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "training_meta.json")):
            try:
                with open(os.path.join(output_dir, "training_meta.json"), "r") as f:
                    meta = json.load(f)

                manifest = _load_lora_manifest()
                manifest.append({
                    "id": adapter_id,
                    "name": request.name,
                    "dataset_id": request.dataset_id,
                    "epochs": meta.get("epochs", request.epochs),
                    "final_loss": meta.get("final_loss"),
                    "sample_count": meta.get("num_samples"),
                    "lora_r": meta.get("lora_r"),
                    "lr": meta.get("lr"),
                    "created": time.time(),
                })
                _save_lora_manifest(manifest)
                logger.info(f"LoRA adapter registered: {adapter_id}")
            except Exception as e:
                logger.error(f"Failed to update LoRA manifest: {e}")

    background_tasks.add_task(on_training_complete)
    return {"status": "started", "adapter_id": adapter_id}

@app.get("/api/lora/models")
async def lora_list_models():
    """List trained LoRA adapters."""
    return _load_lora_manifest()

@app.delete("/api/lora/models/{adapter_id}")
async def lora_delete_model(adapter_id: str):
    """Delete a trained LoRA adapter."""
    manifest = _load_lora_manifest()
    entry = next((m for m in manifest if m["id"] == adapter_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Adapter not found")

    # Delete adapter directory
    adapter_dir = os.path.join(LORA_MODELS_DIR, adapter_id)
    if os.path.isdir(adapter_dir):
        shutil.rmtree(adapter_dir)

    # Remove from manifest
    manifest = [m for m in manifest if m["id"] != adapter_id]
    _save_lora_manifest(manifest)

    logger.info(f"LoRA adapter deleted: {adapter_id}")
    return {"status": "deleted", "adapter_id": adapter_id}

@app.post("/api/lora/test")
async def lora_test_model(request: LoraTestRequest):
    """Generate test audio using a LoRA adapter."""
    manifest = _load_lora_manifest()
    entry = next((m for m in manifest if m["id"] == request.adapter_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Adapter not found")

    adapter_dir = os.path.join(LORA_MODELS_DIR, request.adapter_id)
    if not os.path.isdir(adapter_dir):
        raise HTTPException(status_code=404, detail="Adapter files not found")

    engine = project_manager.get_engine()
    if not engine:
        raise HTTPException(status_code=500, detail="Failed to initialize TTS engine")

    try:
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
            "audio_url": f"/lora_models/{request.adapter_id}/{output_filename}",
        }
    except Exception as e:
        logger.error(f"LoRA test generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

## ── Dataset Builder ──────────────────────────────────────────

def _load_builder_state(name):
    """Load per-sample state from dataset builder working directory."""
    state_path = os.path.join(DATASET_BUILDER_DIR, name, "state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"samples": []}

def _save_builder_state(name, state):
    """Save per-sample state to dataset builder working directory."""
    work_dir = os.path.join(DATASET_BUILDER_DIR, name)
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

@app.post("/api/dataset_builder/generate_sample")
async def dataset_builder_generate_sample(request: DatasetSampleGenRequest):
    """Generate a single dataset sample using VoiceDesign."""
    engine = project_manager.get_engine()
    if not engine:
        raise HTTPException(status_code=500, detail="Failed to initialize TTS engine")

    work_dir = os.path.join(DATASET_BUILDER_DIR, request.dataset_name)
    os.makedirs(work_dir, exist_ok=True)

    try:
        wav_path, sr = engine.generate_voice_design(
            description=request.description,
            sample_text=request.text,
            seed=request.seed,
        )

        dest_filename = f"sample_{request.sample_index:03d}.wav"
        dest_path = os.path.join(work_dir, dest_filename)
        shutil.copy2(wav_path, dest_path)

        # Update state
        state = _load_builder_state(request.dataset_name)
        samples = state.get("samples", [])
        # Ensure list is large enough
        while len(samples) <= request.sample_index:
            samples.append({"status": "pending"})
        samples[request.sample_index] = {
            "status": "done",
            "audio_url": f"/dataset_builder/{request.dataset_name}/{dest_filename}",
            "text": request.text,
            "description": request.description,
        }
        state["samples"] = samples
        _save_builder_state(request.dataset_name, state)

        return {
            "status": "done",
            "sample_index": request.sample_index,
            "audio_url": f"/dataset_builder/{request.dataset_name}/{dest_filename}",
        }
    except Exception as e:
        logger.error(f"Dataset builder sample generation failed: {e}")
        # Mark as error in state
        state = _load_builder_state(request.dataset_name)
        samples = state.get("samples", [])
        while len(samples) <= request.sample_index:
            samples.append({"status": "pending"})
        samples[request.sample_index] = {"status": "error", "error": str(e)}
        state["samples"] = samples
        _save_builder_state(request.dataset_name, state)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dataset_builder/generate_batch")
async def dataset_builder_generate_batch(request: DatasetBatchGenRequest):
    """Batch generate dataset samples as a background task."""
    if process_state["dataset_builder"]["running"]:
        raise HTTPException(status_code=400, detail="Dataset generation already running")

    if not request.samples or len(request.samples) == 0:
        raise HTTPException(status_code=400, detail="No samples provided")

    safe_name = re.sub(r'[^\w\- ]', '', request.name).strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid dataset name")

    work_dir = os.path.join(DATASET_BUILDER_DIR, safe_name)
    os.makedirs(work_dir, exist_ok=True)
    root_desc = request.description.strip()

    # Determine which indices to generate
    if request.indices is not None:
        to_generate = request.indices
    else:
        to_generate = list(range(len(request.samples)))

    total = len(to_generate)

    # Snapshot request data for the thread (request object may not survive)
    samples_snapshot = [(s.emotion.strip(), s.text.strip()) for s in request.samples]
    global_seed = request.global_seed
    per_seeds = request.seeds

    def task():
        process_state["dataset_builder"]["running"] = True
        process_state["dataset_builder"]["logs"] = []
        process_state["dataset_builder"]["cancel"] = False

        engine = project_manager.get_engine()
        if not engine:
            process_state["dataset_builder"]["logs"].append("[ERROR] Failed to initialize TTS engine")
            process_state["dataset_builder"]["running"] = False
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

            # Mark as generating
            samples_state[idx] = {"status": "generating", "text": text, "description": description}
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

                wav_path, sr = engine.generate_voice_design(
                    description=description,
                    sample_text=text,
                    seed=seed,
                )
                dest_filename = f"sample_{idx:03d}.wav"
                dest_path = os.path.join(work_dir, dest_filename)
                shutil.copy2(wav_path, dest_path)

                samples_state[idx] = {
                    "status": "done",
                    "audio_url": f"/dataset_builder/{safe_name}/{dest_filename}",
                    "text": text,
                    "description": description,
                }
                completed += 1
            except Exception as e:
                logger.error(f"Dataset builder sample {idx} failed: {e}")
                process_state["dataset_builder"]["logs"].append(f"  Error: {e}")
                samples_state[idx] = {"status": "error", "error": str(e), "text": text}

            state["samples"] = samples_state
            _save_builder_state(safe_name, state)

        process_state["dataset_builder"]["logs"].append(
            f"[DONE] Generated {completed}/{total} samples"
        )
        process_state["dataset_builder"]["running"] = False

    threading.Thread(target=task, daemon=True).start()
    return {"status": "started", "dataset_name": safe_name, "total": total}

@app.post("/api/dataset_builder/cancel")
async def dataset_builder_cancel():
    """Cancel ongoing batch dataset generation."""
    if process_state["dataset_builder"]["running"]:
        process_state["dataset_builder"]["cancel"] = True
        return {"status": "cancelling"}
    return {"status": "not_running"}

@app.get("/api/dataset_builder/status/{name}")
async def dataset_builder_status(name: str):
    """Get per-sample generation status for a dataset builder project."""
    state = _load_builder_state(name)
    return {
        "samples": state.get("samples", []),
        "running": process_state["dataset_builder"]["running"],
        "logs": process_state["dataset_builder"]["logs"],
    }

@app.post("/api/dataset_builder/save")
async def dataset_builder_save(request: DatasetSaveRequest):
    """Finalize dataset builder project as a training dataset."""
    safe_name = re.sub(r'[^\w\- ]', '', request.name).strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid dataset name")

    work_dir = os.path.join(DATASET_BUILDER_DIR, safe_name)
    if not os.path.exists(work_dir):
        raise HTTPException(status_code=404, detail="Dataset builder project not found")

    state = _load_builder_state(safe_name)
    samples = state.get("samples", [])

    # Collect completed samples
    done_samples = [(i, s) for i, s in enumerate(samples) if s.get("status") == "done"]
    if not done_samples:
        raise HTTPException(status_code=400, detail="No completed samples to save")

    # Check ref_index is valid
    ref_idx = request.ref_index
    ref_sample = next((s for i, s in done_samples if i == ref_idx), None)
    if ref_sample is None:
        # Fall back to first completed sample
        ref_idx = done_samples[0][0]
        ref_sample = done_samples[0][1]

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
            if not os.path.exists(src_path):
                continue

            dest_filename = f"sample_{i:03d}.wav"
            shutil.copy2(src_path, os.path.join(dataset_dir, dest_filename))

            metadata_lines.append(json.dumps({
                "audio_filepath": dest_filename,
                "text": sample.get("text", ""),
                "ref_audio": "ref.wav",
            }, ensure_ascii=False))

        # Copy ref sample and save its text for correct clone prompt alignment
        ref_src = os.path.join(work_dir, f"sample_{ref_idx:03d}.wav")
        if os.path.exists(ref_src):
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
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/dataset_builder/{name}")
async def dataset_builder_delete(name: str):
    """Discard a dataset builder working project."""
    work_dir = os.path.join(DATASET_BUILDER_DIR, name)
    if not os.path.exists(work_dir):
        raise HTTPException(status_code=404, detail="Dataset builder project not found")
    shutil.rmtree(work_dir, ignore_errors=True)
    logger.info(f"Dataset builder project discarded: {name}")
    return {"status": "deleted", "name": name}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4200, access_log=False)
