import os
import sys
import gc
import json
import shutil
import signal
import logging
import torch
import select
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import re
import time
import threading
import zipfile
import subprocess
import aiofiles
from html.parser import HTMLParser
import xml.etree.ElementTree as ET

# Import ProjectManager
from project import ProjectManager
from default_prompts import load_default_prompts
from review_prompts import load_review_prompts
from hf_utils import fetch_builtin_manifest, download_builtin_adapter, is_adapter_downloaded

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlexandriaUI")

app = FastAPI(title="Alexandria Audiobook")

# --- System Helpers ---

def get_gpu_stats():
    """Get current GPU memory and utilization stats."""
    if not torch.cuda.is_available():
        return None

    stats = {}
    try:
        # Memory stats (works for both NVIDIA and AMD ROCm)
        allocated = torch.cuda.memory_allocated() / 1e9  # GB
        reserved = torch.cuda.memory_reserved() / 1e9    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB

        stats['allocated_gb'] = allocated
        stats['reserved_gb'] = reserved
        stats['total_gb'] = total
        stats['allocated_percent'] = (allocated / total * 100) if total > 0 else 0

        # Try to get utilization via rocm-smi for AMD GPUs
        try:
            result = subprocess.run(
                ['/opt/rocm/bin/rocm-smi', '--showuse', '--json'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                json_lines = [line for line in result.stdout.split('\n') if line.strip().startswith('{')]
                if json_lines:
                    data = json.loads(json_lines[0])
                    for card_key, card_data in data.items():
                        gpu_use_str = card_data.get('GPU use (%)', 'N/A')
                        if gpu_use_str != 'N/A':
                            stats['utilization_percent'] = float(gpu_use_str)
                        break 
        except:
            stats['utilization_percent'] = None

    except Exception as e:
        logger.debug(f"Could not get GPU stats: {e}")
        return None

    return stats

def check_disk_space(path, required_gb):
    """Check if disk has enough space. Returns (has_space, free_gb)."""
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024 ** 3)
        return free_gb >= required_gb, free_gb
    except:
        return True, 0

# --- Text Matching Helpers ---

def _normalize_filename_tokens(stem):
    """Simple alphanumeric tokenizer for matching audio files to text sources."""
    return re.findall(r'[a-z0-9]+', stem.lower())

def _fuzzy_score(audio_tokens, book_tokens):
    if not audio_tokens or not book_tokens: return 0.0
    a, b = set(audio_tokens), set(book_tokens)
    common = a & b
    if not common: return 0.0
    precision = len(common) / len(b)
    recall    = len(common) / len(a)
    return 2 * precision * recall / (precision + recall)

@app.get("/api/preparer/suggest_source")
async def suggest_source(audio_filename: str):
    """Suggest the best matching EPUB/TXT from the uploads directory."""
    audio_stem = os.path.splitext(audio_filename)[0]
    audio_tokens = _normalize_filename_tokens(audio_stem)
    
    best_match = None
    best_score = 0.0
    
    if not os.path.exists(UPLOADS_DIR):
        return {"suggested": None}

    for f in os.listdir(UPLOADS_DIR):
        if f.lower().endswith(('.epub', '.txt')):
            # Check exact match first
            f_stem = os.path.splitext(f)[0]
            if f_stem == audio_stem:
                return {"suggested": f, "score": 1.0, "match_type": "exact"}
            
            # Fuzzy match
            f_tokens = _normalize_filename_tokens(f_stem)
            score = _fuzzy_score(audio_tokens, f_tokens)
            if score > best_score:
                best_score = score
                best_match = f

    if best_match and best_score > 0.4: # Low threshold for suggestion
        return {"suggested": best_match, "score": round(best_score, 2), "match_type": "fuzzy"}
    
    return {"suggested": None}

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
VOICES_PATH = os.path.join(ROOT_DIR, "voices.json")
VOICE_CONFIG_PATH = os.path.join(ROOT_DIR, "voice_config.json")
SCRIPT_PATH = os.path.join(ROOT_DIR, "annotated_script.json")
AUDIOBOOK_PATH = os.path.join(ROOT_DIR, "cloned_audiobook.mp3")
M4B_PATH = os.path.join(ROOT_DIR, "audiobook.m4b")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
CHUNKS_PATH = os.path.join(ROOT_DIR, "chunks.json")
DESIGNED_VOICES_DIR = os.path.join(ROOT_DIR, "designed_voices")
CLONE_VOICES_DIR = os.path.join(ROOT_DIR, "clone_voices")
LORA_MODELS_DIR = os.path.join(ROOT_DIR, "lora_models")
LORA_DATASETS_DIR = os.path.join(ROOT_DIR, "lora_datasets")
BUILTIN_LORA_DIR = os.path.join(ROOT_DIR, "builtin_lora")
BUILTIN_LORA_MANIFEST = os.path.join(BUILTIN_LORA_DIR, "manifest.json")
DATASET_BUILDER_DIR = os.path.join(ROOT_DIR, "dataset_builder")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(SCRIPTS_DIR, exist_ok=True)
os.makedirs(DESIGNED_VOICES_DIR, exist_ok=True)
os.makedirs(CLONE_VOICES_DIR, exist_ok=True)
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

# Clone voices directory for user-uploaded reference audio
app.mount("/clone_voices", StaticFiles(directory=CLONE_VOICES_DIR), name="clone_voices")

# LoRA models directory for trained adapter test audio
app.mount("/lora_models", StaticFiles(directory=LORA_MODELS_DIR), name="lora_models")

# Built-in LoRA adapters directory
os.makedirs(BUILTIN_LORA_DIR, exist_ok=True)
app.mount("/builtin_lora", StaticFiles(directory=BUILTIN_LORA_DIR), name="builtin_lora")

# Dataset builder directory for preview audio
app.mount("/dataset_builder", StaticFiles(directory=DATASET_BUILDER_DIR), name="dataset_builder")

# Initialize Project Manager
project_manager = ProjectManager(ROOT_DIR)

# Reset any chunks stuck in "generating" from a prior interrupted session
_startup_chunks = project_manager.load_chunks()
if _startup_chunks:
    _reset_count = 0
    for chunk in _startup_chunks:
        if chunk.get("status") == "generating":
            chunk["status"] = "pending"
            _reset_count += 1
    if _reset_count:
        project_manager.save_chunks(_startup_chunks)
        print(f"Startup: reset {_reset_count} stuck 'generating' chunk(s) to 'pending'")
    del _startup_chunks, _reset_count

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/system/stats")
async def get_system_stats():
    """Return GPU and Disk statistics."""
    gpu = get_gpu_stats()
    # Check root dir for disk space
    has_space, free_gb = check_disk_space(ROOT_DIR, 1.0) # 1GB threshold for generic warning
    
    return {
        "gpu": gpu,
        "disk": {
            "free_gb": round(free_gb, 2),
            "low_space": not has_space
        }
    }

# Data Models
class LLMConfig(BaseModel):
    base_url: str
    api_key: str
    model_name: str

class TTSConfig(BaseModel):
    mode: str = "local"  # "local" or "external"
    url: str = "http://127.0.0.1:7860"  # external mode only
    device: str = "auto"  # local mode: "auto", "cuda:0", "cpu", etc.
    language: str = "English"  # TTS language
    parallel_workers: int = 2  # concurrent TTS workers
    batch_seed: Optional[int] = None  # Single seed for batch mode, None/-1 = random
    compile_codec: bool = False  # torch.compile the codec for ~3-4x batch throughput (slow first run)
    sub_batch_enabled: bool = True  # split batch by text length to reduce padding waste
    sub_batch_min_size: int = 4  # minimum chunks per sub-batch before allowing a split
    sub_batch_ratio: float = 5.0  # max longest/shortest length ratio before splitting
    sub_batch_max_items: int = 0  # hard cap on sequences per sub-batch (0 = auto from VRAM estimate)
    batch_group_by_type: bool = False  # group chunks by voice type for efficient batching
    pause_between_speakers_ms: int = 500  # silence (ms) between different speakers during merge
    pause_same_speaker_ms: int = 250  # silence (ms) when same speaker continues during merge

class GenerationConfig(BaseModel):
    chunk_size: int = 3000
    max_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.8
    top_k: int = 0
    min_p: float = 0
    presence_penalty: float = 0.0
    banned_tokens: List[str] = []
    merge_narrators: bool = False

class PromptConfig(BaseModel):
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    review_system_prompt: Optional[str] = None
    review_user_prompt: Optional[str] = None

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

class ChunkUpdate(BaseModel):
    text: Optional[str] = None
    instruct: Optional[str] = None
    speaker: Optional[str] = None
    pause_after: Optional[int] = None

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

class PreparerConfig(BaseModel):
    audio_filename: str
    source_filename: Optional[str] = None
    output_filename: str = "alexandria_dataset.zip"
    model: Optional[str] = None
    fallback_model: Optional[str] = None
    source_threshold: float = 0.65
    keep_unaligned: bool = False
    chunk_size: float = 10.0
    lang: str = "en"
    resume: bool = False
    skip_annotation: bool = False
    source_start: Optional[int] = None
    source_start_text: Optional[str] = None
    no_auto_anchor: bool = False
    # Optimization: LLM annotation batch size (3 = ~25% faster)
    batch_size: int = 1
    # LLM enrichment
    enrich_with_llm: bool = False
    llm_model_path: Optional[str] = None
    enrich_speaker_attribution: bool = False
    enrich_narration_style: bool = False
    enrich_emotional_tone: bool = False
    # Quality filtering
    min_chunk_duration: float = 2.0
    min_confidence: float = 0.85
    min_snr: int = 25

class LoraTrainingRequest(BaseModel):
    name: str
    dataset_id: str
    epochs: int = 5
    lr: float = 5e-6
    batch_size: int = 1
    lora_r: int = 32
    lora_alpha: int = 128
    gradient_accumulation_steps: int = 8
    language: str = "english"

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

class DatasetBuilderCreateRequest(BaseModel):
    name: str

class DatasetBuilderUpdateMetaRequest(BaseModel):
    name: str
    description: str = ""
    global_seed: str = ""

class DatasetBuilderUpdateRowsRequest(BaseModel):
    name: str
    rows: List[dict]  # [{emotion, text, seed}]

class BatchPreparerTask(BaseModel):
    audio_filename: str
    source_filename: Optional[str] = None
    output_filename: str

class BatchPreparerRequest(BaseModel):
    tasks: List[BatchPreparerTask]
    lang: str = "en"
    min_confidence: float = 0.85
    min_snr: int = 25
    keep_unaligned: bool = False

# Global state for process tracking
process_state = {
    "script": {"running": False, "logs": [], "error_snapshot": [], "progress_pct": 0, "eta": ""},
    "voices": {"running": False, "logs": [], "error_snapshot": [], "progress_pct": 0, "eta": ""},
    "audio": {"running": False, "logs": [], "cancel": False, "error_snapshot": [], "progress_pct": 0, "eta": ""},
    "audacity_export": {"running": False, "logs": [], "error_snapshot": [], "progress_pct": 0, "eta": ""},
    "m4b_export": {"running": False, "logs": [], "error_snapshot": [], "progress_pct": 0, "eta": ""},
    "review": {"running": False, "logs": [], "error_snapshot": [], "progress_pct": 0, "eta": ""},
    "lora_training": {"running": False, "logs": [], "error_snapshot": [], "progress_pct": 0, "eta": ""},
    "dataset_gen": {"running": False, "logs": [], "error_snapshot": [], "progress_pct": 0, "eta": ""},
    "dataset_builder": {"running": False, "logs": [], "cancel": False, "error_snapshot": [], "progress_pct": 0, "eta": ""},
    "preparer": {
        "running": False,
        "logs": [],
        "status": "idle",       # "idle" | "running" | "done" | "failed" | "cancelled"
        "return_code": None,
        "pid": None,
        "output_file": None,
        "cancel": False,
        "error_snapshot": [],
        "progress_pct": 0,
        "eta": "",
    },
    "batch_preparer": {
        "running": False,
        "logs": [],
        "status": "idle",
        "tasks": [],            # List of task statuses
        "current_task_idx": -1,
        "cancel": False,
        "error_snapshot": [],
        "progress_pct": 0,
        "eta": "",
    }
}

# Clone voices directory for user-uploaded reference audio
CLONE_VOICES_MANIFEST = os.path.join(CLONE_VOICES_DIR, "manifest.json")
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".txt", ".epub"}

def run_process(command: List[str], task_name: str, cwd: str = None, timeout: Optional[int] = 3600 * 24):
    """Run a subprocess and capture logs, progress, and error context."""
    state = process_state[task_name]
    state["running"] = True
    state["logs"] = []
    state["error_snapshot"] = []
    state["progress_pct"] = 0
    state["eta"] = ""
    
    if "status"      in state: state["status"]      = "running"
    if "return_code" in state: state["return_code"] = None
    if "pid"         in state: state["pid"]         = None
    if "cancel"      in state: state["cancel"]      = False

    logger.info(f"Starting task {task_name}: {' '.join(command)}")

    return_code = None
    start_time = time.monotonic()
    
    try:
        env = os.environ.copy()
        # Unbuffered output is preferred for real-time capture
        if "-u" not in command and "python" in command[0]:
            command.insert(1, "-u")
            
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd or BASE_DIR,
            env=env,
        )

        if "pid" in state:
            state["pid"] = process.pid

        timeout_at = start_time + timeout if timeout else None
        buf = ""

        while True:
            # Check timeout on every iteration (works even with no output)
            if timeout_at and time.monotonic() > timeout_at:
                logger.error(f"Task {task_name} timed out after {timeout} seconds")
                process.kill()
                state["logs"].append(f"ERROR: Task timed out after {timeout} seconds")
                break

            reads, _, _ = select.select([process.stdout], [], [], 1.0)
            if not reads:
                # Check if process exited while we were waiting
                if process.poll() is not None:
                    break
                continue

            # Read available bytes. Binary read() with select won't block indefinitely.
            raw_data = process.stdout.read(4096)
            if not raw_data:
                break  # EOF

            buf += raw_data.decode('utf-8', errors='replace')
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                log_line = line.strip()
                if log_line:
                    state["logs"].append(log_line)

                    # Maintain error snapshot (last 20 lines)
                    state["error_snapshot"].append(log_line)
                    if len(state["error_snapshot"]) > 20:
                        state["error_snapshot"].pop(0)

                    # History limit
                    if len(state["logs"]) > 2000:
                        state["logs"].pop(0)

                    # Progress parsing
                    # LoRA: [EPOCH] 1/5 avg_loss=4.5
                    epoch_match = re.search(r'\[EPOCH\]\s*(\d+)/(\d+)', log_line)
                    if epoch_match:
                        curr, total = map(int, epoch_match.groups())
                        state["progress_pct"] = int((curr / total) * 100)

                    # Preparer: [1/3] Loading...
                    prep_match = re.search(r'\[(\d+)/(\d+)\]', log_line)
                    if prep_match:
                        curr, total = map(int, prep_match.groups())
                        state["progress_pct"] = int((curr / total) * 100)

                    # Generic Step: Step 10/100
                    step_match = re.search(r'Step\s*(\d+)/(\d+)', log_line, re.IGNORECASE)
                    if step_match:
                        curr, total = map(int, step_match.groups())
                        state["progress_pct"] = int((curr / total) * 100)

        process.wait()
        return_code = process.returncode

        if return_code == 0:
            state["logs"].append(f"Task {task_name} completed successfully.")
            state["progress_pct"] = 100
            if "status" in state: state["status"] = "done"
        elif return_code < 0:
            state["logs"].append(f"Task {task_name} was cancelled or timed out (signal {-return_code}).")
            if "status" in state: state["status"] = "cancelled"
        else:
            state["logs"].append(f"Task {task_name} failed with return code {return_code}.")
            if "status" in state: state["status"] = "failed"

    except Exception as e:
        logger.error(f"Error running {task_name}: {e}")
        state["logs"].append(f"Error: {str(e)}")
        if "status" in state: state["status"] = "failed"
    finally:
        state["running"] = False
        if "return_code" in state: state["return_code"] = return_code
        if "pid"         in state: state["pid"]         = None


PREPARER_SCRIPT_PATH = os.path.join(ROOT_DIR, "alexandria_preparer_rocm_compatible.py")
DEFAULT_PREP_MODEL = os.path.join(ROOT_DIR, "Qwen2.5-14B-Instruct-Q6_K.gguf")

def _run_preparer_task(config: PreparerConfig, audio_file_path: str, source_file_path: Optional[str] = None):
    """Internal function to run the preparer script in a subprocess."""
    preparer_cmd = [sys.executable, PREPARER_SCRIPT_PATH]
    preparer_cmd.extend(["--audio", audio_file_path])
    preparer_cmd.extend(["--output", os.path.join(ROOT_DIR, config.output_filename)])

    if config.source_filename and source_file_path:
        preparer_cmd.extend(["--source", source_file_path])
        preparer_cmd.extend(["--source-threshold", str(config.source_threshold)])
        if config.keep_unaligned:
            preparer_cmd.append("--keep-unaligned")
        if config.source_start is not None:
            preparer_cmd.extend(["--source-start", str(config.source_start)])
        if config.source_start_text:
            preparer_cmd.extend(["--source-start-text", config.source_start_text])
        if config.no_auto_anchor:
            preparer_cmd.append("--no-auto-anchor")

    # Use specified model or project default
    model_path = config.model or DEFAULT_PREP_MODEL
    preparer_cmd.extend(["--model", model_path])
    
    if config.fallback_model:
        preparer_cmd.extend(["--fallback-model", config.fallback_model])

    preparer_cmd.extend(["--chunk-size", str(config.chunk_size)])
    preparer_cmd.extend(["--lang", config.lang])
    if config.resume:
        preparer_cmd.append("--resume")
    if config.skip_annotation:
        preparer_cmd.append("--skip-annotation")

    # Batch annotation optimization (default: 1 = per-chunk, 3 = ~25% faster)
    if config.batch_size > 1:
        preparer_cmd.extend(["--batch-size", str(config.batch_size)])

    # LLM enrichment
    if config.enrich_with_llm and config.llm_model_path:
        preparer_cmd.append("--enrich-with-llm")
        preparer_cmd.extend(["--llm-model-path", config.llm_model_path])
        if config.enrich_speaker_attribution:
            preparer_cmd.append("--enrich-speaker-attribution")
        if config.enrich_narration_style:
            preparer_cmd.append("--enrich-narration-style")
        if config.enrich_emotional_tone:
            preparer_cmd.append("--enrich-emotional-tone")

    # Quality filtering
    if config.min_chunk_duration != 2.0:
        preparer_cmd.extend(["--min-chunk-duration", str(config.min_chunk_duration)])
    if config.min_confidence != 0.85:
        preparer_cmd.extend(["--min-confidence", str(config.min_confidence)])
    if config.min_snr != 25:
        preparer_cmd.extend(["--min-snr", str(config.min_snr)])

    output_path = os.path.join(ROOT_DIR, config.output_filename)
    process_state["preparer"]["output_file"] = None

    # Run from ROOT_DIR so dataset_temp/ and scratch WAVs land in the project root
    run_process(preparer_cmd, "preparer", cwd=ROOT_DIR)

    if process_state["preparer"]["status"] == "done" and os.path.exists(output_path):
        process_state["preparer"]["output_file"] = config.output_filename


@app.post("/api/preparer/batch/start")
async def start_batch_preparer(request: BatchPreparerRequest, background_tasks: BackgroundTasks):
    """Start sequential processing of multiple audiobooks."""
    if process_state["batch_preparer"]["running"]:
        raise HTTPException(status_code=400, detail="Batch preparer already running")

    # Disk Space Check (Require 5GB for a batch run to be safe)
    has_space, free_gb = check_disk_space(ROOT_DIR, 5.0)
    if not has_space:
         raise HTTPException(status_code=400, detail=f"Insufficient disk space ({free_gb:.1f}GB available, 5.0GB recommended for batch)")

    def batch_task():
        state = process_state["batch_preparer"]
        state["running"] = True
        state["status"] = "running"
        state["logs"] = [f"Starting batch of {len(request.tasks)} tasks..."]
        state["tasks"] = [{"audio": t.audio_filename, "status": "pending"} for t in request.tasks]
        state["cancel"] = False
        
        for i, task in enumerate(request.tasks):
            if state["cancel"]:
                state["logs"].append("Batch processing cancelled by user.")
                state["status"] = "cancelled"
                break
                
            state["current_task_idx"] = i
            state["tasks"][i]["status"] = "running"
            
            # Base progress is (completed_tasks / total_tasks)
            base_progress = (i / len(request.tasks)) * 100
            state["progress_pct"] = int(base_progress)
            
            audio_path = os.path.join(UPLOADS_DIR, task.audio_filename)
            source_path = os.path.join(UPLOADS_DIR, task.source_filename) if task.source_filename else None
            
            if not os.path.exists(audio_path):
                state["logs"].append(f"Skipping task {i+1}: Audio file not found {task.audio_filename}")
                state["tasks"][i]["status"] = "failed"
                continue

            state["logs"].append(f"--- Processing [{i+1}/{len(request.tasks)}] {task.audio_filename} ---")
            
            # Setup command
            preparer_cmd = [sys.executable, "-u", PREPARER_SCRIPT_PATH]
            preparer_cmd.extend(["--audio", audio_path])
            preparer_cmd.extend(["--output", os.path.join(ROOT_DIR, task.output_filename)])
            preparer_cmd.extend(["--lang", request.lang])
            preparer_cmd.extend(["--min-confidence", str(request.min_confidence)])
            preparer_cmd.extend(["--min-snr", str(request.min_snr)])
            preparer_cmd.extend(["--model", DEFAULT_PREP_MODEL])
            
            if request.keep_unaligned:
                preparer_cmd.append("--keep-unaligned")
            
            if source_path and os.path.exists(source_path):
                preparer_cmd.extend(["--source", source_path])

            try:
                process = subprocess.Popen(
                    preparer_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT_DIR,
                    env=os.environ.copy()
                )
                
                buf = ""
                while True:
                    if state["cancel"]:
                        process.kill()
                        break
                        
                    reads, _, _ = select.select([process.stdout], [], [], 1.0)
                    if not reads:
                        if process.poll() is not None:
                            break
                        continue
                        
                    raw_data = process.stdout.read(4096)
                    if not raw_data:
                        break # EOF
                        
                    buf += raw_data.decode('utf-8', errors='replace')
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        log_line = line.strip()
                        if log_line:
                            state["logs"].append(f"[{i+1}] {log_line}")
                            if len(state["logs"]) > 5000: state["logs"].pop(0)
                            
                            # Try to parse inner progress [X/Y] and update global progress
                            inner_match = re.search(r'\[(\d+)/(\d+)\]', log_line)
                            if inner_match:
                                curr_step, total_steps = map(int, inner_match.groups())
                                inner_pct = (curr_step / total_steps) * (100 / len(request.tasks))
                                state["progress_pct"] = int(base_progress + inner_pct)
                
                process.wait()
                if process.returncode == 0:
                    state["tasks"][i]["status"] = "done"
                    state["logs"].append(f"Successfully processed {task.audio_filename}")
                else:
                    state["tasks"][i]["status"] = "failed"
                    state["logs"].append(f"Failed task {i+1} with code {process.returncode}")
                    
            except Exception as e:
                state["logs"].append(f"Error in task {i+1}: {e}")
                state["tasks"][i]["status"] = "failed"

        state["running"] = False
        if state["status"] == "cancelled":
            pass  # keep progress as-is; don't overwrite
        else:
            state["progress_pct"] = 100
            state["status"] = "done"
        state["logs"].append("Batch processing finished.")

    background_tasks.add_task(batch_task)
    return {"status": "batch_started", "task_count": len(request.tasks)}

@app.post("/api/preparer/batch/cancel")
async def cancel_batch_preparer():
    process_state["batch_preparer"]["cancel"] = True
    return {"status": "cancel_requested"}

@app.post("/api/preparer/start")
async def start_preparer(
    background_tasks: BackgroundTasks,
    config_json: str = Form(...),
    audio_file: UploadFile = File(...),
    source_file: Optional[UploadFile] = File(None)
):
    """
    Start the Alexandria Preparer process to generate a TTS dataset from an audiobook.
    """
    try:
        config = PreparerConfig(**json.loads(config_json))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid config JSON: {e}")

    if process_state["preparer"]["running"]:
        raise HTTPException(status_code=400, detail="Preparer process is already running.")

    # Disk Space Check (Require at least 2GB for the preparer process)
    has_space, free_gb = check_disk_space(ROOT_DIR, 2.0)
    if not has_space:
        raise HTTPException(status_code=400, detail=f"Insufficient disk space ({free_gb:.1f}GB available, 2.0GB required for preparer)")

    # Save uploaded audio file
    audio_upload_path = os.path.join(UPLOADS_DIR, config.audio_filename)
    async with aiofiles.open(audio_upload_path, "wb") as f:
        while contents := await audio_file.read(1024 * 1024):
            await f.write(contents)

    source_upload_path = None
    if source_file:
        # Use original filename for source if config.source_filename is not provided
        actual_source_filename = config.source_filename or source_file.filename
        source_upload_path = os.path.join(UPLOADS_DIR, actual_source_filename)
        async with aiofiles.open(source_upload_path, "wb") as f:
            while contents := await source_file.read(1024 * 1024):
                await f.write(contents)

    background_tasks.add_task(_run_preparer_task, config, audio_upload_path, source_upload_path)
    return {"status": "Preparer started", "config": config.dict()}


@app.get("/api/preparer/status")
async def get_preparer_status(log_offset: int = Query(0)):
    """Get the current status and logs of the Alexandria Preparer process.

    Pass log_offset to receive only new lines since the last poll, avoiding
    resending the full buffer on every request.
    """
    state = process_state["preparer"]
    all_logs = state["logs"]
    return {
        "running":     state["running"],
        "status":      state["status"],
        "return_code": state["return_code"],
        "pid":         state["pid"],
        "output_file": state["output_file"],
        "log_total":   len(all_logs),
        "logs":        all_logs[log_offset:],
    }


@app.post("/api/preparer/cancel")
async def cancel_preparer():
    """Send SIGTERM to the running preparer subprocess."""
    state = process_state["preparer"]
    if not state["running"]:
        raise HTTPException(status_code=400, detail="No preparer is currently running.")
    pid = state.get("pid")
    if not pid:
        raise HTTPException(status_code=400, detail="Preparer PID not available yet.")
    try:
        os.kill(pid, signal.SIGTERM)
        state["cancel"] = True
    except ProcessLookupError:
        raise HTTPException(status_code=400, detail="Preparer process already exited.")
    return {"status": "cancel signal sent", "pid": pid}


@app.get("/api/preparer/list")
async def list_preparer_outputs():
    """List completed dataset ZIP files available for download."""
    files = []
    for directory in [ROOT_DIR, os.path.join(ROOT_DIR, "test_corpus_output")]:
        if not os.path.isdir(directory):
            continue
        for fname in sorted(os.listdir(directory)):
            if not fname.endswith(".zip"):
                continue
            fpath = os.path.join(directory, fname)
            files.append({
                "filename": fname,
                "path":     os.path.relpath(fpath, ROOT_DIR),
                "size_mb":  round(os.path.getsize(fpath) / (1024 * 1024), 1),
                "modified": os.path.getmtime(fpath),
            })
    return {"files": files}


@app.get("/api/preparer/download/{filename:path}")
async def download_preparer_output(filename: str):
    """Download a generated dataset ZIP file."""
    # Resolve and guard against path traversal
    root = os.path.realpath(ROOT_DIR)
    file_path = os.path.realpath(os.path.join(ROOT_DIR, filename))
    if not file_path.startswith(root + os.sep) and file_path != root:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found or preparer not finished.")
    return FileResponse(file_path, media_type="application/zip", filename=os.path.basename(file_path))


@app.get("/")
async def read_index():
    return FileResponse(
        os.path.join(STATIC_DIR, "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

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
            "mode": "local",
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
        try:
            rev_sys, rev_usr = load_review_prompts()
            default_config["prompts"]["review_system_prompt"] = rev_sys
            default_config["prompts"]["review_user_prompt"] = rev_usr
        except RuntimeError:
            pass
        config = default_config
    else:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)

    # Ensure prompts section exists with defaults from file
    if "prompts" not in config:
        sys_prompt, usr_prompt = load_default_prompts()
        prompts = {"system_prompt": sys_prompt, "user_prompt": usr_prompt}
        try:
            rev_sys, rev_usr = load_review_prompts()
            prompts["review_system_prompt"] = rev_sys
            prompts["review_user_prompt"] = rev_usr
        except RuntimeError:
            pass
        config["prompts"] = prompts
    else:
        if not config["prompts"].get("system_prompt") or not config["prompts"].get("user_prompt"):
            sys_prompt, usr_prompt = load_default_prompts()
            if not config["prompts"].get("system_prompt"):
                config["prompts"]["system_prompt"] = sys_prompt
            if not config["prompts"].get("user_prompt"):
                config["prompts"]["user_prompt"] = usr_prompt
        if not config["prompts"].get("review_system_prompt") or not config["prompts"].get("review_user_prompt"):
            try:
                rev_sys, rev_usr = load_review_prompts()
                if not config["prompts"].get("review_system_prompt"):
                    config["prompts"]["review_system_prompt"] = rev_sys
                if not config["prompts"].get("review_user_prompt"):
                    config["prompts"]["review_user_prompt"] = rev_usr
            except RuntimeError:
                pass  # review_prompts.txt missing or malformed — leave fields empty

    # Always include current_file (null when no state or file missing)
    config["current_file"] = None
    state_path = os.path.join(ROOT_DIR, "state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as sf:
                state = json.load(sf)
            input_path = state.get("input_file_path", "")
            if input_path and os.path.exists(input_path):
                config["current_file"] = os.path.basename(input_path)
        except (json.JSONDecodeError, ValueError):
            pass

    return config

@app.get("/api/default_prompts")
async def get_default_prompts():
    system_prompt, user_prompt = load_default_prompts()
    result = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }
    try:
        review_sys, review_usr = load_review_prompts()
        result["review_system_prompt"] = review_sys
        result["review_user_prompt"] = review_usr
    except RuntimeError:
        pass
    return result

@app.post("/api/config")
async def save_config(config: AppConfig):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2, ensure_ascii=False)
    # Reset engine so it picks up new TTS settings on next use
    project_manager.engine = None
    return {"status": "saved"}

class _HTMLTextExtractor(HTMLParser):
    """Strip HTML tags from EPUB content, preserving block-level structure."""
    BLOCK_TAGS = frozenset({
        'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'li', 'blockquote', 'br', 'hr', 'tr', 'section', 'article',
    })
    SKIP_TAGS = frozenset({'style', 'script'})

    def __init__(self):
        super().__init__()
        self.parts = []
        self._pending_newline = False
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self.BLOCK_TAGS:
            self._pending_newline = True

    def handle_endtag(self, tag):
        if tag.lower() in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        if self._pending_newline and self.parts:
            self.parts.append('\n')
            self._pending_newline = False
        self.parts.append(data)

    def get_text(self):
        return ''.join(self.parts)


def extract_epub_text(epub_path: str) -> str:
    """Extract plain text from an EPUB file, ordered by spine (reading order).

    Parses the EPUB ZIP structure directly using stdlib only:
    META-INF/container.xml -> .opf manifest+spine -> XHTML content files.
    """
    with zipfile.ZipFile(epub_path, 'r') as zf:
        # 1. Find the OPF file path from container.xml
        container_xml = zf.read('META-INF/container.xml')
        container = ET.fromstring(container_xml)
        ns = {'c': 'urn:oasis:names:tc:opendocument:xmlns:container'}
        rootfile_el = container.find('.//c:rootfile', ns)
        if rootfile_el is None:
            raise ValueError("Invalid EPUB: no rootfile found in container.xml")
        opf_path = rootfile_el.get('full-path')

        # 2. Parse the OPF to get manifest (id->href) and spine (reading order)
        opf_xml = zf.read(opf_path)
        opf = ET.fromstring(opf_xml)
        # Detect OPF namespace (varies between EPUB 2 and 3)
        opf_ns = opf.tag.split('}')[0] + '}' if '}' in opf.tag else ''

        # Build manifest: id -> href (resolve relative to OPF directory)
        opf_dir = opf_path.rsplit('/', 1)[0] + '/' if '/' in opf_path else ''
        manifest = {}
        for item in opf.findall(f'.//{opf_ns}item'):
            item_id = item.get('id')
            href = item.get('href')
            media_type = item.get('media-type', '')
            if item_id and href and 'html' in media_type:
                manifest[item_id] = opf_dir + href

        # Get spine order
        spine_ids = []
        for itemref in opf.findall(f'.//{opf_ns}itemref'):
            idref = itemref.get('idref')
            if idref:
                spine_ids.append(idref)

        # 3. Extract text from each spine item in order
        chapters = []
        for item_id in spine_ids:
            href = manifest.get(item_id)
            if href is None:
                continue
            try:
                html_bytes = zf.read(href)
            except KeyError:
                continue
            html_content = html_bytes.decode('utf-8', errors='replace')
            extractor = _HTMLTextExtractor()
            extractor.feed(html_content)
            text = extractor.get_text().strip()
            if text:
                chapters.append(text)

    return '\n\n'.join(chapters)


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOADS_DIR, file.filename)
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    # Convert EPUB to plain text
    if file.filename.lower().endswith('.epub'):
        try:
            text = extract_epub_text(file_path)
        except Exception as e:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Failed to process EPUB: {e}")
        if not text.strip():
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="No readable text content found in EPUB.")
        txt_path = file_path.rsplit('.', 1)[0] + '.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        file_path = txt_path

    # Save input path to state.json to be compatible with original scripts if needed
    state_path = os.path.join(ROOT_DIR, "state.json")
    state = {}
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            try:
                state = json.load(f)
            except (json.JSONDecodeError, ValueError):
                pass

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
    state = process_state[task_name]
    response = {
        "running": state["running"],
        "logs": state["logs"],
        "progress_pct": state.get("progress_pct", 0),
        "eta": state.get("eta", ""),
        "error_snapshot": state.get("error_snapshot", []),
        "status": state.get("status", "idle")
    }
    # Include batch-specific fields for batch_preparer
    if "tasks" in state:
        response["tasks"] = state["tasks"]
        response["current_task_idx"] = state.get("current_task_idx", -1)
    return response

@app.get("/api/voices")
async def get_voices():
    # Parse voices directly from the current script (no stale cache)
    voices_list = []
    if os.path.exists(SCRIPT_PATH):
        try:
            with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
                script_data = json.load(f)
            voices_set = set()
            for entry in script_data:
                speaker = (entry.get("speaker") or entry.get("type") or "").strip()
                if speaker:
                    voices_set.add(speaker)
            voices_list = sorted(voices_set)
            # Update voices.json for compatibility with other tools
            with open(VOICES_PATH, "w", encoding="utf-8") as f:
                json.dump(voices_list, f, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            pass

    if not voices_list:
        return []

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
            except (json.JSONDecodeError, ValueError):
                pass

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

class ChunkRestoreRequest(BaseModel):
    chunk: dict
    at_index: int

@app.post("/api/chunks/restore")
async def restore_chunk(request: ChunkRestoreRequest):
    """Re-insert a previously deleted chunk at a specific index."""
    chunks = project_manager.restore_chunk(request.at_index, request.chunk)
    if chunks is None:
        raise HTTPException(status_code=400, detail="Failed to restore chunk")
    return {"status": "ok", "total": len(chunks)}

@app.post("/api/chunks/{index}")
async def update_chunk(index: int, update: ChunkUpdate):
    data = update.model_dump(exclude_unset=True)
    logger.info(f"Updating chunk {index} with data: {data}")
    chunk = project_manager.update_chunk(index, data)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    logger.info(f"Chunk {index} updated, instruct is now: '{chunk.get('instruct', '')}'")
    return chunk

@app.post("/api/chunks/{index}/insert")
async def insert_chunk(index: int):
    """Insert an empty chunk after the given index."""
    chunks = project_manager.insert_chunk(index)
    if chunks is None:
        raise HTTPException(status_code=404, detail="Invalid chunk index")
    return {"status": "ok", "total": len(chunks)}

@app.delete("/api/chunks/{index}")
async def delete_chunk(index: int):
    """Delete a chunk at the given index."""
    result = project_manager.delete_chunk(index)
    if result is None:
        raise HTTPException(status_code=400, detail="Cannot delete chunk (invalid index or last remaining chunk)")
    deleted, chunks = result
    return {"status": "ok", "deleted": deleted, "total": len(chunks)}

@app.post("/api/chunks/{index}/generate")
async def generate_chunk_endpoint(index: int, background_tasks: BackgroundTasks):
    chunks = project_manager.load_chunks()
    if not (0 <= index < len(chunks)):
        raise HTTPException(status_code=404, detail="Invalid chunk index")
    if not chunks[index].get("text", "").strip():
        raise HTTPException(status_code=400, detail="Cannot generate audio for an empty line")

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
        process_state["audio"]["status"] = "running"
        process_state["audio"]["progress_pct"] = 0
        try:
            success, msg = project_manager.merge_audio()
            if success:
                process_state["audio"]["logs"].append(f"Merge complete: {msg}")
                process_state["audio"]["status"] = "done"
                process_state["audio"]["progress_pct"] = 100
            else:
                process_state["audio"]["logs"].append(f"Merge failed: {msg}")
                process_state["audio"]["status"] = "failed"
        except Exception as e:
            process_state["audio"]["logs"].append(f"Merge error: {e}")
            process_state["audio"]["status"] = "failed"
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
        process_state["audacity_export"]["status"] = "running"
        try:
            success, msg = project_manager.export_audacity()
            if success:
                process_state["audacity_export"]["logs"].append(f"Export complete: {msg}")
                process_state["audacity_export"]["status"] = "done"
                process_state["audacity_export"]["progress_pct"] = 100
            else:
                process_state["audacity_export"]["logs"].append(f"Export failed: {msg}")
                process_state["audacity_export"]["status"] = "failed"
        except Exception as e:
            process_state["audacity_export"]["logs"].append(f"Export error: {e}")
            process_state["audacity_export"]["status"] = "failed"
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

class M4bExportRequest(BaseModel):
    per_chunk_chapters: bool = False
    title: str = ""
    author: str = ""
    narrator: str = ""
    year: str = ""
    description: str = ""

@app.post("/api/merge_m4b")
async def merge_m4b_endpoint(request: M4bExportRequest, background_tasks: BackgroundTasks):
    if process_state["m4b_export"]["running"]:
        raise HTTPException(status_code=400, detail="M4B export already running")

    # Disk Space Check (Require at least 1GB for the M4B export process)
    has_space, free_gb = check_disk_space(ROOT_DIR, 1.0)
    if not has_space:
        raise HTTPException(status_code=400, detail=f"Insufficient disk space ({free_gb:.1f}GB available, 1.0GB required for M4B export)")

    def task():
        process_state["m4b_export"]["running"] = True
        process_state["m4b_export"]["logs"] = ["Starting M4B export..."]
        process_state["m4b_export"]["status"] = "running"
        try:
            meta = {
                "title": request.title,
                "author": request.author,
                "narrator": request.narrator,
                "year": request.year,
                "description": request.description,
                "cover_path": os.path.join(ROOT_DIR, "m4b_cover.jpg") if os.path.exists(os.path.join(ROOT_DIR, "m4b_cover.jpg")) else "",
            }
            success, msg = project_manager.merge_m4b(per_chunk_chapters=request.per_chunk_chapters, metadata=meta)
            if success:
                process_state["m4b_export"]["logs"].append(f"Export complete: {msg}")
                process_state["m4b_export"]["status"] = "done"
                process_state["m4b_export"]["progress_pct"] = 100
            else:
                process_state["m4b_export"]["logs"].append(f"Export failed: {msg}")
                process_state["m4b_export"]["status"] = "failed"
        except Exception as e:
            process_state["m4b_export"]["logs"].append(f"Export error: {e}")
            process_state["m4b_export"]["status"] = "failed"
        finally:
            process_state["m4b_export"]["running"] = False

    background_tasks.add_task(task)
    return {"status": "started"}

@app.get("/api/audiobook_m4b")
async def get_audiobook_m4b():
    if not os.path.exists(M4B_PATH):
        raise HTTPException(status_code=404, detail="M4B audiobook not found. Export it first.")
    return FileResponse(M4B_PATH, filename="audiobook.m4b", media_type="audio/mp4")

@app.post("/api/m4b_cover")
async def upload_m4b_cover(file: UploadFile = File(...)):
    """Upload a cover image for M4B export."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    cover_path = os.path.join(ROOT_DIR, "m4b_cover.jpg")
    content = await file.read()
    with open(cover_path, "wb") as f:
        f.write(content)
    return {"status": "uploaded", "path": cover_path}

@app.delete("/api/m4b_cover")
async def delete_m4b_cover():
    """Remove the uploaded cover image."""
    cover_path = os.path.join(ROOT_DIR, "m4b_cover.jpg")
    if os.path.exists(cover_path):
        os.remove(cover_path)
    return {"status": "removed"}

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
        except (json.JSONDecodeError, ValueError):
            pass

    indices = request.indices
    total = len(indices)

    def progress_callback(completed, failed, total):
        """Update logs with progress."""
        process_state["audio"]["logs"].append(
            f"Progress: {completed + failed}/{total} ({completed} done, {failed} failed)"
        )

    def cancel_check():
        return process_state["audio"]["cancel"]

    def task():
        process_state["audio"]["running"] = True
        process_state["audio"]["cancel"] = False
        process_state["audio"]["status"] = "running"
        process_state["audio"]["progress_pct"] = 0
        process_state["audio"]["logs"] = [
            f"Starting parallel generation of {total} chunks with {workers} workers..."
        ]
        try:
            results = project_manager.generate_chunks_parallel(
                indices, workers, progress_callback, cancel_check=cancel_check
            )
            completed = len(results["completed"])
            failed = len(results["failed"])
            cancelled = results.get("cancelled", 0)
            msg = f"Batch generation complete: {completed} succeeded, {failed} failed"
            if cancelled:
                msg += f", {cancelled} cancelled"
            process_state["audio"]["logs"].append(msg)
            process_state["audio"]["status"] = "done" if failed == 0 else "failed"
            process_state["audio"]["progress_pct"] = 100
            if results["failed"]:
                for idx, err in results["failed"]:
                    process_state["audio"]["logs"].append(f"  Chunk {idx} failed: {err}")
        except Exception as e:
            logger.error(f"Batch generation error: {e}")
            process_state["audio"]["logs"].append(f"Batch generation error: {e}")
            process_state["audio"]["status"] = "failed"
        finally:
            process_state["audio"]["running"] = False
            process_state["audio"]["cancel"] = False

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
    batch_group_by_type = False
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                tts_cfg = cfg.get("tts", {})
                seed_val = tts_cfg.get("batch_seed")
                if seed_val is not None and seed_val != "":
                    batch_seed = int(seed_val)
                batch_size = max(1, tts_cfg.get("parallel_workers", 4))
                batch_group_by_type = tts_cfg.get("batch_group_by_type", False)
        except (json.JSONDecodeError, ValueError):
            pass

    indices = request.indices
    total = len(indices)

    def progress_callback(completed, failed, total):
        process_state["audio"]["logs"].append(
            f"Progress: {completed + failed}/{total} ({completed} done, {failed} failed)"
        )

    def cancel_check():
        return process_state["audio"]["cancel"]

    def task():
        process_state["audio"]["running"] = True
        process_state["audio"]["cancel"] = False
        process_state["audio"]["status"] = "running"
        process_state["audio"]["progress_pct"] = 0
        process_state["audio"]["logs"] = [
            f"Starting batch generation of {total} chunks (batch_size={batch_size}, seed={batch_seed})..."
        ]
        try:
            results = project_manager.generate_chunks_batch(
                indices, batch_seed, batch_size, progress_callback,
                batch_group_by_type=batch_group_by_type,
                cancel_check=cancel_check,
            )
            completed = len(results["completed"])
            failed = len(results["failed"])
            cancelled = results.get("cancelled", 0)
            msg = f"Batch generation complete: {completed} succeeded, {failed} failed"
            if cancelled:
                msg += f", {cancelled} cancelled"
            process_state["audio"]["logs"].append(msg)
            process_state["audio"]["status"] = "done" if failed == 0 else "failed"
            process_state["audio"]["progress_pct"] = 100
            if results["failed"]:
                for idx, err in results["failed"]:
                    process_state["audio"]["logs"].append(f"  Chunk {idx} failed: {err}")
        except Exception as e:
            logger.error(f"Batch generation error: {e}")
            process_state["audio"]["logs"].append(f"Batch generation error: {e}")
            process_state["audio"]["status"] = "failed"
        finally:
            process_state["audio"]["running"] = False
            process_state["audio"]["cancel"] = False

    background_tasks.add_task(task)
    return {"status": "started", "batch_seed": batch_seed, "batch_size": batch_size, "total_chunks": total}

@app.post("/api/cancel_audio")
async def cancel_audio():
    """Cancel ongoing audio generation and reset in-progress chunks."""
    if process_state["audio"]["running"]:
        process_state["audio"]["cancel"] = True
        process_state["audio"]["logs"].append("[CANCEL] Cancellation requested")
        return {"status": "cancelling"}
    # Not running — still reset any stuck "generating" chunks (e.g. from a crash)
    chunks = project_manager.load_chunks()
    if chunks:
        reset_count = 0
        for chunk in chunks:
            if chunk.get("status") == "generating":
                chunk["status"] = "pending"
                reset_count += 1
        if reset_count:
            project_manager.save_chunks(chunks)
    return {"status": "not_running", "reset_chunks": reset_count if chunks else 0}

## ── Saved Scripts ──────────────────────────────────────────────

def _sanitize_name(name: str) -> str:
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

    safe_name = _sanitize_name(request.name)
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

def _load_manifest(path):
    """Load a JSON manifest file, returning [] on missing or corrupt file."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass
    return []

def _save_manifest(path, manifest):
    """Write a JSON manifest file."""
    with open(path, "w", encoding="utf-8") as f:
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

    safe_name = _sanitize_name(request.name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid voice name")

    # Generate unique ID
    voice_id = f"{safe_name}_{int(time.time())}"
    dest_filename = f"{voice_id}.wav"
    dest_path = os.path.join(DESIGNED_VOICES_DIR, dest_filename)

    shutil.copy2(preview_path, dest_path)

    # Update manifest
    manifest = _load_manifest(DESIGNED_VOICES_MANIFEST)
    manifest.append({
        "id": voice_id,
        "name": request.name,
        "description": request.description,
        "sample_text": request.sample_text,
        "filename": dest_filename,
    })
    _save_manifest(DESIGNED_VOICES_MANIFEST, manifest)

    logger.info(f"Designed voice saved: '{request.name}' as {dest_filename}")
    return {"status": "saved", "voice_id": voice_id}

@app.get("/api/voice_design/list")
async def voice_design_list():
    """List all saved designed voices."""
    return _load_manifest(DESIGNED_VOICES_MANIFEST)

@app.delete("/api/voice_design/{voice_id}")
async def voice_design_delete(voice_id: str):
    """Delete a saved designed voice."""
    manifest = _load_manifest(DESIGNED_VOICES_MANIFEST)
    entry = next((v for v in manifest if v["id"] == voice_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Voice not found")

    # Delete WAV file
    wav_path = os.path.join(DESIGNED_VOICES_DIR, entry["filename"])
    if os.path.exists(wav_path):
        os.remove(wav_path)

    # Remove from manifest
    manifest = [v for v in manifest if v["id"] != voice_id]
    _save_manifest(DESIGNED_VOICES_MANIFEST, manifest)

    logger.info(f"Designed voice deleted: {voice_id}")
    return {"status": "deleted", "voice_id": voice_id}

## ── Clone Voice Uploads ───────────────────────────────────────

CLONE_VOICES_MANIFEST = os.path.join(CLONE_VOICES_DIR, "manifest.json")
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}

@app.get("/api/clone_voices/list")
async def clone_voices_list():
    """List all uploaded clone voices."""
    return _load_manifest(CLONE_VOICES_MANIFEST)

@app.post("/api/clone_voices/upload")
async def clone_voices_upload(file: UploadFile = File(...)):
    """Upload an audio file for voice cloning."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_AUDIO_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported format. Use: {', '.join(ALLOWED_AUDIO_EXTS)}")

    base_name = os.path.splitext(file.filename)[0]
    safe_name = _sanitize_name(base_name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    voice_id = f"{safe_name}_{int(time.time())}"
    dest_filename = f"{voice_id}{ext}"
    dest_path = os.path.join(CLONE_VOICES_DIR, dest_filename)

    async with aiofiles.open(dest_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    manifest = _load_manifest(CLONE_VOICES_MANIFEST)
    manifest.append({
        "id": voice_id,
        "name": base_name,
        "filename": dest_filename,
    })
    _save_manifest(CLONE_VOICES_MANIFEST, manifest)

    logger.info(f"Clone voice uploaded: '{base_name}' as {dest_filename}")
    return {"status": "uploaded", "voice_id": voice_id, "filename": dest_filename}

@app.delete("/api/clone_voices/{voice_id}")
async def clone_voices_delete(voice_id: str):
    """Delete an uploaded clone voice."""
    manifest = _load_manifest(CLONE_VOICES_MANIFEST)
    entry = next((v for v in manifest if v["id"] == voice_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Clone voice not found")

    wav_path = os.path.join(CLONE_VOICES_DIR, entry["filename"])
    if os.path.exists(wav_path):
        os.remove(wav_path)

    manifest = [v for v in manifest if v["id"] != voice_id]
    _save_manifest(CLONE_VOICES_MANIFEST, manifest)

    logger.info(f"Clone voice deleted: {voice_id}")
    return {"status": "deleted", "voice_id": voice_id}

## ── LoRA Training ──────────────────────────────────────────────

LORA_MODELS_MANIFEST = os.path.join(LORA_MODELS_DIR, "manifest.json")

def _load_builtin_lora_manifest():
    """Load built-in LoRA manifest from HF (with local fallback). Returns ALL entries with download status."""
    entries = fetch_builtin_manifest(BUILTIN_LORA_DIR)
    result = []
    for entry in entries:
        entry = dict(entry)  # avoid mutating cached list
        local_id = entry["id"] if entry["id"].startswith("builtin_") else f"builtin_{entry['id']}"
        downloaded = is_adapter_downloaded(local_id, BUILTIN_LORA_DIR)
        entry["id"] = local_id
        entry["builtin"] = True
        entry["downloaded"] = downloaded
        entry["adapter_path"] = f"builtin_lora/{local_id}" if downloaded else None
        result.append(entry)
    return result

@app.post("/api/lora/upload_dataset")
async def lora_upload_dataset(file: UploadFile = File(...)):
    """Upload a ZIP containing WAV files and metadata.jsonl."""
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a .zip archive")

    # Disk Space Check (Require at least 0.5GB for the upload and extraction)
    has_space, free_gb = check_disk_space(ROOT_DIR, 0.5)
    if not has_space:
        raise HTTPException(status_code=400, detail=f"Insufficient disk space ({free_gb:.1f}GB available, 0.5GB required for upload)")

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

    safe_name = _sanitize_name(request.name)
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
    safe_name = _sanitize_name(request.name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid adapter name")

    adapter_id = f"{safe_name}_{int(time.time())}"
    output_dir = os.path.join(LORA_MODELS_DIR, adapter_id)

    # Unload TTS engine to free GPU
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
                    "created": time.time(),
                })
                _save_manifest(LORA_MODELS_MANIFEST, manifest)
                logger.info(f"LoRA adapter registered: {adapter_id}")
            except Exception as e:
                logger.error(f"Failed to update LoRA manifest: {e}")

    background_tasks.add_task(on_training_complete)
    return {"status": "started", "adapter_id": adapter_id}

@app.get("/api/lora/models")
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
    return models

@app.delete("/api/lora/models/{adapter_id}")
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

@app.post("/api/lora/download/{adapter_id}")
async def lora_download_builtin(adapter_id: str):
    """Download a built-in LoRA adapter from HuggingFace."""
    manifest = fetch_builtin_manifest(BUILTIN_LORA_DIR)
    hf_name = adapter_id.replace("builtin_", "", 1)
    entry = next((e for e in manifest if e["id"] == hf_name or e["id"] == adapter_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Unknown built-in adapter: {adapter_id}")

    if is_adapter_downloaded(adapter_id, BUILTIN_LORA_DIR):
        return {"status": "already_downloaded", "adapter_id": adapter_id}

    try:
        download_builtin_adapter(adapter_id, BUILTIN_LORA_DIR)
        logger.info(f"Built-in adapter downloaded: {adapter_id}")
        return {"status": "downloaded", "adapter_id": adapter_id}
    except Exception as e:
        logger.error(f"Download failed for {adapter_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/lora/test")
async def lora_test_model(request: LoraTestRequest):
    """Generate test audio using a LoRA adapter (built-in or user-trained)."""
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

    if not os.path.isdir(adapter_dir) and is_builtin:
        try:
            download_builtin_adapter(request.adapter_id, BUILTIN_LORA_DIR)
            adapter_dir = os.path.join(BUILTIN_LORA_DIR, request.adapter_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Auto-download failed: {e}")
    elif not os.path.isdir(adapter_dir):
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
            "audio_url": f"{audio_url_prefix}/{output_filename}",
        }
    except Exception as e:
        logger.error(f"LoRA test generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

LORA_PREVIEW_TEXT = "The ancient library stood at the crossroads of two forgotten paths, its weathered stone walls covered in ivy that had been growing for centuries."

@app.post("/api/lora/preview/{adapter_id}")
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

    if not os.path.isdir(adapter_dir) and is_builtin:
        try:
            download_builtin_adapter(adapter_id, BUILTIN_LORA_DIR)
            adapter_dir = os.path.join(BUILTIN_LORA_DIR, adapter_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Auto-download failed: {e}")
    elif not os.path.isdir(adapter_dir):
        raise HTTPException(status_code=404, detail="Adapter files not found")

    preview_path = os.path.join(adapter_dir, "preview_sample.wav")

    # Return cached if exists
    if os.path.exists(preview_path):
        return {"status": "cached", "audio_url": f"{url_prefix}/preview_sample.wav"}

    # Generate preview
    engine = project_manager.get_engine()
    if not engine:
        raise HTTPException(status_code=500, detail="Failed to initialize TTS engine")

    try:
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
    except Exception as e:
        logger.error(f"LoRA preview generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

## ── Dataset Builder ──────────────────────────────────────────

def _load_builder_state(name):
    """Load project state from dataset builder working directory."""
    state_path = os.path.join(DATASET_BUILDER_DIR, name, "state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            # Ensure new fields exist for backward compat
            state.setdefault("description", "")
            state.setdefault("global_seed", "")
            state.setdefault("samples", [])
            return state
        except Exception:
            pass
    return {"description": "", "global_seed": "", "samples": []}

def _save_builder_state(name, state):
    """Save per-sample state to dataset builder working directory."""
    work_dir = os.path.join(DATASET_BUILDER_DIR, name)
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

@app.get("/api/dataset_builder/list")
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

@app.post("/api/dataset_builder/create")
async def dataset_builder_create(request: DatasetBuilderCreateRequest):
    """Create a new dataset builder project."""
    safe_name = _sanitize_name(request.name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid dataset name")
    work_dir = os.path.join(DATASET_BUILDER_DIR, safe_name)
    if os.path.exists(work_dir):
        raise HTTPException(status_code=400, detail=f"Project '{safe_name}' already exists")
    _save_builder_state(safe_name, {"description": "", "global_seed": "", "samples": []})
    return {"name": safe_name}

@app.post("/api/dataset_builder/update_meta")
async def dataset_builder_update_meta(request: DatasetBuilderUpdateMetaRequest):
    """Update project description and global seed without touching samples."""
    safe_name = _sanitize_name(request.name)
    work_dir = os.path.join(DATASET_BUILDER_DIR, safe_name)
    if not os.path.exists(work_dir):
        raise HTTPException(status_code=404, detail="Project not found")
    state = _load_builder_state(safe_name)
    state["description"] = request.description
    state["global_seed"] = request.global_seed
    _save_builder_state(safe_name, state)
    return {"status": "ok"}

@app.post("/api/dataset_builder/update_rows")
async def dataset_builder_update_rows(request: DatasetBuilderUpdateRowsRequest):
    """Update row definitions, preserving existing generation status/audio."""
    safe_name = _sanitize_name(request.name)
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

        # Update state (cache-bust URL so browser loads fresh audio on regen)
        cache_bust = int(time.time())
        audio_url = f"/dataset_builder/{request.dataset_name}/{dest_filename}?t={cache_bust}"
        state = _load_builder_state(request.dataset_name)
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
        _save_builder_state(request.dataset_name, state)

        return {
            "status": "done",
            "sample_index": request.sample_index,
            "audio_url": audio_url,
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

    safe_name = _sanitize_name(request.name)
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

                wav_path, sr = engine.generate_voice_design(
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
        "description": state.get("description", ""),
        "global_seed": state.get("global_seed", ""),
        "samples": state.get("samples", []),
        "running": process_state["dataset_builder"]["running"],
        "logs": process_state["dataset_builder"]["logs"],
    }

@app.post("/api/dataset_builder/save")
async def dataset_builder_save(request: DatasetSaveRequest):
    """Finalize dataset builder project as a training dataset."""
    safe_name = _sanitize_name(request.name)
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
    host = os.environ.get("ALEXANDRIA_HOST", "127.0.0.1")
    port = int(os.environ.get("ALEXANDRIA_PORT", "4200"))
    uvicorn.run(app, host=host, port=port, access_log=False)
