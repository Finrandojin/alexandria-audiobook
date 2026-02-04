import os
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
import subprocess

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

os.makedirs(UPLOADS_DIR, exist_ok=True)

# Mount static files with absolute path
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Create voicelines directory if it doesn't exist to prevent startup error
VOICELINES_DIR = os.path.join(ROOT_DIR, "voicelines")
os.makedirs(VOICELINES_DIR, exist_ok=True)
app.mount("/voicelines", StaticFiles(directory=VOICELINES_DIR), name="voicelines")

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
    url: str

class AppConfig(BaseModel):
    llm: LLMConfig
    tts: TTSConfig

class VoiceConfigItem(BaseModel):
    type: str = "custom"
    voice: Optional[str] = "Ryan"
    default_style: Optional[str] = ""
    seed: Optional[str] = "-1"
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None

class ProcessStatus(BaseModel):
    running: bool
    logs: List[str]

# Global state for process tracking
process_state = {
    "script": {"running": False, "logs": []},
    "voices": {"running": False, "logs": []},
    "audio": {"running": False, "logs": []}
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
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=BASE_DIR,
            bufsize=1,
            universal_newlines=True
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

@app.get("/api/config")
async def get_config():
    if not os.path.exists(CONFIG_PATH):
        # Return defaults if no config
        return {
            "llm": {
                "base_url": "http://localhost:1234/v1",
                "api_key": "local",
                "model_name": "local-model"
            },
            "tts": {
                "url": "http://127.0.0.1:7860"
            }
        }
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

@app.post("/api/config")
async def save_config(config: AppConfig):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config.dict(), f, indent=2)
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
        with open(state_path, "r") as f:
            try:
                state = json.load(f)
            except: pass

    state["input_file_path"] = file_path
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    return {"filename": file.filename, "path": file_path}

@app.post("/api/generate_script")
async def generate_script(background_tasks: BackgroundTasks):
    # Get input file from state.json
    state_path = os.path.join(ROOT_DIR, "state.json")
    if not os.path.exists(state_path):
        raise HTTPException(status_code=400, detail="No input file selected")

    with open(state_path, "r") as f:
        state = json.load(f)
        input_file = state.get("input_file_path")

    if not input_file:
         raise HTTPException(status_code=400, detail="No input file found in state")

    if process_state["script"]["running"]:
         raise HTTPException(status_code=400, detail="Script generation already running")

    background_tasks.add_task(run_process, ["python", "generate_script.py", input_file], "script")
    return {"status": "started"}

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

    with open(VOICES_PATH, "r") as f:
        voices_list = json.load(f)

    # Combine with config
    voice_config = {}
    if os.path.exists(VOICE_CONFIG_PATH):
        with open(VOICE_CONFIG_PATH, "r") as f:
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

    background_tasks.add_task(run_process, ["python", "parse_voices.py"], "voices")
    return {"status": "started"}

@app.post("/api/save_voice_config")
async def save_voice_config(config_data: Dict[str, VoiceConfigItem]):
    # Read existing to preserve any fields not sent?
    # For now, we assume frontend sends full config or we just overwrite specific keys

    current_config = {}
    if os.path.exists(VOICE_CONFIG_PATH):
        with open(VOICE_CONFIG_PATH, "r") as f:
            try:
                current_config = json.load(f)
            except: pass

    # Update current config with new data
    for voice_name, config in config_data.items():
        # Convert Pydantic model to dict
        current_config[voice_name] = config.dict()

    with open(VOICE_CONFIG_PATH, "w") as f:
        json.dump(current_config, f, indent=2)

    return {"status": "saved"}

@app.post("/api/generate_audiobook")
async def generate_audiobook_endpoint(background_tasks: BackgroundTasks):
    if process_state["audio"]["running"]:
         raise HTTPException(status_code=400, detail="Audio generation already running")

    background_tasks.add_task(run_process, ["python", "generate_audiobook.py"], "audio")
    return {"status": "started"}

@app.get("/api/audiobook")
async def get_audiobook():
    if not os.path.exists(AUDIOBOOK_PATH):
        raise HTTPException(status_code=404, detail="Audiobook not found")
    return FileResponse(AUDIOBOOK_PATH)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4200)
