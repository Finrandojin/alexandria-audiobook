<img width="475" height="467" alt="Alexandria Logo" src="https://github.com/user-attachments/assets/fa2c36d3-a5f3-49ab-9dfe-30933359dfbd" />

# Alexandria Audiobook Generator

Transform any book or novel into a fully-voiced audiobook using AI-powered script annotation and text-to-speech. Features a built-in Qwen3-TTS engine with batch processing and a browser-based editor for fine-tuning every line before final export.

## Example: [audiobook.mp3](https://github.com/user-attachments/files/25156435/audiobook.mp3)

## Screenshots

<img src="https://github.com/user-attachments/assets/874b5e30-56d2-4292-b754-4408fc53f5d6" width="30%"></img> <img src="https://github.com/user-attachments/assets/488cde02-6b93-47fa-874b-97a618ae482c" width="30%"></img> <img src="https://github.com/user-attachments/assets/4c0805a6-bb9d-42c1-a9ff-79bb29d0613c" width="30%"></img> <img src="https://github.com/user-attachments/assets/8e58a5bf-ed8f-4864-8545-1e3d9681b0cf" width="30%"></img> <img src="https://github.com/user-attachments/assets/531830da-8668-4189-a0dc-020e6661bfb6" width="30%"></img> 

## Features

### AI-Powered Pipeline
- **Local & Cloud LLM Support** - Use any OpenAI-compatible API (LM Studio, Ollama, OpenAI, etc.)
- **Automatic Script Annotation** - LLM parses text into JSON with speakers, dialogue, and TTS instruct directions
- **Smart Chunking** - Groups consecutive lines by speaker (up to 500 chars) for natural flow
- **Context Preservation** - Maintains character consistency across chunks during generation

### Voice Generation
- **Built-in TTS Engine** - Qwen3-TTS runs locally with no external server required
- **External Server Mode** - Optionally connect to a remote Qwen3-TTS Gradio server
- **Custom Voices** - 9 pre-trained voices with instruct-based emotion/tone control
- **Voice Cloning** - Clone any voice from a 5-15 second reference audio sample
- **Batch Processing** - Generate dozens of chunks simultaneously with 3-6x real-time throughput
- **Codec Compilation** - Optional `torch.compile` optimization for 3-4x faster batch decoding
- **Non-verbal Sounds** - LLM writes natural vocalizations ("Ahh!", "Mmm...", "Haha!") with context-aware instruct directions
- **Natural Pauses** - Intelligent delays between speakers (500ms) and same-speaker segments (250ms)

### Web UI Editor
- **5-Tab Interface** - Setup, Script Generation, Voice Configuration, Editor, Results
- **Chunk Editor** - Edit speaker, text, and instruct for any line
- **Selective Regeneration** - Re-render individual chunks without regenerating everything
- **Batch Processing** - Two render modes: standard parallel and fast batch
- **Live Progress** - Real-time logs and status tracking for all operations
- **Audio Preview** - Play individual chunks or sequence through the entire audiobook
- **Script Library** - Save and load annotated scripts with voice configurations

### Export Options
- **Combined Audiobook** - Single MP3 with all voices and natural pauses
- **Individual Voicelines** - Separate MP3 per line for DAW editing (Audacity, etc.)
- **Audacity Export** - One-click zip with per-speaker WAV tracks, LOF project file, and labels for automatic multi-track import into Audacity

## Requirements

- [Pinokio](https://pinokio.computer/)
- LLM server (one of the following):
  - [LM Studio](https://lmstudio.ai/) (local) - recommended: Qwen3 or similar
  - [Ollama](https://ollama.ai/) (local)
  - [OpenAI API](https://platform.openai.com/) (cloud)
  - Any OpenAI-compatible API
- GPU with 16+ GB VRAM recommended (NVIDIA or AMD)
  - CPU mode available but significantly slower

> **Note:** No external TTS server is required. Alexandria includes a built-in Qwen3-TTS engine that loads models directly. Model weights are downloaded automatically on first use (~3.5 GB).

## Installation

1. Install [Pinokio](https://pinokio.computer/) if you haven't already
2. In Pinokio, click **Download** and paste this URL:
   ```
   https://github.com/Finrandojin/alexandria-audiobook
   ```
3. Click **Install** to set up dependencies
4. Click **Start** to launch the web interface

## Quick Start

1. **Setup Tab** - Configure your LLM and TTS:
   - **LLM Base URL**: `http://localhost:1234/v1` (LM Studio) or `http://localhost:11434/v1` (Ollama)
   - **LLM API Key**: Your API key (use `local` for local servers)
   - **LLM Model Name**: The model to use (e.g., `qwen2.5-14b`)
   - **TTS Mode**: `local` (built-in, recommended) or `external` (Gradio server)

2. **Script Tab** - Upload your book (.txt or .md) and click "Generate Annotated Script"

3. **Voices Tab** - Click "Refresh Voices" then configure each speaker:
   - Choose Custom Voice (with instruct) or Clone Voice (from reference audio)
   - Set voice parameters and save

4. **Editor Tab** - Review and edit chunks:
   - Select "Batch (Fast)" mode and click "Batch Render Pending" for fastest generation
   - Edit any chunk's text/instruct/speaker and regenerate individually
   - Click "Merge All" when satisfied

5. **Result Tab** - Download your finished audiobook

## Web Interface

### Setup Tab
Configure connections to your LLM and TTS engine.

**TTS Settings:**
- **Mode** - `local` (built-in engine) or `external` (connect to Gradio server)
- **Device** - `auto` (recommended), `cuda`, `cpu`, or `mps`
- **Parallel Workers** - Batch size for fast batch rendering (higher = more VRAM usage)
- **Batch Seed** - Fixed seed for reproducible batch output (leave empty for random)
- **Compile Codec** - Enable `torch.compile` for 3-4x faster batch decoding (adds ~30-60s warmup on first generation)
- **Sub-batching** - Split batches by text length to reduce wasted GPU compute on padding (enabled by default)
- **Min Sub-batch Size** - Minimum chunks per sub-batch before allowing a split (default: 4)
- **Length Ratio** - Maximum longest/shortest text length ratio before forcing a sub-batch split (default: 5)

**Prompt Settings (Advanced):**
- **Generation Settings** - Chunk size and max tokens for LLM responses
- **LLM Sampling Parameters** - Temperature, Top P, Top K, Min P, and Presence Penalty
- **Banned Tokens** - Comma-separated list of tokens to ban from LLM output (useful for disabling thinking mode on models like GLM4, DeepSeek-R1, etc.)
- **Prompt Customization** - System and user prompts used for script generation

### Script Tab
Upload a text file and generate the annotated script. The LLM converts your book into a structured JSON format with:
- Speaker identification (NARRATOR vs character names)
- Dialogue text with natural vocalizations (written as pronounceable text, not tags)
- Style directions for TTS delivery

### Voices Tab
After script generation, parse voices to see all speakers. For each:

**Custom Voice Mode:**
- Select from 9 pre-trained voices: Aiden, Dylan, Eric, Ono_anna, Ryan, Serena, Sohee, Uncle_fu, Vivian
- Set a default style (e.g., "calm, professional narrator")
- Optionally set a seed for reproducible output

**Clone Voice Mode:**
- Upload 5-15 seconds of clear reference audio
- Provide the exact transcript of the reference
- Note: Instruct directions are ignored for cloned voices

### Editor Tab
Fine-tune your audiobook before export:
- **View all chunks** in a table with status indicators
- **Edit inline** - Click to modify speaker, text, or instruct
- **Generate single** - Regenerate just one chunk after editing
- **Batch render** - Process all pending chunks (see Render Modes below)
- **Play sequence** - Preview audio playback in order
- **Merge all** - Combine chunks into final audiobook

### Render Modes

Alexandria offers two methods for batch rendering audio:

#### Render Pending (Standard)
The default rendering mode. Sends individual TTS calls in parallel using the configured worker count.

- **Per-speaker seeds** - Each voice uses its configured seed for reproducible output
- **Voice cloning support** - Works with both custom voices and cloned voices

#### Batch (Fast)
High-speed rendering that sends multiple lines to the TTS engine in a single batched call. Chunks are sorted by text length and processed in optimized sub-batches to minimize padding waste.

- **3-6x real-time throughput** - With codec compilation enabled, batches of 20-60 chunks process at 3-6x real-time speed
- **Sub-batching** - Automatically groups similarly-sized chunks together for efficient GPU utilization
- **Single seed** - All voices share the `Batch Seed` from config (set empty for random)
- **Custom voices only** - Clone voices fall back to individual calls
- **Parallel Workers** setting controls batch size (higher values use more VRAM)

### Result Tab
Download your completed audiobook as MP3, or click **Export to Audacity** to download a zip with per-speaker WAV tracks that import as separate Audacity tracks. Unzip and open `project.lof` in Audacity to load all tracks, then import `labels.txt` via File > Import > Labels for chunk annotations.

## Performance

### Recommended Settings for Batch Generation

| Setting | Recommended | Notes |
|---------|-------------|-------|
| TTS Mode | `local` | Built-in engine, no external server |
| Compile Codec | `true` | 3-4x faster decoding after one-time warmup |
| Parallel Workers | 20-60 | Higher = more throughput, more VRAM |
| Render Mode | Batch (Fast) | Uses batched TTS calls |

### Benchmarks

Tested on AMD RX 7900 XTX (24 GB VRAM, ROCm 6.3):

| Configuration | Throughput |
|--------------|------------|
| Standard mode (sequential) | ~1x real-time |
| Batch mode, no codec compile | ~2x real-time |
| Batch mode + compile_codec | **3-6x real-time** |

A 273-chunk audiobook (~54 minutes of audio) generates in approximately 16 minutes with batch mode and codec compilation enabled.

### ROCm (AMD GPU) Notes

Alexandria automatically applies ROCm-specific optimizations when running on AMD GPUs:
- **MIOpen fast-find mode** - Prevents workspace allocation failures that cause slow GEMM fallback
- **Triton AMD flash attention** - Enables native flash attention for the whisper encoder
- **triton_key compatibility shim** - Fixes `torch.compile` on pytorch-triton-rocm

These are applied transparently and require no configuration.

## Script Format

The generated script is a JSON array with `speaker`, `text`, and `instruct` fields:

```json
[
  {"speaker": "NARRATOR", "text": "The door creaked open slowly.", "instruct": "Calm, even narration."},
  {"speaker": "ELENA", "text": "Ah! Who's there?", "instruct": "Fearful, sharp whisper."},
  {"speaker": "MARCUS", "text": "Haha... did you miss me?", "instruct": "Menacing, slow and smug."}
]
```

- **`instruct`** — Short voice direction (3-8 words) sent directly to the TTS engine. The TTS responds best to simple emotion keywords like "Angry, slow and threatening." rather than long descriptions.

### Non-verbal Sounds
Vocalizations are written as real pronounceable text that the TTS speaks directly — no bracket tags or special tokens. The LLM generates natural onomatopoeia with short instruct directions:
- Gasps: "Ah!", "Oh!" with instruct like "Fearful, sharp gasp."
- Moans: "Mmm...", "Ahh...", "Ohhh..."
- Sighs: "Haah...", "Hff..."
- Laughter: "Haha!", "Ahaha..."
- Crying: "Hic... sniff..."

## Output Files

**Final Audiobook:**
- `cloned_audiobook.mp3` - Combined audiobook with natural pauses

**Individual Voicelines (for DAW editing):**
```
voicelines/
├── voiceline_0001_narrator.mp3
├── voiceline_0002_elena.mp3
├── voiceline_0003_marcus.mp3
└── ...
```

Files are numbered in timeline order with speaker names for easy:
- Import into Audacity or other DAWs
- Placement on separate character tracks
- Fine-tuning of timing and effects

**Audacity Export (per-speaker tracks):**
```
audacity_export.zip
├── project.lof       # Open this in Audacity to import all tracks
├── labels.txt        # Import via File > Import > Labels for chunk annotations
├── narrator.wav      # Full-length track with only NARRATOR audio
├── elena.wav         # Full-length track with only ELENA audio
├── marcus.wav        # Full-length track with only MARCUS audio
└── ...
```

Each WAV track is padded to the same total duration with silence where other speakers are talking. Playing all tracks simultaneously sounds identical to the merged MP3.

## API Reference

Alexandria exposes a REST API for programmatic access:

### Configuration
```bash
# Get current config
curl http://127.0.0.1:4200/api/config

# Save config
curl -X POST http://127.0.0.1:4200/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "llm": {"base_url": "...", "api_key": "...", "model_name": "..."},
    "tts": {
      "mode": "local",
      "device": "auto",
      "parallel_workers": 25,
      "batch_seed": 12345,
      "compile_codec": true,
      "sub_batch_enabled": true,
      "sub_batch_min_size": 4,
      "sub_batch_ratio": 5
    },
    "generation": {"chunk_size": 3000, "max_tokens": 4096, "temperature": 0.6, "top_p": 0.8, "top_k": 20, "min_p": 0, "presence_penalty": 0.0, "banned_tokens": []}
  }'
```

### Script Generation
```bash
# Upload text file
curl -X POST http://127.0.0.1:4200/api/upload \
  -F "file=@mybook.txt"

# Generate script (returns task ID)
curl -X POST http://127.0.0.1:4200/api/generate_script

# Check status
curl http://127.0.0.1:4200/api/status/script_generation
```

### Voice Management
```bash
# Get voices and config
curl http://127.0.0.1:4200/api/voices

# Parse voices from script
curl -X POST http://127.0.0.1:4200/api/parse_voices

# Save voice config
curl -X POST http://127.0.0.1:4200/api/save_voice_config \
  -H "Content-Type: application/json" \
  -d '{"NARRATOR": {"type": "custom", "voice": "Ryan", "default_style": "calm"}}'
```

### Chunk Management
```bash
# Get all chunks
curl http://127.0.0.1:4200/api/chunks

# Update a chunk
curl -X POST http://127.0.0.1:4200/api/chunks/5 \
  -H "Content-Type: application/json" \
  -d '{"text": "Updated dialogue", "instruct": "Excited, bright energy."}'

# Generate audio for single chunk
curl -X POST http://127.0.0.1:4200/api/chunks/5/generate

# Standard batch render (parallel individual calls)
curl -X POST http://127.0.0.1:4200/api/generate_batch \
  -H "Content-Type: application/json" \
  -d '{"indices": [0, 1, 2, 3, 4]}'

# Fast batch render (batched TTS calls, much faster)
curl -X POST http://127.0.0.1:4200/api/generate_batch_fast \
  -H "Content-Type: application/json" \
  -d '{"indices": [0, 1, 2, 3, 4]}'

# Merge all chunks into final audiobook
curl -X POST http://127.0.0.1:4200/api/merge
```

### Saved Scripts
```bash
# List saved scripts
curl http://127.0.0.1:4200/api/scripts

# Save current script
curl -X POST http://127.0.0.1:4200/api/scripts/save \
  -H "Content-Type: application/json" \
  -d '{"name": "my-novel"}'

# Load a saved script
curl -X POST http://127.0.0.1:4200/api/scripts/load \
  -H "Content-Type: application/json" \
  -d '{"name": "my-novel"}'
```

### Audio Download
```bash
# Download audiobook (after merging in editor)
curl http://127.0.0.1:4200/api/audiobook --output audiobook.mp3

# Export to Audacity (per-speaker tracks + LOF + labels)
curl -X POST http://127.0.0.1:4200/api/export_audacity

# Poll for completion
curl http://127.0.0.1:4200/api/status/audacity_export

# Download the zip
curl http://127.0.0.1:4200/api/export_audacity --output audacity_export.zip
```

## Python Integration

```python
import requests

BASE = "http://127.0.0.1:4200"

# Upload and generate script
with open("mybook.txt", "rb") as f:
    requests.post(f"{BASE}/api/upload", files={"file": f})

requests.post(f"{BASE}/api/generate_script")

# Poll for completion
import time
while True:
    status = requests.get(f"{BASE}/api/status/script_generation").json()
    if status.get("status") in ["completed", "error"]:
        break
    time.sleep(2)

# Configure voices
voice_config = {
    "NARRATOR": {"type": "custom", "voice": "Ryan", "default_style": "calm narrator"},
    "HERO": {"type": "custom", "voice": "Aiden", "default_style": "brave, determined"}
}
requests.post(f"{BASE}/api/save_voice_config", json=voice_config)

# Fast batch render all chunks
chunks = requests.get(f"{BASE}/api/chunks").json()
indices = [c["id"] for c in chunks]
requests.post(f"{BASE}/api/generate_batch_fast", json={"indices": indices})
# ... poll until all chunks status == "done" ...
requests.post(f"{BASE}/api/merge")

# Download
with open("output.mp3", "wb") as f:
    f.write(requests.get(f"{BASE}/api/audiobook").content)

# Export to Audacity
requests.post(f"{BASE}/api/export_audacity")
# ... poll /api/status/audacity_export until not running ...
with open("audacity_export.zip", "wb") as f:
    f.write(requests.get(f"{BASE}/api/export_audacity").content)
```

## JavaScript Integration

```javascript
const BASE = "http://127.0.0.1:4200";

// Upload file
const formData = new FormData();
formData.append("file", fileInput.files[0]);
await fetch(`${BASE}/api/upload`, { method: "POST", body: formData });

// Generate script
await fetch(`${BASE}/api/generate_script`, { method: "POST" });

// Poll for completion
async function waitForTask(taskName) {
  while (true) {
    const res = await fetch(`${BASE}/api/status/${taskName}`);
    const data = await res.json();
    if (data.status === "completed" || data.status === "error") return data;
    await new Promise(r => setTimeout(r, 2000));
  }
}
await waitForTask("script_generation");

// Configure and generate
await fetch(`${BASE}/api/save_voice_config`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    NARRATOR: { type: "custom", voice: "Ryan", default_style: "calm" }
  })
});

// Fast batch render all chunks
const chunks = await (await fetch(`${BASE}/api/chunks`)).json();
const indices = chunks.map(c => c.id);
await fetch(`${BASE}/api/generate_batch_fast`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ indices })
});
// ... poll until all chunks done ...

// Merge into final audiobook
await fetch(`${BASE}/api/merge`, { method: "POST" });

// Export to Audacity
await fetch(`${BASE}/api/export_audacity`, { method: "POST" });
// ... poll /api/status/audacity_export until not running ...
// Download zip from GET /api/export_audacity
```

## Recommended LLM Models

For script generation, non-thinking models work best:
- **Qwen3-next** (80B-A3B-instruct) - Excellent JSON output and instruct directions
- **Gemma3** (27B recommended) - Strong JSON output and instruct directions
- **Qwen2.5** (any size) - Reliable JSON output
- **Qwen3** (non-thinking variant)
- **Llama 3.1/3.2** - Good character distinction
- **Mistral/Mixtral** - Fast and reliable

**Thinking models** (DeepSeek-R1, GLM4-air, etc.) can interfere with JSON output. If you must use one, add `<think>` to the **Banned Tokens** field in Setup to disable thinking mode.

## Troubleshooting

### Script generation fails
- Check LLM server is running and accessible
- Verify model name matches what's loaded
- Try a different model - some struggle with JSON output

### TTS generation fails
- Check the Pinokio terminal for model loading errors
- Ensure sufficient VRAM (16+ GB recommended for bfloat16)
- For external mode, ensure the Gradio TTS server is running at the configured URL
- Check voice_config.json has valid settings for all speakers
- For clone voices, verify reference audio exists and transcript is accurate

### Slow batch generation
- Enable **Compile Codec** in Setup (adds warmup time but 3-4x faster after)
- Increase **Parallel Workers** (batch size) if VRAM allows
- Use **Batch (Fast)** render mode instead of Standard
- If you see MIOpen warnings on AMD, these are handled automatically

### Out of memory errors
- Reduce **Parallel Workers** (batch size)
- Close other GPU-intensive applications
- Try `device: cpu` as a fallback (much slower)

### Audio quality issues
- Use 5-15 second clear reference audio for cloning
- Avoid background noise in reference samples
- Try different seeds for custom voices

### Mojibake characters in output
- The system automatically fixes common encoding issues
- If problems persist, ensure your input text is UTF-8 encoded

## Project Structure

```
Alexandria/
├── app/
│   ├── app.py                 # FastAPI server
│   ├── tts.py                 # TTS engine (local + external backends)
│   ├── generate_script.py     # LLM script annotation
│   ├── project.py             # Chunk management & batch generation
│   ├── parse_voices.py        # Voice extraction
│   ├── config.json            # Runtime configuration
│   ├── static/index.html      # Web UI
│   └── requirements.txt       # Python dependencies
├── install.js                 # Pinokio installer
├── start.js                   # Pinokio launcher
├── reset.js                   # Reset script
├── pinokio.js                 # Pinokio UI config
├── pinokio.json               # Pinokio metadata
└── README.md
```

## License

MIT
