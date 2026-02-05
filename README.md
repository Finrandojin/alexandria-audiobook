<img width="475" height="467" alt="Alexandria Logo" src="https://github.com/user-attachments/assets/fa2c36d3-a5f3-49ab-9dfe-30933359dfbd" />

# Alexandria Audiobook Generator

Transform any book or novel into a fully-voiced audiobook using AI-powered script annotation and styled text-to-speech. Features a browser-based editor for fine-tuning every line before final export.

## Features

### AI-Powered Pipeline
- **Local & Cloud LLM Support** - Use any OpenAI-compatible API (LM Studio, Ollama, OpenAI, etc.)
- **Automatic Script Annotation** - LLM parses text into JSON with speakers, dialogue, and style directions
- **Smart Chunking** - Groups consecutive lines by speaker (up to 500 chars) for natural flow
- **Context Preservation** - Maintains character consistency across chunks during generation

### Voice Generation
- **Custom Voices** - 9 pre-trained voices with full style direction support
- **Voice Cloning** - Clone any voice from a 5-15 second reference audio sample
- **Non-verbal Sounds** - Supports `[laughs]`, `[sighs]`, `[gasps]`, and 20+ vocalizations
- **Natural Pauses** - Intelligent delays between speakers (500ms) and same-speaker segments (250ms)

### Web UI Editor
- **5-Tab Interface** - Setup, Script Generation, Voice Configuration, Editor, Results
- **Chunk Editor** - Edit speaker, text, and style for any line
- **Selective Regeneration** - Re-render individual chunks without regenerating everything
- **Batch Processing** - Render all pending chunks or regenerate entire audiobook
- **Live Progress** - Real-time logs and status tracking for all operations
- **Audio Preview** - Play individual chunks or sequence through the entire audiobook

### Export Options
- **Combined Audiobook** - Single MP3 with all voices and natural pauses
- **Individual Voicelines** - Separate MP3 per line for DAW editing (Audacity, etc.)

## Requirements

- [Pinokio](https://pinokio.computer/)
- LLM server (one of the following):
  - [LM Studio](https://lmstudio.ai/) (local) - recommended: Qwen3 or similar
  - [Ollama](https://ollama.ai/) (local)
  - [OpenAI API](https://platform.openai.com/) (cloud)
  - Any OpenAI-compatible API
- [Qwen3 TTS](https://github.com/Qwen/Qwen3-TTS) server running locally (Gradio interface)

## Installation

1. Install [Pinokio](https://pinokio.computer/) if you haven't already
2. In Pinokio, click **Download** and paste this URL:
   ```
   https://github.com/Finrandojin/alexandria-audiobook
   ```
3. Click **Install** to set up dependencies
4. Click **Start** to launch the web interface

## Quick Start

1. **Setup Tab** - Configure your LLM and TTS servers:
   - **LLM Base URL**: `http://localhost:1234/v1` (LM Studio) or `http://localhost:11434/v1` (Ollama)
   - **LLM API Key**: Your API key (use `local` for local servers)
   - **LLM Model Name**: The model to use (e.g., `qwen2.5-14b`)
   - **TTS Server URL**: Default `http://127.0.0.1:7860`

2. **Script Tab** - Upload your book (.txt or .md) and click "Generate Annotated Script"

3. **Voices Tab** - Click "Refresh Voices" then configure each speaker:
   - Choose Custom Voice (with style) or Clone Voice (from reference audio)
   - Set voice parameters and save

4. **Editor Tab** - Review and edit chunks:
   - Click "Batch Render Pending" to generate all audio
   - Edit any chunk's text/style/speaker and regenerate individually
   - Click "Merge All" when satisfied

5. **Result Tab** - Download your finished audiobook

## Web Interface

### Setup Tab
Configure connections to your LLM and TTS servers. Optionally customize the system and user prompts used for script generation.

### Script Tab
Upload a text file and generate the annotated script. The LLM converts your book into a structured JSON format with:
- Speaker identification (NARRATOR vs character names)
- Dialogue text with non-verbal cues
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
- Note: Style directions are ignored for cloned voices

### Editor Tab
Fine-tune your audiobook before export:
- **View all chunks** in a table with status indicators
- **Edit inline** - Click to modify speaker, text, or style
- **Generate single** - Regenerate just one chunk after editing
- **Batch render** - Process all pending chunks (see Render Modes below)
- **Play sequence** - Preview audio playback in order
- **Merge all** - Combine chunks into final audiobook

### Render Modes

Alexandria offers two methods for batch rendering audio:

#### Render Pending (Standard)
The default rendering mode. Uses parallel workers to make individual TTS API calls.

- **Per-speaker seeds** - Each voice uses its configured seed for reproducible output
- **Voice cloning support** - Works with both custom voices and cloned voices
- **Parallel Workers** setting controls concurrency

#### Batch (Fast) ⚗️
An experimental high-speed rendering mode that sends multiple lines to the TTS server in a single request.

- **Significantly faster** - Reduces API overhead, ~5x speedup in testing
- **Single seed** - All voices share the `Batch Seed` from config (set empty for random)
- **Custom voices only** - Clone voices fall back to individual calls
- **Parallel Workers** setting controls batch size

> **Note:** The Batch (Fast) mode requires a custom build of Qwen3-TTS with the `/generate_batch` endpoint. This is not available in the standard Qwen3-TTS release. A pull request to add this feature is pending at [SUP3RMASS1VE/Qwen3-TTS](https://github.com/SUP3RMASS1VE/Qwen3-TTS). If you need this feature before it's merged, please open an issue to request access.

### Result Tab
Download your completed audiobook as MP3.

## Script Format

The generated script is a JSON array:

```json
[
  {"speaker": "NARRATOR", "text": "The door creaked open slowly.", "style": "tense, suspenseful"},
  {"speaker": "ELENA", "text": "[gasps] Who's there?", "style": "frightened, whispered"},
  {"speaker": "MARCUS", "text": "[chuckles] Did you miss me?", "style": "smug, menacing"}
]
```

### Supported Non-verbal Sounds
`[laughs]`, `[chuckles]`, `[giggles]`, `[scoffs]`, `[sighs]`, `[gasps]`, `[groans]`, `[moans]`, `[whimpers]`, `[sobs]`, `[cries]`, `[sniffs]`, `[whispers]`, `[shouts]`, `[screams]`, `[yells]`, `[clears throat]`, `[coughs]`, `[pauses]`, `[hesitates]`, `[stammers]`, `[gulps]`, `[snorts]`, `[hums]`, `[growls]`, `[purrs]`, `[shivers]`

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

## API Reference

Alexandria exposes a REST API for programmatic access:

### Configuration
```bash
# Get current config
curl http://127.0.0.1:4200/api/config

# Save config
curl -X POST http://127.0.0.1:4200/api/config \
  -H "Content-Type: application/json" \
  -d '{"llm": {"base_url": "...", "api_key": "...", "model_name": "..."}}'
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
  -d '{"text": "Updated dialogue", "style": "excited"}'

# Generate audio for single chunk
curl -X POST http://127.0.0.1:4200/api/chunks/5/generate

# Merge all chunks into final audiobook
curl -X POST http://127.0.0.1:4200/api/merge
```

### Audio Generation
```bash
# Generate full audiobook (legacy, bypasses editor)
curl -X POST http://127.0.0.1:4200/api/generate_audiobook

# Download audiobook
curl http://127.0.0.1:4200/api/audiobook --output audiobook.mp3
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

# Generate and download
requests.post(f"{BASE}/api/generate_audiobook")
# ... poll status ...
with open("output.mp3", "wb") as f:
    f.write(requests.get(f"{BASE}/api/audiobook").content)
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

await fetch(`${BASE}/api/generate_audiobook`, { method: "POST" });
await waitForTask("audiobook_generation");
```

## Recommended LLM Models

For script generation, non-thinking models work best:
- **Qwen2.5** (any size) - Excellent JSON output
- **Qwen3** (non-thinking variant)
- **Llama 3.1/3.2** - Good character distinction
- **Mistral/Mixtral** - Fast and reliable

**Avoid** thinking models (DeepSeek-R1, GLM4-air, etc.) as they can interfere with JSON output.

## Troubleshooting

### Script generation fails
- Check LLM server is running and accessible
- Verify model name matches what's loaded
- Try a different model - some struggle with JSON output

### TTS generation fails
- Ensure Qwen3-TTS is running at the configured URL
- Check voice_config.json has valid settings for all speakers
- For clone voices, verify reference audio exists and transcript is accurate

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
│   ├── generate_script.py     # LLM script annotation
│   ├── generate_audiobook.py  # Batch TTS generation
│   ├── tts.py                 # TTS abstraction layer
│   ├── project.py             # Chunk management
│   ├── parse_voices.py        # Voice extraction
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
