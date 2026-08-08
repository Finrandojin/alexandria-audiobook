<img width="475" height="467" alt="Alexandria Logo" src="https://github.com/user-attachments/assets/fa2c36d3-a5f3-49ab-9dfe-30933359dfbd" />

# Alexandria Audiobook Generator

English | [中文](README_CN.md)

> **A note to new users:** Alexandria has recently seen a sudden surge of attention and new users. As a small project, I may not be able to respond to every issue promptly. Before opening an issue, please read this README and the [Wiki](https://github.com/Finrandojin/alexandria-audiobook/wiki) thoroughly — most common questions are already answered there. Thank you for your patience!

Transform any book or novel into a fully-voiced audiobook using AI-powered script annotation and text-to-speech. Features a built-in Qwen3-TTS engine with batch processing and a browser-based editor for fine-tuning every line before final export.

## Example: [sample.mp3](https://github.com/user-attachments/files/25276110/sample.mp3)


## Screenshots

<img src="https://github.com/user-attachments/assets/874b5e30-56d2-4292-b754-4408fc53f5d6" width="30%"></img> <img src="https://github.com/user-attachments/assets/488cde02-6b93-47fa-874b-97a618ae482c" width="30%"></img> <img src="https://github.com/user-attachments/assets/4c0805a6-bb9d-42c1-a9ff-79bb29d0613c" width="30%"></img> <img src="https://github.com/user-attachments/assets/8e58a5bf-ed8f-4864-8545-1e3d9681b0cf" width="30%"></img> <img src="https://github.com/user-attachments/assets/531830da-8668-4189-a0dc-020e6661bfb6" width="30%"></img> 

## Features

### AI-Powered Pipeline
- **Local & Cloud LLM Support** - Use any OpenAI-compatible API (LM Studio, Ollama, OpenAI, etc.)
- **Verbatim Script Annotation** - Code splits each chunk into quoted/unquoted spans and the LLM returns only labels (`id`, `speaker`, `role`, `instruct`). The script text is reassembled from the source by offset, so the author's words are never rewritten
- **LLM Script Review** - Optional second LLM pass that checks only two things per entry: whether `speaker` is the correct attribution and whether `instruct` is a usable voice direction. Entry text is always re-imposed from the original, so review cannot alter a single character of the book
- **Persona Generation** - LLM analyzes the script to create voice descriptions for each character, then generates reference audio via VoiceDesign and assigns clone voices automatically — one click to go from script to fully-voiced cast
- **Speaker Canonicalization** - Speaker labels are normalized to a canonical UPPERCASE form (accents folded, parentheticals and wrapping quotes removed, honorifics dropped). Spellings that differ only in spaces, hyphens or apostrophes ("ABBEMARIGNAN" / "ABBE MARIGNAN") are unified; similar-but-distinct names (JON/JOHN) never are
- **Speaker Aliases** - Map multiple speaker names to the same voice (e.g. "YOUNG ELENA" → "ELENA") so variants share a single voice configuration
- **Smart Chunking** - Groups consecutive lines by speaker (up to 500 chars) for natural flow
- **Context Preservation** - Passes a capped character roster (50 most recently spoken names by default) and short snippets of the last 3 script entries between chunks for name and style continuity

#### Verbatim Fidelity

The goal is an industry-standard audiobook, not an audio drama: the author's text is read word-for-word. Quoted dialogue gets character voices; everything else — including attribution tags like "said Marcus" — is read by the narrator. The LLM is an analytical classifier, never a generator of book text.

- **Verbatim reassembly** - Each chunk is tokenized into contiguous quoted/unquoted spans that tile the source exactly. The LLM sees numbered spans and returns only labels; code rebuilds the entries from the source string by `(start, end)` offset. A truncated or malformed response costs labels, never prose
- **Byte-identity is asserted, not hoped for** - Every chunk's entries must concatenate back to the chunk byte-for-byte; a mismatch raises rather than warns. Source → `annotated_script.json` word coverage is exactly 1.0
- **NARRATOR fallback** - Any span the LLM failed to label keeps its text and is attributed to NARRATOR. Enumerated placeholder labels ("SPEAKER 1", "VOICE 3") are rejected the same way
- **Loud degradation** - Script generation exits with code **3** when the script was written but some spans fell back to NARRATOR — distinct from 0 (clean) and 1 (nothing produced). The task log reports it as "completed with degradations" and the run summary names every degraded chunk

### Voice Generation
- **Built-in TTS Engine** - Qwen3-TTS runs locally with no external server required
- **External Server Mode** - Optionally connect to a remote Qwen3-TTS Gradio server
- **Multi-Language Support** - English, Chinese, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish, or Auto-detect
- **Custom Voices** - 9 pre-trained voices with instruct-based emotion/tone control
- **Voice Cloning** - Clone any voice from a 5-15 second reference audio sample
- **Voice Designer** - Create new voices from text descriptions (e.g. "A warm, deep male voice with a calm and steady tone")
- **LoRA Voice Training** - Fine-tune the Base model on custom voice datasets to create persistent voice identities with instruct-following
- **Built-in LoRA Presets** - Pre-trained voice adapters included out of the box, ready to assign to characters
- **Dataset Builder** - Interactive tool for creating LoRA training datasets with per-sample text, emotion, and audio preview
- **Batch Processing** - Generate dozens of chunks simultaneously with 3-6x real-time throughput
- **Codec Compilation** - Optional `torch.compile` optimization for 3-4x faster batch decoding
- **Non-verbal Sounds** - Vocalizations the author wrote ("Ahh!", "Mmm...", "Haha!") are spoken as-is with context-aware instruct directions — no bracket tags or special tokens. The LLM never adds sounds that aren't in the book; add your own in the Editor if you want them
- **Natural Pauses** - Configurable silence between speakers (default 500ms) and same-speaker segments (default 250ms)

### Web UI Editor
- **Streamlined Interface** - 5-step core pipeline (Setup, Script, Voices, Editor, Result) plus advanced tools (Designer, Dataset, Training)
- **Chunk Editor** - Edit speaker, text, and instruct for any line
- **Selective Regeneration** - Re-render individual chunks without regenerating everything
- **Batch Processing** - Optimized batch rendering with sub-batching for efficient GPU utilization
- **Live Progress** - Real-time logs and status tracking for all operations
- **Audio Preview** - Play individual chunks or sequence through the entire audiobook
- **Script Library** - Save and load annotated scripts with voice configurations

### Export Options
- **Combined Audiobook** - Single MP3 with all voices and natural pauses
- **Individual Voicelines** - Separate MP3 per line for DAW editing (Audacity, etc.)
- **Audacity Export** - One-click zip with per-speaker WAV tracks, LOF project file, and labels for automatic multi-track import into Audacity
- **M4B Audiobook** - Chaptered M4B (AAC) with per-chunk or auto-detected chapter markers for audiobook players (Audiobookshelf, Apple Books, VLC, etc.)

## Requirements

- [Pinokio](https://pinokio.computer/)
- LLM server (one of the following):
  - [LM Studio](https://lmstudio.ai/) (local) - recommended: Qwen3 or similar
  - [Ollama](https://ollama.ai/) (local)
  - [OpenAI API](https://platform.openai.com/) (cloud)
  - Any OpenAI-compatible API
- **GPU:** 8 GB VRAM minimum, 16 GB+ recommended — see compatibility table below
  - Each TTS model uses ~3.4 GB; remaining VRAM determines batch size
  - CPU mode available on all platforms but significantly slower
- **RAM:** 16 GB recommended (8 GB minimum)
- **Disk:** ~20 GB (8 GB venv/PyTorch, ~7 GB for model weights, working space for audio)

### GPU Compatibility

| GPU | OS | Status | Driver Requirement | Notes |
|-----|-----|--------|-------------------|-------|
| **NVIDIA** | Windows | Full support | Driver 550+ (CUDA 12.8) | Flash attention included for faster encoding |
| **NVIDIA** | Linux | Full support | Driver 550+ (CUDA 12.8) | Flash attention + triton included |
| **AMD** | Linux | Full support | ROCm 6.3+ | ROCm optimizations applied automatically |
| **AMD** | Windows | CPU only | N/A | GPU acceleration is not supported — the app runs in CPU mode. For GPU acceleration with AMD, use Linux |
| **Apple Silicon** | macOS | CPU only | N/A | MPS acceleration is not currently supported. Functional but slow |
| **Intel** | macOS | CPU only | N/A | |

> **Note:** No external TTS server is required. Alexandria includes a built-in Qwen3-TTS engine that loads models directly. Model weights are downloaded automatically on first use (~3.5 GB per model variant).

> **Documentation:** For in-depth guidance on voice types, LoRA training, batch generation, and more, see the [Wiki](https://github.com/Finrandojin/alexandria-audiobook/wiki).

## Installation

### Option A: Pinokio (Recommended)

1. Install [Pinokio](https://pinokio.computer/) if you haven't already
2. Open Alexandria on Pinokio: **[Install via Pinokio](https://beta.pinokio.co/apps/github-com-finrandojin-alexandria-audiobook)**
   - Or manually: in Pinokio, click **Download** and paste `https://github.com/Finrandojin/alexandria-audiobook`
3. Click **Install** to set up dependencies
4. Click **Start** to launch the web interface

### Option B: Google Colab (No Install Required)

No GPU or wrong OS? Run Alexandria on a free T4 GPU in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Finrandojin/alexandria-audiobook/blob/main/alexandria_colab.ipynb)

Requires a free [ngrok account](https://dashboard.ngrok.com/signup) for the web UI tunnel. See the notebook for full instructions.

### Option C: Docker (NVIDIA GPU)

For integration into automated pipelines or server deployments:

```bash
git clone https://github.com/Finrandojin/alexandria-audiobook.git
cd alexandria-audiobook
docker compose up --build
```

Requires [Docker](https://docs.docker.com/get-docker/) with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). The web UI is available at `http://localhost:4200`. TTS models download on first use and are cached in a Docker volume. User data (uploads, voice configs, trained LoRA adapters, audio output) persists via bind mounts to the project directory.

## First Launch — What to Expect

If this is your first time running Alexandria, read this before anything else.

### 1. You Need an LLM Server Running First

Alexandria does **not** include an LLM — it connects to one over an API. Before generating a script, you must have one of these running:

| Server | Default URL | Install |
|--------|-------------|---------|
| [LM Studio](https://lmstudio.ai/) | `http://localhost:1234/v1` | Download, load a model, start server |
| [Ollama](https://ollama.ai/) | `http://localhost:11434/v1` | `ollama run qwen3` |
| [OpenAI API](https://platform.openai.com/) | `https://api.openai.com/v1` | Get an API key |

If the LLM server isn't running when you click "Generate Script", the generation will fail. Check the Pinokio terminal for error details.

### 2. First TTS Generation Downloads ~3.5 GB

The TTS models are **not** included in the install. They download automatically from Hugging Face the first time you generate audio. This is normal:

- **Each model variant is ~3.5 GB** (CustomVoice, Base/Clone, VoiceDesign)
- Only the variant you use gets downloaded (most users start with CustomVoice)
- Downloads happen in the background — **check the Pinokio terminal** for progress
- The web UI may appear frozen during this time. It is not — it's waiting for the download to finish
- After the first download, models are cached locally and load in seconds

> **Tip:** If the download seems stuck, check your internet connection. If it fails, restart the app and try again — it will resume from where it left off.

### 3. First Batch Has Extra Warmup Time

The very first batch generation in a session takes longer than subsequent ones:

- **MIOpen autotuning** (AMD GPUs): The GPU kernel optimizer runs once per session, adding 30-60 seconds
- **Codec compilation** (if enabled): One-time ~30-60 second warmup, then 3-4x faster for all remaining batches
- **This is expected.** After the first batch, generation speed stabilizes

### 4. VRAM Determines What You Can Do

| Available VRAM | What Works |
|---------------|------------|
| 8 GB | One model at a time, small batches (2-5 chunks), CPU offload may be needed |
| 16 GB | Comfortable for most use cases, batches of 10-20 chunks |
| 24 GB+ | Full speed, batches of 40-60 chunks with codec compilation |

- If you run out of VRAM, reduce **Parallel Workers** or **Max Chars/Batch** in the Setup tab
- Close other GPU applications (games, other AI tools) before generating
- Switching between voice types (Custom → Clone → LoRA) unloads and reloads models, which temporarily frees VRAM

### 5. Where to Look When Something Goes Wrong

The web UI shows high-level status, but **detailed logs are in the Pinokio terminal**:

- Click **Terminal** in the Pinokio sidebar to see real-time output
- Model loading, download progress, VRAM estimates, and errors all appear here
- If generation fails silently in the UI, the terminal will show why

For common issues and solutions, see [Troubleshooting](https://github.com/Finrandojin/alexandria-audiobook/wiki/Troubleshooting).

---

## Quick Start

The interface is split into a **5-step core pipeline** (green tabs, numbered) and **advanced tools** (blue tabs, unnumbered). You only need the core pipeline to produce an audiobook.

### Core Pipeline

**Step 1 — Setup**
Configure your LLM connection and TTS engine. At minimum you need:
- **LLM Base URL**: `http://localhost:1234/v1` (LM Studio) or `http://localhost:11434/v1` (Ollama)
- **LLM API Key**: Your API key (use `local` for local servers)
- **LLM Model Name**: The model to use (e.g., `qwen2.5-14b`)
- **TTS Mode**: `local` (built-in, recommended) — loads models directly, no external server needed
- Click **Save Configuration** when done

**Step 2 — Script**
- Select your book file (.txt, .md, or .epub) using the file picker — it uploads automatically
- Click **Generate Annotated Script** — this sends the book to your LLM to split it into annotated chunks with speaker labels and voice directions
- *(Optional)* Click **Review Script** if the generated script has issues — this runs a second LLM pass to fix speaker misattributions and weak instruct directions. It cannot change the text
- You can save the script for later use with the Save feature below

**Step 3 — Voices**
Each character detected in the script gets a voice card. For each speaker:
- Choose a voice type: Custom Voice (easiest), Clone Voice, LoRA Voice, or Voice Design
- For Custom Voice, pick from 9 presets (Ryan, Serena, Aiden, etc.) and optionally set a character style (e.g., "Heavy Scottish accent")
- **Generate Personas** — Click to have the LLM analyze the script, create voice descriptions for each character, generate reference audio, and assign clone voices automatically. Toggle "Advanced" for batch size control. This is the fastest way to assign unique voices to all characters
- **Speaker Aliases** — Use the "Alias of" dropdown on any voice card to map a speaker to another character's voice (e.g., set "YOUNG ELENA" as alias of "ELENA"). Aliased speakers use the target's voice config during generation
- Changes save automatically — see [Voice Types](https://github.com/Finrandojin/alexandria-audiobook/wiki/Voice-Types) for guidance on each type

**Step 4 — Editor**
- Click **Render Pending** to generate audio for all chunks in batch
- Listen to individual chunks or click **Play Sequence** to preview in order
- Edit any chunk's text, speaker, or instruct inline and regenerate it individually
- When satisfied, click **Merge All** to combine everything into the final audiobook

**Step 5 — Result**
- Listen to the finished audiobook in the browser
- Download as MP3, or click **Export to Audacity** for per-speaker WAV tracks

### Advanced Tools (Optional)

These tabs are for power users who want more control over voice creation:

- **Designer** — Create new voices from text descriptions (e.g., "A warm elderly woman with a gentle raspy voice"). Save them to use as clone references in the Voices tab
- **Dataset** — Build LoRA training datasets interactively, one sample at a time with audio preview
- **Training** — Train LoRA adapters on voice datasets to create persistent voice identities that follow instruct directions

## Web Interface

### Setup Tab
Configure connections to your LLM and TTS engine.

**TTS Settings:**
- **Mode** - `local` (built-in engine) or `external` (connect to Gradio server)
- **Device** - `auto` (recommended), `cuda`, `cpu`, or `mps`
- **Language** - TTS synthesis language: English (default), Chinese, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish, or Auto (let the model detect)
- **Parallel Workers** - Batch size for fast batch rendering (higher = more VRAM usage)
- **Batch Seed** - Fixed seed for reproducible batch output (leave empty for random)
- **Compile Codec** - Enable `torch.compile` for 3-4x faster batch decoding (adds ~30-60s warmup on first generation)
- **Sub-batching** - Split batches by text length to reduce wasted GPU compute on padding (enabled by default)
- **Min Sub-batch Size** - Minimum chunks per sub-batch before allowing a split (default: 4)
- **Length Ratio** - Maximum longest/shortest text length ratio before forcing a sub-batch split (default: 5)
- **Speaker Change Pause** - Silence in milliseconds between different speakers during merge (default: 500)
- **Same Speaker Pause** - Silence in milliseconds when the same speaker continues during merge (default: 250)

**Prompt Settings (Advanced):**
- **Generation Settings** - Chunk size and max tokens for LLM responses
- **LLM Sampling Parameters** - Temperature, Top P, Top K, Min P, and Presence Penalty
- **Banned Tokens** - Comma-separated list of tokens to ban from LLM output (useful for disabling thinking mode on models like GLM4, DeepSeek-R1, etc.)
- **Prompt Customization** - System and user prompts used for script generation. Defaults are loaded from `default_prompts.txt` and can be customized per-session in the UI. Click "Reset to Defaults" to reload the file-based defaults (picks up edits without restarting the app)

**Advanced Keys (edit `app/config.json` directly):**

These have no UI control and are read straight from `app/config.json` by the generation/review/TTS code. Note that clicking **Save Configuration** rewrites `config.json` from the UI's form fields and will drop them, so set them after your last save (or re-add them afterwards).

| Key | Default | What it does |
|-----|---------|--------------|
| `generation.num_ctx` | unset | Serving context window requested from the LLM server per call, for servers that honor it (Ollama). Set this when you see the prompt-truncation warning |
| `generation.max_context_roster_names` | 50 | Cap on how many character names the per-chunk roster block may list. Kept names are the most recently spoken; the prompt says so when the list is partial |
| `generation.review_batch_char_budget` | 12000 | Rendered-JSON character budget per review batch, applied alongside `review_batch_size`. Keeps a few very long entries from producing a prompt that overflows a modest serving window |
| `tts.enable_nemo_normalization` | `false` | Optional text normalization ("Dr." → "doctor") applied **only** at the TTS call boundary — never to `instruct`, never to the script file. Uses `nemo_text_processing` where installable, else `wetext` (which has prebuilt Windows wheels). Off by default because whether it helps is book- and voice-specific |
| `pronunciation_dict.json` (repo root) | absent | Optional per-book file of flat `{"from": "to"}` literal replacements, applied after the normalizer to correct its mistakes (e.g. `{"Marlborough Dr.": "Marlborough Drive"}`). Only applied when normalization is enabled |

Run `python tools/verify_tts_normalization.py` to see raw vs. normalized vs. dictionary-applied text for a test sentence before deciding, or `--synthesize` to render the sentence both ways and listen.

### Script Tab
Upload a text file (.txt, .md, or .epub) and generate the annotated script. EPUB files are automatically converted to plain text on upload, in spine (reading) order; a per-chapter character count is printed to the terminal so you can confirm nothing was lost. A spine item whose file cannot be resolved, or an `idref` with no manifest entry, aborts the extraction rather than silently dropping a chapter. Non-text spine items (e.g. an SVG cover page) and the EPUB3 navigation document are skipped with a notice.

The LLM annotates your book into a structured JSON format with:
- Speaker identification (NARRATOR vs character names)
- The author's text, reassembled verbatim from the source — the LLM only labels spans, it never emits book text
- Style directions for TTS delivery

**Review Script** - After generation, click "Review Script" to run a second LLM pass. It checks exactly two fields per entry:
1. `speaker` — is the entry attributed to the right character?
2. `instruct` — is it a usable voice direction (not a physical action, not empty)?

Entry `text` is always re-imposed from the original entry, so a review pass cannot reword, split, merge or strip anything. A batch whose response has a different entry count than it was sent is rejected wholesale and the originals are kept. Speaker corrections are canonicalized and snapped back onto the spelling already established in the script.

Review prompts are customizable in `review_prompts.txt` (same format as `default_prompts.txt`).

> **Note:** `default_prompts.txt` carries a `span-labels-v1` marker and `review_prompts.txt` a `verbatim-review-v1` marker. A custom prompt saved in `config.json` before this pipeline existed asks the LLM for something it can no longer use, so a saved prompt missing its marker is rejected with a loud warning and the built-in default is used instead. If you customize a prompt, keep the marker line.

### Voices Tab
After script generation, voices are automatically loaded from the annotated script. For each speaker:

**Persona Generation:**
Click **Generate Personas** to automatically assign voices to all characters. The LLM analyzes dialogue in the script and produces a voice description and sample text for each speaker. These are fed to the VoiceDesign model to generate reference audio, which is saved and assigned as a clone voice. The result is a fully-voiced cast with no manual configuration.

- **Standard mode** — Processes all speakers at once
- **Advanced mode** — Toggle "Advanced" to control batch size (default 40) for large casts. In advanced mode, speakers are processed in discovery batches, with the LLM receiving dialogue samples to infer voice characteristics

**Speaker Aliases:**
Each voice card has an "Alias of" dropdown. Setting a speaker as an alias of another speaker means it will use the target's voice configuration during audio generation. Useful for:
- Age variants (e.g., "YOUNG ELENA" → "ELENA")
- Nickname or spelling variants the automatic rules deliberately don't merge (e.g. "JON" / "JOHN" — the pipeline never auto-merges similar names, because they are often two different people)
- Reducing the number of voices to configure

Aliases resolve transitively (A → B → C uses C's config) with cycle detection.

Honorific variants ("DR. SMITH" → "SMITH") no longer need an alias — canonicalization strips a leading honorific automatically.

> **Known limitation:** because honorifics are stripped, `MR SMITH` and `MRS SMITH` both canonicalize to `SMITH`, so a husband and wife share one voice card and one voice. There is currently no way to split them back apart. If a book distinguishes two characters only by title, edit their `speaker` values in the Editor to something unambiguous before configuring voices.

`GET /api/voices/alias_suggestions` returns advisory fuzzy-similarity suggestions (JON/JOHN, ELLA/BELLA) for the current roster. It is purely advisory — nothing is merged — and is not yet surfaced in the UI.

**Custom Voice Mode:**
- Select from 9 pre-trained voices: Aiden, Dylan, Eric, Ono_anna, Ryan, Serena, Sohee, Uncle_fu, Vivian
- Set a character style that appends persistent traits to every TTS instruct (e.g., "Heavy Scottish accent", "Refined aristocratic tone")
- Optionally set a seed for reproducible output

**Clone Voice Mode:**
- Select a designed voice or enter a custom reference audio path
- Provide the exact transcript of the reference
- Note: Instruct directions are ignored for cloned voices

**LoRA Voice Mode:**
- Select a trained LoRA adapter from the Training tab
- Set a character style (same as Custom — appended to every instruct)
- Combines voice identity from training with instruct-following from the Base model

**Voice Design Mode:**
- Set a base voice description (e.g., "Young strong soldier")
- Each line's instruct is appended as delivery/emotion direction
- Generates voice on-the-fly using the VoiceDesign model — ideal for minor characters

### Voice Designer Tab
Create new voices from text descriptions without needing reference audio.

- **Describe a voice** in natural language (e.g., "A warm elderly woman with a gentle, raspy voice and a slight Southern drawl")
- **Preview** the voice with sample text before saving
- **Save to library** for use as clone voice references in the Voices tab
- Uses the Qwen3-TTS VoiceDesign model to synthesize voice characteristics from descriptions

### Training Tab
Train LoRA adapters on the Base model to create custom voice identities. Several built-in LoRA presets are included out of the box and appear alongside your trained adapters.

**Dataset:**
- **Upload ZIP** — WAV files (24kHz mono) + `metadata.jsonl` with `audio_filepath` and `text` fields
- **Generate Dataset** — Auto-generate training samples from a Voice Designer description with custom sample texts
- **Dataset Builder** — Interactive tool in its own tab (see below) for building datasets sample-by-sample with preview

**Training Configuration:**
- **Adapter Name** — Identifier for the trained model
- **Epochs** — Full passes over the dataset (15-30 recommended for 20+ samples)
- **Learning Rate** — Default 5e-6 (conservative). Higher trains faster but risks instability
- **LoRA Rank** — Adapter capacity. High (64+) locks voice identity strongly but can flatten delivery. Low (8-16) preserves expressiveness
- **LoRA Alpha** — Scaling factor. Effective strength = alpha / rank. Common starting point: alpha = 2x rank
- **Language** — Language for the codec prefix token. Match this to your training data's language (English, Chinese, Korean, Japanese, etc.). Mismatched language can cause the adapter to lose speaker identity
- **Batch Size / Grad Accum** — Batch 1 with gradient accumulation 8 is typical for 24GB cards

**Training tips:**
- Include samples with varied emotions (happy, sad, angry, calm) for expressive voices
- Neutral-only training data produces flat voices that resist instruct prompting
- The settings info panel in the UI explains each parameter's effect on voice quality

### Dataset Builder Tab
Build LoRA training datasets interactively, one sample at a time.

- **Create a project** with a voice description and optional global seed
- **Define samples** — Set text and emotion/style per row
- **Preview audio** — Generate and listen to individual samples or batch-generate all at once
- **Cancel batch** — Stop a running batch generation without losing completed samples
- **Save as dataset** — Export the project as a training-ready dataset that appears in the Training tab
- Designed voices and Voice Designer descriptions drive the audio generation via Qwen3-TTS VoiceDesign model

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
- **All voice types supported** - Custom, Clone, and LoRA voices are batched; Voice Design is sequential
- **Parallel Workers** setting controls batch size (higher values use more VRAM)

### Result Tab
Download your completed audiobook as MP3, export as **M4B** with chapter markers for audiobook players, or click **Export to Audacity** for per-speaker WAV tracks.

- **Download MP3** - Standard merged audiobook
- **Export M4B** - AAC audiobook with chapter markers. By default, chapters are auto-detected from headings in the script (e.g. "Chapter 1", "Prologue"). Toggle **Per-chunk chapters** for fine-grained navigation where every line becomes a chapter.
- **Export to Audacity** - Zip with per-speaker WAV tracks. Unzip and open `project.lof` in Audacity to load all tracks, then import `labels.txt` via File > Import > Labels for chunk annotations.

> **Note:** Some Linux audiobook players (e.g. Cozy) have limited M4B support and may not detect the file. The M4B output has been tested with VLC, Haruna, and Audiobookshelf.

## Performance

### Recommended Settings for Batch Generation

| Setting | Recommended | Notes |
|---------|-------------|-------|
| TTS Mode | `local` | Built-in engine, no external server |
| Compile Codec | `true` | 3-4x faster decoding after one-time warmup |
| Parallel Workers | 20-60 | Higher = more throughput, more VRAM |
| Render Mode | Batch (Fast) | Uses batched TTS calls |

### Benchmarks

Tested on AMD RX 7900 XTX (24 GB VRAM, ROCm 6.3/7.2):

| Configuration | Throughput |
|--------------|------------|
| Standard mode (sequential) | ~1x real-time |
| Batch mode, no codec compile | ~2x real-time |
| Batch mode + compile_codec | **3-6x real-time** |

A 273-chunk audiobook (~54 minutes of audio) generates in approximately 16 minutes with batch mode and codec compilation enabled.

### ROCm (AMD GPU) Notes

> **Linux only.** AMD GPU acceleration requires ROCm 6.3+ on Linux. AMD GPUs on Windows run in CPU mode — see [GPU Compatibility](#gpu-compatibility).

Alexandria automatically applies ROCm-specific optimizations when running on AMD GPUs:
- **MIOpen fast-find mode** - Prevents workspace allocation failures that cause slow GEMM fallback
- **Triton AMD flash attention** - Enables native flash attention for the whisper encoder
- **triton_key compatibility shim** - Fixes `torch.compile` on pytorch-triton-rocm

These are applied transparently and require no configuration.

> **ROCm 7.x GPU downclocking fix:** ROCm 7.x has a regression where the GPU's DPM controller aggressively downclocks the shader engine between autoregressive generation steps, causing batch generation to slow to a crawl or appear to hang. The fix is to set the GPU power profile to COMPUTE, which enforces a minimum clock frequency floor:
>
> ```bash
> echo 5 | sudo tee /sys/class/drm/card1/device/pp_power_profile_mode
> ```
>
> This needs to be run once per boot (it does not persist across reboots). You can add it to your system startup or run it manually before launching Alexandria. To verify it's active, check for `COMPUTE*` in the output of:
>
> ```bash
> cat /sys/class/drm/card1/device/pp_power_profile_mode
> ```
>
> ROCm 6.x users and NVIDIA users are not affected.

## Script Format

The generated script is a JSON array with `speaker`, `text`, and `instruct` fields:

```json
[
  {"speaker": "NARRATOR", "text": "The door creaked open slowly. ", "instruct": "Calm, even narration."},
  {"speaker": "ELENA", "text": "\"Ah! Who's there?\"", "instruct": "Startled and fearful, sharp whispered question, voice cracking with panic."},
  {"speaker": "NARRATOR", "text": " she whispered.\n\n", "instruct": "Neutral, even narration."},
  {"speaker": "MARCUS", "text": "\"Haha... did you miss me?\"", "instruct": "Menacing confidence, low smug drawl with a dark chuckle, savoring the moment."}
]
```

- **`speaker`** — canonical UPPERCASE character name, or `NARRATOR`. The casing is load-bearing: the pipeline compares against the literal `"NARRATOR"`
- **`text`** — verbatim from the source book, including quotation marks and the whitespace between entries. Concatenating every entry's `text` in order reproduces the book exactly, which is why attribution tags ("she whispered") are their own NARRATOR entries rather than being dropped
- **`instruct`** — 2-3 sentence TTS voice direction sent directly to the engine. Set tone, describe delivery, then give specific references. Example: "Devastated by grief, Sniffing between words and pausing to collect herself, end with a wracking sob."

### Non-verbal Sounds
Vocalizations are real pronounceable text that the TTS speaks directly — no bracket tags or special tokens. Because the script is verbatim, these come from the book itself (or from your own edits in the Editor), and the LLM gives them a short instruct direction:
- Gasps: "Ah!", "Oh!" with instruct like "Fearful, sharp gasp."
- Sighs: "Haah...", "Hff..."
- Laughter: "Haha!", "Ahaha..."
- Crying: "Hic... sniff..."
- Exclamations: "Mmm...", "Hmm...", "Ugh..."

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

**M4B Audiobook (chaptered):**
- `audiobook.m4b` - AAC audiobook with embedded chapter markers
- Chapters auto-detected from script headings, or per-chunk when toggled
- Compatible with Audiobookshelf, Apple Books, VLC, Haruna, and most audiobook players

## API Reference

Alexandria exposes a REST API for programmatic access:

### Configuration
```bash
# Get current config (empty prompts fall through to file defaults)
curl http://127.0.0.1:4200/api/config

# Get file-based default prompts (hot-reloads from default_prompts.txt)
curl http://127.0.0.1:4200/api/default_prompts

# Save config
curl -X POST http://127.0.0.1:4200/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "llm": {"base_url": "...", "api_key": "...", "model_name": "..."},
    "tts": {
      "mode": "local",
      "device": "auto",
      "language": "English",
      "parallel_workers": 25,
      "batch_seed": 12345,
      "compile_codec": true,
      "sub_batch_enabled": true,
      "sub_batch_min_size": 4,
      "sub_batch_ratio": 5,
      "pause_between_speakers_ms": 500,
      "pause_same_speaker_ms": 250
    }
  }'
```

### Script Generation
```bash
# Upload text file (supports .txt, .md, .epub)
curl -X POST http://127.0.0.1:4200/api/upload \
  -F "file=@mybook.epub"

# Generate script (returns task ID)
curl -X POST http://127.0.0.1:4200/api/generate_script

# Check status
curl http://127.0.0.1:4200/api/status/script_generation

# Review script (fix speaker attribution and instruct fields; text is never altered)
curl -X POST http://127.0.0.1:4200/api/review_script

# Check review status
curl http://127.0.0.1:4200/api/status/review
```

### Voice Management
```bash
# Get voices and config
curl http://127.0.0.1:4200/api/voices

# Parse voices from script
curl -X POST http://127.0.0.1:4200/api/parse_voices

# Advisory alias suggestions for similar roster names (never merges anything)
# Kept off the /api/voices hot path — it is O(n^2) in roster size
curl http://127.0.0.1:4200/api/voices/alias_suggestions

# Save voice config
curl -X POST http://127.0.0.1:4200/api/save_voice_config \
  -H "Content-Type: application/json" \
  -d '{"NARRATOR": {"type": "custom", "voice": "Ryan", "character_style": "calm"}}'
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

### Persona Generation
```bash
# Generate personas (LLM + VoiceDesign, assigns clone voices automatically)
curl -X POST http://127.0.0.1:4200/api/generate_personas

# Generate personas in advanced mode with custom batch size
curl -X POST http://127.0.0.1:4200/api/generate_personas \
  -H "Content-Type: application/json" \
  -d '{"advanced": true, "batch_size": 40}'

# Check persona generation status
curl http://127.0.0.1:4200/api/status/persona

# Cancel persona generation
curl -X POST http://127.0.0.1:4200/api/cancel_persona
```

### Voice Designer
```bash
# Preview a voice from text description
curl -X POST http://127.0.0.1:4200/api/voice_design/preview \
  -H "Content-Type: application/json" \
  -d '{"description": "A warm, deep male voice", "text": "Hello world."}'

# Save a designed voice
curl -X POST http://127.0.0.1:4200/api/voice_design/save \
  -H "Content-Type: application/json" \
  -d '{"name": "warm_narrator", "description": "A warm, deep male voice", "text": "Hello world."}'

# List saved designed voices
curl http://127.0.0.1:4200/api/voice_design/list

# Delete a designed voice
curl -X DELETE http://127.0.0.1:4200/api/voice_design/delete/voice_id_here
```

### LoRA Training
```bash
# Upload a training dataset (ZIP with WAV + metadata.jsonl)
curl -X POST http://127.0.0.1:4200/api/lora/upload_dataset \
  -F "file=@dataset.zip" -F "name=my_voice"

# Generate a dataset from Voice Designer description
curl -X POST http://127.0.0.1:4200/api/lora/generate_dataset \
  -H "Content-Type: application/json" \
  -d '{"name": "warm_voice", "description": "A warm male voice", "texts": ["Hello.", "Goodbye."]}'

# List uploaded datasets
curl http://127.0.0.1:4200/api/lora/datasets

# Delete a dataset
curl -X DELETE http://127.0.0.1:4200/api/lora/datasets/dataset_id_here

# Start LoRA training
curl -X POST http://127.0.0.1:4200/api/lora/train \
  -H "Content-Type: application/json" \
  -d '{"name": "narrator_warm", "dataset_id": "my_voice", "epochs": 25, "lr": "5e-6", "lora_r": 32, "lora_alpha": 64}'

# Check training status
curl http://127.0.0.1:4200/api/status/lora_training

# List trained adapters
curl http://127.0.0.1:4200/api/lora/models

# Test a trained adapter
curl -X POST http://127.0.0.1:4200/api/lora/test \
  -H "Content-Type: application/json" \
  -d '{"adapter_id": "narrator_warm_1234567890", "text": "Test line.", "instruct": "Calm narration."}'

# Delete an adapter
curl -X DELETE http://127.0.0.1:4200/api/lora/models/adapter_id_here
```

### Dataset Builder
```bash
# List all dataset builder projects
curl http://127.0.0.1:4200/api/dataset_builder/list

# Create a new project
curl -X POST http://127.0.0.1:4200/api/dataset_builder/create \
  -H "Content-Type: application/json" \
  -d '{"name": "my_voice_dataset"}'

# Update project metadata (description and global seed)
curl -X POST http://127.0.0.1:4200/api/dataset_builder/update_meta \
  -H "Content-Type: application/json" \
  -d '{"name": "my_voice_dataset", "description": "A warm male narrator", "global_seed": "42"}'

# Update sample rows
curl -X POST http://127.0.0.1:4200/api/dataset_builder/update_rows \
  -H "Content-Type: application/json" \
  -d '{"name": "my_voice_dataset", "rows": [{"text": "Hello world.", "emotion": "cheerful"}]}'

# Generate a single sample preview
curl -X POST http://127.0.0.1:4200/api/dataset_builder/generate_sample \
  -H "Content-Type: application/json" \
  -d '{"name": "my_voice_dataset", "description": "A warm male voice", "sample_index": 0, "samples": [{"text": "Hello.", "emotion": "cheerful"}]}'

# Batch generate all samples
curl -X POST http://127.0.0.1:4200/api/dataset_builder/generate_batch \
  -H "Content-Type: application/json" \
  -d '{"name": "my_voice_dataset", "description": "A warm male voice", "samples": [{"text": "Hello.", "emotion": "cheerful"}]}'

# Check batch generation status
curl http://127.0.0.1:4200/api/dataset_builder/status/my_voice_dataset

# Cancel a running batch generation
curl -X POST http://127.0.0.1:4200/api/dataset_builder/cancel \
  -H "Content-Type: application/json" \
  -d '{"name": "my_voice_dataset"}'

# Save project as a training dataset
curl -X POST http://127.0.0.1:4200/api/dataset_builder/save \
  -H "Content-Type: application/json" \
  -d '{"name": "my_voice_dataset", "ref_sample_index": 0}'

# Delete a project
curl -X DELETE http://127.0.0.1:4200/api/dataset_builder/my_voice_dataset
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
    "NARRATOR": {"type": "custom", "voice": "Ryan", "character_style": "calm narrator"},
    "HERO": {"type": "custom", "voice": "Aiden", "character_style": "brave, determined"}
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
    NARRATOR: { type: "custom", voice: "Ryan", character_style: "calm" }
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

### Script generation "completed with degradations" (return code 3)
This is not a failure: the script was written and no text was lost, but some spans could not be classified and are attributed to NARRATOR. The run summary lists every degraded chunk. Review those entries in the Editor, or re-run after fixing the cause below.

### Speakers are mostly NARRATOR, or `prompt_tokens` never changes
Your LLM server is silently truncating the prompt. A server whose context window is smaller than the prompt does not error — it drops the overflow and answers from what's left, so the model is asked to classify spans it was never shown. Symptoms in the terminal:
- A `PROMPT LIKELY TRUNCATED BY THE SERVER` warning on individual chunks
- A `SERVER CONTEXT WINDOW SUSPECT` line in the run summary, reporting that `prompt_tokens` was pinned to one exact value across prompts of wildly different sizes (Ollama's default `num_ctx=2048` produces exactly this)
- Lots of retries and degraded chunks with an otherwise green run

Fix it by raising the **server's** context window:
- **Ollama:** set `OLLAMA_CONTEXT_LENGTH=8192` in the environment, bake `num_ctx` into the model's Modelfile, or set `generation.num_ctx` in `app/config.json` (forwarded per request to servers that honor it)
- **llama.cpp / vLLM:** raise `-c` / `--max-model-len`
- Or reduce the prompt: lower `generation.chunk_size` or `generation.max_context_roster_names`

### Model download fails or is very slow
- TTS models (~3.5 GB each) are downloaded from Hugging Face on first use
- If downloads are slow or fail due to network restrictions (common in mainland China), set a Hugging Face mirror before launching:
  - Set the environment variable `HF_ENDPOINT=https://hf-mirror.com` before starting the app
  - Or in Pinokio, add it to start.js `env` field: `env: { HF_ENDPOINT: "https://hf-mirror.com" }`
- If you hit rate limits, create a free [Hugging Face account](https://huggingface.co/join) and set `HF_TOKEN` to your access token
- Downloads resume automatically if interrupted — just restart the app

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

### Broken or tiny MP3 files (428 bytes)
Conda's bundled ffmpeg on Windows often lacks the MP3 encoder (libmp3lame). Alexandria now detects this and automatically falls back to WAV, but if you want MP3 output:
- Install ffmpeg with MP3 support: `conda install -c conda-forge ffmpeg`
- Or remove conda's ffmpeg to use your system one: `conda remove ffmpeg`
- Verify with: `ffmpeg -encoders 2>/dev/null | grep mp3`

### Audio quality issues
- Use 5-15 second clear reference audio for cloning
- Avoid background noise in reference samples
- Try different seeds for custom voices

### Mojibake characters in output
- The system automatically fixes common encoding issues
- If problems persist, ensure your input text is UTF-8 encoded

## Prompt Customization

LLM prompts are stored in plain-text files at the project root, split into system prompt and user prompt sections by a `---SEPARATOR---` delimiter:

- **`default_prompts.txt`** — Prompts for script generation (annotation)
- **`review_prompts.txt`** — Prompts for script review (error correction)

**How it works:**
- `app/default_prompts.py` and `app/review_prompts.py` read their respective files and export the prompts
- Prompts hot-reload from disk on every request, so edits take effect immediately without restarting the app
- `config.json` stores user overrides for generation prompts — when its prompt fields are empty, the file defaults are used
- The "Reset to Defaults" button in the Web UI fetches the latest file defaults via `/api/default_prompts`

**To customize prompts:**
1. **Temporary (per-session):** Edit generation prompts directly in the Setup tab's Prompt Customization section
2. **Permanent (all sessions):** Edit `default_prompts.txt` or `review_prompts.txt` directly — changes are picked up on the next request

**Non-English books:** The default LLM prompts are written for English text and reference English-specific conventions (attribution tags like "said he", quotation marks, etc.). When processing books in other languages, you'll get better results by editing the prompts to match that language's dialogue conventions — for example, French guillemets (« »), Japanese brackets (「」), or language-appropriate attribution patterns. Set the TTS **Language** dropdown to match as well.

> **Limitation:** the span tokenizer recognizes straight and curly double quotes (`"` `“` `”`) and single quotes (`'` `‘` `’`) as dialogue delimiters. Guillemets (« ») and CJK brackets (「」) are **not** recognized, so dialogue marked that way is currently classified as narration and read by the narrator. No text is lost — it is the safe failure direction — but such a book will not get character voices until the tokenizer learns those marks. Prompt customization cannot work around this, since the LLM only labels the spans code gives it.

## Testing

The test suites are standalone scripts (not pytest) and run fully offline — they use fake LLM clients and programmatically built EPUB fixtures, so no LLM server, no TTS model and no network are required. Run them from the `app/` directory:

```bash
cd app
python test_api.py --offline        # pipeline invariants (verbatim, coverage == 1.0) — ~1 second
python test_span_tokenizer.py       # span FSM: tiling, quote disambiguation
python test_span_integration.py     # span classification -> verbatim entries end to end
python test_speaker_canon.py        # canonicalization, roster keys, alias suggestions
python test_canon_wiring.py         # canonicalization wired through generation/review/API
python test_review_verbatim.py      # positional overlay, batch sizing, text-loss guards
python test_epub_extract.py         # EPUB href resolution and loud-failure behavior
python test_integration_fixes.py    # regression checks for previously shipped fixes
python test_tts_normalization.py    # normalizer no-op-when-off and backend chain
```

`python test_api.py` without `--offline` additionally exercises the HTTP API and needs the server running.

## Project Structure

```
Alexandria/
├── app/
│   ├── app.py                 # FastAPI server
│   ├── tts.py                 # TTS engine (local + external backends)
│   ├── tts_normalizer.py      # Optional TTS-boundary text normalization (off by default)
│   ├── train_lora.py          # LoRA training subprocess script
│   ├── generate_script.py     # LLM span classification -> verbatim script
│   ├── span_tokenizer.py      # FSM splitting text into quoted/unquoted spans
│   ├── speaker_canon.py       # Speaker-label canonicalization and alias suggestions
│   ├── generate_personas.py   # LLM persona generation + VoiceDesign voice assignment
│   ├── review_script.py       # LLM script review (second pass, speaker/instruct only)
│   ├── utils.py               # Shared utilities (atomic JSON writes)
│   ├── test_*.py              # Standalone offline test suites (see Testing above)
│   ├── default_prompts.py     # Generation prompt loader (reads default_prompts.txt)
│   ├── review_prompts.py      # Review prompt loader (reads review_prompts.txt)
│   ├── project.py             # Chunk management & batch generation
│   ├── config.json            # Runtime configuration (gitignored)
│   ├── static/index.html      # Web UI
│   └── requirements.txt       # Python dependencies
├── builtin_lora/              # Pre-trained LoRA voice presets
├── dataset_builder/           # Dataset builder project workspace (gitignored)
├── designed_voices/           # Saved Voice Designer outputs (gitignored)
├── lora_datasets/             # Uploaded/generated training datasets (gitignored)
├── lora_models/               # Trained LoRA adapters (gitignored)
├── tools/                     # Standalone diagnostic tools (TTS normalization check)
├── default_prompts.txt        # LLM prompts for script generation
├── review_prompts.txt         # LLM prompts for script review
├── install.js                 # Pinokio installer
├── start.js                   # Pinokio launcher
├── reset.js                   # Reset script
├── pinokio.js                 # Pinokio UI config
├── pinokio.json               # Pinokio metadata
└── README.md
```

## Acknowledgements

- [Ayush Naphade](https://github.com/aayushnaphade) — Persona generation, speaker alias resolution, and contextual script review ([PR #42](https://github.com/Finrandojin/alexandria-audiobook/pull/42)). Check out his project [Lily](https://lily.rayoneai.in/) — looking forward to seeing where it goes!
- [Michii](https://github.com/on22s) — System Health Dashboard with real-time GPU/disk monitoring ([PR #45](https://github.com/Finrandojin/alexandria-audiobook/pull/45)), cross-platform subprocess runner ([PR #46](https://github.com/Finrandojin/alexandria-audiobook/pull/46)), Voice Training Dataset Preparer tab ([PR #47](https://github.com/Finrandojin/alexandria-audiobook/pull/47))

## License

MIT

### Third-Party Licenses

- [qwen_tts](https://github.com/Qwen/Qwen3-TTS) — Apache License 2.0, Copyright Alibaba Qwen Team
