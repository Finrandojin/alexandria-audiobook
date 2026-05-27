# Alexandria Preparer Guide

Process a single audiobook into an annotated TTS training dataset with one command.

## Overview

The preparer takes an audio file (audiobook) and produces a ZIP archive containing:
- Word-aligned audio segments (`sample_NNNN.wav` at 24 kHz)
- `metadata.jsonl` with text annotations, durations, and time ranges
- Ready for use as a TTS training dataset

# Alexandria Preparer Guide

## What This Tool Does (Really Simple Version)

The Preparer converts an audiobook (audio file) into a training dataset for the Alexandria TTS engine. It does three things:

1. **Listens to the audio** and writes down every word with exact timestamps (called "transcription" or "ASR")
2. **Optionally matches the words against the original book** so the text uses the book's spelling (not what the AI misheard)
3. **Cuts the audio into small chunks** (about 10 seconds each) and saves them as WAV files with a metadata file

The output is a `.zip` file that you can feed into Alexandria's LoRA Training tab to teach the TTS engine a new voice.

### Who Is This For?

- **You have an audiobook** (WAV file) and want to clone a voice
- **You want to train a custom voice** using the Training tab
- **You don't need this** if you're just using Alexandria to generate audiobooks from books — the web UI handles everything automatically

---

## Quick Start for Non-Programmers

### Prerequisites

Before running the Preparer, make sure:

1. **You have Pinokio installed** and Alexandria is installed in Pinokio
2. **You have an audiobook WAV file** — if you have an MP3, convert it to WAV first (you can use an online converter or Audacity)
3. **You have the original book file** (.epub or .txt) — this is optional but highly recommended for better results

### Step 1: Open a Terminal

- **Windows:** Press `Win + R`, type `cmd`, press Enter
- **Mac:** Open Spotlight (Cmd + Space), type `terminal`, press Enter
- **Linux:** Open your applications menu and search for "Terminal"

### Step 2: Navigate to the Alexandria Folder

Type this command and press Enter:

```bash
cd ~/.pinokio/api/alexandria-audiobook.git
```

If you installed Alexandria in a different location, use that path instead.

### Step 3: Run the Preparer

Copy and paste this command (replace the paths with your actual file paths):

```bash
./app/env/bin/python alexandria_preparer_rocm_compatible.py \
  --audio "path/to/your/audiobook.wav" \
  --model Qwen2.5-14B-Instruct-Q6_K.gguf \
  --source "path/to/your/book.epub"
```

**What each part means:**
- `./app/env/bin/python` — This is the Python that comes with Alexandria (don't use your system Python)
- `--audio` — The path to your audiobook WAV file
- `--model` — The AI model that labels each chunk with speaker and voice directions
- `--source` — The original book file (optional, but improves accuracy)

### Step 4: Wait

The preparer will:
1. Transcribe the audio (takes ~1 hour per 5 hours of audio)
2. Match words against the book (if you provided `--source`)
3. Cut into chunks and save the ZIP

When it finishes, you'll see:
```
✓ Dataset written to: alexandria_dataset.zip
```

### Step 5: Use the Output

1. Unzip `alexandria_dataset.zip` — it contains WAV files and a `metadata.jsonl` file
2. In Alexandria's web UI, go to the **Dataset** tab
3. Upload the `metadata.jsonl` file and the WAV files
4. Use it to train a custom voice in the **Training** tab

---

## What This Tool Does (Really Simple Version)

Imagine you have a long audiobook — someone reading a whole novel out loud,
maybe 60 minutes or 6 hours long. To teach a computer how to talk like that
narrator, you need lots and lots of **short little clips** of them speaking,
each one labelled with **the exact words they said**. This tool makes those
clips for you, automatically.

Think of it like this:

1. **You give it the audiobook** (one big audio file).
2. **The tool listens** and figures out where every word starts and stops.
3. **It chops the audiobook into snack-sized clips** of about 10 seconds each,
   trying to cut at natural pauses (like at the end of a sentence) so each
   clip doesn't end mid-word.
4. **For each clip, it writes down the words** that were spoken in it.
5. **A small AI assistant adds little hints** to the words — like marking the
   *important* ones, or where the narrator took a long pause `...`
6. **You get back a zip file** full of clips + a list of what each clip says.

That zip is what you feed to a TTS-training program later, so the computer can
learn to copy that narrator's voice.

### The big new trick: using the book

You can also give the tool **the actual book** (the EPUB or text file).

Why? Because the listener-AI (the part that turns audio into words) makes
mistakes — especially with character names. It might write "**Yudy**" when the
narrator said "**Yurie**", or "**Kudou**" might come out as "**Coodo**".

When you give it the book, the tool can **look up the right spelling** for
every clip. Like having the answer key next to your homework: the listener-AI
hears the audio, then the tool checks the book and goes "ah, that bit says
'Miyo Saimori', let me use that spelling instead of whatever I guessed."

It can also **throw away clips that aren't in the book** — like when the
narrator says "Audible presents…" or reads the chapter announcement out loud,
or any other bits that don't match the actual story.

## Simplest possible example (copy/paste)

### Just turn an audiobook into clips (no book file)

```bash
app/env/bin/python alexandria_preparer_rocm_compatible.py \
  --audio "/path/to/audiobook.wav" \
  --model models/Qwen2.5-14B-Instruct-Q6_K.gguf
```

What this does, in plain English:
- Reads the audiobook at `/path/to/audiobook.wav`.
- Uses the AI helper at `models/Qwen2.5-14B-Instruct-Q6_K.gguf` to label the clips.
- Saves the result as `alexandria_dataset.zip` in the current folder.

### Now do the same thing but use the book too (recommended)

```bash
app/env/bin/python alexandria_preparer_rocm_compatible.py \
  --audio "/path/to/audiobook.wav" \
  --model models/Qwen2.5-14B-Instruct-Q6_K.gguf \
  --source "/path/to/books/My Happy Marriage - Volume 01.epub"
```

The only difference: that last line tells the tool where the book is. Names
will come out spelt right, and audio-only stuff (credits, intros) will get
thrown away.

### "I want to give you the book but please don't throw anything away"

```bash
app/env/bin/python alexandria_preparer_rocm_compatible.py \
  --audio "/path/to/audiobook.wav" \
  --model models/Qwen2.5-14B-Instruct-Q6_K.gguf \
  --source "/path/to/books/book.epub" \
  --keep-unaligned
```

The `--keep-unaligned` flag means "if a clip doesn't match the book, that's
okay — just use the listener-AI's best guess and keep the clip anyway."

### "I lost track of where it was — start over"

Just don't use `--resume`. Every run starts fresh by default. If the previous
attempt left junk behind, it gets cleaned up automatically.

### "It crashed three hours in — please pick up where you stopped"

```bash
app/env/bin/python alexandria_preparer_rocm_compatible.py \
  --audio "/path/to/audiobook.wav" \
  --model models/Qwen2.5-14B-Instruct-Q6_K.gguf \
  --resume
```

Same command, just add `--resume` at the end. It'll skip clips it already
made and continue from where it left off. (If you used `--source` originally,
add it here too — must match the original run.)

## How long does it take?

Rough rule of thumb on an AMD Radeon RX 9070 XT:

- **Listening to the audio** (turning sounds into words): ~1 minute per hour
  of audiobook.
- **Labelling the clips** (the slow part): about 10 hours per 1 hour of
  audiobook. So a typical 6-hour audiobook takes around 60 hours of
  computer time. You can leave it running overnight; if you need the GPU
  back, press **Ctrl+C** to stop and use `--resume` later.



**Key Features:**
- **Wav2Vec2 ASR (primary)** with true CTC-aligned word timestamps
- Insanely Fast Whisper / WhisperX-CPU fallbacks
- **LLM Pre-Processing (Enrichment)** — optional phase that adds speaker attribution, narration style, and emotional tone metadata to transcript chunks using a local GGUF LLM (e.g. Gemma-4-E2B)
- Gemma 4 / Qwen LLM annotations on GPU
- **Source-guided chunking** (`--source`): align ASR against an EPUB/TXT, use source spelling for chunk text, drop audio-only material
- **Pause-aware chunk boundaries**: chunks end at sentence punctuation or natural narrator pauses in the last 30% of each chunk, not wherever the duration threshold landed
- **3-phase pipeline** (ASR → Enrich → Annotate) with subprocess isolation for ROCm safety
- Resume capability — never lose work from a crash
- Periodic checkpointing every 50 segments
- Dynamic ETA based on actual measured throughput
- Single-pass audio loading (resample in memory)
- 24 kHz audio spilled to disk during annotation (~1.8 GB RAM saved on long audiobooks)
- GPU memory + utilization logging via `rocm-smi`

## Recommended Models

The preparer uses an LLM to annotate text chunks for TTS. Recommended for AMD 16 GB+ GPUs:

| Role | Model | VRAM (Q6_K) | Why |
|---|---|---|---|
| **Primary** | `Qwen2.5-14B-Instruct-Q6_K.gguf` | ~12 GB | Strong instruction following, terse structured output |
| **Fallback** | `Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf` | ~5 GB | Backup if Qwen fails to load |

## Usage

```bash
python alexandria_preparer_rocm_compatible.py \
  --audio audiobook.wav \
  --model Qwen2.5-14B-Instruct-Q6_K.gguf \
  --fallback-model Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf \
  --chunk-size 10.0 \
  --lang en
```

### With LLM enrichment (recommended for character-heavy books)

```bash
python alexandria_preparer_rocm_compatible.py \
  --audio audiobook.wav \
  --model Qwen2.5-14B-Instruct-Q6_K.gguf \
  --enrich-with-llm \
  --llm-model-path /path/to/gemma-4-E2B-it-Uncensored-MAX.BF16.gguf
```

This runs the **full 3-phase pipeline** (ASR → Enrich → Annotate). The enrichment phase uses a small local LLM to add `speaker_attribution`, `narration_style`, and `emotional_tone` metadata to every transcript chunk before annotation.

## Command Line Options

### Required
- `--audio PATH` — Input audio file (.wav, .mp3, .m4a, .flac, .ogg)
- `--model PATH` — Primary GGUF model path (omit only with `--skip-annotation`)

### Optional
- `--fallback-model PATH` — Backup GGUF model used if `--model` fails to load
- `--chunk-size SECONDS` — Target duration per chunk in seconds (default: `10.0`)
- `--lang CODE` — Language code for transcription (default: `en`)
- `--output PATH` — Output ZIP path (default: `alexandria_dataset.zip`)
- `--resume` — Resume from existing `dataset_temp/` instead of starting over
- `--skip-annotation` — Stop after transcription (not fully implemented)

### LLM Pre-Processing (Enrichment)
Add metadata to transcript chunks before annotation using a local GGUF LLM.
- `--enrich-with-llm` — Enable the enrichment phase (runs between ASR and annotation)
- `--llm-model-path PATH` — Path to the GGUF LLM model file for enrichment (required if `--enrich-with-llm` is set; recommended: Gemma-4-E2B or similar small model)
- `--enrich-speaker-attribution` — Instruct LLM to extract speaker attribution (e.g. "main character", "narrator", "secondary character")
- `--enrich-narration-style` — Instruct LLM to extract narration style (e.g. "calm", "energetic", "sad", "questioning")
- `--enrich-emotional-tone` — Instruct LLM to extract emotional tone (e.g. "happy", "anxious", "neutral", "excited")

### Source-guided chunking (optional)
Pass an EPUB or text file matching the audiobook to fix ASR mistranscriptions
at source and drop audio-only material before the LLM step.
- `--source PATH` — EPUB or TXT of the same book the audio narrates
- `--source-threshold N` — Minimum alignment ratio to keep a chunk (default: `0.65`)
- `--keep-unaligned` — When a chunk falls below `--source-threshold`, write the
  ASR text instead of dropping (default: drop)
- `--source-start N` — Start source alignment at word N (skip auto-anchor)
- `--source-start-text TEXT` — Fuzzy-search source for TEXT and start there
- `--no-auto-anchor` — Disable automatic source-anchor detection (start at word 0)

## Examples

### Standard run (Qwen primary, Gemma fallback)
```bash
python alexandria_preparer_rocm_compatible.py \
  --audio audiobook.wav \
  --model Qwen2.5-14B-Instruct-Q6_K.gguf \
  --fallback-model Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf
```

### Custom output path
```bash
python alexandria_preparer_rocm_compatible.py \
  --audio book1.wav \
  --model model.gguf \
  --output datasets/book1.zip
```

### Pause and resume (e.g., to game or free up GPU)

Press **Ctrl+C** at any time to stop the script. Your progress is saved automatically:

```
⚠ Process interrupted by user
Partial results preserved in dataset_temp/ - rerun with --resume to continue
```

When you're ready to continue, rerun with the same arguments plus `--resume`:

```bash
python alexandria_preparer_rocm_compatible.py \
  --audio audiobook.wav \
  --model model.gguf \
  --resume
```

The preparer picks up exactly where it left off — no repeated work.

### Resume after crash
If the script crashes 50 hours into a 60-hour annotation, the recovery is identical — rerun with `--resume`:

```bash
python alexandria_preparer_rocm_compatible.py \
  --audio audiobook.wav \
  --model model.gguf \
  --resume
```

The preparer will:
1. Read existing `dataset_temp/metadata.jsonl`
2. Find the last completed segment and its end time
3. Skip already-processed audio
4. Continue with new segments from that point

### Different language
```bash
python alexandria_preparer_rocm_compatible.py \
  --audio spanish_book.wav \
  --model model.gguf \
  --lang es
```

### Larger chunks
```bash
python alexandria_preparer_rocm_compatible.py \
  --audio audiobook.wav \
  --model model.gguf \
  --chunk-size 15.0
```

## Source-Guided Chunking

When you pass `--source <epub-or-txt>`, the preparer fuzzy-aligns each ASR
chunk against the source text and either:

- **Replaces** the chunk's ASR text with the source's spelling when the
  alignment ratio meets `--source-threshold` (default 0.65). The LLM then
  annotates already-correct text. Character names, dialect spellings, and
  punctuation come out exactly as the EPUB has them.
- **Drops** the chunk entirely when the alignment is below threshold. No
  WAV file, no JSONL entry. This is what strips audio-only material:
  narrator intros, "this audiobook is a work of fiction" credits, chapter
  announcements that aren't in the text, etc.

If you'd rather keep low-confidence chunks (using their ASR text), pass
`--keep-unaligned`. Cursor doesn't advance on those, so the next chunk
still tries to align from where the prose was last known to match.

### Example

```bash
app/env/bin/python alexandria_preparer_rocm_compatible.py \
  --audio audiobook.wav \
  --model Qwen2.5-14B-Instruct-Q6_K.gguf \
  --source "/path/to/books/My Happy Marriage - Volume 01.epub"
```

### What you'll see in the logs

At startup, the preparer reports source loading + lexicon build:
```
▶ Loading source for guided chunking: /path/to/book.epub
  ├─ Source: 284,125 characters
  ├─ 29 recurring proper nouns (gifted, godou, grotesqueries, hana, kanoko, kaya +23 more)
  ├─ 49,686 source words
  └─ Auto-anchor: entry 3 → source word 287 (62.1% match)
```

Per dropped chunk during annotation:
```
↪ DROPPED chunk at 421.36s (source ratio 0.42 < 0.65)
```

Per low-confidence kept chunk (only with `--keep-unaligned`):
```
↪ chunk 42 kept (ASR text); source ratio 0.51 < 0.65
```

### Wrong-source-file detection

Before LLM annotation starts, the preparer pre-scans the first ~30 ASR
chunks against the source. If the average alignment is below 50% or more
than 40% of chunks fall below 60%, the preparer **aborts with an error**
rather than waste hours of LLM time on a mismatched source:

```
⚠ Source/audio divergence too high to proceed:
  Sampled 30 chunks — avg alignment 38%, 21 (70%) below 60%.
  Usually means a wrong edition or different translation.
  Re-run without --source, or pass --keep-unaligned to accept the ASR
  text for low-confidence chunks.
```

This typically means you've supplied the wrong EPUB (different
translation, abridged edition, or a different book entirely). Re-anchor
manually with `--source-start N` / `--source-start-text "..."`, swap to
the right source, or run without `--source` to fall back to the legacy
ASR-only flow.

### When to use vs. skip

**Use `--source` when:** you have the matching EPUB/TXT for the
audiobook AND you want clean character names + dialect spellings in the
training dataset AND you're okay dropping chunks that don't appear in
the source (chapter intros, narrator notes, paratext).

**Skip `--source` when:** you don't have the source text, or you're
processing audio that's known to be loosely based on the source
(adaptations, rewrites, abridgments) where strict dropping would lose
content you want to keep.

### Effect on compare-review downstream

Source-guided mode dramatically reduces what you'll see when reviewing
the resulting JSONL with `alexandria_compare.py`. Most of the edits
users typically make (character-name ASR fixes, dialect spelling fixes)
have already been applied at preparation time. Compare-review becomes a
prosody sanity check — usually a few minutes per book instead of an
hour.

## Pause-Aware Chunk Boundaries

The chunker emits a chunk when the accumulated duration reaches
`--chunk-size`, but it doesn't cut at exactly that word. It looks back
across the last 30% of the chunk's duration and prefers to cut at:

1. A word ending in sentence punctuation (`.`, `!`, `?`) — best for TTS
   coherence; picks the LATEST one in the window.
2. The longest natural pause (gap ≥ 0.25s between ASR word end + next
   word start) in the same window.
3. The final word (current word at the threshold) — only when neither
   of the above is available.

Words trimmed off by the look-back become the seed of the next chunk;
no audio is lost. Net effect: chunks end on breath/clause/sentence
breaks instead of mid-phrase. Combined with `--source` mode, the
resulting WAVs are usable for TTS training with minimal hand-editing.

## Processing Flow

1. **Validate inputs** — Check audio file and model exist
2. **Load audio** — Single read at native sample rate, resample to 16 kHz (for ASR) and 24 kHz (for output)
3. **Spill 24 kHz to disk** — Write `.alexandria_audio_24k.wav` and free RAM
4. **Transcribe audio** (Wav2Vec2 → IFW → WhisperX-CPU):
   - Wav2Vec2 processes audio in 30 s chunks with 3 s overlap
   - Each chunk decoded with `output_word_offsets=True` for true CTC-frame alignment
   - Overlap region words deduplicated by word-center ownership
5. **Annotate chunks**:
   - Load Gemma 4 model fully to GPU (`n_gpu_layers=-1`)
   - Walk word segments, accumulate into ~10 s chunks
   - Generate TTS annotation per chunk with 3-segment context window
   - Write each `sample_NNNN.wav` and append to `metadata.jsonl` immediately (crash-safe)
   - Log dynamic ETA every 10 segments
6. **Create output dataset** — Zip `dataset_temp/` contents into output path
7. **Cleanup** — Remove scratch audio + temp dir (only on success)

## Resume Capability

The preparer writes each segment's metadata to `dataset_temp/metadata.jsonl` **immediately** after processing (not buffered to end). Each entry includes:

```json
{
  "audio_filepath": "sample_0042.wav",
  "text": "annotated text...",
  "duration": 10.04,
  "start": 421.36,
  "end": 431.40
}
```

On `--resume`:
- Existing entries are loaded
- The highest `end` time becomes the resume point
- Words with `start < resume_time` are skipped
- New entries are appended (existing entries untouched)
- The last 5 entries' text becomes the LLM context

**On intentional pause (Ctrl+C) or crash**: `dataset_temp/` is preserved (not deleted). Rerun with `--resume` and the same `--audio` and `--model` arguments to continue from where you stopped.

**On success**: `dataset_temp/` is deleted after the zip is created.

### Source-file safety check

A `.source` marker is written into `dataset_temp/` containing the absolute path of the audio file being processed. On `--resume`:
- If the marker matches the current `--audio` → resume safely.
- If the marker is missing or points to a different file → **wipe `dataset_temp/` and start fresh** (logged as a warning).

This prevents accidental corruption when reusing the working directory for different audiobooks. The batch processor relies on this guarantee to safely pass `--resume` on every retry.

## GPU Monitoring

The preparer logs GPU stats at key points:
- After Wav2Vec2 model load
- Before/during chunk processing (every 50 chunks)
- After Gemma model load
- During annotation (every 10 segments)

Each log entry shows:
```
GPU Usage (chunk 50/725):
  ├─ Memory: 4.52 GB / 17.10 GB (26.4%)
  └─ Utilization: 87.5%
```

Memory is from `torch.cuda.memory_allocated()`. Utilization is from `rocm-smi --showuse --json` (AMD GPUs). Note: when Gemma is the only GPU consumer, `torch.cuda.memory_allocated()` reports 0 because llama-cpp-python manages GPU memory outside PyTorch — check `rocm-smi` for the real picture.

## Progress Tracking

The pipeline runs in **three phases**, each launched as a separate subprocess for ROCm isolation:

### Phase 1: Wav2Vec2 (ASR)
```
↳ Chunk 50/725 | avg 0.51s/chunk | ETA 5m 47s
```

### Phase 2: LLM Enrichment (optional, requires `--enrich-with-llm`)
```
▶ Created 72 chunks for LLM enrichment
▶ Running LLM enrichment: python llm_enricher.py --model-path ... --input-file ... --output-file ...
Enriching chunk: 10.00s - 20.14s
...
✓ LLM Enrichment Phase completed successfully.
```

### Phase 3: Annotation
```
↳ Progress: 50/4713 chunks | Avg: 46.2s/chunk | Elapsed: 38m 42s | ETA: 59h 31m
```

Time format adapts: `Nh Mm` if > 1 hour, `Mm Ss` if > 1 min, `Ns` otherwise.

## Logs

Detailed logs are written to `logs/alexandria_preparer_YYYYMMDD_HHMMSS.log`:
- Full DEBUG-level entries for every step
- GPU stat snapshots
- Subprocess command lines and outputs
- Exception tracebacks (if any)

Tail the log in another terminal:
```bash
tail -f logs/alexandria_preparer_*.log
```

## ASR Method Priority

| # | Method | Device | Context | Word Timestamps |
|---|---|---|---|---|
| 1 | **Wav2Vec2** (primary) | GPU | 30 s chunks + 3 s overlap | True (CTC frame alignment) |
| 2 | Insanely Fast Whisper | GPU | 30 s fixed chunks | Native |
| 3 | WhisperX-CPU | CPU | n/a | Aligned via forced alignment |

Wav2Vec2 is selected first because of its overlapping-chunk context preservation and CTC-aligned per-word timestamps. Falls back if not installed or if the run errors out.

## Performance Tips

- **GPU memory** — Gemma uses ~8 GB; Wav2Vec2 uses ~2 GB. Both fit comfortably on a 16 GB+ GPU.
- **RAM** — Spilling 24 kHz audio to disk saves ~1.8 GB during the multi-hour annotation phase.
- **Disk space** — Allow ~5 – 10 GB free per audiobook (scratch WAV + segment WAVs + final zip).
- **Run time** — On AMD Radeon RX 9070 XT, expect ~6 min Wav2Vec2 + ~60 h Gemma annotation for a ~5.4 h audiobook with 10 s chunks.

## Troubleshooting

### Audio file not found
```
Error: Audio file not found: audiobook.wav
```
- Check path is correct; the script's CWD is wherever you launched it from.

### Model file not found
```
Error: Model file not found: ...
```
- Download the Gemma GGUF model (e.g., `download_model.py`) or supply `--model PATH`.

### `rocm-smi unavailable`
- Verify `/opt/rocm/bin/rocm-smi` exists and is executable.
- GPU memory still works without it; only utilization % is unavailable.

### `expandable_segments not supported`
- Informational warning from PyTorch — AMD ROCm doesn't support that memory optimization. PyTorch falls back to standard allocation. No action needed.

### Resume picked the wrong file
- The preparer tracks the source via `dataset_temp/.source` marker. Mismatched markers cause `dataset_temp/` to be wiped automatically (with a warning) before fresh processing.
- If you want to force a fresh start regardless, delete `dataset_temp/` manually or omit `--resume`.

### Out of disk space
- Check `du -sh dataset_temp/ .alexandria_audio_24k.wav`.
- Free up space, then rerun with `--resume`.

### Oversized WAV (>4 GiB) — header reports a tiny duration
Symptom: a multi-GB audiobook WAV reports a duration of a few minutes instead
of hours. Example startup log:
```
[WARNING] ⚠ Oversized WAV detected: header says 135.3s (2.3 min) but file size
implies 24477.2s (6.80 hr). WAV data-chunk-size field is 32-bit and has
wrapped — ffmpeg `-ignore_length 1` will be used to read the full audio.
```

Why it happens: standard WAV uses a 32-bit unsigned field for the `data`
chunk size, which overflows at 4 GiB. At 44.1 kHz stereo PCM_16 that hits at
~6.8 hours, so any full-length audiobook WAV produced by a tool that didn't
switch to RF64/WAV64 has a wrapped header. `soundfile` and `librosa.load`
honor the bogus header and silently truncate to `filesize mod 4 GiB` of
audio — every "complete" run on an oversized file is actually just the
first slice (e.g. 2 min of a 6.8 hr file, 22 min of a 27 hr file).

What the preparer does about it: `validate_inputs` compares file size
against header-implied bytes, prints the warning above when they
disagree, and routes the load through ffmpeg (`-ignore_length 1`). The
24 kHz scratch is streamed straight to disk and the 16 kHz ASR buffer
is piped into a float32 numpy array. No native-rate intermediate is
materialised, so peak RAM stays bounded regardless of the audio length.

What you should do: nothing, normally — the fix is automatic when the
preparer detects the wrap. If you'd rather re-encode the WAV once to
fix it permanently (and avoid the duplicate ffmpeg decode), convert to
FLAC or RF64-WAV, e.g.:
```
ffmpeg -ignore_length 1 -i input.wav -c:a flac output.flac
```

## API (programmatic invocation)

### Python (subprocess)
```python
import subprocess

result = subprocess.run([
    "python", "alexandria_preparer_rocm_compatible.py",
    "--audio", "audiobook.wav",
    "--model", "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf",
    "--chunk-size", "10.0",
    "--lang", "en",
    "--output", "my_dataset.zip",
], check=True)
```

### JavaScript (Node.js)
```javascript
const { execFile } = require("child_process");

execFile("python", [
  "alexandria_preparer_rocm_compatible.py",
  "--audio", "audiobook.wav",
  "--model", "model.gguf",
  "--output", "my_dataset.zip",
], { maxBuffer: 1024 * 1024 * 100 }, (err, stdout, stderr) => {
  if (err) throw err;
  console.log(stdout);
});
```

### Curl (not applicable)
The preparer is a CLI tool, not an HTTP service. Wrap it in a Flask/FastAPI server if you need network access.

## See Also

- `BATCH_PROCESSOR_GUIDE.md` — Process multiple audiobooks sequentially with resume across files
- `README.md` — Project overview
