# Alexandria Batch Processor Guide

Process multiple audiobook files sequentially with a single command.

## What This Tool Does (Plain English)

The Batch Processor runs the Preparer (see `PREPARER_GUIDE.md`) on **multiple audiobook files at once**. Instead of running the preparer one file at a time manually, you give it a list of files (or a folder full of files) and it processes them all in order, one after another.

### Who Is This For?

- **You have multiple audiobooks** and want to create training datasets for all of them
- **You don't need this** if you only have one audiobook — use the Preparer instead

### How It Relates to the Preparer

The Batch Processor is just a wrapper around the Preparer. It:
1. Takes a list of audio files
2. Runs the Preparer on each one
3. Saves each output with a unique name so they don't overwrite each other
4. Shows you a summary when all files are done

---

## Quick Start for Non-Programmers

### Step 1: Gather Your Files

Put all the audiobook WAV files you want to process into one folder. For example:
```
/home/fakemitch/Desktop/my_audiobooks/
  ├── book1.wav
  ├── book2.wav
  └── book3.wav
```

### Step 2: Open a Terminal

- **Windows:** Press `Win + R`, type `cmd`, press Enter
- **Mac:** Open Spotlight (Cmd + Space), type `terminal`, press Enter
- **Linux:** Open your applications menu and search for "Terminal"

### Step 3: Run the Batch Processor

Copy and paste this command (replace the folder path with your actual path):

```bash
cd ~/.pinokio/api/alexandria-audiobook.git
./app/env/bin/python alexandria_batch_processor.py \
  --folder "/path/to/your/audiobooks" \
  --model Qwen2.5-14B-Instruct-Q6_K.gguf
```

### Step 4: Wait

Each file takes about 1 hour per 5 hours of audio. If you have 3 audiobooks that are each 5 hours long, expect about 3 hours total.

### Step 5: Check the Results

When finished, each audiobook will have its own output folder with a `.zip` dataset file.

---

## Overview

The batch processor allows you to queue multiple audio files for processing. Each file is processed one after another (sequentially), creating a complete annotated dataset for each.

**Key Features:**
- Process multiple files with one command
- Sequential processing (one at a time, no parallel overhead)
- Automatic output file naming (includes original filename)
- Comprehensive logging and progress tracking
- JSON results summary with statistics
- GPU acceleration for all files
- Time estimates for each file

## Quick Start

Pick one of three input modes:

**A. List files individually** (positional, space-separated)
```bash
python alexandria_batch_processor.py book1.wav book2.wav book3.wav --model Qwen2.5-14B-Instruct-Q6_K.gguf
```

**B. Scan an entire folder** (use `--folder` flag — the space matters)
```bash
python alexandria_batch_processor.py --folder /path/to/audiobooks --model Qwen2.5-14B-Instruct-Q6_K.gguf
```

**C. Mix both** (folder + extra files)
```bash
python alexandria_batch_processor.py extra.wav --folder /path/to/audiobooks --model Qwen2.5-14B-Instruct-Q6_K.gguf
```

> **Paths with spaces must be quoted.** If your folder is `/home/you/Desktop/New folder`, the shell will split it into two arguments unless you wrap it in quotes:
> ```bash
> # WRONG — shell splits at the space
> python alexandria_batch_processor.py --folder /home/you/Desktop/New folder --model model.gguf
>
> # RIGHT — quotes keep the path as one argument
> python alexandria_batch_processor.py --folder "/home/you/Desktop/New folder" --model model.gguf
> ```

> **Don't forget the space after `--folder`.** Writing `--/home/...` (no space) makes argparse treat the whole path as an unknown flag name.

## Usage

### Single File (Original Script)
```bash
python alexandria_preparer_rocm_compatible.py \
  --audio audiobook.wav \
  --model Qwen2.5-14B-Instruct-Q6_K.gguf \
  --fallback-model Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf \
  --chunk-size 10.0 \
  --lang en
```

### Multiple Files (Batch Processor)
```bash
python alexandria_batch_processor.py \
  audiobook1.wav audiobook2.wav audiobook3.wav \
  --model Qwen2.5-14B-Instruct-Q6_K.gguf \
  --fallback-model Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf \
  --chunk-size 10.0 \
  --lang en
```

### Folder Scan (Batch Processor)
```bash
python alexandria_batch_processor.py \
  --folder "/path/to/audiobooks" \
  --model Qwen2.5-14B-Instruct-Q6_K.gguf \
  --fallback-model Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf \
  --chunk-size 10.0 \
  --lang en
```
Quote the folder path if it contains spaces.

## Command Line Options

### Positional Arguments
- `audio_files` - One or more audio files to process (required, space-separated)

### Optional Arguments
- `--folder DIR` - Scan a folder for all supported audio files (`.wav .flac .ogg`), sorted alphabetically; can be combined with individually listed files
- `--model PATH` - Path to primary GGUF model file (required, recommended: `Qwen2.5-14B-Instruct-Q6_K.gguf`)
- `--fallback-model PATH` - Optional backup GGUF model if `--model` fails to load (e.g., `Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf`)
- `--chunk-size SIZE` - Target duration per chunk in seconds (default: 10.0)
- `--lang CODE` - Language code for transcription (default: en)
- `--force` - Reprocess files even if `alexandria_dataset_<name>.zip` already exists (default: skip)

### Source-Guided Mode (Optional)

When enabled, the batch processor forwards source-guided chunking flags
to each preparer subprocess. See `PREPARER_GUIDE.md` for what the mode
does in detail; the batch-level summary is: characters' names come out
correctly spelt (from the EPUB), and audio-only passages (credits,
chapter intros) get dropped automatically.

- `--source-folder DIR` - Folder of source `.epub` or `.txt` files. Each
  audio file is matched to a source by basename: `audio/Book1.wav` looks
  for `<source-folder>/Book1.epub` (preferred) or `<source-folder>/Book1.txt`.
  Audio files with no matching source run in legacy ASR-only mode with a
  warning. Mutually exclusive with `--source`.
- `--source PATH` - A single source file applied to **every** audio file
  in the batch. Useful when you're processing multiple recordings of
  the same book (e.g., different narrators). Mutually exclusive with
  `--source-folder`.
- `--source-threshold N` - Minimum alignment ratio to keep a chunk
  (default: 0.65). Forwarded to the preparer's `--source-threshold`.
- `--keep-unaligned` - Don't drop low-confidence chunks; use the ASR
  text instead. Forwarded to the preparer's `--keep-unaligned`.

At the start of the batch the processor prints which files matched and
how many didn't, so you can spot rename issues before the run grinds
through 60 hours of audio:

```
├─ Source-guided: matching from /path/to/books/
│   3 matched, 1 no match (legacy ASR-only for those)
│     no match: 'Audiobook4'
├─ Source threshold: 0.65 (strict-drop)
```

## Pause and Resume

### Pausing mid-run (e.g., to game or free up GPU)

Press **Ctrl+C** at any time to stop the batch processor. The current file's checkpoint in `dataset_temp/` is preserved automatically. When you're ready to continue, just rerun the exact same command — no extra flags needed:

```bash
# Stopped mid-run with Ctrl+C — just rerun the same command to resume
python alexandria_batch_processor.py book1.wav book2.wav book3.wav --model model.gguf
```

The batch processor will:
- **Skip** files whose output zip already exists (completed files)
- **Resume** the interrupted file from its last checkpoint (no repeated work)
- **Process fresh** any files not yet started

## Resume Capability

The batch processor supports **two layers of resume**:

**1. File-level skip** — Files whose output zip (`alexandria_dataset_<name>.zip`) already exists are skipped on subsequent runs.

**2. Mid-file resume** — If a file was interrupted mid-annotation (by Ctrl+C or a crash), the batch processor passes `--resume` to the preparer automatically. The preparer reads its checkpoint and continues from the last completed segment.

The preparer's `.source` marker in `dataset_temp/` ensures we never accidentally resume into a different file's partial work — if the marker doesn't match the current audio file, `dataset_temp/` is wiped and processing starts fresh.

```bash
# First run - processes all 3 files
python alexandria_batch_processor.py book1.wav book2.wav book3.wav --model model.gguf
# (book1 completes, book2 crashes 50h into annotation)

# Second run:
#  - book1.wav skipped (zip exists)
#  - book2.wav resumes from last checkpoint (continues from segment ~4000)
#  - book3.wav processed fresh
python alexandria_batch_processor.py book1.wav book2.wav book3.wav --model model.gguf

# Force reprocess everything (no skip, no resume)
python alexandria_batch_processor.py book1.wav book2.wav book3.wav --model model.gguf --force
```

## Real-Time Output Streaming

The batch processor streams the main script's output **in real-time** so you can monitor:
- Transcription progress (Wav2Vec2 chunks being processed)
- GPU utilization percentage
- Annotation progress with elapsed/remaining time estimates
- Any warnings or errors as they happen

No more waiting hours wondering if processing is stuck!

## Examples

### Process 3 audiobooks
```bash
python alexandria_batch_processor.py \
  book1.wav book2.wav book3.wav \
  --model Qwen2.5-14B-Instruct-Q6_K.gguf \
  --fallback-model Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf
```

### Process with custom chunk size
```bash
python alexandria_batch_processor.py \
  book1.wav book2.wav \
  --model model.gguf \
  --chunk-size 15.0
```

### Process in different language
```bash
python alexandria_batch_processor.py \
  spanish_book.wav french_book.wav \
  --model model.gguf \
  --lang es
```

### Source-guided: batch with one EPUB per audiobook

Each audio file is matched to a sibling source by basename. So
`audio/MyHappyMarriage.wav` pairs with `books/MyHappyMarriage.epub`,
etc. Audio files that don't have a matching source still get processed —
just in legacy ASR-only mode (with a startup warning).

```bash
app/env/bin/python alexandria_batch_processor.py \
  --folder /home/fakemitch/Desktop/audiobooks/ \
  --source-folder /home/fakemitch/Desktop/books/ \
  --model models/Qwen2.5-14B-Instruct-Q6_K.gguf
```

### Source-guided: same EPUB applied to every file in batch

When you have multiple recordings of the same book (different narrators,
different versions), use `--source` instead of `--source-folder` so the
same source file is applied to all of them:

```bash
app/env/bin/python alexandria_batch_processor.py \
  narrator_a.wav narrator_b.wav narrator_c.wav \
  --source "/home/fakemitch/Desktop/books/My Happy Marriage - Volume 01.epub" \
  --model models/Qwen2.5-14B-Instruct-Q6_K.gguf
```

### Source-guided with stricter threshold

If the default 0.65 threshold is dropping passages you want to keep,
loosen it (or use `--keep-unaligned` to never drop):

```bash
# Loosen threshold to 0.55 — keep more borderline chunks
app/env/bin/python alexandria_batch_processor.py \
  book.wav --model model.gguf \
  --source book.epub \
  --source-threshold 0.55

# Or: keep everything (use ASR text for low-confidence chunks)
app/env/bin/python alexandria_batch_processor.py \
  book.wav --model model.gguf \
  --source book.epub \
  --keep-unaligned
```

## Output Files

For each input file, the batch processor creates:
- `alexandria_dataset_<filename>.zip` - Annotated dataset
- Automatically renamed from `alexandria_dataset.zip` to avoid conflicts

Example:
```
Input:  audiobook1.wav
Output: alexandria_dataset_audiobook1.zip

Input:  audiobook2.wav  
Output: alexandria_dataset_audiobook2.zip
```

## Logging

The batch processor creates detailed logs in the `logs/` directory:
- `logs/alexandria_batch_YYYYMMDD_HHMMSS.log` - Full processing log

## Results Summary

After processing completes, a JSON summary is created:
- `batch_results_YYYYMMDD_HHMMSS.json` - Processing results

Contains:
- Individual file results (success/failure)
- Output file paths and sizes
- Processing times for each file
- Total processing time
- Timestamp

Example:
```json
{
  "timestamp": "2026-05-16T16:30:00",
  "total_time_seconds": 14400,
  "results": {
    "succeeded": [
      {
        "file": "book1.wav",
        "output": "alexandria_dataset_book1.zip",
        "output_size_mb": 250.5,
        "time": "2h 15m",
        "time_seconds": 8100
      }
    ],
    "failed": [],
    "skipped": []
  }
}
```

## Processing Flow

1. **Validation** - Checks all files exist and are valid audio formats
2. **Model Check** - Verifies Gemma model is accessible
3. **GPU Monitoring** - Logs GPU memory and utilization at start
4. **Sequential Processing** - Processes each file one by one:
   - Loads audio (16kHz + 24kHz)
   - **Transcribes with Wav2Vec2** (GPU primary, 30s overlapping chunks for context preservation)
     - Falls back to Insanely Fast Whisper if needed
     - Falls back to WhisperX-CPU if needed
   - **Annotates chunks with Gemma** (GPU acceleration)
     - GPU memory and utilization monitored every 10 segments
     - Elapsed time shown in hours/minutes format
     - Remaining time estimated with real-time updates
   - Creates audio segments with annotations
   - Packages as ZIP file
   - Renames output to avoid conflicts
   - GPU stats logged between files
5. **Summary** - Generates processing report with batch statistics

## GPU Monitoring & Progress Tracking

### Real-Time Monitoring
- **GPU Memory**: Tracks allocated/total memory and percentage during:
  - After Wav2Vec2 model loads
  - Before/during Wav2Vec2 chunk processing (every 50 chunks)
  - After Gemma model loads
  - During annotation (every 10 segments)
  - Between files during batch processing

- **GPU Utilization**: Shows actual GPU workload percentage via rocm-smi
  - Helps verify GPU is actually being used
  - Detects false positives where GPU claims to be used but isn't
  - Example: `Utilization: 45.3%`

### Progress Tracking
- **Time Format**: All time estimates shown in hours/minutes format
  - Estimated total: `~60h 13m` instead of just seconds
  - Elapsed time: `Elapsed: 7m 40s`
  - Remaining time: `Remaining: ~60h 12m`
  - Updated every 10 annotation segments

### Example Progress Output
```
GPU Usage (chunk 50/725):
  ├─ Memory: 4.52GB / 16.00GB (28.3%)
  └─ Utilization: 87.5%

Progress: 10/4713 chunks | Elapsed: 7m 40s | Remaining: ~60h 12m
```

## Performance Tips

- **GPU Acceleration** - All transcription and annotation uses GPU with real-time monitoring
- **Sequential Processing** - No parallel overhead, each file gets full GPU access
- **Disk Space** - Ensure ~5-10 GB free space per audiobook
- **Monitor Memory** - GPU cache is cleared between files, memory usage logged

## Time Estimates

Based on ~5.5 hours per 19,500-second audiobook:
- Transcription: ~15-20 minutes (GPU accelerated)
- Annotation: ~4-5 hours (GPU accelerated, ~46s per chunk)

Total = ~4.5-5.5 hours per audiobook

For 3 audiobooks: ~13-16 hours total

## Troubleshooting

### "unrecognized arguments" or "--folder path is not a directory"
```
error: unrecognized arguments: --/home/you/Desktop/New
Error: --folder path is not a directory: /home/you/Desktop/New
```
Almost always a shell quoting issue, not a bug:
- **Missing space:** `--/home/...` should be `--folder /home/...` (with a space between the flag and the path)
- **Unquoted spaces:** wrap any path containing spaces in double quotes: `--folder "/home/you/Desktop/New folder"`
- **Trailing backslash + space:** in multi-line commands, make sure the `\` is the very last character on the line (no trailing spaces)

### File Not Found
```
Error: File not found: audiobook.wav
```
- Check file path is correct
- Use absolute paths if file is in different directory
- Quote any path containing spaces: `"/path with spaces/book.wav"`

### Unsupported Format
```
Warning: Unsupported format: audiobook.aac
```
- Supported formats: `.wav`, `.flac`, `.ogg` (natively via soundfile/libsndfile)
- `.mp3` and `.m4a` are not reliably supported — convert to `.wav` first: `ffmpeg -i input.mp3 output.wav`

### Model Not Found
```
Error: Model file not found
```
- Ensure model path is correct
- Download Gemma-4 model if missing

### GPU Not Available
```
Warning: GPU not available - running on CPU (slower)
```
- Check CUDA/ROCm installation
- Verify GPU drivers are installed
- Single file will take much longer on CPU

### Out of Space
```
Error: No space left on device
```
- Clean up old datasets
- Free up disk space (need ~10GB per file)
- Check `du -sh alexandria_dataset*.zip` for old files

### Datasets look complete but cover only a tiny fraction of each audiobook
If your `.wav` files are larger than ~4 GiB each and you see preparer
summaries like "Total audio in dataset: 1.5 min" on what should be a
multi-hour book, you've hit the >4 GiB WAV header-wrap bug — the WAV
data-chunk-size field is 32-bit and silently truncates everything past
the wrap. The preparer detects this automatically now and routes the
load through ffmpeg, but you should re-run any datasets that were built
with the old loader. See PREPARER_GUIDE.md → "Oversized WAV" for details.

## Advanced Usage

### Process all files in a folder
```bash
python alexandria_batch_processor.py \
  --folder /path/to/audiobooks \
  --model model.gguf
```

Files are picked up alphabetically. Supported formats: `.wav .flac .ogg` — these are natively supported by soundfile/libsndfile. Convert `.mp3` or `.m4a` files to `.wav` first (e.g., `ffmpeg -i input.mp3 output.wav`).

You can also mix a folder with individually listed files:
```bash
python alexandria_batch_processor.py extra.wav \
  --folder /path/to/audiobooks \
  --model model.gguf
```

### Process with Error Recovery
The batch processor automatically:
- Validates all files before starting
- Continues if one file fails
- Reports which files succeeded/failed
- Saves results to JSON

### Monitor Progress
```bash
# In another terminal
tail -f logs/alexandria_batch_*.log
```

## Development Notes

The batch processor:
- Wraps `alexandria_preparer_rocm_compatible.py`
- Handles file naming conflicts
- Tracks timing and results
- Logs all operations
- Cleans up temporary files

To modify behavior, edit `alexandria_batch_processor.py` directly.
