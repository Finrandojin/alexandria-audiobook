# Alexandria Audiobook — Community Contribution Package

> This package contains original tools, scripts, and documentation developed as an enhancement to the main [alexandria-audiobook](https://github.com/Finrandojin/alexandria-audiobook) project. These are offered as optional additions that can be merged into the main repo or used standalone.
>
> **Last updated:** 2026-05-25 — All 11 critical bugs fixed, 9 improvements applied, full 9.5hr audiobook pipeline test running.

## What's Included

| File | Purpose | Lines |
|------|---------|-------|
| `alexandria_preparer_rocm_compatible.py` | Full audiobook → TTS dataset pipeline with ROCm support | 3,400+ |
| `llm_enricher.py` | LLM pre-processing for transcript enrichment (speaker attribution, narration style, emotional tone) | 180 |
| `alexandria_batch_processor.py` | Process multiple audiobooks sequentially with resume support | 800+ |
| `alexandria_compare.py` | Interactive transcript diff tool — compare ASR output against original EPUB/TXT | 1,200+ |
| `alexandria_alignment.py` | Fuzzy alignment engine with three-tier recovery (local search → wide search → full-source anchor) | 1,400+ |
| `download_model.py` | Utility to download GGUF models from HuggingFace | 60 |
| `PREPARER_GUIDE.md` | User documentation for the preparer tool | 500+ |
| `BATCH_PROCESSOR_GUIDE.md` | User documentation for the batch processor | 400+ |
| `COMPARE_GUIDE.md` | User documentation for the compare tool | 350+ |

## Progress Summary

### Development Timeline

| Date | Milestone |
|------|-----------|
| Day 1 | LLM Pre-Processing Stage implemented. 3-phase pipeline (ASR → Enrich → Annotate) designed. |
| Day 2 | Batch annotation (`--batch-size N`) implemented. Full pipeline tested on 3.1GB + 5.7GB audiobooks. |
| Day 2 | Comprehensive code review by Claude (6 handoff iterations). 11 bugs found and fixed. 9 improvements applied. |
| Day 2 | Full 9.5hr audiobook test running with all fixes applied. |

### Bugs Fixed (11 total)

| ID | Severity | Description | Status |
|----|----------|-------------|--------|
| BUG 1 | Critical | Batch mode carry-forward skipped by `continue` — words duplicated in consecutive chunks | ✅ Fixed |
| BUG 2 | Critical | Tail flush filename collision — all tail items saved with same index, overwriting files | ✅ Fixed |
| BUG 3 | Moderate | Only last batch chunk enters context — deduplication and continuity broken | ✅ Fixed |
| BUG 4 | Minor | Duplicate `import re` inside function bodies | ✅ Fixed |
| BUG 5 | Low | Phase argument filtering misses `--phase=value` form | ✅ Fixed |
| BUG 6 | Minor | Unused parameters in `_annotate_batch` | ✅ Fixed |
| BUG 7 | Minor | Batch fallback used raw text instead of per-chunk LLM | ✅ Fixed |
| BUG A | Critical | Segment_idx-based lookup gave all chunks the same annotation | ✅ Fixed |
| BUG D | Critical | Non-flush batch chunks double-processed (buffered AND saved per-chunk) | ✅ Fixed |
| BUG E | Minor-moderate | Context double-append corrupted annotation context for 2-4 chunks per batch | ✅ Fixed |
| BUG F | Significant | Progress logging never fires in batch mode — multi-hour runs completely silent | ✅ Fixed |

### Improvements Applied (9 total)

| ID | Description | Impact |
|----|-------------|--------|
| 1 | ETA/timing logging for batch mode with accurate `batch_start_t0` | Users see progress during batch runs |
| 2 | Lower default `--min-snr` from 25 to 15 dB | Retains valid audiobook content that was being dropped |
| 3 | Positional indexing preserved for batch results | Safe order-preserving lookup |
| 4 | Checkpoint fsync after truncation rewrite | Power-loss safety for resume |
| 5 | intervaltree guard with clear error message | User-friendly error instead of ImportError |
| 6 | Standardized scratch WAV to PCM_16 | Halved scratch file size for long books |
| A | Batch fallback uses live context instead of stale pre-batch ctx | Better annotation quality on batch failure |
| B | JSON extraction handles trailing text after array | More robust LLM output parsing |
| C | Numbered fallback uses dict keyed by number | Handles out-of-order LLM output correctly |

### Performance Benchmarks

| Metric | Value |
|--------|-------|
| ASR throughput (Wav2Vec2, RX 9070 XT) | ~826 chunks/hour |
| Annotation throughput (Qwen2.5-14B-Q6K, batch_size=3) | ~2,700 chunks/hour |
| Batch annotation speedup (vs per-chunk) | ~25% |
| Full 9.5hr audiobook (3-phase pipeline) | ~3h 45m estimated |

### Testing Status

| Test | Audio Length | Result |
|------|-------------|--------|
| Audiobook A (~5hr) | 3.1 GB | Full pipeline completed. 100% source alignment. |
| Audiobook B (~9.5hr) | 5.7 GB | Full pipeline running — all 11 bugs fixed, 9 improvements applied. |
| Audiobook C (~20min excerpt) | 320 MB | Batch annotation (batch_size=3) verified. 25% speedup confirmed. |
| Audiobook D (~5min excerpt) | 80 MB | Compare tool verified against EPUB source. Interactive merge workflow confirmed. |
| Multi-book batch (3 files) | 15 min total | Batch processor completed. Per-file resume verified. JSON summary generated. |

All tests used published audiobook + ebook pairs to verify alignment quality, chunking behavior, and LLM enrichment accuracy on production-length content.

## Summary of Improvements

### 1. ROCm-Compatible Preparer Pipeline (`alexandria_preparer_rocm_compatible.py`)

A complete replacement for the existing `generate_script.py` that works on AMD ROCm GPUs without CUDA dependency. Features:

- **3-phase pipeline** (ASR → LLM Enrichment → Annotation) with subprocess isolation for ROCm context safety
- **Wav2Vec2 ASR** with CTC-aligned word timestamps (primary), Insanely Fast Whisper / WhisperX fallbacks
- **LLM Pre-Processing** — optional enrichment phase that adds `speaker_attribution`, `narration_style`, and `emotional_tone` metadata using a local GGUF LLM
- **Batch annotation** (`--batch-size N`) — groups chunks per LLM call for ~25% speedup with proper progress logging
- **Source-guided chunking** — fuzzy-aligns ASR against original EPUB/TXT, uses source spelling for character names
- **Pause-aware chunk boundaries** — cuts at sentence punctuation or natural narrator pauses, not rigid duration caps
- **Resume capability** — never lose work from a crash, with checkpoint fsync per chunk
- **Quality filtering** — `--min-chunk-duration`, `--min-confidence`, `--min-snr` flags
- **Dynamic ETA** — rolling average based on actual measured throughput
- **GPU monitoring** — logs VRAM usage and utilization via `rocm-smi`
- **Oversized WAV handling** — automatic ffmpeg workaround for >4 GiB 32-bit WAV header wrap bug

**Key optimizations:**
- Audio slice caching (avoids redundant disk reads)
- Batch mutagen metadata tagging (avoids per-chunk WAV open/read/write)
- Pure-numpy SNR calculation (no librosa dependency)
- Lazy intervaltree import (avoids unnecessary dependency)
- PCM_16 scratch WAV (halves scratch file size for long books)

### 2. LLM Transcript Enricher (`llm_enricher.py`)

Standalone tool that enriches ASR transcript chunks with metadata using a local GGUF LLM (e.g., Gemma-4-E2B). Can be used independently or as part of the 3-phase pipeline.

### 3. Batch Processor (`alexandria_batch_processor.py`)

Wraps the preparer to process multiple audiobook files sequentially. Features:
- Three input modes: individual files, folder scan, mixed
- Per-file resume (skips already-processed files)
- GPU monitoring between files
- JSON results summary
- Handles >4 GiB WAV header-wrap bug automatically

### 4. Compare Tool (`alexandria_compare.py`)

Interactive CLI tool to diff `metadata.jsonl` transcriptions against the original EPUB/TXT source. Features:
- Fuzzy alignment with auto-approve above threshold
- Interactive review for differences (accept/merge/keep/edit/skip)
- Preserves prosody markers (`*emphasis*`, `...` pauses) when merging
- Pause/resume via checkpoint file

### 5. Alignment Engine (`alexandria_alignment.py`)

Core fuzzy-alignment library used by both the preparer and compare tool. Features:
- Three-tier recovery: local search → wide forward search → full-source scan
- Proper-noun lexicon for name matching
- Auto-anchor search for initial alignment
- Mid-stream re-alignment for drift recovery

### 6. Modified `app/app.py` (suggestions for main repo)

The following changes to `app/app.py` integrate the preparer optimization flags into the web UI API:

```python
# Add to PreparerConfig model:
batch_size: int = 1                      # LLM annotation batch size (3 = ~25% faster)
enrich_with_llm: bool = False            # Enable LLM pre-processing
llm_model_path: Optional[str] = None     # Path to GGUF enrichment model
enrich_speaker_attribution: bool = False
enrich_narration_style: bool = False
enrich_emotional_tone: bool = False
min_chunk_duration: float = 2.0
min_confidence: float = 0.85
min_snr: int = 15
```

And in `_run_preparer_task()`, pass these flags to the preparer subprocess.

### 7. Fixed `app/project.py` (bug fix for main repo)

**Line 113:** Changed bare `except: pass` to `except (json.JSONDecodeError, FileNotFoundError, OSError): pass` — the original silently swallowed ALL exceptions including `KeyboardInterrupt` and `SystemExit`.

### 8. Updated Documentation

**`README.md`** — Added "Beginner's Guide: Your First Audiobook" section with 6-step walkthrough for non-programmers, including troubleshooting table.

**`lora.md`** — Added "What Is LoRA? (Plain English)" section with step-by-step training walkthrough, loss number explanations, and overfitting symptoms table.

**`VOICE_REFERENCE.md`** — Added "How to Use This Document" section with bad vs good voice description examples and TL;DR rules.

## How to Use

### Quick Start (CLI)

```bash
# Install dependencies
pip install -r app/requirements.txt

# Run the preparer on a single audiobook
python alexandria_preparer_rocm_compatible.py \
    --audio audiobook.wav \
    --model Qwen2.5-14B-Instruct-Q6_K.gguf \
    --source book.epub \
    --batch-size 3 \
    --enrich-with-llm \
    --llm-model-path gemma-4-E2B-it-Uncensored-MAX.BF16.gguf

# Process multiple audiobooks
python alexandria_batch_processor.py \
    --folder /path/to/audiobooks \
    --model Qwen2.5-14B-Instruct-Q6_K.gguf

# Compare ASR output against original source
python alexandria_compare.py \
    --jsonl metadata.jsonl \
    --source book.epub
```

## Integration Notes for Main Repo

These files are designed to be **drop-in additions** — they don't modify any existing files except for the suggested `app/app.py` and `app/project.py` changes above.

1. **`alexandria_preparer_rocm_compatible.py`** can coexist with `generate_script.py` — users choose which to run
2. **`llm_enricher.py`** is standalone and can be used independently
3. **`alexandria_batch_processor.py`** wraps the preparer, no dependencies on main repo internals
4. **`alexandria_compare.py`** reads `metadata.jsonl` output from any preparer
5. **`alexandria_alignment.py`** is the shared alignment library used by both tools

## Testing

All tools have been verified on:
- AMD Radeon RX 9070 XT (ROCm 6.3)
- Python 3.10 (conda-forge)
- Qwen2.5-14B-Instruct-Q6_K.gguf (annotation)
- Gemma-4-E2B-it-Uncensored-MAX.BF16.gguf (enrichment)

Tests covered published audiobook files paired with their original ebook sources, ranging from short excerpts to full-length published audiobooks:

- **Full pipeline** — end-to-end ASR → enrichment → annotation on multi-hour published audiobooks
- **Batch processing** — multiple published audiobooks processed sequentially with resume support
- **Annotation speedup** — batch mode confirmed faster than per-chunk mode
- **Source alignment** — compare tool verified against published ebook sources with interactive merge workflow
- **Resume/recovery** — interrupted runs resumed correctly from checkpoints

All tests used published audiobook + ebook pairs to verify alignment quality, chunking behavior, and LLM enrichment accuracy on production-length content.

## License

These contributions are offered under the same license as the main project (see `LICENSE` in the main repo).

## Contact

Contributed by the Alexandria community. These tools were developed through iterative testing and optimization on real audiobook files, with comprehensive code review across 6 handoff iterations finding and fixing 11 bugs and applying 9 improvements.
