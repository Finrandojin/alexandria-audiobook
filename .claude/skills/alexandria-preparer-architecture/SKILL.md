---
name: alexandria-preparer-architecture
description: High-level map of alexandria_preparer_rocm_compatible.py — what each pipeline phase does, which functions implement it, and where state files live. Use this BEFORE reading the preparer file for any edit, debug, or code question, so you can jump straight to the relevant 50–200 lines instead of scanning all ~2,000+ lines. Triggers when the user asks to modify, debug, explain, or extend the preparer, or asks about the dataset-prep pipeline, ASR, alignment, annotation, chunking, resume, or scratch directory.
---

# Alexandria preparer architecture

`alexandria_preparer_rocm_compatible.py` is the dataset-prep pipeline. It takes one audiobook WAV + one EPUB/TXT source and emits a directory of aligned `(audio_chunk.wav, annotated_text)` pairs ready for TTS dataset use. The file is ~2,000+ lines; this skill is the map.

## Pipeline phases

Read top-to-bottom — each phase consumes the previous phase's output.

**Orchestration mode** (`args.phase is None`): the main function launches three subprocesses sequentially for ROCm isolation:

| # | Phase | Script | Entry point | Notes |
|---|---|---|---|---|
| 1 | Validate + WAV-wrap detection | same | `validate_inputs()` → `_wav_overflow_info()` | Computes true duration from filesize; flags 32-bit chunk-size wrap on >4 GiB WAVs |
| 2 | Load audio (24 kHz + 16 kHz) | same | `main()` load step | If oversized: ffmpeg → scratch WAV (24k) + ffmpeg → numpy (16k). Otherwise: librosa.load |
| 3 | ASR transcription | same | `choose_and_transcribe()` → one of `transcribe_with_*()` | Wav2Vec2 primary (CTC-aligned word timestamps), insanely-fast-whisper / whisperx fallbacks |
| 4 | **LLM Enrichment (optional)** | `llm_enricher.py` | `LLMEnricher.enrich_transcript_chunk()` | Groups ASR words into 10s chunks, runs small GGUF LLM to add `speaker_attribution`, `narration_style`, `emotional_tone` |
| 5 | Source loading + tokenization | same | `_build_source_state()` → `_COMPOUND_SPLIT`, `normalize()` | Loads EPUB/TXT, expands compounds, normalizes for alignment |
| 6 | Source-guided chunking | same | `_find_best_cut()` + `_provisional_entries_for_anchor()` | Finds natural sentence breaks near the ASR's word-timed segments |
| 7 | Multi-tier alignment recovery | same | `find_best_match` → `realign` → `find_anchor_position` | Drift-resistant: fuzzy, then local search, then full-source re-anchor |
| 8 | LLM prosody annotation | same | `_load_llm()` → `annotate_chunks()` | Qwen 2.5 14B Q6_K primary, Gemma fallback. `_sanitize_annotation()` cleans output |
| 9 | Write outputs | same | end of `annotate_chunks()` and `main()` | Atomic write to scratch, then promote to dataset folder |

Use `grep -nE "^def |^class " alexandria_preparer_rocm_compatible.py` to get current line numbers — the file evolves, so don't trust hardcoded ones in this skill.

## Key helpers (where to grep for them)

- **`_wav_overflow_info(path)`** — true_dur vs header_dur; returns `is_oversized` (filesize > 2³² AND true > header × 1.5)
- **`_ffmpeg_decode_to_wav(src, dst, sr, mono)`** — ffmpeg `-ignore_length 1 -c:a pcm_s16le` streamed to scratch WAV. Used at 24 kHz to avoid materializing the float32 array in RAM
- **`_ffmpeg_decode_to_numpy(src, sr, mono)`** — pipes s16le PCM into numpy float32 [-1, 1]. Used at 16 kHz for ASR input
- **`_load_existing_checkpoint(temp_dir)`** + **`_check_source_marker` / `_write_source_marker`** — durable resume. Source marker prevents cross-file `dataset_temp/` corruption
- **`_wipe_temp_dir(temp_dir)`** — only runs when source marker mismatch confirmed
- **`_sanitize_annotation(text)`** — strips angle-brackets, normalizes whitespace; runs on every LLM response before write
- **`_find_best_cut()`** — chunk-boundary picker; respects word boundaries from ASR's word-timed segments

## State and scratch files

- **`dataset_temp/`** — scratch dir; `sample_NNNN.wav` chunks + partial `metadata.jsonl`. Gitignored. Source marker (`.source.txt`) keeps cross-file corruption from happening
- **`dataset_temp/asr_segments.json`** — Wav2Vec2 output: `{word_segments: [...], detected_lang: str}`. Protected from `_wipe_temp_dir`.
- **`dataset_temp/enriched_segments.json`** — LLM enrichment output: chunks with `speaker_attribution`, `narration_style`, `emotional_tone` fields. Protected from `_wipe_temp_dir`.
- **`dataset_temp/asr_chunks_for_enrich.json`** — 10s chunked input for `llm_enricher.py`. Protected from `_wipe_temp_dir`.
- **`dataset_temp/metadata.jsonl`** — partial output; entries appended atomically per chunk
- **Final output** — `<book>-converted/` next to source WAV: `metadata.jsonl` + `sample_NNNN.wav` files
- **Checkpoint** — implicit; partial `metadata.jsonl` + sample count is the resume cursor

## Resume behavior

Re-running on the same audio source reuses scratch:
- Source marker matches → continue from `len(samples) → next_chunk_idx`
- Source marker mismatches → wipe scratch, start fresh (fixes the cross-file `dataset_temp/` corruption bug from prior incidents)
- No checkpoint file is loaded for partial chunks mid-annotation; the smallest unit of resume is one complete chunk

## Common edit hotspots

When the user asks for a fix in one of these areas, jump straight to the named function — don't read the whole file:

| User says | Function to read |
|---|---|
| "loader is reading 0 audio" / "duration looks wrong" | `_wav_overflow_info`, then load step in `main()` |
| "ASR is wrong / slow / picking the wrong backend" | `choose_and_transcribe()` |
| "alignment drifted" / "wrong chunk boundary" | `_find_best_cut`, then `find_best_match`/`realign`/`find_anchor_position` |
| "LLM annotation is empty / weird" | `_sanitize_annotation`, then `annotate_chunks()` |
| "resume isn't picking up" / "scratch was wiped" | `_check_source_marker`, `_load_existing_checkpoint`, `_wipe_temp_dir` |
| "ETA in the progress line is wrong" | `ProgressTracker` class (~line 71) |

## Companions

- **`alexandria_batch_processor.py`** — drives the preparer over a list of (audio, source) pairs from `pairs.json`. Writes `batch_results_<ts>.json` summary
- **`alexandria_compare.py`** — post-hoc review tool that aligns the preparer's `metadata.jsonl` against source. Has its own skill: `alexandria-compare-review`
- **`build_test_corpus.sh`** + **`run_subset.sh`** + **`run_with_restart.sh`** — shell drivers for long-running multi-book runs

## Anti-patterns

- **Don't read the whole file** to answer a phase-specific question. Use the table above to jump.
- **Don't trust line numbers in this skill.** Run `grep -n` first.
- **Don't propose changes to the WAV-wrap path** without checking `_wav_overflow_info` and the load-step branch in `main()` together — the two are coupled.
- **Don't touch `dataset_temp/` from outside the preparer.** The source marker + per-chunk atomic write are the resume contract.
