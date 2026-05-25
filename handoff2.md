# Handoff: Alexandria Preparer Code Review

**Project:** `/home/fakemitch/pinokio/api/alexandria-audiobook2.git`  
**File under review:** `alexandria_preparer_rocm_compatible.py` (3314 lines)  
**Date:** 2026-05-25  
**Prepared by:** Claude Sonnet 4.6 for follow-up agent (Qwen or Gemini)

---

## What this file does

`alexandria_preparer_rocm_compatible.py` is the master audiobook dataset preparation pipeline. It runs in three sequential subprocess-isolated phases to avoid ROCm HIP context conflicts between PyTorch (Wav2Vec2 ASR) and llama-cpp-python (GGUF LLM annotation):

1. **ASR phase** — loads audio, resamples, transcribes to word-level timestamps, saves `dataset_temp/asr_segments.json`
2. **Enrich phase** (optional) — passes ASR chunks to `llm_enricher.py` for speaker/tone attribution
3. **Annotate phase** — chunks the word segments by duration+pause heuristics, runs each chunk through a GGUF LLM for TTS annotation (`*emphasis*`, `...` pauses), saves `sample_NNNN.wav` + `metadata.jsonl`, then zips into a dataset

Key features: resume/checkpoint via `dataset_temp/`, source-guided alignment against EPUB text, oversized WAV support (>4 GiB), SNR/confidence quality filtering, deduplication, speaker diarization.

---

## CRITICAL BUGS (must fix before trusting batch mode output)

### BUG 1 — Batch mode: tail carry-forward is silently skipped  
**Severity: Critical. Produces duplicated/scrambled chunk content.**

When `--batch-size N > 1` and a full batch is ready to flush, the code at line 2243 does:
```python
continue  # Skip the per-chunk code below
```

This `continue` exits the inner `for idx, word_data in enumerate(word_segments):` iteration. The problem is that the "carry-forward" block at lines 2334–2340, which trims `current_words` to only the tail after the cut point, is **inside** the same `if is_final or is_good_break or duration >= max_dur:` block and is **skipped** by the `continue`.

```python
# Lines 2334-2340 — SKIPPED when batch continue fires
current_words       = current_words[cut_at + 1:]
current_word_starts = current_word_starts[cut_at + 1:]
current_word_ends   = current_word_ends[cut_at + 1:]
current_start = current_word_starts[0] if current_word_starts else chunk_end_time
```

**Effect:** After every full batch flush, `current_words` retains all words including the ones just emitted. The next chunk will be built on top of them, causing words to appear in two consecutive chunks.

**Fix:** Move the carry-forward block outside the `if chunk_words and chunk_duration >= 1.0 and not drop_chunk:` guard (but still inside `if is_final or is_good_break or duration >= max_dur:`), so it executes even when `continue` is taken:

```python
if is_final or is_good_break or duration >= max_dur:
    ...
    cut_at, cut_strategy = ...
    
    # (all chunk processing — batch buffer, per-chunk LLM, etc.)
    
    # ← carry-forward MUST be here, unconditionally, even for batched chunks
    current_words       = current_words[cut_at + 1:]
    current_word_starts = current_word_starts[cut_at + 1:]
    current_word_ends   = current_word_ends[cut_at + 1:]
    current_start = current_word_starts[0] if current_word_starts else chunk_end_time
```

---

### BUG 2 — Batch mode tail flush: all items get the same segment index → filename collision  
**Severity: Critical. Overwrites WAV files. Only affects the last partial batch.**

Each item added to `batch_buffer` stores `segment_idx` at append time (line 2206):
```python
batch_buffer.append({
    "segment_idx": segment_idx,
    ...
})
```

But `segment_idx` is only incremented by `+= batch_size` **after** a full batch completes. So all items accumulated in a partial tail batch share the same `segment_idx` value.

**Example with batch_size=3, 5 chunks total:**
- Chunks 0,1,2 → full batch → saved as 0,1,2 → `segment_idx` becomes 3
- Chunk 3: appended with `segment_idx=3`
- Chunk 4: appended with `segment_idx=3` ← same as chunk 3!
- Tail flush (lines 2351-2361): both items use `idx = item["segment_idx"]` = 3
  → both try to write `sample_0003.wav` → second write silently overwrites first

**Fix the tail flush** (lines 2351–2361):
```python
# BEFORE (broken):
for i, item in enumerate(batch_buffer):
    idx = item["segment_idx"]
    _save_chunk_metadata(item, annotated, ..., idx, temp_dir)

# AFTER (correct):
for i, item in enumerate(batch_buffer):
    _save_chunk_metadata(item, annotated, ..., segment_idx + i, temp_dir)
segment_idx += len(batch_buffer)
```

---

### BUG 3 — Batch mode: only the last chunk's text enters context/deduplication  
**Severity: Moderate. Context continuity breaks across batch boundaries.**

After processing a full batch (line 2240–2241):
```python
prev_raw_text = text   # ← only the LAST chunk in the batch
context.append(text)   # ← only the LAST chunk
```

For a batch of 4 chunks, chunks 1–3 are never added to `context` and never used as `prev_raw_text`. This means:
- The LLM prompt's "Previous context" window loses 3 chunks of continuity per batch
- The deduplication check (`similarity > 0.85`) only compares the next chunk against the last chunk of the previous batch, not the last-of-the-last-batch

**Fix:**
```python
# After the batch save loop, update context with all batch items:
for item in batch_buffer:
    context.append(item["text"])
prev_raw_text = batch_buffer[-1]["text"]
```

---

## MINOR BUGS

### BUG 4 — `import re` inside function bodies (lines 815, 1631)
Both `transcribe_with_insanely_fast_whisper` (line 815) and `_annotate_batch` (line 1631) do `import re` or `import re as _re` inside the function body, even though `re` is imported at module level (line 44). Harmless (Python caches imports), but messy — remove the local imports.

### BUG 5 — Phase argument forwarding doesn't handle `--phase=value` form  
Lines 2882–2883, 2906–2907, 2917–2918 filter out phase tokens with:
```python
if arg not in ["--phase", "asr", "enrich", "annotate"]:
```
This misses `--phase=asr` as a single token. argparse itself normalizes to `--phase asr` (two tokens) when called from Python, so this only matters if the user manually passes `--phase=asr` on the shell command line. Low risk, but worth a `arg.startswith("--phase=")` guard.

### BUG 6 — Unused `segment_idx_start` parameter in `_annotate_batch` (line 1564)  
The function signature is:
```python
def _annotate_batch(llm, batch_data, source_words_list, alignment, batch_size, timing, stats, segment_idx_start):
```
`segment_idx_start` is never read inside the function body. `source_words_list` is also always passed as `None` and never used. Both parameters should be removed.

### BUG 7 — Batch fallback uses raw text instead of per-chunk LLM annotation  
When `_annotate_batch` returns `None` (JSON parse failure), the caller (lines 2226–2229) falls back to `item["text"]` (unannotated ASR text):
```python
else:
    annotated = item["text"]  # Fallback
```
The `_annotate_batch` docstring says "Falls back to per-chunk annotation if batch parsing fails" but the actual implementation skips the LLM entirely on failure. The annotation is silently dropped. Consider actually running per-chunk LLM calls on failure rather than using bare text.

---

## IMPROVEMENTS (non-bug quality items)

### IMPROVEMENT 1 — ETA/timing logs skip entirely in batch mode  
The `chunk_times.append(...)` and ETA logging at lines 2296–2328 are inside the per-chunk path. Batch mode's `continue` skips them. Add equivalent timing after batch processing so the user can see ETA during long batch runs.

### IMPROVEMENT 2 — SNR floor of 25 dB is too aggressive for audiobooks  
The default `--min-snr 25` (line 2779) drops any chunk below 25 dB SNR. A typical professionally-recorded audiobook at 44.1 kHz has SNR ~30–45 dB in the vocal passages, but background music beds, chapter intros with ambience, or narrator recordings at home studios can easily measure 18–24 dB. Dropping these loses real content. Suggest defaulting to 15 or adding `--no-snr-filter` flag.

### IMPROVEMENT 3 — `_annotate_batch` returns `(segment_idx, annotated)` tuples but main loop indexes by position  
The batch result is a list of `(segment_idx, annotated)` tuples (line 1657), but the caller at line 2227 accesses them as `batch_results[i][1]` (pure positional). If `_annotate_batch` returns fewer results than `batch_data` items (partial success), the mapping breaks. Either always return exactly `len(batch_data)` entries, or use the `segment_idx` key to look up results.

### IMPROVEMENT 4 — Checkpoint rewrite on truncation doesn't fsync  
Lines 1386–1388 rewrite the checkpoint after finding a corrupt line:
```python
with open(checkpoint_path, "w", encoding="utf-8") as f:
    f.writelines(good_lines)
```
This write is not followed by `os.fsync()`. A power loss immediately after could leave the checkpoint in a partially-written state on the next resume. Add:
```python
f.flush()
os.fsync(f.fileno())
```

### IMPROVEMENT 5 — `_assign_speakers_to_words` doesn't guard against missing `intervaltree`  
If `INTERVALTREE_AVAILABLE = False` but `--diarize` is used, `_assign_speakers_to_words` calls `_lazy_import_intervaltree()` (line 2611), which will raise `ImportError` without a user-friendly message. Should check `INTERVALTREE_AVAILABLE` at call site and raise a clear error.

### IMPROVEMENT 6 — The ASR phase scratch WAV is always float32 (large), annotation phase may rewrite as PCM_16  
Line 3003: `sf.write(audio_24k_scratch, audio_24k, 24000, subtype="FLOAT")` writes float32.  
Line 3184: `sf.write(audio_24k_path, audio_24k, 24000, subtype="PCM_16")` writes int16 (during scratch recreation).  
This inconsistency means the WAV format changes if the annotation phase has to recreate the scratch file. `_read_audio_segment` reads it back with `dtype="float32"` either way (soundfile handles the conversion), so no data loss, but it's confusing. Standardize on `PCM_16` to halve the scratch file size for long books.

---

## CODE STRUCTURE NOTES FOR EDITOR

- **Lines 1–200**: imports, logging setup, lazy-import helpers, WAV overflow detection
- **Lines 200–500**: GPU stats, ASR methods (WhisperX, Wav2Vec2, Insanely Fast Whisper)
- **Lines 982–1050**: `choose_and_transcribe` — ASR method selection with fallback chain
- **Lines 1050–1230**: annotation sanitization, chunk boundary heuristics
- **Lines 1230–1530**: source state/alignment, audio segment reader, checkpoint I/O
- **Lines 1530–1760**: LLM loading, annotation system prompts, `_annotate_batch`, `_save_chunk_metadata`
- **Lines 1760–2520**: `annotate_chunks` — main chunking/annotation loop (the core of the pipeline)
- **Lines 2520–2760**: output naming, ZIP dataset creation with train/val split
- **Lines 2764–3314**: `main()` — argument parsing, phase orchestration, phase execution

The three-phase subprocess isolation (lines 2866–2939) is important: PyTorch's ROCm context and llama-cpp's ggml_cuda context conflict if initialized in the same process. The orchestrator spawns ASR, enrich, and annotate as child processes.

---

## TESTING SUGGESTIONS

To verify the batch mode bugs before fixing:
```bash
# Run with batch-size=3 and limit=7 (2 full batches + 1 tail)
python alexandria_preparer_rocm_compatible.py \
  --audio test.wav --model model.gguf \
  --batch-size 3 --limit 7 --chunk-size 10

# Check: are sample_0003.wav and sample_0004.wav DIFFERENT files?
# Check: does dataset_temp/metadata.jsonl have 7 unique entries?
# Check: are there any duplicate audio_filepath values?
python -c "
import json
entries = [json.loads(l) for l in open('dataset_temp/metadata.jsonl')]
names = [e['audio_filepath'] for e in entries]
dupes = [n for n in names if names.count(n) > 1]
print(f'{len(entries)} entries, {len(dupes)} duplicates: {set(dupes)}')
"
```

---

## FILE SUMMARY

| File | Role |
|------|------|
| `alexandria_preparer_rocm_compatible.py` | Main pipeline (this review) |
| `alexandria_alignment.py` | Fuzzy source alignment, EPUB loading, `find_best_match`, `realign`, `find_anchor_position` |
| `alexandria_compare.py` | Manual review tool for comparing ASR vs source text; produces `metadata_corrected_review_log.jsonl` |
| `llm_enricher.py` | LLM-based transcript enrichment (speaker attribution, tone) |
| `alexandria_batch_processor.py` | Batch processing utilities |
| `prepare_dataset.py` | Dataset preparation utilities |

The `alexandria_alignment` module is imported at line 52 and used extensively in the annotate phase for source-guided chunking. Its functions (`find_best_match`, `realign`, `find_anchor_position`, `merge_annotations_with_source`) are called from inside `annotate_chunks`.
