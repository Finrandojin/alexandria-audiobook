# Handoff 4: Alexandria Preparer — Post-handoff3 Verification

**Project:** `/home/fakemitch/pinokio/api/alexandria-audiobook2.git`  
**File under review:** `alexandria_preparer_rocm_compatible.py` (3404 lines)  
**Date:** 2026-05-25  
**Prepared by:** Claude Sonnet 4.6 — verifying Claude's commit `ea8449d` (handoff3 fixes)

---

## Handoff3 Fixes: All Confirmed Correct

| handoff3 item | Status |
|---|---|
| BUG A — positional indexing restored | ✓ Fixed: `batch_results[i][1]` at line 2231 |
| BUG B — tail segment_idx not incremented | ✓ Fixed: `tail_count = len(batch_buffer)` + `segment_idx += tail_count` after clear |
| BUG C — tail single-item gets raw text | ✓ Fixed: per-chunk LLM fallback added in tail flush |
| BUG 5 — `--phase=value` filter | ✓ Fixed: all 3 occurrences (asr/enrich/annotate phase forwarding) |

---

## CRITICAL NEW BUG (pre-existing since batch mode added, not introduced by ea8449d)

### BUG D — Non-flush batch chunks fall through to per-chunk code

**Severity: Critical. In batch mode, every chunk that doesn't trigger a full flush is ALSO processed by the per-chunk path. Each such chunk is saved twice with different indices.**

**Location: Lines 2204–2302 (structural indentation issue)**

**Root cause:**

The `if batch_size > 1:` block is at indent 20 inside `if chunk_words... and not drop_chunk:` (indent 16). After the `if batch_size > 1:` block ends, the per-chunk code begins at indent 20 — a **sibling**, not an else-branch:

```python
if batch_size > 1:          # indent 20
    batch_buffer.append({}) # indent 24
    if len(batch_buffer) >= batch_size:  # indent 24
        # ... full flush ...            # indent 28
        continue             # indent 28 → skips per-chunk only for flush trigger

# PER-CHUNK CODE — indent 20 (always runs unless continue was hit!)
user_prompt = ...            # indent 20
...
_save_chunk_metadata(...)    # indent 20
segment_idx += 1             # indent 20
context.append(text)         # indent 20
```

**The `continue` at line 2302 (indent 28) only fires for the last chunk of a full batch.** For chunks 0 through batch_size−2, the batch-not-full branch hits no `continue` and falls straight through to the per-chunk code.

**Effect for batch_size=4 with chunks A,B,C,D:**

| Chunk | batch_buffer | Action |
|---|---|---|
| A | [A] | buffered + per-chunk save → sample_0000.wav, segment_idx → 1 |
| B | [A,B] | buffered + per-chunk save → sample_0001.wav, segment_idx → 2 |
| C | [A,B,C] | buffered + per-chunk save → sample_0002.wav, segment_idx → 3 |
| D | [A,B,C,D] | FULL FLUSH: batch saves A→sample_0003, B→sample_0004, C→sample_0005, D→sample_0006; segment_idx → 7 |

**Result:** 7 metadata entries + 7 WAV files for 4 input chunks. A/B/C appear twice with different annotations and indices. Batch mode is NOT faster than per-chunk — it runs (batch_size−1) extra per-chunk LLM calls.

**This bug existed in commit `8038c48` (first clean batch implementation) and in all subsequent commits including ea8449d.**

**Fix — inside `if batch_size > 1:` (indent 24), after the `if len(batch_buffer) >= batch_size:` block, add:**

```python
                        # Batch not full yet — carry forward and skip per-chunk
                        prev_raw_text = text
                        context.append(text)
                        current_words       = current_words[cut_at + 1:]
                        current_word_starts = current_word_starts[cut_at + 1:]
                        current_word_ends   = current_word_ends[cut_at + 1:]
                        current_start = current_word_starts[0] if current_word_starts else chunk_end_time
                        continue  # skip per-chunk code below
```

This should be at indent 24 (inside `if batch_size > 1:`) immediately after the closing of the `if len(batch_buffer) >= batch_size:` block (i.e., after line 2302). The `continue` applies to the main word-iterating loop and skips the per-chunk code at indent 20. No `segment_idx` increment for buffered non-flush items — the full flush handles that with `segment_idx += batch_size`.

**Location to insert: after line 2302 (`continue  # Skip the per-chunk code below`), still inside `if batch_size > 1:` at indent 24.**

---

## Minor Issues

### Dead Variable: `batch_annotations`
**Severity: Cosmetic / dead code**  
**Location: Line 1868**

```python
batch_annotations = {}  # segment_idx -> annotated_text (populated after batch LLM call)
```

This variable is initialized but never read or written anywhere else in the file. It is a leftover from the original batch implementation (commit `06d02bb`) that was replaced by positional `batch_results` lookup. Safe to delete.

---

### ETA Timing Inaccuracy in Batch Mode
**Severity: Minor (display only, no data impact)**  
**Location: Lines 2265–2277**

`chunk_t0` is reset by `chunk_t0 = time.monotonic()` at line 2010 for every chunk that enters the `if chunk_words...` block. In batch mode, by the time the batch flushes on chunk N, `chunk_t0` holds the timestamp from when chunk N entered — not when chunk 1 entered. So `chunk_times.append(time.monotonic() - chunk_t0)` measures only the time from the last chunk's entry to the end of the batch LLM call, not the full batch cycle time.

After fixing BUG D, the ETA will be more meaningful since non-flush chunks won't run per-chunk anymore. Can optionally fix by snapping a `batch_start_t0 = time.monotonic()` when the first item is added to a fresh batch buffer and using that in the batch ETA log.

---

### Tail Flush: Context Not Updated Between Fallback Items
**Severity: Minor quality (tail items only)**  
**Location: Lines 2419–2441**

In the tail flush's LLM fallback, `ctx_tail` is computed once using the current `context` deque, then reused for all tail items. Items 2, 3, etc. in the tail don't see each other's text as context. Since tail batches are small (typically 1–3 items) and context continuity is low-priority for trailing chunks, this is low priority.

**Fix:** After each tail item's LLM call, append `item["text"]` to `context` and recompute `ctx_tail`.

---

## Fix Priority

| Bug | Severity | Lines | Action |
|---|---|---|---|
| BUG D — non-flush batch double-process | **CRITICAL** | after 2302 | Add carry-forward + `continue` at indent 24 inside `if batch_size > 1:` |
| Dead `batch_annotations` var | Cosmetic | 1868 | Delete the line |
| ETA timing off by (N-1) chunks | Minor | 2265 | Snap `batch_start_t0` on first buffer append |
| Tail context not updated | Minor quality | 2419–2441 | `context.append(item["text"])` + recompute `ctx_tail` after each fallback |

---

## CODE STRUCTURE REFERENCE (3404-line file)

- **Lines 1–200**: imports, logging, lazy-import helpers, WAV overflow detection
- **Lines 200–500**: GPU stats, ASR methods (WhisperX, Wav2Vec2, IFW)
- **Lines 982–1050**: `choose_and_transcribe` — ASR method selection chain
- **Lines 1050–1530**: annotation sanitization, chunk boundary heuristics, alignment I/O, checkpoint I/O
- **Lines 1530–1760**: LLM loading, TTS system prompts, `_annotate_batch`, `_save_chunk_metadata`
- **Lines 1760–2500**: `annotate_chunks` — main chunking/annotation loop
  - Lines 1864–1868: `batch_buffer` init + dead `batch_annotations` dict
  - Lines 2200–2302: batch collection, full-batch flush
  - **← BUG D: lines 2302–2303 (between `continue` and per-chunk code)**
  - Lines 2304–2395: per-chunk mode path (runs for batch_size=1 AND broken non-flush batch items)
  - Lines 2401–2451: tail-batch flush (BUG B+C fixed in ea8449d)
- **Lines 2500–2760**: output naming, ZIP dataset creation with train/val split
- **Lines 2764–3404**: `main()` — argument parsing, phase orchestration

---

## HOW TO VERIFY BUG D

```bash
# Run with batch_size=4, limit=5 (1 full batch + 1 tail item)
python alexandria_preparer_rocm_compatible.py \
  --audio test.wav --model model.gguf \
  --batch-size 4 --limit 5 --chunk-size 10

# Count metadata entries vs WAV files
python3 -c "
import json, glob
entries = [json.loads(l) for l in open('dataset_temp/metadata.jsonl')]
wavs = sorted(glob.glob('dataset_temp/sample_*.wav'))
print(f'Metadata entries: {len(entries)}  (expected: 5)')
print(f'WAV files:        {len(wavs)}     (expected: 5)')
print()
# Check for duplicated audio (BUG D: chunks 0-2 saved twice with different names)
names = [e['audio_filepath'] for e in entries]
print('Files in metadata:', names)
"
# BUG D present: you will see 8 entries (3 per-chunk + 4 batch + 1 tail)
# BUG D fixed: you will see 5 entries
```
