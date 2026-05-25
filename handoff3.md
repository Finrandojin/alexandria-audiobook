# Handoff 3: Alexandria Preparer — Post-Qwen Review

**Project:** `/home/fakemitch/pinokio/api/alexandria-audiobook2.git`  
**File under review:** `alexandria_preparer_rocm_compatible.py` (3382 lines)  
**Date:** 2026-05-25  
**Prepared by:** Claude Sonnet 4.6 — verifying Qwen's commits `b1b75e8` + `43fe1d1`

---

## What Qwen Fixed Correctly

The following items from handoff2 were applied and verified in the code:

| handoff2 item | Status |
|---|---|
| BUG 1 — carry-forward skipped by `continue` | ✓ Fixed: carry-forward moved to lines 2302–2305, before the `continue` |
| BUG 2 — tail flush filename collision | ✓ Fixed: `idx = segment_idx + i` at line 2417 |
| BUG 3 — only last batch chunk in context | ✓ Fixed: `context.append(item["text"])` for each item at line 2226 |
| BUG 4 — duplicate `import re` inside functions | ✓ Fixed: only module-level `import re` at line 44 remains |
| BUG 6 — unused `source_words_list`, `segment_idx_start` params | ✓ Fixed: both removed from `_annotate_batch` signature and call sites |
| BUG 7 — batch fallback used raw text instead of LLM | ✓ Fixed: per-chunk LLM call added in main loop fallback (lines 2238–2259) |
| IMPROVEMENT 1 — ETA logging missing in batch mode | ✓ Fixed: `chunk_times.append()` + ETA log at lines 2269–2296 |
| IMPROVEMENT 2 — `--min-snr` default 25 → 15 | ✓ Fixed: line 2844 now `default=15` |
| IMPROVEMENT 4 — fsync after checkpoint truncation | ✓ Fixed: `f.flush(); os.fsync(f.fileno())` at lines 1387–1388 |
| IMPROVEMENT 5 — intervaltree guard before `_assign_speakers_to_words` | ✓ Fixed: `if not INTERVALTREE_AVAILABLE: sys.exit(1)` at line 3287 |
| IMPROVEMENT 6 — standardize scratch WAV to PCM_16 | ✓ Fixed: `subtype="PCM_16"` at line 3068 |

---

## CRITICAL NEW BUG (introduced by "IMPROVEMENT 3" fix)

### BUG A — Full batch flush: segment_idx-based lookup gives all chunks the same annotation
**Severity: Critical. Every chunk in a batch gets annotated with the first chunk's annotation.**  
**Location: Lines 2229–2235**

**Root cause:** All items appended to `batch_buffer` share the same `segment_idx` value. The counter is only incremented by `+= batch_size` **after** the batch finishes (line 2298). So for `batch_size=4` starting at `segment_idx=0`, all four items have `"segment_idx": 0`.

`_annotate_batch` returns `[(0, ann1), (0, ann2), (0, ann3), (0, ann4)]` — all four tuples have the same key `0`.

Qwen's "IMPROVEMENT 3" fix then does:
```python
seg_idx = item["segment_idx"]  # = 0 for ALL items
for result_idx, result_annotated in batch_results:
    if result_idx == seg_idx:   # 0 == 0 → always matches first tuple
        annotated = result_annotated   # ann1 for ALL items!
        break
```

**Effect:** Every chunk in the batch writes `ann1` as its annotation text. Chunks 1–3 are silently overwritten with chunk 0's annotation.

**The OLD positional approach was correct.** `_annotate_batch` guarantees `len(results) == len(batch_data)` and returns results in the same order as input (see line 1644: `for i, (item, annotated_raw) in enumerate(zip(batch_data, annotations))`). Positional lookup is safe.

**Fix** — replace lines 2228–2235 with:
```python
annotated = None
if batch_results is not None and i < len(batch_results):
    annotated = batch_results[i][1]   # positional — same order as input
```

Note: the **tail flush** at line 2418–2419 was NOT changed by Qwen and still uses the correct positional approach. The inconsistency is: main loop = broken, tail flush = correct.

---

## MINOR NEW BUG

### BUG B — After tail flush, `segment_idx` is not incremented
**Severity: Minor (display only). Final summary count is wrong by the tail batch size.**  
**Location: Lines 2407–2428**

After saving all tail-batch items, `segment_idx` remains at the pre-tail value. Line 2498 then computes:
```python
chunks_emitted_this_run = segment_idx - next_segment_idx
```
...which is short by `len(batch_buffer)` items.

The saved WAV files are correctly named (`idx = segment_idx + i`), so no data is lost. Only the summary log is wrong.

**Fix** — add after line 2428 (`batch_buffer.clear()`):
```python
segment_idx += len(batch_buffer_size_at_flush)
```

More simply, capture the count before `clear()`:
```python
tail_count = len(batch_buffer)
batch_buffer.clear()
segment_idx += tail_count
```

---

## MINOR REGRESSION

### BUG C — Tail flush single-item batches get unannotated raw text
**Severity: Minor. Same behavior as before BUG 7 was fixed, but only for tail batches of size 1.**  
**Location: Lines 2409–2421**

`_annotate_batch` returns `None` when `len(batch_data) == 1` (line 1572–1574). The tail flush at line 2420–2421 then falls back to `annotated = item["text"]` — raw unannotated ASR text.

The main loop's BUG 7 fix added per-chunk LLM fallback (lines 2238–2259), but this fix was not applied to the tail flush path. A book that generates exactly `(N * batch_size) + 1` chunks will have the last chunk unannotated.

**Fix** — replace the tail flush fallback block:
```python
else:
    annotated = item["text"]  # Fallback
```
...with a per-chunk LLM call matching lines 2238–2259 in the main loop.

---

## STILL NOT FIXED FROM HANDOFF2

### BUG 5 — Phase argument filtering misses `--phase=value` form
**Severity: Low (only affects manual shell invocations, not Python orchestrator calls).**  
**Location: Lines 2947, 2971, 2983**

The filter:
```python
if arg not in ["--phase", "asr", "enrich", "annotate"]:
```
passes `--phase=asr` through as a forwarded argument, causing the child subprocess to see a duplicate `--phase` flag. argparse won't error on this (last value wins), but it's sloppy.

**Fix:**
```python
if arg not in ["--phase", "asr", "enrich", "annotate"] and not arg.startswith("--phase="):
```

---

## FIX PRIORITY

| Bug | Severity | Lines | Action |
|---|---|---|---|
| BUG A — IMPROVEMENT 3 broken lookup | **CRITICAL** | 2229–2235 | Replace with `batch_results[i][1]` |
| BUG B — tail `segment_idx` not incremented | Minor | after 2428 | `segment_idx += tail_count` |
| BUG C — tail single-item gets raw text | Minor | 2418–2421 | Add per-chunk LLM fallback |
| BUG 5 — `--phase=value` filter | Low | 2947, 2971, 2983 | `.startswith("--phase=")` guard |

---

## CODE STRUCTURE REFERENCE (updated for 3382-line file)

- **Lines 1–200**: imports, logging, lazy-import helpers, WAV overflow detection
- **Lines 200–500**: GPU stats, ASR methods (WhisperX, Wav2Vec2, IFW)
- **Lines 982–1050**: `choose_and_transcribe` — ASR method selection chain
- **Lines 1050–1530**: annotation sanitization, chunk boundary heuristics, alignment I/O, checkpoint I/O
- **Lines 1530–1760**: LLM loading, TTS system prompts, `_annotate_batch`, `_save_chunk_metadata`
- **Lines 1760–2500**: `annotate_chunks` — main chunking/annotation loop
  - Lines 1867–1868: `batch_buffer` and `batch_annotations` init
  - Lines 2200–2307: batch collection, full-batch flush ← **BUG A is here**
  - Lines 2308–2395: per-chunk mode path
  - Lines 2405–2428: tail-batch flush ← **BUG B and BUG C are here**
- **Lines 2500–2760**: output naming, ZIP dataset creation with train/val split
- **Lines 2764–3382**: `main()` — argument parsing, phase orchestration

---

## TESTING TO CONFIRM BUG A

```bash
# Run with batch_size=3, limit=4 (1 full batch + 1 tail)
python alexandria_preparer_rocm_compatible.py \
  --audio test.wav --model model.gguf \
  --batch-size 3 --limit 4 --chunk-size 10

# Confirm BUG A: do sample_0001 and sample_0002 have the same annotation as sample_0000?
cat dataset_temp/metadata.jsonl | python -c "
import json, sys
rows = [json.loads(l) for l in sys.stdin]
texts = [r['text'] for r in rows[:3]]
print('BUG A present:', texts[0] == texts[1] == texts[2])
print('sample_0000:', texts[0][:80])
print('sample_0001:', texts[1][:80])
print('sample_0002:', texts[2][:80])
"

# Confirm BUG B: does reported count match actual files?
python -c "
import json, glob
entries = [json.loads(l) for l in open('dataset_temp/metadata.jsonl')]
wavs = glob.glob('dataset_temp/sample_*.wav')
print(f'metadata entries: {len(entries)}, wav files: {len(wavs)}')
"
```
