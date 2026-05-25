# Handoff 5: Alexandria Preparer — Post-handoff4 Verification

**Project:** `/home/fakemitch/pinokio/api/alexandria-audiobook2.git`  
**File under review:** `alexandria_preparer_rocm_compatible.py` (3414 lines)  
**Date:** 2026-05-25  
**Prepared by:** Claude Sonnet 4.6 — verifying Qwen's commit `97a62bd` (handoff4 fixes)

---

## Handoff4 Fixes: Status

| handoff4 item | Status |
|---|---|
| BUG D — non-flush batch double-process | ✓ Mechanism correct: `continue` added at indent 24 to skip per-chunk code |
| Dead `batch_annotations` var | ✓ Removed at line 1868 |
| Tail context not updated between items | ✓ `context.append(item["text"])` added at line 2451 inside tail for-loop |
| ETA timing off by (N-1) chunks | ✗ NOT fixed — still open (see below) |

---

## NEW BUG INTRODUCED (BUG E) — Context Double-Append in Batch Mode

**Severity: Minor-moderate quality regression (annotation context corrupted for 2–4 chunks per batch)**  
**Introduced by: commit `97a62bd` (handoff4 fix)**  
**Location: Line 2309**

### Root cause

Qwen's BUG D fix adds `context.append(text)` inside the non-flush carry-forward path (line 2309), at indent 24 inside `if batch_size > 1:`. But the **full-flush path** at line 2225 **already appends all batch items** to context when the batch completes:

```python
# Full-flush path (line 2223–2225) — already updates context for ALL items
for i, item in enumerate(batch_buffer):
    context.append(item["text"])   # ← item 0..N all get appended here

# BUG D non-flush path (line 2309) — NEW, conflicts with above
context.append(text)  # ← items 0..N-2 already appended here, then again above
continue
```

Similarly, the **tail flush path** at line 2451 does `context.append(item["text"])` for each tail item. Any item that entered via the non-flush path also gets appended there — double-append for tail items too.

### Effect for batch_size=4 with chunks A, B, C, D

`context = deque(maxlen=5)`

| Step | Action | context after |
|---|---|---|
| A non-flush | `context.append("A")` | [A] |
| B non-flush | `context.append("B")` | [A, B] |
| C non-flush | `context.append("C")` | [A, B, C] |
| D full-flush | loop appends A, B, C, D | [B, C, A, B, C] → [C, A, B, C, D] |

After the batch: context = `[C, A, B, C, D]` instead of correct `[A, B, C, D]`.

The last-2 slice used for `ctx` is still `"C D"` (correct for the immediate next chunk) but the context window contains stale duplicates that corrupt chunk 2–4 of the *next* batch.

### Effect on tail items

Tail items (chunks remaining in batch_buffer after the main loop, not enough for a full batch) enter via the non-flush path, so each gets `context.append(text)` at line 2309. Then the tail loop at line 2451 appends them again. Result: all tail items are double-appended.

### Effect on tail fallback ctx

The tail LLM fallback computes `ctx_tail = " ".join(list(context)[-2:])` at line 2427 AFTER the non-flush path has already appended ALL tail items. So tail item 1's fallback context already contains tail items 2, 3, etc. — temporally wrong (looking at future chunks as "previous context").

Note: this ctx_tail issue only manifests when the batch annotation fails and per-chunk fallback runs for multiple tail items. In the normal case (batch annotation succeeds), the tail context is irrelevant.

### Fix

**Remove line 2309 (`context.append(text)`)** from the BUG D non-flush carry-forward block.

```python
                        # Batch not full yet — buffer this chunk and skip per-chunk processing (BUG D fix)
                        current_words       = current_words[cut_at + 1:]
                        current_word_starts = current_word_starts[cut_at + 1:]
                        current_word_ends   = current_word_ends[cut_at + 1:]
                        current_start = current_word_starts[0] if current_word_starts else chunk_end_time
                        prev_raw_text = text
                        # DO NOT context.append(text) here — full-flush and tail paths handle context updates
                        continue  # Skip the per-chunk code below
```

The full-flush path (line 2225) and tail path (line 2451) already update context correctly. The trade-off: within a single batch, items 1..N-1 all see the pre-batch context instead of progressive within-batch context. This is acceptable because `_annotate_batch` sends all chunks together in one prompt so the LLM sees all texts anyway; the `ctx` field is just for the fallback per-chunk path.

**Correction to handoff4 suggested fix:** handoff4.md's suggested code block included `context.append(text)` in the non-flush fix. That suggestion was incorrect. The correct non-flush block omits it.

---

## Still Open from Handoff4

### ETA Timing Inaccuracy in Batch Mode  
**Severity: Minor (display only)**  
**Location: Lines 2264–2276**

`chunk_times.append(time.monotonic() - chunk_t0)` at line 2264 measures time from when the **flush-triggering chunk** entered the valid-chunk block (line 2009), not from when the first item in the batch was buffered. For batch_size=4, this measures ~1/4 of the real batch cycle time.

**Fix:** Snap `batch_start_t0 = time.monotonic()` when the first item is appended to a fresh batch buffer, and replace `chunk_t0` with `batch_start_t0` in the timing append at line 2264.

---

## Cosmetic Issues (no functional impact)

Three misindented comments in the tail flush section (Python ignores comment indentation but it's confusing):

- Line 2421: `# Get annotation from batch results or fallback` — at 12 spaces, should be 16
- Line 2450: `# Update context after each tail item so subsequent items see it` — at 12 spaces, should be 16
- Line 2401: `# Carry the post-cut tail forward...` — at 12 spaces, should be 16 (pre-existing)

---

## Fix Priority

| Issue | Severity | Location | Action |
|---|---|---|---|
| BUG E — context double-append | Minor-moderate quality | Line 2309 | Delete `context.append(text)` from non-flush block |
| ETA timing off | Minor display | Lines 2264 | Snap `batch_start_t0` on first buffer append |
| Misindented comments | Cosmetic | Lines 2421, 2450, 2401 | Re-indent to match code at 16 spaces |

---

## CODE STRUCTURE REFERENCE (3414-line file)

- **Lines 1–200**: imports, logging, lazy-import helpers, WAV overflow detection
- **Lines 200–500**: GPU stats, ASR methods (WhisperX, Wav2Vec2, IFW)
- **Lines 982–1050**: `choose_and_transcribe` — ASR method selection chain
- **Lines 1050–1530**: annotation sanitization, chunk boundary heuristics, alignment I/O, checkpoint I/O
- **Lines 1530–1760**: LLM loading, TTS system prompts, `_annotate_batch`, `_save_chunk_metadata`
- **Lines 1760–2500**: `annotate_chunks` — main chunking/annotation loop
  - Lines 1864–1868: `batch_buffer` init (dead `batch_annotations` dict removed by 97a62bd)
  - Lines 2199–2310: batch collection, full-batch flush, non-flush carry-forward
    - Lines 2203–2215: `batch_buffer.append({...})`
    - Lines 2218–2301: full-flush path (if `len(batch_buffer) >= batch_size:`)
    - **← BUG E: line 2309 (`context.append(text)`) in non-flush path**
  - Lines 2312–2407: per-chunk mode path (batch_size=1 only, BUG D fixed)
  - Lines 2409–2460: tail-batch flush (BUG B+C+tail-context fixed in ea8449d+97a62bd)
- **Lines 2500–2760**: output naming, ZIP dataset creation with train/val split
- **Lines 2764–3414**: `main()` — argument parsing, phase orchestration

---

## HOW TO VERIFY BUG E

```bash
# After fixing (removing context.append from line 2309), run a batch test:
python3 -c "
from collections import deque

# Simulate batch_size=4 with BUG E (current code)
ctx_bugged = deque(maxlen=5)
texts = ['A', 'B', 'C', 'D']

# Non-flush appends (lines 2309 - BUG E)
for t in texts[:-1]:  # A, B, C (non-flush)
    ctx_bugged.append(t)

# Full-flush appends (line 2225)
for t in texts:  # A, B, C, D (all flush)
    ctx_bugged.append(t)

print('BUG E result:', list(ctx_bugged))  # [C, A, B, C, D] — WRONG

# Simulate fixed code (no context.append in non-flush)
ctx_fixed = deque(maxlen=5)
for t in texts:  # only full-flush appends
    ctx_fixed.append(t)
print('Fixed result:', list(ctx_fixed))  # [A, B, C, D] — CORRECT
"
```
