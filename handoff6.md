# Handoff 6: Alexandria Preparer — Post-handoff5 Review

**Project:** `/home/fakemitch/pinokio/api/alexandria-audiobook2.git`  
**File under review:** `alexandria_preparer_rocm_compatible.py` (3414 lines)  
**Date:** 2026-05-25  
**Prepared by:** Claude Sonnet 4.6 — reviewing Qwen's commit `b5bb910` (handoff5 BUG E fix)

---

## Handoff5 Fixes: Status

| handoff5 item | Status |
|---|---|
| BUG E — context double-append in batch mode | ✓ Fixed correctly. Line 2309: `context.append(text)` replaced with a comment. One-line change, no side-effects. |
| ETA timing off by (N-1) chunks | ✗ Still open (see below) |
| Misindented comments | ✗ Still open (L2417, L2421, L2450, L2459) |

---

## NEW BUG INTRODUCED — None

Qwen's commit `b5bb910` is a clean minimal fix. The single changed line is correct. No new bugs were introduced by this commit.

---

## NEW BUG FOUND (BUG F) — Progress Logging Never Fires in Batch Mode

**Severity: Significant operational bug (batch mode runs completely silently)**  
**Introduced by: Original batch implementation (pre-handoff1) — never fixed**  
**Location: Lines 2265 and 2277**

### Root cause

The batch-mode ETA/progress checks use `(segment_idx + 1) % N == 0`. In batch mode, `segment_idx` steps by `batch_size` (line 2293: `segment_idx += batch_size`), not by 1. The check fires only when `segment_idx + 1` is an exact multiple of `N`. But `segment_idx` takes values `0, batch_size, 2*batch_size, ...` — so `segment_idx + 1` is always `1, batch_size+1, 2*batch_size+1, ...` — never a multiple of 10 or 100 for any `batch_size > 1`.

**Proof for batch_size=4:**

| Batch | segment_idx before flush | (segment_idx+1) % 10 | Fires? |
|---|---|---|---|
| 1 | 0 | 1 | No |
| 2 | 4 | 5 | No |
| 3 | 8 | 9 | No |
| 4 | 12 | 3 | No |
| 5 | 16 | 7 | No |
| 6 | 20 | 1 | No |
| … | … | … | **Never** |

This holds for **every** `batch_size > 1` — the cycle repeats without ever hitting 0.

### Effect

In batch mode (`--batch-size N` with `N > 1`):
- `"↳ Progress: X/Y chunks"` log **never fires**
- `log_gpu_stats(...)` **never fires** — GPU VRAM is never logged during a batch run
- `"⏱ Timing breakdown"` log **never fires**

A multi-hour batch run produces zero progress output. Users cannot monitor ETA, VRAM, or timing.

In per-chunk mode (`batch_size=1`), everything is correct because `segment_idx` steps by 1.

### Also: reported chunk count would be wrong even if check fired

The log message at line 2271:
```python
f"  ↳ Progress: {segment_idx + 1}/{estimated_chunks_total} chunks "
```

`segment_idx` here is the **pre-increment** value (before `+= batch_size` at line 2293). After a 4-item batch starting at index 8, the log would say "chunk 9" but 12 chunks have actually been processed. Off by `batch_size - 1`.

Similarly, `remaining_chunks = max(0, estimated_chunks_total - segment_idx - 1)` at line 2268 is off by `batch_size - 1`.

### Fix

Replace the pre-increment modulo checks with a "crossed a decade/century boundary?" check:

```python
# Line 2265 — was: if (segment_idx + 1) % 10 == 0:
if ((segment_idx + batch_size) % 10) < batch_size:
    avg_chunk_s = sum(chunk_times) / len(chunk_times)
    elapsed_s = time.monotonic() - annotation_start_time
    completed = segment_idx + batch_size                  # post-increment count
    remaining_chunks = max(0, estimated_chunks_total - completed)
    remaining_s = remaining_chunks * avg_chunk_s
    logger.info(
        f"  ↳ Progress: {completed}/{estimated_chunks_total} chunks "
        f"| Avg: {avg_chunk_s:.1f}s/chunk "
        f"| Elapsed: {format_duration(elapsed_s)} "
        f"| ETA: {format_duration(remaining_s)}"
    )
    log_gpu_stats(f"annotation segment {completed}/{estimated_chunks_total}")

# Line 2277 — was: if (segment_idx + 1) % 100 == 0:
if ((segment_idx + batch_size) % 100) < batch_size:
    total_timed = timing['audio_read'] + timing['snr_calc'] + timing['alignment'] + \
                  timing['llm_infer'] + timing['sanitize'] + timing['wav_write']
    completed = segment_idx + batch_size
    logger.info(
        f"  ⏱ Timing breakdown (chunk {completed}): "
        f"audio_read={timing['audio_read']:.1f}s "
        f"snr={timing['snr_calc']:.1f}s "
        f"alignment={timing['alignment']:.1f}s "
        f"llm={timing['llm_infer']:.1f}s "
        f"sanitize={timing['sanitize']:.1f}s "
        f"wav_write={timing['wav_write']:.1f}s "
        f"total_timed={total_timed:.1f}s "
        f"dropped={timing['dropped_chunks']} "
        f"kept={timing['kept_chunks']}"
    )
```

**Why `((segment_idx + batch_size) % N) < batch_size` works:**  
It fires when we have just crossed a multiple of `N`. For `batch_size=4`, `N=10`: fires at `segment_idx=8` (batch 8-11, crossing 10), `segment_idx=16` (batch 16-19, crossing 20), `segment_idx=36` (batch 36-39, crossing 40), etc. For `batch_size=1`: equivalent to the existing `(segment_idx + 1) % N == 0`. Safe for per-chunk too.

---

## Still Open from Handoff5

### ETA Timing Inaccuracy in Batch Mode
**Severity: Minor (display only)**  
**Location: Lines 2204–2264**

`chunk_times.append(time.monotonic() - chunk_t0)` at line 2264 measures time from when the **flush-triggering item** (item N-1) entered the valid-chunk block (line 2009), not from when the first item of the batch was buffered. For batch_size=4, this captures only item 3's audio processing time + the full batch LLM inference — missing items 0–2's audio processing.

`chunk_t0` is set at line 2009 (overwritten each outer-loop iteration, including non-flush items), then again at line 2295 after each flush. So for the batch starting at items 0–3: `chunk_t0` is overwritten at items 0, 1, 2, and 3. By the time the flush executes at line 2264, `chunk_t0` reflects item 3's start time only.

**Fix: snap `batch_start_t0` when the first item enters a fresh batch:**

```python
# Around line 2203 — before batch_buffer.append():
if batch_size > 1:
    if not batch_buffer:                         # NEW: fresh batch starting
        batch_start_t0 = time.monotonic()
    batch_buffer.append({...})
```

Initialize `batch_start_t0 = 0.0` before the main while loop (near `chunk_t0` init at line 2009's surrounding scope).

Then at line 2264:
```python
# was:  chunk_times.append(time.monotonic() - chunk_t0)
chunk_times.append((time.monotonic() - batch_start_t0) / batch_size)
```

Dividing by `batch_size` keeps `chunk_times` in per-chunk units (seconds/chunk) so the ETA formula `remaining_chunks * avg_chunk_s` remains correct.

**Note:** Fix this AFTER BUG F is fixed. Both affect line ~2264. Do them together to avoid a double-touch.

---

### Misindented Comments (Cosmetic)
**Severity: Cosmetic only (no functional impact)**  
**Location: Lines 2417, 2421, 2450, 2459**

Python ignores comment indentation; these don't affect execution. But they are misleading:

| Line | Actual indent | Should be | Comment text |
|---|---|---|---|
| 2417 | 8 (outside `if batch_buffer:`) | 12 (inside `if`) | `# Save audio and write metadata for remaining batch chunks` |
| 2421 | 12 (outside for-loop body) | 16 (inside for-loop) | `# Get annotation from batch results or fallback` |
| 2450 | 12 (outside for-loop body) | 16 (inside for-loop) | `# Update context after each tail item so subsequent items see it` |
| 2459 | 8 (outside `if batch_buffer:`) | 12 (inside `if`) | `# BUG B fix: increment segment_idx for tail batch` |

---

## Improvement Opportunities (Non-Bug)

### IMPROVEMENT A: Batch fallback ctx should use current context, not pre-batch ctx

**Location: Line 2233**  
**Severity: Minor quality improvement**

In the full-flush fallback path (when batch annotation fails for some items), the per-item LLM fallback uses:
```python
user_prompt = f"Previous context: {ctx}\n\n..."   # ctx = outer loop variable (pre-batch)
```

Since context is updated at line 2225 (`context.append(item["text"])`) BEFORE the fallback check, items 1..N-1 already have preceding batch items in `context`. Using the live context would give them better preceding context:
```python
ctx_fallback = " ".join(list(context)[-2:]) if context else ""
user_prompt = f"Previous context: {ctx_fallback}\n\n..." if ctx_fallback else f"..."
```

This only matters when batch_results is None (entire batch annotation failed and ALL items fall back to per-chunk). In that case all N items would get the stale pre-batch `ctx` instead of progressively updated context.

### IMPROVEMENT B: JSON extraction in `_annotate_batch` handles trailing text

**Location: Lines 1603–1618 in `_annotate_batch`**  
**Severity: Minor robustness improvement**

When `raw_output.startswith("[")`, the raw output is used as-is for `json.loads`. If the LLM adds trailing text (e.g., `["text1", "text2"] (end of output)`), json.loads fails and we silently fall through to the numbered-line fallback. The numbered fallback is weaker — add a repair step before giving up:

```python
if raw_output.startswith("["):
    json_match = raw_output
    if "[" not in json_match or "]" not in json_match:
        json_match = None
    else:
        # Try to extract [first..last] in case there's trailing text
        try:
            json.loads(json_match)
        except json.JSONDecodeError:
            try:
                end = json_match.rindex("]") + 1
                json_match = json_match[:end]
            except ValueError:
                json_match = None
```

### IMPROVEMENT C: Numbered fallback in `_annotate_batch` handles out-of-order results

**Location: Lines 1629–1637 in `_annotate_batch`**  
**Severity: Minor robustness improvement**

The numbered fallback appends items in order of appearance. If the LLM outputs them out of order (e.g., "1. ... / 3. ... / 2. ..."), annotations end up in the wrong order. Fix: use a dict keyed by number, then sort:

```python
annotated_by_num = {}
for line in lines:
    match = re.match(r"^\s*(\d+)[\.\)]\s*(.+)$", line)
    if match:
        num = int(match.group(1))
        if 1 <= num <= len(batch_data) and num not in annotated_by_num:
            annotated_by_num[num] = match.group(2).strip()
if len(annotated_by_num) == len(batch_data):
    annotations = [annotated_by_num[i+1] for i in range(len(batch_data))]
```

---

## Fix Priority for Next Agent

| Issue | Severity | Location | Action |
|---|---|---|---|
| BUG F — progress log never fires in batch mode | Significant operational | Lines 2265, 2277 | Replace `(segment_idx + 1) % N == 0` with `((segment_idx + batch_size) % N) < batch_size`; use `segment_idx + batch_size` in log messages |
| ETA timing underestimate | Minor display | Lines 2203–2264 | Snap `batch_start_t0`; use `(now - batch_start_t0) / batch_size` at line 2264 |
| Misindented comments | Cosmetic | Lines 2417, 2421, 2450, 2459 | Re-indent 4 spaces right |
| Improvement A (fallback ctx) | Minor quality | Line 2233 | Use live `context` in fallback instead of pre-batch `ctx` |
| Improvement B (JSON trailing text) | Minor robustness | Lines 1603–1618 | Try `rindex("]")` trim when startswith fallthrough fails |
| Improvement C (numbered fallback ordering) | Minor robustness | Lines 1629–1637 | Use dict keyed by number |

---

## Verification Scripts

### Verify BUG F

```python
# Prove that (segment_idx + 1) % 10 never fires for batch_size=4
batch_size = 4
total = 100
fired = []
for s in range(0, total, batch_size):
    if (s + 1) % 10 == 0:
        fired.append(s + 1)
print("Old check fires at:', fired)  # Prints: [] — never fires

# Prove the fix does fire
fired_fixed = []
for s in range(0, total, batch_size):
    if ((s + batch_size) % 10) < batch_size:
        fired_fixed.append(s + batch_size)
print('Fixed check fires at:', fired_fixed)  # Prints: [12, 20, 32, 40, 52, 60, 72, 80, 92, 100]
```

### Verify ETA timing fix

```python
import time

# Simulate batch_size=4 timing comparison
# Old behavior: only captures last item's time
# New behavior: captures full batch time / batch_size

# The batch_start_t0 should be set when batch_buffer was empty
# Then (time.monotonic() - batch_start_t0) / batch_size gives per-chunk average
print("batch_start_t0 approach gives per-chunk avg across full batch cycle")
```

---

## CODE STRUCTURE REFERENCE (3414-line file)

- **Lines 1–200**: imports, logging, lazy-import helpers, WAV overflow detection
- **Lines 200–500**: GPU stats, ASR methods (WhisperX, Wav2Vec2, IFW)
- **Lines 982–1050**: `choose_and_transcribe` — ASR method selection chain
- **Lines 1050–1530**: annotation sanitization, chunk boundary heuristics, alignment I/O, checkpoint I/O
- **Lines 1530–1760**: LLM loading, TTS system prompts, `_annotate_batch`, `_save_chunk_metadata`
  - Lines 1565–1668: `_annotate_batch`
  - Lines 1603–1618: JSON extraction (IMPROVEMENT B target)
  - Lines 1629–1637: numbered fallback (IMPROVEMENT C target)
  - Lines 1671–1727: `_save_chunk_metadata`
- **Lines 1760–2500**: `annotate_chunks` — main chunking/annotation loop
  - Lines 1864–1868: `batch_buffer` init
  - Lines 2199–2310: batch collection, full-flush, non-flush carry-forward
    - Lines 2203–2215: `batch_buffer.append({...})` ← ETA timing fix: snap `batch_start_t0` here
    - Lines 2218–2301: full-flush path
      - Line 2225: `context.append(item["text"])` — context update for batch items
      - Line 2233: fallback `ctx` ← IMPROVEMENT A target
      - Line 2264: `chunk_times.append(time.monotonic() - chunk_t0)` ← ETA timing fix
      - Lines 2265, 2277: `% 10` and `% 100` checks ← **BUG F fix target**
      - Line 2293: `segment_idx += batch_size`
    - Lines 2303–2310: non-flush carry-forward (BUG D fixed, BUG E fixed)
  - Lines 2312–2407: per-chunk mode path (batch_size=1 only)
    - Lines 2366, 2381: `% 10` and `% 100` checks (correct in per-chunk mode)
    - Line 2397: `segment_idx += 1`
    - Line 2400: `context.append(text)`
  - Lines 2409–2460: tail-batch flush
    - Lines 2417, 2421, 2450, 2459: misindented comments ← cosmetic fix target
- **Lines 2500–2760**: output naming, ZIP dataset creation with train/val split
- **Lines 2764–3414**: `main()` — argument parsing, phase orchestration
