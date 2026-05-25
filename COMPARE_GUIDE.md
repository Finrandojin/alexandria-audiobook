# Alexandria Compare Guide

Diff your `metadata.jsonl` transcriptions against the original EPUB or text file.
Every change requires your sign-off — nothing is written automatically.

## What This Tool Does (Plain English)

After the Preparer listens to your audiobook and writes down what it hears, sometimes it gets words wrong — especially names, unusual words, or words that sound similar. The Compare tool lets you **check what the AI heard against what the book actually says**, and fix mistakes one by one.

### Who Is This For?

- **You ran the Preparer** and got a `metadata.jsonl` file
- **You have the original book** (.epub or .txt) that the audiobook was read from
- **You want to fix transcription errors** before using the data for training

### Why Bother?

If the AI misheard "Sherlock" as "sure lock", the TTS training will learn the wrong word. Fixing these errors makes your trained voice more accurate.

---

## Quick Start for Non-Programmers

### Step 1: Extract Your Dataset

If your preparer output is a `.zip` file, unzip it first:
```bash
unzip alexandria_dataset.zip -d my_dataset/
```

You should now have a folder with `metadata.jsonl` and many `sample_NNNN.wav` files.

### Step 2: Open a Terminal

- **Windows:** Press `Win + R`, type `cmd`, press Enter
- **Mac:** Open Spotlight (Cmd + Space), type `terminal`, press Enter
- **Linux:** Open your applications menu and search for "Terminal"

### Step 3: Run the Compare Tool

Copy and paste this command (replace paths with your actual paths):

```bash
cd ~/.pinokio/api/alexandria-audiobook.git
./app/env/bin/python alexandria_compare.py \
  --jsonl /path/to/my_dataset/metadata.jsonl \
  --source /path/to/original/book.epub
```

### Step 4: Review and Fix Mistakes

The tool will show you each chunk where the AI's transcription differs from the book. You'll see something like:

```
[ANNOTATED] sure lock looked at the clock
[ORIGINAL]  Sherlock looked at the clock
[a]ccept [m]erge [k]eep [e]dit [s]kip [q]uit
```

**What each key does:**
- **`a` (accept):** The AI was right — keep the AI's version
- **`m` (merge):** Combine both versions — this is usually the best choice because it keeps the book's correct spelling AND any voice direction markers the AI added
- **`k` (keep):** Keep the AI's version unchanged
- **`e` (edit):** Type a custom correction
- **`s` (skip):** Skip this one for now (you can come back later)
- **`q` (quit):** Save your progress and exit

**Pro tip:** Press **`m` (merge)** most of the time. It keeps the book's correct words while preserving any voice directions (like "*whispered*" or "*excitedly*") that the AI added.

### Step 5: Use the Corrected Data

When the compare tool finishes, it saves a corrected `metadata.jsonl`. Use this corrected file instead of the original when training your voice.

---

## Overview

The compare tool:
1. Loads your JSONL and the original source text
2. Fuzzy-aligns each ~10 s chunk against the matching passage in the source
3. **Auto-approves** entries that already match well (configurable threshold)
4. **Shows you** entries that differ and waits for your decision
5. Writes a corrected JSONL only with the changes you approved

## Dependencies

Plain text sources need no extra packages. EPUB support requires:
```bash
uv pip install ebooklib beautifulsoup4
# or
pip install ebooklib beautifulsoup4
```

The compare script also imports `alexandria_alignment.py` (a sibling module
in the same directory) for source loading, fuzzy alignment, and the
proper-noun lexicon. Both files must stay next to each other.

## Did you use `--source` in the preparer?

If you produced this JSONL with `alexandria_preparer_rocm_compatible.py
--source ...`, character-name ASR mistranscriptions and dialect spellings
have already been corrected at preparation time — the chunk text is the
source's spelling, not what the ASR transcribed. The compare-review step
in that case is mostly a prosody sanity check: skim for `*emphasis*` and
`...` pause markers that look off, fix a few outliers with `[e]dit`, done.

If you produced the JSONL without `--source`, every chunk's text is the
raw ASR output and you'll be using compare-review to fix names, dialect,
and any audio-only material the preparer didn't know to drop. That's the
classic workflow this script was originally designed for.

## Important: Run in a Real Terminal

This script is **interactive** — it waits for your keyboard input at each entry.
You must run it in a real terminal session, not through an AI coding tool or any
other environment that doesn't support stdin.

```bash
cd /path/to/alexandria-audiobook2
app/env/bin/python alexandria_compare.py \
  --jsonl /path/to/metadata.jsonl \
  --source /path/to/original.txt
```

Then type your choice (`a`, `k`, `e`, `s`, or `q`) directly at the `>` prompt.

## Usage

```bash
python alexandria_compare.py \
  --jsonl dataset_temp/metadata.jsonl \
  --source original_book.epub
```

Or with a plain text file:
```bash
python alexandria_compare.py \
  --jsonl dataset_temp/metadata.jsonl \
  --source original_book.txt
```

If your dataset is already zipped, extract `metadata.jsonl` from it first:
```bash
unzip alexandria_dataset.zip dataset_temp/metadata.jsonl
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--jsonl PATH` | required | Path to `metadata.jsonl` |
| `--source PATH` | required | Original `.epub` or `.txt` |
| `--output PATH` | `<jsonl>_corrected.jsonl` | Where to write the corrected JSONL |
| `--threshold N` | `0.90` | Similarity ratio above which entries are auto-approved |
| `--review-all` | off | Show every entry regardless of similarity |
| `--reset` | off | Discard saved progress and start from the beginning |
| `--source-start N` | off | Manually start at source word N (skip auto-anchor) |
| `--source-start-text "..."` | off | Fuzzy-search for a phrase in source and start there |
| `--no-auto-anchor` | off | Disable auto-anchor; start at source word 0 |
| `--review-preanchor` | off | Review pre-anchor entries individually (default: auto-keep) |

## Source Loading

At startup you'll see one or two diagnostic lines like:

```
Loading source  : /path/to/book.epub
  284,125 characters
  29 recurring proper nouns (gifted, godou, grotesqueries, hana, kanoko, kaya, keiko, kiyoka +21 more)
```

The proper-noun line is a quick sanity check that the script identified the
right character names from the source. It scans Title-cased words that appear
multiple times mid-sentence — character names, place names, story-specific
terms like `Grotesqueries` or `Spirit Sight`. The alignment uses these to
relax the boundary acceptance bar when an ASR-mangled chunk word (`coodo`,
`youth`, `caia`) sits next to one of these names in the source. If the sample
looks wrong (missing main characters, full of common words), the alignment
will still work but may need more manual `[e]dit`s for proper-noun boundaries.

The script also auto-cleans two recurring EPUB extraction artifacts during
source load:
- **Letter-digit-letter glitches** (`thos1e` → `those`, `Kars1a` → `Karsa`)
  caused by OCR inserting stray digits between letters.
- **Split diacritics** (`fianc é` → `fiancé`, `fiancé e` → `fiancée`) caused
  by EPUB extraction separating precomposed Latin diacritics from their stems.

You don't need to do anything for these — they happen automatically.

## Initial Alignment (Audio Intro vs Text Front-Matter)

Audiobooks and source texts rarely start at the same point:
- The audio may open with **credits** ("Audible presents…"), a **narrator intro**,
  or a **chapter announcement** that isn't in the text.
- The text may open with **copyright pages**, a **table of contents**, or a
  **dedication** that the narrator doesn't read.

On a fresh run, the compare tool performs an **auto-anchor search**: it scans
the entire source for the position that best matches the first solid JSONL
entry, and starts alignment there.

You'll see output like:
```
🔍 Searching for initial alignment anchor...
  ✓ JSONL entry 3 anchors at source word 287 (62.1% match)
  Source preview: "It was a warm Saturday morning when Sakuta..."

  ⚠ 3 JSONL entries before the anchor have no matching source text
     (likely audio intro/credits not present in the text)
     ✓ Auto-kept as-is (use --review-preanchor to review them individually)
```

### When auto-anchor gets it wrong

If the anchor lands in the wrong place (e.g., a phrase repeats in the book and
it found the wrong instance), override it manually:

```bash
# Skip the first 412 words of source (e.g., copyright + TOC + dedication)
app/env/bin/python alexandria_compare.py \
  --jsonl metadata.jsonl --source book.txt --source-start 412

# Or search for a distinctive opening phrase
app/env/bin/python alexandria_compare.py \
  --jsonl metadata.jsonl --source book.txt \
  --source-start-text "It was a warm Saturday morning"

# Or disable auto-anchor entirely and start at word 0
app/env/bin/python alexandria_compare.py \
  --jsonl metadata.jsonl --source book.txt --no-auto-anchor
```

### Pre-anchor entries

JSONL entries that come before the anchor point have no corresponding source
text. By default they're auto-kept as-is (since they're usually audio-only
intro material you don't want to overwrite). Use `--review-preanchor` to
review each one individually instead.

Note: the auto-anchor only runs on the **first** run. When resuming from
checkpoint, the saved cursor position is used directly — your prior alignment
decisions are preserved.

## Mid-Stream Re-Alignment

Even with a good initial anchor, the source and audio can drift apart
mid-book (e.g., audio reads a chapter title that isn't in the text, or
skips a passage that is). The script automatically detects this:

- When an entry scores poorly against its near-cursor source span, the
  script runs a **wider search ahead** (up to 3,000 words).
- If a confident match is found, the cursor **jumps forward** to it.
- If no confident match exists anywhere ahead, the entry is flagged as
  having no source equivalent and shown with:

  ```
  ORIGINAL  :  (no matching passage found in source within search range)
  ```

  For these entries the `[a]` accept-original action is disabled; use
  `[k]` keep, `[e]` edit, or `[s]` skip.

This typically catches: chapter announcements only spoken in audio, audio
outro credits, and any audio-only narrator inserts.

## Review Interface

Each entry that falls below the threshold is shown like this:

```
────────────────────────────────────────────────────────────────────────
Entry 42/4713  sample_0042.wav  7m 1s → 7m 11s  Match: 78.3%
────────────────────────────────────────────────────────────────────────
ANNOTATED :  He *walked* ... slowly across the room .... pausing to look
ORIGINAL  :  He walked slowly across the room, pausing to look

  TRANS diff:  He walked [slowly] across the room [pausing] to look
  ORIG  diff:  He walked [slowly] across the room [,] [pausing] to look

  [a] accept original   [k] keep annotation   [e] edit manually   [s] skip for now   [q] quit & save
  >
```

**Diff colours:**
- `RED` words — in the transcription but not in the original (extra/wrong)
- `GREEN` words — in the original but not in the transcription (missing)
- Normal — matching words

**ANNOTATED** shows the full text including TTS markers (`...`, `*word*`).
The diff rows compare stripped/normalised words so punctuation noise doesn't
cloud the comparison.

## Your Choices

| Key | What happens |
|---|---|
| `a` | Replace JSONL text with the plain source text. **Strips all prosody markers.** |
| `m` | **Merge** — use source words but keep the LLM's `*emphasis*` and `...` pauses. Only shown when the annotation has markers worth preserving. |
| `k` | Keep the current annotated text exactly as-is |
| `e` | Type a custom replacement (you write exactly what goes in the JSONL) |
| `s` | Skip this entry — it stays unchanged and won't appear again this session (will show on next run) |
| `q` | Save progress and exit — rerun the same command to continue |

## Don't Let the Voice Go Flat: Prefer `[m]erge`

The preparer's LLM adds prosody markers to the JSONL:
- `*word*` — **emphasis** on stressed words
- `...` — natural pause, `....` — longer pause

These markers are what make the trained TTS voice expressive. If you `[a]ccept
original` for many entries, you replace the annotated text with the plain
source words — **stripping all the prosody hints** and pushing the voice
toward flat/monotone output.

The `[m]erge` option fixes the words while keeping the markers:

```
ANNOTATED :  HE *WALKED* SLOWLY... ACROSS THE ROOM
ORIGINAL  :  He walked slowly across the room,
MERGED    :  He *walked* slowly ... across the room,
            (correct spelling + capitalisation + the LLM's emphasis and pause)
```

The merge handles several tricky shapes automatically:
- **Multi-word emphasis** — `*Trull Sengar*` is expanded to `*Trull* *Sengar*`
  before alignment, so both names carry the marker into the source spelling.
- **Dot-joined chains** — chunks like `YOU...*WALKED*...HOME` (where the LLM
  used `...` as a word separator instead of whitespace) are split correctly
  so each `*word*` survives the merge.
- **ASR-mashed names** — a single chunk word like `*Tralesengar*` covering a
  two-word source phrase `Trull Sengar` will broadcast its emphasis across
  both source words via concatenation-fuzzy.

Emphasis can still be dropped when ASR mistranscription is severe enough that
neither the per-word nor concatenated fuzzy ratio clears the threshold —
those cases show up in the review and you can fix them with `[e]dit`.
Pauses between words are preserved either way.

At the end of the session, if you accepted many originals without merging,
you'll see a warning suggesting `[m]erge` next time.

## Pause and Resume

Press `q` at any review prompt to stop. A checkpoint file is saved automatically
next to your JSONL:

```
.metadata_compare_progress.json
```

Rerun the exact same command to pick up where you left off:

```bash
python alexandria_compare.py \
  --jsonl dataset_temp/metadata.jsonl \
  --source original_book.epub
```

To start completely fresh (discard all prior decisions):
```bash
python alexandria_compare.py \
  --jsonl dataset_temp/metadata.jsonl \
  --source original_book.epub \
  --reset
```

## Output

The corrected JSONL is written to `metadata_corrected.jsonl` by default (next to
the input JSONL). Entries where you chose `[k]` keep or were auto-approved are
unchanged. Only `[a]` accept and `[e]` edit entries have different text.

You can specify a different output path:
```bash
python alexandria_compare.py \
  --jsonl dataset_temp/metadata.jsonl \
  --source book.epub \
  --output my_reviewed_dataset.jsonl
```

## Threshold Tuning

The default threshold of `0.90` means entries where ≥ 90 % of words match the
original are silently approved. Lower it to see more entries:

```bash
# See everything with less than 95% match
python alexandria_compare.py --jsonl ... --source ... --threshold 0.95

# See everything, no auto-approve
python alexandria_compare.py --jsonl ... --source ... --review-all
```

## How Alignment Works

The tool maintains a **cursor** in the source word list. For each JSONL entry:
1. Strips TTS annotation markers (`...`, `*word*`) from the transcription
2. Normalises both the transcription and source words (lowercase, no punctuation)
3. Slides a window ±200 words around the cursor, scoring each position with
   fuzzy sequence matching
4. Advances the cursor to the end of the best match

This means **alignment accuracy depends on the transcription being in order**,
which it always is since it was produced sequentially from the audio. If
individual chunks have very bad ASR errors, alignment may drift slightly —
you'll notice this as unexpectedly low match scores on surrounding entries.

## Summary at Completion

```
Complete!
  Auto-approved (≥90%) : 4512
  Kept annotation      : 83
  Accepted original    : 78
  Edited manually      : 12
  Skipped              : 28
  Re-run to review 28 skipped entries.
```
