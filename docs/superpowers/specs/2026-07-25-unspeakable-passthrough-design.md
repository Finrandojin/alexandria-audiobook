# Unspeakable Character Passthrough — Design

_Written 2026-07-24 for a fresh session on 2026-07-25. Prepared so the next
session can implement without re-deriving the investigation._

## The problem in one paragraph

Symbols and code-switched text pass every gate in the pipeline and reach TTS as
literal text. The segment fidelity gate verifies text is **preserved**, never
that it is **speakable**, so a scene-break glyph or a pictographic kana sails
through while the run reports success. On `mushoku16`, **57 of 76 `■`
survived into the final script** alongside `'Eris pouts, her mouth へ.'`, and
the run reported a single failure.

## What the evidence does and does not support

Measured against the 356 diagnostic failures in
`ab_test_runtime/results/collect_all_20260723-040555`:

| claim | verdict |
|---|---|
| Symbols cause chunk failures | **No.** Excluding index18 (whose "symbols" are U+FFFD), 3 of 152 failures contain a symbol = 2%, *below* the 18% baseline rate for random chunks. |
| Code-switching causes failures | **No.** 0 of 356 failures contain CJK, despite `arc4_volume10wn` holding 340 CJK characters. |
| These corrupt the output silently | **Yes.** 28 of 1,995 `mushoku16` entries reached the script carrying unvoiceable characters. |

Do not implement this expecting failure counts to improve. They will not move.
The win is audio quality, and it is invisible to every metric currently
collected.

## Where this goes, and why

**`project.py`, in and around `get_speakable_entries` (line 66). Not source
normalization.**

The codebase has already made this decision explicitly. That function's
docstring reads:

> Punctuation-only and block-glyph dialogue remains in annotated_script.json
> for fidelity, but sending it to TTS produces noise or failures. A nonverbal
> entry extends the preceding speakable entry's pause without mutating caller
> data.

So the established convention is: **the script stays faithful to the source;
the transformation happens on the way to TTS.** Follow it (CLAUDE.md Rule 13 —
don't introduce a paradigm the codebase isn't already using).

Doing this in `source_normalization.py` instead would rewrite the source before
chunking, which changes what the user reviews and edits in the script editor,
and discards the distinction between "what the book says" and "what the
narrator speaks".

### Why the existing helper misses these

`utils.is_nonverbal_text` (line 19) is:

```python
return not any(char.isalnum() for char in str(text or ""))
```

It only fires when an entry has **no** alphanumerics at all. `mushoku16`'s
glyphs are embedded inside otherwise-speakable prose:

```
'I wonder if she hates mice\n\n■\n\nIt seems that a cat infected with...'
```

That entry is speakable overall, so the whole thing passes through with the
`■` intact. The gap is embedded glyphs, not whole-entry glyphs.

Leave `is_nonverbal_text` alone — `three_pass_generate.py:151` and `:1026` also
depend on its current meaning for deterministic NARRATOR assignment.

## Four classes, four different fixes

Do not build one mapping table. These need different handling:

| class | characters | fix |
|---|---|---|
| **Scene break** | `■ ─ ━ ○` | Split the entry and convert to `pause_after`. Not speech. |
| **Verbalize** | `∞ ♪ ♫ ← → ↑ ° © ×` | Replace with spoken words ("infinity", "music note"). |
| **Delivery** | `~` (Japanese vowel elongation) | Belongs in the `instruct` field, not the text. Strip from text. |
| **Human judgement** | pictographic kana (`へ` as a mouth shape) | No rule can fix. Flag for review. |

The fourth class is why this cannot be fully automated. `'her mouth へ'` means
"her mouth made a へ shape" — there is no pronunciation, only a translation
("her mouth twisted into a pout"). Detect and report; do not guess.

## Implementation

### Task 1: classify unspeakable characters

**Files:** create `app/verbalization.py`, test `app/test_verbalization.py`

```python
"""Map characters that cannot be spoken into speech, pauses, or review flags.

The script keeps the source's characters for fidelity; this module decides what
the narrator actually voices. Split by class because the classes need different
handling - a scene break is a pause, an arrow is a word, and a pictographic
kana needs a human.
"""

import unicodedata

# Scene / section breaks: silence, never speech.
SCENE_BREAK_CHARS = frozenset("■─━□◆◇○●▪▫")

# Spoken renderings. Deliberately small and explicit rather than a
# category-wide rule, so every substitution is auditable.
VERBALIZE = {
    "∞": "infinity", "♪": "music note", "♫": "music note",
    "←": "left arrow", "→": "right arrow", "↑": "up arrow", "↓": "down arrow",
    "°": "degrees", "©": "copyright", "×": "times", "÷": "divided by",
    "±": "plus or minus", "≠": "not equal to", "≈": "approximately",
    "→": "right arrow", "★": "star", "☆": "star", "♥": "heart",
}

# Vowel elongation in translated Japanese prose ("Yaaay~"). A delivery cue,
# not a word.
ELONGATION_CHARS = frozenset("~～")

_SYMBOL_CATEGORIES = frozenset({"So", "Sm", "Sk"})


def is_pictographic_kana(char, following):
    """Kana used as a picture rather than as language.

    Detects the manga convention where a kana describes a shape, as in
    "her mouth へ". Heuristic: a lone kana with no other kana adjacent.
    """
    if not ("぀" <= char <= "ヿ"):
        return False
    return not any("぀" <= c <= "ヿ" for c in following)


def classify(char):
    """Return one of: scene_break, verbalize, elongation, review, speakable."""
    if char in SCENE_BREAK_CHARS:
        return "scene_break"
    if char in VERBALIZE:
        return "verbalize"
    if char in ELONGATION_CHARS:
        return "elongation"
    if unicodedata.category(char) in _SYMBOL_CATEGORIES:
        return "review"
    return "speakable"
```

Tests to write:
- each scene-break char classifies as `scene_break`
- `∞` and `♪` classify as `verbalize`
- `~` classifies as `elongation`
- an unmapped symbol (e.g. `⌘`) classifies as `review`, not silently dropped
- ordinary letters, digits and punctuation classify as `speakable`
- `is_pictographic_kana("へ", " .")` is True; `is_pictographic_kana("へ", "んな")` is False

### Task 2: apply it on the way to TTS

**Files:** modify `app/project.py` (`get_speakable_entries`, line 66),
test `app/test_project_chunks.py` (already imports that function)

Add a helper beside it:

```python
def split_on_unspeakable(entry, default_pause_ms):
    """Split one entry into speakable parts, converting glyphs per class.

    Returns (parts, review_flags). A scene break inside prose becomes a
    boundary carrying pause_after rather than a character the narrator reads
    aloud. Verbalized symbols become words. Elongation marks are dropped from
    the text (they belong in `instruct`). Anything unmapped is left in place
    and reported, so an unknown glyph is visible rather than silently voiced.
    """
```

Then in `get_speakable_entries`, for entries that are **not** wholly nonverbal,
run them through `split_on_unspeakable` and extend `speakable` with the parts,
preserving the existing `pause_after` max() merge behaviour.

Key constraints:
- Do not mutate caller data — the existing function copies with `dict(source_entry)`; keep that.
- `EXPLICIT_SILENCE_MS = 1000` (`script_repair.py:17`) is the established scene-break-scale pause; `DEFAULT_PAUSE_MS = 500` (`tts.py:35`) is the between-speakers pause. Use `EXPLICIT_SILENCE_MS` for scene breaks.
- Preserve `speaker` and `instruct` on every split part.

Tests to write:
- `'mice\n\n■\n\nIt seems'` becomes two entries, the first carrying `pause_after >= EXPLICIT_SILENCE_MS`
- a leading scene break produces no orphan pause (mirrors the existing "no audio anchor" rule)
- `'the value is ∞.'` becomes `'the value is infinity.'`
- `'Yaaay~'` becomes `'Yaaay'`
- an entry with no unspeakable characters is returned unchanged and `is` equal in content
- a pictographic kana entry is returned **unchanged** but appears in `review_flags`

### Task 3: surface the review flags

Report unresolved cases rather than hiding them. `script_preflight.audit_script`
is where script-level findings already live — add an `advisory` finding
(not `blocking`) so the user sees "3 entries contain characters that cannot be
voiced" without failing the run.

## Verification

Against a real book rather than fixtures:

```bash
cd app && ./env/bin/python -c "
import json
from project import get_speakable_entries
entries = [e for e in json.load(open(
  '../ab_test_runtime/probe_thinking/off/result.json.threepass_checkpoint.json'
))['named'] if e]
out = get_speakable_entries(entries)
bad = [e for e in out if any(c in '■─━' for c in e.get('text',''))]
print('entries still carrying a scene break:', len(bad))
assert not bad
"
```

Expected before: 28 entries carry unvoiceable characters. Expected after: 0
scene breaks remain, `∞`/`♪` are words, and any pictographic kana is flagged
rather than altered.

Also confirm the extended-corpus summary's `UNSPEAKABLE PASSTHROUGH` counts
(added 2026-07-24 to `ab_test_runtime/probe_thinking/run_extended.sh`) drop to
zero for the scene-break and verbalize classes across all 11 books.

## Scope notes

- `annotated_script.json` is deliberately **unchanged** by this work. Only the
  TTS path is affected. If the user later wants the editor to show verbalized
  text, that is a separate decision.
- Check the extended-corpus numbers before sizing this. If `mushoku16`'s 28
  affected entries turns out to be an outlier across the 11 books, the work is
  smaller than it looks; if it is typical, this touches every book produced.
