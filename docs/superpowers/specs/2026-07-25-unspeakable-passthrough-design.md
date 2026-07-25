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

## Library prevalence — measured, not estimated

Scanned 2026-07-24 across all three library roots: 7,100 files, **5,850 unique
books** after content-hash dedupe, **4,195 real novels** after dropping code,
build logs, textbooks and RPG rulebooks. Exclusion is by **symbol density**
(`(review_symbol + verbalize) / chars < 0.002`), not by title:
`101CProgrammingChallenges.epub`, `lib_burst_generated.txt` and both
`KonoSuba TRPG` rulebooks are excluded on content, while `Katanagatari` is
kept.

Do not filter on title keywords. Light novels routinely mimic textbook
phrasing — "The Genius Prince's Guide to Raising a Nation Out of Debt",
"Before the Tutorial Starts", "Min-Maxing My TRPG Build in Another World" —
and a keyword filter drops 27 genuine novels while catching nothing the
density filter misses.

The top unfiltered "symbol" hits were a Unity build log and a C++ textbook,
so never quote unfiltered counts.

| class | novels affected | occurrences |
|---|---:|---:|
| verbalize (`∞ ← → °`) | 82.2% | 17,115 |
| review symbol | 62.5% | 82,818 |
| **scene break** (`─ ◇ ◆ ■`) | 17.2% | **105,897** |
| elongation (`~`) | 15.9% | 11,070 |
| music (`♪`) | 8.0% | 2,854 |
| pictographic kana (`へ`) | 1.8% | 945 |

**Paragraph impact: median 0.26%, p75 0.74%, p90 1.80%, p99 18.85%.** Only
**62 novels (1.5%)** have 10% or more of their paragraphs affected.

Two consequences for sizing:

- **This is broad but shallow.** Nearly every book has a few; almost none have
  many. `mushoku16`'s 1.40% sits at the **86th percentile** — it was already a
  worse-than-typical case, not a representative one.
- **Scene breaks are the whole game.** 105,897 occurrences against 17,115 for
  verbalization. Build the pause conversion first; the `VERBALIZE` table is a
  long tail that can be deferred without losing much.

The severe tail is almost entirely **Nisio Issin**: Katanagatari vols 1-4
(29-38% of paragraphs) and the whole Monogatari series (24-38%), which use `◇`
and `─` as scene separators at enormous density. Note that
**`owarimonogatari3` is already in the A/B matrix** — model benchmarking has
been running against one of the most symbol-dense books in the library.

### Character frequency (250 sampled affected novels)

```
─  7716   ◇ 1463   | 1394   >  1083   ~   863   <   802
◆   774   ■  767   + 487    ＋  470   ©   442   °   431
✧   300   🍖 210   ♪  204   🎂 198   🍵  192   ❄   183
🐺  150   □  142   ＜ 128    ＞ 128    🐅  120   ○   119
```

`─` alone outweighs every other character combined. Size the work off `─` and
`◇`, not `■` — the earlier draft led with `■`, which is 10x rarer.

## Classes the first draft missed

Sampling real usage turned up four more, all the same insight as `♪`: **these
mark _how_ to speak, not _what_ to speak.**

| pattern | real example | what it is | fix |
|---|---|---|---|
| `><` `^^` `orz` | `"can't see you during lunch >< Sorry!"` | Japanese emoticon | emotion cue -> `instruct`, strip from text |
| `<...>` wrapping a whole line | `<Today's temperature stands at −7°C...>` | system/PA announcement | delivery cue -> `instruct`, strip brackets |
| `\|` in CIP blocks | `Names: Ōmori, Fujino, author. \| Haimura, Kiyotaka, illustrator.` | library cataloging front matter | not content - belongs in `strip_known_front_matter` |
| 🍖 🎂 🍵 🐺 ❄ ✧ | decorative / chat-scene emoji | emoji | verbalize or drop; **decide deliberately** |

The emoji case needs a policy call before coding. `🍖` in a chat message might
warrant "meat emoji"; the same glyph as a chapter-heading decoration should be
dropped. Prevalence is low enough (a few hundred across 250 books) that
flagging for review is defensible instead of guessing.

Fullwidth variants (`＋ ＜ ＞`) appear alongside their ASCII forms and should be
normalized together, not handled as separate entries.

## Four classes, four different fixes

Do not build one mapping table. These need different handling:

| class | characters | fix |
|---|---|---|
| **Scene break** | `─ ◇ ◆ ■ □ ○ ━ █ △` (ordered by real frequency) | Split the entry and convert to `pause_after`. Not speech. **Highest value: 105,897 occurrences.** |
| **Verbalize** | `∞ ← → ↑ ° © ×` | Replace with spoken words ("infinity", "degrees"). |
| **Delivery** | `~ ～` (vowel elongation), `♪ ♫` when they bracket a line | **Move** into `instruct`; do not merely strip. |
| **Human judgement** | pictographic kana (`へ` as a mouth shape) | No rule can fix. Flag for review. |

Two refinements that matter, both easy to get wrong:

**Delivery marks must move, not vanish.** Stripping `~` from `"Yaaay~"` throws
away the only signal that the line is drawn out. The classifier returns an
instruct hint and the caller appends it to the entry's existing `instruct`, so
the cue survives in the field that controls delivery. Deleting it silently
downgrades the audio while looking like a clean fix.

**`♪` is usually not a word.** In light novels a music note most often
*brackets* sung dialogue — `♪ La la la ♪`. Verbalizing that yields "music note
la la la music note", which is worse than leaving it in. Rule: when `♪`/`♫`
appears at both the start and end of an entry, strip both and add a sung hint
to `instruct`; only a lone, mid-sentence occurrence is a candidate for a
spoken word, and even then prefer flagging it for review.

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
# Ordered by measured frequency across 4,195 novels: U+2500 alone outweighs
# every other symbol combined, and the diamonds beat the square 3:1.
SCENE_BREAK_CHARS = frozenset("─◇◆■□○━█△●▪▫✧❄")

# Spoken renderings. Deliberately small and explicit rather than a
# category-wide rule, so every substitution is auditable.
VERBALIZE = {
    "∞": "infinity",
    "←": "left arrow", "→": "right arrow", "↑": "up arrow", "↓": "down arrow",
    "°": "degrees", "©": "copyright", "×": "times", "÷": "divided by",
    "±": "plus or minus", "≠": "not equal to", "≈": "approximately",
    "★": "star", "☆": "star", "♥": "heart",
}

# Delivery cues, NOT words. These move into `instruct`; see MUSIC_CHARS below
# for why a music note is deliberately absent from VERBALIZE.
ELONGATION_CHARS = frozenset("~～")
MUSIC_CHARS = frozenset("♪♫")

# Hints appended to the entry's existing `instruct` when a cue is removed from
# the text, so the signal survives instead of being silently deleted.
ELONGATION_HINT = "Drawn-out, elongated delivery."
SUNG_HINT = "Sung rather than spoken."

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
    """Return one of: scene_break, verbalize, elongation, music, review,
    speakable."""
    if char in SCENE_BREAK_CHARS:
        return "scene_break"
    if char in VERBALIZE:
        return "verbalize"
    if char in ELONGATION_CHARS:
        return "elongation"
    if char in MUSIC_CHARS:
        return "music"
    if unicodedata.category(char) in _SYMBOL_CATEGORIES:
        return "review"
    return "speakable"


def extract_delivery_cues(text):
    """Strip delivery cues from text, returning (text, instruct_hints).

    A music note bracketing an entry marks sung dialogue: "♪ La la la ♪"
    verbalized becomes "music note la la la music note", which is worse than
    leaving it alone. Bracketing pairs become a sung hint; a lone mid-sentence
    note is left in place and reported for review instead of guessed at.
    """
    hints = []
    stripped = text.strip()
    if (len(stripped) > 1 and stripped[0] in MUSIC_CHARS
            and stripped[-1] in MUSIC_CHARS):
        text = stripped[1:-1].strip()
        hints.append(SUNG_HINT)
    if any(c in ELONGATION_CHARS for c in text):
        text = "".join(c for c in text if c not in ELONGATION_CHARS)
        hints.append(ELONGATION_HINT)
    return text, hints
```

Tests to write:
- each scene-break char classifies as `scene_break`
- `∞` and `♪` classify as `verbalize`
- `~` classifies as `elongation`
- an unmapped symbol (e.g. `⌘`) classifies as `review`, not silently dropped
- ordinary letters, digits and punctuation classify as `speakable`
- `is_pictographic_kana("へ", " .")` is True; `is_pictographic_kana("へ", "んな")` is False
- `extract_delivery_cues("♪ La la la ♪")` returns `("La la la", [SUNG_HINT])`
- `extract_delivery_cues("Yaaay~")` returns `("Yaaay", [ELONGATION_HINT])`
- a lone mid-sentence `♪` is **not** stripped and yields no hint (review case)
- text with no cues returns unchanged with an empty hint list

### Task 2: apply it on the way to TTS

**Files:** modify `app/project.py` (`get_speakable_entries`, line 66),
test `app/test_project_chunks.py` (already imports that function)

Add a helper beside it:

```python
def split_on_unspeakable(entry, default_pause_ms):
    """Split one entry into speakable parts, converting glyphs per class.

    Returns (parts, review_flags). A scene break inside prose becomes a
    boundary carrying pause_after rather than a character the narrator reads
    aloud. Verbalized symbols become words. Delivery cues move into
    `instruct` via extract_delivery_cues rather than being deleted. Anything
    unmapped is left in place and reported, so an unknown glyph is visible
    rather than silently voiced.
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
- `'Yaaay~'` becomes `'Yaaay'` **and its entry's `instruct` gains the elongation hint** (assert both; stripping alone is the bug this guards)
- `'♪ La la la ♪'` becomes `'La la la'` with a sung hint, not "music note la la la music note"
- an entry that already has an `instruct` keeps it, with the hint appended rather than replacing it
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
- Sizing is now measured rather than guessed (see Library prevalence above):
  median book 0.26% of paragraphs, only 1.5% of novels above 10%. The extended
  corpus will likely read milder than `mushoku16`, which is at the 86th
  percentile.
- Scene-break conversion alone captures ~86% of all occurrences. If time is
  limited, build that and defer verbalization, emoji policy and emoticons.
- `|` in CIP front matter is a `strip_known_front_matter` job, not a
  verbalization one. Do not add it to any mapping table.
