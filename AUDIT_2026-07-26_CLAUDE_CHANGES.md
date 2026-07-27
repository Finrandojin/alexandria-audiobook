# Code audit — post-PR-210 changes and narrator work

Date: 2026-07-26  
Repository: `alexandria-audiobook2.git`  
Reviewed baseline: merge commit `589c0c2` (PR #210) through `a78d40b`
(PR #231), plus untracked `app/narrator.py` and
`app/test_narrator_detection.py`.

## Executive summary

The committed changes are generally well tested, and the quick API suite
passed. The audit found two high-impact defects, nine medium-impact defects,
and two low-impact comparison defects.

The most urgent problems are:

1. Inline scene-break pauses are removed when TTS chunks are regrouped.
2. The new narrator feature is incomplete and currently fails the release
   verifier.
3. Unknown symbols identified for human review are discarded instead of
   reported.
4. Several attribution evaluation tools can report misleading accuracy,
   agreement, or coverage.

No application code, tests, configuration, or runtime data were changed during
this audit. This Markdown report is the only file added.

## Verification performed

- Reviewed the complete `589c0c2..a78d40b` diff: 35 files, approximately 7,143
  insertions and 58 deletions.
- Included the two current untracked narrator files.
- Ran `app/verify_release.py --json-report ...`.
  - 165 Python files compiled.
  - 929 unit tests ran.
  - 4 errors, all in `RosterPromptTest`, because
    `three_pass_generate.format_roster` does not exist.
  - The verifier stopped before its API gate.
- Ran `app/run_isolated_api_tests.py` separately in quick mode.
  - 70 passed, 0 failed, 12 expected `--full` skips.
- Ran focused narrator tests independently.
  - 6 narrator-detection tests passed.
  - 4 narrator prompt/integration tests errored.
- Reproduced the scene-break pause loss with a real
  `get_speakable_entries`/`group_into_chunks` call.
- Verified duplicate identities in the attribution gold fixture.
- Performed independent passes for line correctness, removed behavior,
  cross-file behavior, security/input validation, reuse, simplification,
  efficiency, architecture, and test gaps.

## Findings

### 1. High — TTS regrouping removes inline scene-break silence

File: `app/project.py:181`

`split_on_unspeakable` correctly turns an inline scene break into two parts and
places `pause_after=1000` on the part before the break. `group_into_chunks`
then merges adjacent entries when speaker and instruction match without
checking whether the current entry already has `pause_after`.

During the merge, the pause is moved to the end of the combined text:

```text
"First sentence. ■ Second sentence."
-> ["First sentence." pause=1000, "Second sentence."]
-> ["First sentence. Second sentence." pause=1000]
```

The intended silence between sentences disappears and is played after both
sentences.

Recommended fix: prevent a merge whenever the current chunk has a positive
`pause_after`, or represent internal pauses explicitly rather than as a
property that grouping can move.

Required regression test: run an inline scene break through both
`get_speakable_entries` and `group_into_chunks`; assert two chunks and assert
the pause remains between them.

### 2. High — Narrator detection is not integrated and fails verification

Files:

- `app/narrator.py`
- `app/three_pass_generate.py:198`
- `app/test_narrator_detection.py:61`

The narrator module is never imported or called by the generation pipeline.
`attribute_batch` still renders the roster with `", ".join(roster)`.
`three_pass_generate` has no `format_roster` function, even though four new
tests require it.

Observable effects:

- Narrator detection cannot influence any attribution prompt.
- The claimed first-person narrator improvement is absent at runtime.
- Four tests fail with `ImportError: cannot import name 'format_roster'`.
- Both narrator files are untracked, so the committed CI inventory cannot
  protect this feature.

Recommended fix: complete the integration as one change: detect the narrator
from the source and accepted roster, format the prompt through one shared
function, pass aliases explicitly, add both files to the test inventory, and
run the release verifier.

Required end-to-end test: first-person source -> captured attribution prompt ->
assert the detected narrator, explanation, and validated aliases are present.

### 3. Medium — Unknown-symbol review findings are discarded

File: `app/project.py:158`

`split_on_unspeakable` returns `(parts, review_chars)`, and its contract says
unmapped characters are reported. The only production caller binds that value
as `_review` and discards it. No other production path consumes it.

Example: `Eris pouts ⌘` returns `["⌘"]` for review, but the character remains
in TTS text and no warning or review record is surfaced.

Recommended fix: propagate review characters to a preflight/reporting layer
before generation. Do not silently remove them; the existing design correctly
prefers review over guessing.

Required regression test: pass an unmapped symbol through the production
speakable-entry path and assert it appears in a surfaced review report.

### 4. Medium — Reasoning-only responses with `content: null` bypass the new policy

File: `app/generate_script.py:674`

The code calls:

```python
text = choice.message.content.strip()
```

before the new reasoning-overflow classifier handles empty content. OpenAI-
compatible responses may represent no visible content as JSON `null`. In that
case `.strip()` raises `AttributeError`, the response is classified as a
generic API error, and the intended one-escalation-then-overflow policy is not
used.

Recommended fix: normalize nullable content before any string operation, then
apply the same reasoning-overflow policy to every attempt.

Required regression test: fake a `finish_reason="length"` response with
`content=None` and positive reasoning tokens; assert one escalation followed
by `reasoning_overflow`, not `api_error`.

### 5. Medium — Accepted speakers bypass the roster attestation threshold

File: `app/three_pass_generate.py:1029`

`build_roster` requires `MIN_ROSTER_ATTESTATIONS=3`, because a bad roster name
propagates into later prompts. On the normal successful attribution path,
however, every returned speaker is appended directly to `roster` without
calling `build_roster` or `is_attested_name`.

The per-entry validation threshold is only two attestations. A name appearing
exactly twice can therefore pass one batch and be advertised as an established
character to every later batch, contrary to the documented roster invariant.
The fallback path rebuilds through the correct gate; the normal path does not.

Recommended fix: use one shared roster-admission function on both paths.

Required regression test: accept a speaker with exactly two occurrences in a
source longer than 5,000 characters, then assert the next batch does not
receive that speaker in its roster.

### 6. Medium — Duplicate gold entries overweight three source lines

File: `app/fixtures/attribution_gold.json:68`

The fixture contains exact duplicate identities:

- `mushoku16-00716`: 3 copies
- `mushoku16-00717`: 2 copies
- `mushoku16-00723`: 2 copies

`score_run` iterates every fixture entry. Three real lines therefore contribute
seven votes to accuracy and confusion results, skewing model comparisons.

Recommended fix: remove duplicate records and make fixture uniqueness a hard
validation rule.

Required regression test: assert both `id` and `(book, entry_index)` are unique
in every gold fixture.

### 7. Medium — Accuracy alignment treats a 60-character prefix as exact identity

File: `app/attribution_accuracy.py:47`

Both index validation and fallback lookup truncate normalized text to the first
60 characters. Two different entries with a shared 60-character prefix are
treated as the same line and can be scored as correct or confused against the
wrong source entry.

Recommended fix: use full normalized text for exact identity. If fuzzy
alignment is needed, store and validate a stronger source identity and report
ambiguity instead of silently choosing.

Required regression test: two lines with the same first 60 normalized
characters and different suffixes must not align.

### 8. Medium — Scoring sheets align repeated dialogue to the first occurrence

File: `app/build_scoring_sheet.py:81`

The code comments that duplicate text is ambiguous, but retains the first
occurrence rather than excluding ambiguous keys. Repeated short dialogue is
common in the corpus. When occurrences have different speakers or context—or
models segment them differently—the sheet compares unrelated occurrences and
shows the wrong surrounding context.

Recommended fix: retain occurrence lists and align by sequence/position, or
exclude duplicate keys from automatic sampling.

Required regression test: repeat identical dialogue at different positions
with different speakers and assert the occurrences are not conflated.

### 9. Medium — Arm comparison reports only one-sided coverage

File: `app/compare_attribution_arms.py:44`

Coverage is `aligned / len(arm_a)`. If all 10 entries in arm A match but arm B
contains 1,000 entries, the tool reports 100% coverage and emits no warning,
even though only 1% of arm B was compared.

Recommended fix: report coverage for both arms and warn based on the lower
coverage, or use intersection divided by the larger arm when one scalar is
required.

Required regression test: compare severely imbalanced arms in both argument
orders and require equivalent low-coverage warnings.

### 10. Medium — Three-character prefix aliasing merges distinct characters

File: `app/narrator.py:58`

Any two names sharing three leading characters are treated as the same person.
This demonstrably groups names such as `ANA`/`ANASTASIA`,
`TOM`/`TOMOE`, or `ALBERT`/`ALBERTA`. Their narrator scores are pooled and
`narrator_aliases` would tell the model they are aliases.

If integrated unchanged, a same-prefix cast can select the wrong narrator and
systematically merge two characters' dialogue.

Recommended fix: use established nickname/alias evidence rather than a raw
three-character prefix. At minimum, require a substantially stronger
relationship and explicitly reject same-prefix full names that both have
independent source evidence.

Required regression test: distinct same-prefix cast members must remain in
separate groups and must not be returned as narrator aliases.

### 11. Medium — Dialogue longer than 400 characters is counted as narration

File: `app/narrator.py:24`

The quote regex accepts only 3–400 characters. A quoted monologue longer than
400 characters is left entirely in the narration bucket. This was reproduced:
a third-person wrapper around a 500-character quoted first-person monologue
produced an empty dialogue bucket and `is_first_person=True`.

If integrated, long dialogue can falsely classify a third-person book as
first-person and put name occurrences in the wrong side of the narrator ratio.

Recommended fix: use the repository's existing quote-region parser instead of
a bounded regex.

Required regression test: a third-person book containing a long first-person
monologue must remain third-person, and the monologue must remain dialogue.

### 12. Low — Formatting-only differences become scoring-sheet disagreements

File: `app/build_scoring_sheet.py:106`

Speaker values are compared raw. `RUDI` and `rudi ` are marked as different,
although the accuracy scorer already treats case and surrounding whitespace as
irrelevant.

Recommended fix: centralize speaker normalization and use it in every scoring
tool while preserving raw values for display.

Required regression test: case/whitespace variants must agree.

### 13. Low — Formatting-only differences become arm disagreements

File: `app/compare_attribution_arms.py:53`

This tool also compares raw speaker strings, creating false disagreements for
case and whitespace variants.

Recommended fix and test: use the same shared speaker normalization and the
same regression case as finding 12.

## Cleanup observations

These are maintainability issues, not verified user-visible defects:

1. `app/compare_attribution_arms.py:83` performs the expensive
   `SequenceMatcher` alignment twice. Pass the already-computed pairs into
   disagreement extraction.
2. `app/compare_attribution_arms.py:11` imports `re` but never uses it.
3. `app/three_pass_generate.py:1290` reads the source with a bare
   `open(...).read()` rather than a context manager.
4. `app/test_pass_quality.py` imports `validate_attribution` twice.

## Additional test gaps

1. `app/test_narrator_detection.py:54` places `unittest.main()` before
   `RosterPromptTest`. Running the file directly executes only the first six
   tests and silently omits the four integration tests. Move the guard to EOF.
2. No composition test covers `split_on_unspeakable` followed by
   `group_into_chunks`; this is why finding 1 survived otherwise good unit
   coverage.
3. No test asserts gold fixture identity uniqueness.
4. No alignment test covers different lines sharing the first 60 characters.
5. No scoring-sheet test covers repeated dialogue occurrences.
6. No comparison test checks normalized speaker equality.
7. No comparison test checks bilateral coverage.
8. No production-path test asserts that unknown-symbol review results are
   surfaced.

## Limitations

- GPU/LLM/TTS-dependent checks were not run. The quick API suite intentionally
  skipped 12 checks requiring `--full`.
- No live model generation was performed; model-quality claims and narrator
  accuracy measurements were not independently reproduced.
- `env_doctor.py`, `CLAUDE.md`, and several earlier changes shown by a diff
  against feature commit `0cf9b32` predate merge commit `589c0c2`; they were
  excluded from the stated post-PR-210 scope.
- The narrator files are untracked work in progress. They were included because
  the audit request covered the current Claude changes, but findings 2, 10, and
  11 should be resolved before those files are committed.

## Suggested repair order

1. Complete or temporarily remove the narrator integration so the release
   verifier is green.
2. Preserve scene-break pauses through chunk grouping.
3. Surface unknown-symbol review results.
4. Fix reasoning-only nullable responses and roster admission.
5. Correct the gold fixture and evaluation/alignment tools before using their
   numbers for model decisions.
6. Harden narrator alias and quote parsing before enabling narrator hints.
7. Add the exact regression tests listed above, then rerun
   `verify_release.py` and the appropriate live qualification.
