# Response to REVIEW_2026-08-04.md

Every finding was reproduced before being fixed. All nine are addressed.
Test suite: **1,177 run, 0 failures** (was 1,172 run, 3 failed).

Fixes landed in `701371d` (findings 1, 2, 9) and `8c4c6b6` (findings 3, 4, 5,
6, 7, 8, plus improvement 4).

---

## Summary

| # | Severity | Finding | Status |
| --- | --- | --- | --- |
| 1 | High | failed GPU lock still launched the command | fixed, `701371d` |
| 2 | High | validation manifest silently excluded generation failures | fixed, `701371d` |
| 3 | Medium | narration gender evidence not connected to production | documented, not wired |
| 4 | Medium | narration regex did not enforce the binding it claimed | fixed, `8c4c6b6` |
| 5 | Medium | failed A/B lines dropped independently per arm | fixed, `8c4c6b6` |
| 6 | Low | `find_scene` never examined the final window | fixed, `8c4c6b6` |
| 7 | Low | `--aliases` accepted and ignored | fixed, `8c4c6b6` |
| 8 | Low | alias lookup missed mixed-case stored keys | fixed, `8c4c6b6` |
| 9 | Medium | required test gates were red | fixed, `701371d` |

The review was correct on every point checked. Three findings were in code
written specifically to prevent the class of problem it then failed to
prevent — the lock that did not gate, the manifest that hid failures, the
evidence module that claimed grammar and implemented proximity.

---

## Finding 1 — a failed GPU lock still launched the command (High)

**Reproduced.** With a fake `flock` returning 73 first on `PATH`, the wrapped
command ran and the script returned 0.

**Fix.** Lock acquisition is now an explicit gate. `exec 9>"$LOCK"` and
`flock 9` are both checked; on failure the queue log records `LOCK_FAILED`, the
command is never invoked, and the script exits 4. `set -e` is still off
deliberately, because the wrapped command's exit code has to survive — which is
exactly why the unchecked `flock` was dangerous.

**Verified after fixing.** Same failing-`flock` harness: command did not run,
exit 4, and the queue log contains `LOCK_FAILED` with no `START` or `OK`.

**Context that makes this worse than it looks.** Earlier the same day, a race in
a *different* wrapper around this primitive let `llama-server` start beside a
training job and OOM it at step 218/1250 — 42 minutes of A6000 time. That race
came from bootstrapping the lock with a `pgrep` wait, the precise pattern
`gpu_job.sh` exists to replace. Both jobs now enter through `gpu_job.sh` from
the start, so there is nothing to bootstrap around.

---

## Finding 2 — the validation manifest excluded generation failures (High)

**Confirmed by reading.** Only successful segments were written; failures lived
in terminal output, capped at the first five.

**Fix.** The manifest is now structured — `selected`, `generated`, `failures`,
`segments`. `tts_output_validation.py` counts generation failures in the
denominator and reports them on their own line, and the builder exits 3 when
any segment failed, so a partial run cannot be read as complete. The old
bare-list shape is still accepted, and re-scoring the existing chapter manifest
under the new code reproduces its numbers exactly.

**This did not affect the run that was live during the review.** 150 selected,
150 generated, 0 failed. The bug was real; the measurement was not corrupted by
it.

---

## Finding 3 — narration gender evidence is not connected to production

**Correct, and it was already the intent** — but the review is right that
nothing in the code said so, and its location in `app/` implied otherwise.

`character_evidence.py` now opens with an explicit **NOT WIRED INTO
PRODUCTION** notice: no caller in `routers/voices.py` uses it, the change that
removed dialogue inference did not replace it with narration evidence, and a
character with no gender-bearing label or persona still resolves `unknown` in
casting.

**Deliberately not wired.** Connecting it changes voice allocation, which is a
product decision rather than a defect fix. The measurement is offline evidence
that the approach works; shipping it is a separate call.

---

## Finding 4 — the regex did not enforce the binding it claimed (Medium)

**The most serious finding, and the one that is squarely an error rather than
an oversight.** The docstring said two constructions "cannot float to another
referent". They can. The review reproduced it exactly: `"Subaru watched Emilia
raise her hand."` ×3 returns `('female', 'medium', 3 feminine)` for Subaru. The
code was proximity wearing the vocabulary of grammar.

**Fix — an intervening-name rule.** A match is discarded when another known
character is named between the target and the construction, because the nearer
name is the likelier subject. This requires a roster; without one the function
degrades to the old proximity behaviour, and that degradation is now asserted
in a test rather than left as a silent surprise.

**It is also better on real data, not merely safer:**

| character | before | after |
| --- | --- | --- |
| Subaru | 90/12 male | 83/11 male |
| Reinhard | 14/2 male | 14/1 male |
| ROM | 9/1 male | 9/0 male |
| Emilia | 0/9 female | 0/9 female |
| SATELLA | 5/13 — abstained | **3/12 female** |

Removing contamination resolved a character that previously could not be
called.

**The docstring now describes what the code does.** The claim of grammatical
binding is replaced with "two constructions that USUALLY bind", plus the
intervening-name rule as the thing that separates it from plain proximity, plus
an explicit statement that it is not coreference and does not resolve
subordinate clauses or coordination.

---

## Finding 5 — A/B arms dropped failed lines independently (Medium)

**Confirmed by reading.** Each arm caught its own exceptions and continued, so
one arm could publish 14 lines against the other's 13 while both looked like a
normal comparison — inviting a listener to attribute to casting a difference
that was actually a missing sentence.

**Fix.** Both arms render fully first. The pair is published only if both
produced the identical complete line set; otherwise the script prints which arm
failed and on which line, writes a manifest with `published: false` and the
per-arm line ids, and exits 3. Published manifests record `line_ids` and
`published: true`.

**Consequence worth stating.** The A/B audio produced before this fix cannot be
guaranteed symmetric, and is being regenerated rather than defended.

---

## Findings 6, 7, 8 — Low

**6 — `find_scene` off-by-one.** `range(len(chunks) - size)` never examined the
final window and performed zero iterations when `len(chunks) == size`. Fixed to
`- size + 1`, with guards for `size <= 0` and empty input. The review's
four-chunk repro now returns index 0.

**7 — `--aliases` ignored.** The CLI accepted the flag and then rebuilt the
default path, so a book-specific map was silently discarded. Now reads
`args.aliases`.

**8 — alias lookup missed mixed-case keys.** `aliases_for('SUBARU',
{'Subaru': 'NATSUKI SUBARU'})` returned an empty set, so evidence was not
pooled across spellings. The whole mapping is casefolded once and looked up in
both directions, returning the original spellings because those are what the
narration contains. Both directions verified.

**Improvement 4** — the unused `combine_audio_with_pauses` import is removed.

---

## Finding 9 — the required test gates were red (Medium)

**1,172 run, 3 failed → 1,177 run, 0 failed.**

1. `quote_fallthrough.py` was untracked, correctly tripping the reproducibility
   guard. Now tracked.
2. `test_character_evidence.py` was missing from `unit_test_inventory.json`.
   Regenerated through the repository's updater.
3. `test_llm_traits_replace_local_traits_only_with_stronger_authority` did not
   merely hold a stale expectation — **it encoded the defect as correct
   behaviour.** It gave a character named "Hero" the line *"She was an old woman
   who entered quietly"* and asserted Hero was therefore female and elderly,
   which is precisely the inverted dialogue inference removed in `88d3ac7`.
   Rewritten around a neutral label so local traits are `unknown`, preserving
   the test's real subject — the authority ordering between local and LLM
   traits — without asserting the bug.

**The process failure behind this.** Targeted tests were run all day; full
discovery was not run once. A test encoding a bug that had just been fixed sat
red for hours.

---

## What the review exposed about how these results were produced

The consistent shape across findings 1, 2 and 4: **outputs were verified,
mechanisms were not.** Every number reported was reproducible. Several
explanations of *why* were wrong — the lock that logged `START` without holding
anything, the manifest whose rate excluded its own failures, the regex
described as grammar.

Three process changes follow, and are already in effect:

1. **Run full test discovery before claiming a suite passes.** Targeted runs
   hid a red gate for a day.
2. **Reproduce a mechanism before describing it.** The review's method —
   construct the adversarial input and run it — found in minutes what days of
   confirming outputs did not.
3. **Guard shapes at every level, not the level that already bit.**
   `collect_results.py` carried a comment reading "the shape is checked, not
   assumed" protecting a field, while the enclosing object was still assumed to
   be a dict; a list-shaped artifact killed the whole index. Same lesson, one
   level up.

---

## Related work found while responding

Not review findings, but discovered in the same window and relevant to the
trustworthiness of anything measured through the TTS path.

**`generate_lora_voice` never read the seed.** 121 lines, zero occurrences of
`seed`, while the three sibling generation paths all read
`voice_data["seed"]` and call `torch.manual_seed`. Every line of every `lora`
voice — 22 characters including NARRATOR, which speaks 1,581 of 2,606 lines —
was an independent draw of the voice.

Found by the user **listening** to A/B audio and saying it sounded like
multiple narrators. The same instability had already been measured twice (the
same adapter producing clips at 97–200 Hz) and misfiled twice, first as YIN
octave error and then as model behaviour.

Fixed in `1629254` and verified: `seed=7` now produces byte-identical output
across three runs (99840 samples, identical hash); `seed=-1` still varies.

**Consequence for the evidence base.** A *rate* measured on unseeded generation
remains valid, because production runs `seed: -1` on 70 of 71 characters —
unseeded is the shipping behaviour. A *comparison* is not, because the arms
differed by random draw. Six comparisons are affected and all are being
re-run seeded rather than defended:

- `instruct_value` — per-line vs per-character vs no instruction
- `prose_vs_nonprose` — the 40% vs 0% failure gap
- `nonprose_split_test` — whether splitting front matter helps
- `casting_ab_audio` — the perceptual comparison
- the bullet-list fix verification — 0/2 → 3/3
- `mean_f0` as a casting metric, whose *withdrawal* also rests on unseeded
  audio and may be reversed by `pitch_stability.py`

Unaffected, because they are not TTS arm comparisons: the 1.3% chapter defect
rate, the non-prose gate's 2.61% coverage, the quote-aware chunker's 77→42
repairs, the Chinese and Japanese harness results, the BookNLP baseline, the
PDNC generalisation numbers, the seven casting conflicts, and the voice-pool
coverage analysis.

---

## Outstanding

**Not addressed from the review's "Tests still needed" list.** Items 1–10 are
largely unwritten. The highest-value ones by the review's own reasoning are
item 1 (lock behaviour under failure and contention — partially covered by the
verification above, but not as a committed test) and item 10 (deterministic
local-LoRA generation, which is now verifiable and is not yet a test).

**Improvements 1, 2, 3 and 5 are not done**: precise server ownership by
PID/start-time rather than `pkill -f`, repository-relative paths instead of the
hard-coded `/home/fakemitch/...` that only works on the cloud box through a
symlink, slice-argument validation, and recording the deployed revision in the
queue log before `START`.

Improvement 2 is the one worth doing next. The hard-coded path is load-bearing
on a machine that is disposable, and the symlink that makes it work is not
recorded anywhere the next person would find it.
