# Alexandria Audiobook — Quality Goals

What "good" means for this app, as numbers a script can check.

Every current value below is a measurement with a source you can re-run. Every
target is a commitment. Where there is no baseline yet, the goal is *to take
the measurement*, and it says so — an unmeasured target is a wish, and this
document does not contain wishes.

**Last updated:** 2026-08-06

## How to read this

Each goal has a **metric** (what is counted), a **probe** (the script that
counts it), a **current** value with its evidence, and a **target**.

Three status markers:

- **MET** — measured at or beyond target. Keep a test so it stays there.
- **OPEN** — measured, below target. The gap is the work.
- **NO BASELINE** — not measured. The first task is the measurement, not the fix.

A target is only listed when something in the measured record suggests it is
reachable — a better arm, a cloud model, a human ceiling. Where the ceiling
itself is unknown, the goal says so rather than inventing a number.

---

## 1. Speaker attribution — who says which line

The core task. Everything downstream inherits its errors: a misattributed line
gets the wrong voice, and no amount of TTS quality repairs it.

### 1.1 Accuracy on the four annotated books

**Metric** — percent of gold-labelled lines assigned the correct speaker.
**Probe** — `app/experiments/` arms, aggregated in `results_index.csv`.
**Current** — best local arm per book, 616 scored arm rows:

| book | best local (qwen3-14b) | best cloud (llama-3.3-70b) | gap |
|---|---|---|---|
| grimgar03 | 84.4% | 86.8% | 2.4 |
| index18 | 81.5% | 82.6% | 1.1 |
| mushoku16 | 72.9% | 74.8% | 1.9 |
| owarimonogatari3 | 69.1% | 69.8% | 0.7 |

**Target — every book ≥ 75% on the local model.** Two of four already clear it;
owarimonogatari3 needs +5.9 and mushoku16 +2.1.

**Why this is obtainable, and why not higher.** The local model is within 2.4
points of a 70B cloud model on every book, so the target is not asking local to
do something only cloud has done. Setting it at 90% would be, since nothing
measured has reached 90% on any book by any method.

**The honest caveat.** Median across all 616 arms is 46–67% depending on the
book. The best arm is not the shipped arm, and the spread between books (69.1
to 84.4 on the same method) is larger than the spread between most methods.
Book identity dominates.

### 1.2 Close the selection gap

**Metric** — of lines where the correct speaker is present in the candidate
roster, the percent where the model picks it.
**Probe** — see memory `attribution_selection_not_recall`.
**Current** — roster contains the right name **85%** of the time; the model
picks it **29.9%** of the time. **OPEN.**

**Target — selection ≥ 50% with roster recall held at ≥ 85%.**

**Why this and not more context.** Two independent measurements say supply is
not the constraint: the answer is already in front of the model in 85% of
cases. Context-supply fixes are treating the wrong problem. This is the single
largest known headroom in the app, which is why it is stated separately from
1.1 rather than folded into it.

### 1.3 Generalisation beyond the four books

**Metric** — accuracy on held-out books never used in development.
**Probe** — PDNC gold sets (`attribution_gold_pdnc_*.json`, 1270 / 640 / 584
rows) plus `attribution_gold_random.json`.
**Current** — the PDNC evaluation was **contaminated**: 25 of 28 books were in
training. Held-out performance measured **−4.4** against the contaminated
figure. **OPEN.**

**Target — a clean held-out number on ≥ 3 books, within 5 points of the
development books' figure.**

A method that only works on the four books it was tuned against is not a
feature of the app. The contamination is already known; what is missing is the
clean run.

---

## 2. Voice — does it sound like the target speaker

### 2.1 Speaker similarity against a human ceiling

**Metric** — ECAPA cosine similarity, generated vs the human reading, read
against `human_vs_human` (same narrator, different held-out line).
**Probe** — `app/experiments/ljspeech_score.py`.
**Current** — 2026-08-06, 150 held-out lines per language:

| language | ceiling | zero-shot clone | LoRA | clone as % of ceiling |
|---|---|---|---|---|
| English | 0.809 | 0.757 | 0.690 | 93% |
| Japanese | 0.796 | 0.779 | 0.755 | 98% |
| Chinese | *0.691 — anchor invalid* | 0.765 | 0.720 | — |

**Target — reach 95% of the ceiling in every language with a valid anchor.**
English is at 93% and Japanese at 98%, so this is close, and it is expressed as
a fraction of what one human achieves against herself rather than as a raw
similarity, because a raw ECAPA number is not interpretable on its own.

**A result worth stating plainly: zero-shot cloning beat the trained LoRA in
all three languages.** That replicates the earlier `clone_vs_lora` finding and
extends it across languages. It is a measured comparison, not a
recommendation — LoRA wins F0 correlation in English and Chinese, so timbre and
pitch-contour disagree about which is better.

### 2.2 Repair the Chinese anchor

**Metric** — `human_vs_human` must exceed every arm it bounds.
**Probe** — `find_invalid_anchors` in `ljspeech_score.py`, tested in
`app/test_score_anchor.py`.
**Current** — Chinese ceiling 0.691 sits **below** both its arms (0.720, 0.765):
the narrator matched herself worse than a synthetic voice matched her. **OPEN.**

**Target — a valid Chinese anchor, or a documented reason the eval set cannot
produce one.**

The leading hypothesis is clip length — Chinese medians 3.17 s against English
7.33 s and Japanese 4.71 s, and ECAPA degrades on short audio. **This is a
hypothesis, not a measurement.** The cheap test is to truncate LJSpeech clips
to ~3 s and see whether its ceiling collapses too. Until that runs, no Chinese
voice conclusion should be quoted.

### 2.3 Adapters that stop talking

**Metric** — median generated duration ÷ human duration, per adapter.
**Probe** — `app/experiments/verify_adapter_stops.py`, gate at 3.0x.
**Current** — 1.01x / 0.87x / 0.94x across the three languages. **MET.**

**Target — hold median within 0.8–1.25x, and never ship an adapter above
3.0x.**

This is the goal that already cost the most. Two adapters trained at the wrong
learning rate produced 163.8 s of audio for a 7.3 s line — every render, hitting
the token ceiling. Training loss looked ordinary (2.9 and 3.4), so only
generated output reveals it. Protected by the gate plus
`app/test_training_defaults.py`.

### 2.4 Duration fidelity in normal use

**Metric** — mean `dur_ratio` across held-out lines.
**Current** — LoRA 0.92 / 0.95 / 0.95; clone 0.97 / **0.76** / 0.90. **OPEN.**

**Target — every arm within 0.90–1.10.**

Japanese zero-shot clone at 0.76 is generating roughly three-quarters of the
expected length. Nothing flags it, and it is the same *class* of defect as the
runaway adapters — wrong duration, silent — but in the opposite direction and
far smaller. The stop-gate catches 3.0x, not 0.76x.

---

## 3. Reliability — does a run finish and produce the right thing

### 3.1 Chunk completion on script generation

**Metric** — chunks completing without exhausting retries.
**Probe** — `logs/review_responses.log`, per-run logs.
**Current** — mixed and model-dependent. In the 2026-08-06 run: mushoku16 9/9
clean, grimgar03 and owarimonogatari3 failed chunk 1 outright. **OPEN.**

**Target — ≥ 99% of chunks complete without manual intervention, on the
shipped model.**

Known failure modes, already characterised: Group A is a trigram-ceiling
near-miss and threshold-fixable; Group B is stochastic collapse with no content
trigger. The current failures are on qwen2.5-14b, which is **not** the model the
experiments were run against (qwen3-14b) — so this number is not yet a fair
measure of the shipped path.

### 3.2 Every generated file is real audio

**Metric** — files that are absent, empty, or unreadable after generation.
**Probe** — `validate_generated_audio` in `app/audio_validation.py`, funnelled
through `_save_wav` — all seven generation paths.
**Current** — 0 known escapes since the funnel was added. **MET.**

**Target — 0. Any regression is a release blocker.**

### 3.3 One character, one voice

**Metric** — distinct roster entries sharing a voice through a name-matching
bug.
**Probe** — `app/test_generate_personas.py`.
**Current** — fixed. Mr/Mrs Bennet and five other couples across 28 books were
merging into one voice; nothing failed, the audiobook was simply wrong. **MET.**

**Target — 0, with the couple case and the case-variant case both tested.**

Case variants (`EMILIA`/`Emilia`) must still merge; married couples must not.
Both directions are asserted, because the first fix broke the first case.

### 3.4 Reproducible output

**Metric** — identical seed and inputs produce byte-identical audio.
**Probe** — waveform SHA-256 comparison.
**Current** — seeded generation confirmed deterministic at temperature 0;
`character_voice_seed` now derives a stable per-character seed where 70 of 71
characters previously ran unseeded. **MET for TTS.**

**Target — extend to the LLM path: same seed, same model, same script output.**

---

## 4. Speed and cost

### 4.1 Faster than real time

**Metric** — generation seconds ÷ audio seconds.
**Current** — median **0.91x / 0.98x / 0.97x**, slowest 1.21x (n=300 per
language, RX 9070 XT). **MET, barely.**

**Target — median ≤ 0.90x, worst case ≤ 1.5x.**

A 10-hour audiobook currently takes about 10 hours of GPU time. That is the
number a user actually feels.

### 4.2 Local should not need the cloud

**Metric** — best local accuracy ÷ best cloud accuracy, per book.
**Current** — 97.2% / 98.7% / 97.5% / 99.0%. **MET.**

**Target — hold local within 5% of cloud on every book.**

Worth protecting deliberately. It means the app has no required cloud
dependency for its core task, and the `confirmIfRemote()` cost prompt stays a
convenience rather than a necessity.

---

## 5. Text handling

### 5.1 Nothing unspeakable reaches the TTS

**Metric** — characters passed to TTS that have no spoken form: `■`, `♪`, `∞`,
pictographic CJK in non-CJK text.
**Current** — **NO BASELINE.** No gate catches these; they reach the engine
silently.

**Target — measure the rate first, then drive it to 0 with a verbalization
pass.**

This is listed without a target number on purpose. The rate is unknown, and a
target invented before the measurement would be a wish.

### 5.2 Names pronounced consistently

**Metric** — character names spoken the same way across a book.
**Probe** — `app/pronunciation.py`, `pronunciation.json` (ships empty).
**Current** — infrastructure exists; the lexicon is empty, so the effective
rate is unmeasured. **NO BASELINE.**

**Target — a populated lexicon for the shipped demo book, and 0 substitutions
that alter a non-name word.**

The hazard is already handled in code and tested: `Felt` the character must not
rewrite `felt` the verb (242 against 65 occurrences in one book), and an alias
(`BETTY` → `BEATRICE`) records identity, never pronunciation — substituting
across it would put a word in the audio that is not in the book.

### 5.3 Three-pass vs single-pass generation

**Metric** — accuracy of `three_pass_generate.py` against the shipped single
pass, paired on line id.
**Probe** — `app/experiments/three_pass_vs_single.py`.
**Current** — **NO BASELINE.** The repository has carried a second generation
architecture, with six settings and three prompt files, that nothing invokes and
nothing has scored.

**Target — one clean comparison, then wire it in or delete it.**

Both outcomes are fine. Carrying an unmeasured alternative indefinitely is not.

---

## 6. Measurement integrity

Goals about the instruments. These earned their place by failing.

### 6.1 A ceiling must bound its arms

Covered at 2.2. Enforced by `find_invalid_anchors`, reported in every score
artifact as `anchor_invalid`.

**Target — 0 comparisons published from an eval set with an invalid anchor.**

### 6.2 One source per decision

**Metric** — settings defined in more than one place.
**Current** — two found and fixed: the training learning rate (four
definitions, three of them wrong, and the wrong one shipped in the UI), and
`is_remote_llm`. One known outstanding: `config["llm"]` versus
`config["llm_local"]`, which cost an hour on 2026-08-06 when a run dialled a
dead endpoint while a working server sat idle. **OPEN.**

**Target — 0 known parallel definitions; each new one gets a test that asserts
the copies agree.**

### 6.3 Indexes describe committed state

**Metric** — index checks passing on a clean checkout.
**Current** — **MET**, after `collect_results.py` was found scoring gitignored
files and stamping rows with file mtimes. It passed locally and was permanently
stale in CI.

**Target — every index check passes from a fresh clone with no untracked
files.**

### 6.4 No skipped tests

**Metric** — tests skipped in the release verifier.
**Current** — 0. **MET.**

**Target — 0. A skip is a failure, per Rule 8.**

Three tests were skipping because they read live on-disk data absent from CI.
Rewriting them as fixtures immediately exposed a real defect in the collision
predicate they were supposed to be testing — the skip had been hiding it.

---

## Priority

If only three things get worked on:

1. **Selection (1.2)** — 85% supply against 29.9% use is the largest known
   headroom anywhere in the app.
2. **Chinese anchor (2.2)** — until it is valid, one third of the voice
   evidence cannot be read.
3. **Three-pass baseline (5.3)** — decide the fate of a whole subsystem.

## Rules for changing this file

- A current value moves only with an artifact and a date.
- A target moves only with a stated reason.
- Never delete an OPEN goal because it proved difficult. Convert it, or record
  why it was abandoned.
- Do not add a target without evidence that it is reachable. `NO BASELINE` is a
  respectable status; an invented number is not.
