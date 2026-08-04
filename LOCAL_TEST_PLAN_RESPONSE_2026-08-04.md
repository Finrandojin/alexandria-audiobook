# Response to LOCAL_TEST_PLAN_2026-08-04.md

Agreement on the plan's shape, one concession where it corrects a claim of
mine, one dependency it treats as blocked that is not, and measured throughput
numbers because two of the stage estimates are large enough to change what gets
run.

I am staying off the GPU while Stage 1 runs.

---

## The plan's central move is the right one

> Test the assumptions behind the current claims **without treating an
> assumption as a result.**

Every error I made today was that substitution. Not a reasoning failure — a
measured number attached to a sentence wider than the measurement:

- "the path migration is finished" — one pattern migrated, 41 references left
- "the queue is chained correctly" — deadlocked on a dead PID at the time
- "the three sibling paths all seed" — two of them are dispatchers with no seed
  logic at all
- "0.20 / 0.20 / 0.41" — the unseeded artifact, quoted in the seeded column

The plan's stage gates are the structural fix for that, and they are stricter
than anything I imposed on myself.

---

## Where the plan corrects me: Stage 4

I wrote, after the ablation came back null:

> the failure is joint, the model is out of distribution on this register as a
> whole, and non-prose should be routed away from TTS rather than repaired —
> the gate is the answer, not a stopgap.

That rests on **8 segments, one adapter (`husky_tenor_30s_m_fantasy`), one
seed.** The plan's gate — no general routing recommendation until the effect
survives adapters, seeds, categories and matched controls — is correct, and my
sentence claimed more than the run supports.

The *measurement* stands: no ablation cleared a single failure, and it
reproduced identically across two runs. The *policy conclusion* does not.

**Two things from my data that should shape Stage 4's design:**

1. **Length is uncontrolled and is a live candidate.** The one segment of eight
   that passed was the shortest at 64 characters, and `digits`, `caps` and
   `syntax` each took it to 0.0% WER. Nothing in my set separates "non-prose"
   from "short fragment".
2. **Segment 7 produced WER above 100%** — 180.0% on three arms. Legitimate for
   WER (insertions can exceed the reference) but it means the model generated
   substantially *more* than a 65-character fragment asked for. Pooled mean WER
   hides that; the plan's "report by adapter and seed, not only pooled" should
   probably also report insertion rate separately from substitution.

---

## Where the plan is more conservative than necessary: Stage 8

The plan calls Stage 8 "conditional on a functioning local speaker-embedding
dependency." That dependency exists on this machine — in the other interpreter.

```
app/env/bin/python                      speechbrain  MISSING
alexandria-audiobook.git/app/env/bin/python
                                        speechbrain  1.1.0
                                        torchaudio   2.7.0+rocm6.3
```

And the fallback is quiet: `voice_data_saturation.embedder()` catches the
ImportError and returns `None`, dropping to acoustic-feature distance. It
prints which metric ran, so it is not concealed — but under `app/env` both
Stage 3 and Stage 8 will score on the crude metric unless someone reads that
line. **The plan's own rule about not converting acoustic metrics into
listener claims is exactly the thing this would silently violate.**

The workable split is **generate under `app/env`** (torch 2.10, transformers
4.57 — required for Qwen3-TTS) and **score under the sibling** (speechbrain +
torchaudio). That is already how the Voice Lab pipeline splits profiling, so it
is an established pattern rather than a new one.

---

## Measured throughput, because two estimates are far off

From today's queue log, separating generation from scoring:

| | s/render | source |
| --- | ---: | --- |
| generation only | **5.8** | `instruct_listening`, 24 renders in 2.3 min |
| generation + whisper | **15.9** | `nonprose_mechanism`, 80 renders in 21.2 min |
| ASR overhead | 10.2 | difference |

Applying those to the plan as written:

| stage | renders implied | measured estimate | plan estimate |
| --- | ---: | ---: | ---: |
| Stage 4 (3 adapters × 3 seeds × 8 passages × 2 classes) | 144 | **0.6 h** | 5–9 h |
| Stage 7 (75 adapters × 8 passages × 3 seeds) | 1,800 | **2.9 h** + tracking | 12–24 h |

Two readings, and I cannot tell which is intended:

- If the estimates assume a much larger corpus than the arithmetic above, the
  plan should say the intended N, because at 0.6 hours Stage 4 could run
  **many** more adapters and seeds for the same budget — and more adapters is
  precisely what would settle whether the finding generalises.
- If the estimates are padding for failure and rerun, that is reasonable, but
  it hides that Stage 7's *marginal* cost per additional adapter is about 2.3
  minutes. Pitch profiling the whole pool is cheap. The reason to defer it is
  that pitch may not gate casting at all — not runtime.

**The practical consequence:** Stage 4's stated blocker is GPU hours, and it
probably is not. Widening it from 3 adapters to 10 costs roughly an hour and
directly attacks the "one adapter" weakness in my result.

---

## Stage 9 partly exists

Already committed and mutation-checked:

- `test_gpu_job.py`, 11 tests — success, failed `flock`, unopenable lock file,
  two-process serialisation, blocking rather than dropping a queued job,
  interruption while waiting, exit-code propagation, queue-log ordering,
  identity-before-START, identity degradation with git and both smi tools
  absent, misuse. Verified against a byte-faithful copy of the pre-fix script:
  exactly the two gate tests fail, the other nine pass.
- `test_generation_guard.py`, 10 tests — stale output removed before
  generation, false returns, missing files, empty files, `None`-on-success.

**Genuine gaps from the Stage 9 list:**

1. **Timeout behaviour.** Nothing tests that a wedged job is killed and the
   lock released. This matters more than it looks: a hung TTS render holds the
   card indefinitely and everything queued behind it waits silently.
2. **Invalid-WAV rejection.** ~~Not implemented.~~ **Now closed** — see the
   second addendum. Taken first because Stage 3's whole point is auditing
   audio, and a validator that runs after Stage 3 validates nothing.

Timeout behaviour remains open.

---

## Smaller notes

**Stage 3 will run slower than the old artifacts suggest.** Both harnesses now
always regenerate — the reuse *was* the defect — so any timing derived from the
cached runs will not transfer.

**Stage 2's classification scheme is the useful part of this plan.** Applying
supported / provisional / exploratory / invalid to 228 artifacts is the thing
that stops a number being quoted three weeks from now with its caveats lost. I
would add one field: **whether the artifact's conclusion was restated anywhere
else** — a wrong number in an artifact is contained; a wrong number copied into
a work log, an index and a reply is what actually happened today.

**On Stage 6 and the human verdict.** The plan is right that automated systems
must not supply it. Concretely, `instruct_listening.py` produces fixed-order,
spoken-labelled files — a listening *aid*, explicitly not a blind test. Stage 6
needs shuffled order and a withheld key, which is a different artifact, not a
relabelling of mine. The four files on disk should not be used as Stage 6
material.

---

## What I would run first, if it were my call

1. **Invalid-WAV rejection** (Stage 9, ~30 min, CPU) — before Stage 3, because
   Stage 3 is auditing audio.
2. **Stage 4 widened to ~10 adapters** (~1 GPU hour by measurement) — it
   directly attacks the weakest link in a conclusion I already stated too
   broadly.
3. **Stage 2's audit** (CPU, off the GPU entirely) in parallel with the above.

Stage 7 and Stage 8 are both larger and both gated on decisions that are not
yet made — whether pitch will gate casting, and whether weight norm predicts
identity. Running them before those decisions is how 12–24 GPU hours gets spent
on a question nobody is waiting on.

---

## Addendum — Stage 1 completed, verified independently

`seed_instruction_controls_rerun` finished 21:51, 3.5 min, and the gate passes.
I re-derived it from the WAVs rather than reading the artifact's own flags.

**Determinism, recomputed from the files:**

| speaker | same-seed hash A | B | |
| --- | --- | --- | --- |
| NARRATOR | `4d0d24fc6120` | `4d0d24fc6120` | match |
| EMILIA | `9c4e0f519c90` | `9c4e0f519c90` | match |
| NATSUKI SUBARU | `d88603a06ac5` | `d88603a06ac5` | match |

Different seed changes the hash in all three. **This is a stronger control than
mine** — mine was same-process, so it could not have caught state carried across
a process boundary. Cross-process byte equality on three adapters closes that.

**The instruction positive control, measured from the audio:**

| speaker | slow | neutral | fast | slow/fast |
| --- | ---: | ---: | ---: | ---: |
| NARRATOR | 4.80 s | 4.08 s | 3.36 s | 1.43× |
| EMILIA | 4.56 s | 3.84 s | 3.36 s | 1.36× |
| NATSUKI SUBARU | 5.12 s | 4.48 s | 3.60 s | 1.42× |

Monotone slow > neutral > fast on every adapter. The plan is right that this is
plumbing, not quality — but it settles something my `instruct_value` run could
not, and it changes how that null should be read.

**Instructions reach the model. They just do not change the words.** The
seeded arms were 0.611% / 0% / 0% WER, and I described that as instructions
having no measurable effect on content. That is still true and now better
supported: an extreme instruction moves duration by 40%+ while content stays
identical. The earlier flat result is evidence about *subtlety*, not about
instructions being ignored — which is a materially different reading, and the
one that justifies Stage 6 rather than abandoning the question.

### One gap in the artifact

The JSON records `duration_order_control_passes: true` for each adapter but
**does not record the renders it is based on**. 18 WAVs exist on disk —
`_instruction_very_slow`, `_instruction_neutral`, `_instruction_very_fast` for
each speaker — while the artifact's `renders` map holds only 9, the seed
controls. The durations above had to be measured from the audio because the
artifact does not carry them.

The control ran and passed; the log confirms `slow_gt_fast= True` for all
three. But a published boolean whose supporting numbers are absent is the same
shape as the problem this whole exercise is about — and by the plan's own
Stage 10 rule ("validate the JSON shape, output files, and provenance"), the
artifact is incomplete. One-line fix: put the three instruction renders in
`renders` alongside the seed ones.

Worth noting what the artifact does well, since it is new: it carries a full
provenance block with commit, dirty-tree file list, harness SHA-256 and seed,
and it states its own limits in an `interpretation` field — "delivery quality
still requires blinded listening". That is the standard the older artifacts
should be brought to.

---

## Second addendum — decodable-WAV validation, and what building it taught me

I flagged this as the highest-value gap and then closed it, since it is my code
and needs no GPU. Card untouched; it is still yours.

**The first version I wrote was wrong, and a real file proved it.** I added
`sf.info()` as the decode check, wrote a test using a header-only stub, and it
passed. Then I truncated an actual render from your Stage 1 output:

```
NARRATOR_same_seed_a.wav   195,884 bytes -> decodes, 97,920 frames
truncated to 5,000 bytes   ->  still DECODES, 2,478 frames
```

**libsndfile returns the frames that happen to be present rather than
raising.** So a run killed mid-write produces a short, valid, wrong file that
`sf.info` accepts — and my decode check would have passed it while I reported
decodability as validated. The synthetic stub in my test was the easy case and
hid the real one.

The fix compares the **declared RIFF size against the bytes on disk**:

```
header declares 195884 bytes, file is 5000 (190884 missing)
```

Tolerant in one direction on purpose — a file *larger* than declared is legal
(trailing metadata chunks are common), so only a shortfall is an error. It also
skips the check when the declared size is near 2^32, because this repo has
already been bitten by the 32-bit WAV size field wrapping on long audiobook
files, where the wrapped value is meaningless rather than a shortfall.

`render()` now rejects: undecodable files, valid headers with zero frames, and
files shorter than their own header claims. **17 tests**, including one that
asserts the truncated file *does* decode — a guard against the guard, so the
test cannot pass merely because the file is unreadable, which is exactly what
made my first version look correct.

**1,225 tests, 0 failures.**

### Stage 10 housekeeping done

`collect_results.py` regenerated: **230 artifacts, 483 arm rows, 68 not
indexed.** `seed_instruction_controls.json` appears under **Not indexed**,
correctly — it is a control artifact, not per-arm attribution accuracy, and the
plan already anticipates that. Absence from the arm table is not absence of a
result.

### What this changes for Stage 3

Stage 3 reruns `clone_vs_lora` and `voice_data_saturation` because their old
harness could reuse stale audio. Both now go through `render()`, so they also
inherit truncation and decodability rejection — which they did not have when I
described them as fixed. That was accurate about the reuse defect and silent
about this one.

Two more places worth pointing the same check at, which I have **not** done:
`tts.py`'s own generation paths write audio without validating it, and
`project.py` assembles chunks into exports without checking that each chunk
decodes. A truncated chunk would reach a finished M4B. That is production code
rather than harness code, so it is a larger change than I would make while your
plan is mid-flight.
