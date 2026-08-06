# What the experiments say to change in the code — 2026-08-05

Ranked by measured value, not by how interesting the work is. Each item states
what was measured, what the measurement does **not** cover, and what changes.

The recurring shape: **most of this week's evidence has not reached production.**
The experiments improved, the harness improved, the tests improved. `app/` is
largely where it was.

---

## 1. The +14.6 attribution gain is measured and unused

**Measured.** `lora_serving_eval__mixed-shippable.json`, `validation=ok`,
547 lines, Q4_K_M base + f16 LoRA through llama.cpp:

| book | base | +LoRA |
| --- | ---: | ---: |
| grimgar03 | 64.4% | **80.3%** |
| owarimonogatari3 | 45.7% | **57.4%** |
| overall | 58.9% | **73.5%** |

And `distill_eval__pdnc_only.json`, 772 lines, positive on all four books,
60.0% → 69.8%, disagreement 149–73 in the adapter's favour.

**The gap.** `grep -rn 'lora-adapters' app/ --include='*.py'` outside
`experiments/` returns nothing. Production attribution builds an OpenAI client
against `base_url` and knows nothing about adapters.

**Why this is the cheapest item on the list.** The adapter lives in the
*serving* layer, not the application. `llama-server` is started with `--lora`
and the scale is toggled via `POST /lora-adapters`. So the application code
does not need to change to benefit — the server it points at does. What the
code *should* gain is:

- the adapter path and scale as `config.json` settings beside `base_url`, so a
  run records which adapter produced it;
- a startup check that the configured adapter is actually loaded, failing loud
  rather than silently scoring at base quality. `lmstudio_settings` already has
  the shape for this.

**Not covered.** Two books, not four, in the serving eval. Nothing measures
whether the adapter helps or hurts on English prose, which is what PDNC
training was drawn from. Do not assume it generalises past light novels.

---

## 2. `tts.py` still accepts audio it has not looked at

**Measured today, on a real file.** Truncating a 195,884-byte render to 5,000
bytes still *decodes* — libsndfile returns the 2,478 frames that happen to be
present rather than raising.

**The gap, and it is now asymmetric.** `project.py` validates (3 call sites,
including per-chunk at line 1167). `tts.py` has **zero** validation calls and
**11** `os.path.exists`/`getsize` checks — the exact pattern that let stale
audio be scored as fresh in six experiment harnesses.

Asymmetric coverage is worse than none, because the presence of validation in
the assembly layer makes the pipeline look guarded when its generation layer is
not.

**Change.** Route `tts.py`'s generation paths through
`audio_validation.validate_generated_audio`, the same function `project.py` and
`experiments/generation.py` already use. One implementation, three callers —
the alternative is three that drift, which is how this class of bug started.

---

## 3. 70 of 71 characters have a random seed

**Measured.** Seeded generation is byte-identical across fresh processes on
three adapters (SHA-256 equality). Unseeded, the same adapter moves across a
**32.4 Hz median band** within a voice, and an extreme instruction moves
duration 1.36–1.43×.

**Current state.** `voice_config.json`: 1 character seeded, **70 random**.

**What that means for a listener.** Every line of a character is an independent
draw of that voice. This is the defect the user identified by ear as "multiple
narrators" — it was fixed in the sense that the seed is now *honoured*, and not
fixed in the sense that production still does not *set* one.

**Change.** Assign a stable per-character seed at cast time. A character's
voice should be a fixed draw for the whole book, not a fresh draw per line.

**Deliberately not:** one global seed. That would make every character the same
draw — reproducible and wrong, which is why `test_seed_plumbing` asserts no
generation path seeds from a literal.

**Not covered.** Nobody has *listened* to a seeded book against an unseeded
one. The 32.4 Hz is a measurement; that it is audible over a chapter is an
inference.

---

## 4. The non-prose gate exists and is not connected

**Measured.** At matched length, with symbols already normalised, non-prose
fails **11/25 (WER 29.40%)** against prose **0/25 (0.89%)**. Splitting does not
fix it (8/8 fail both ways). No ablation — digits, caps, punctuation, syntax —
clears a single failure.

**Current state.** No reference to the gate anywhere in `tts.py`, `app.py` or
`project.py`.

**Change, narrower than I first said.** Connect it to **flag** front matter,
not to route it away. I wrote earlier that non-prose should be routed away from
TTS as general policy; that rested on 8 segments, one adapter, one seed, and
the reviewer was right that it does not support a general policy. Flagging is
supported by the current evidence. Routing needs Stage 4.

---

## 5. `mean_f0` in the manifest is not fit for the use it invites

**Measured.** Declared vs seeded measurement across six adapters spanning the
pool: **12.9 Hz mean absolute error, worst 29.7 Hz**, against a within-voice
spread of 32.4 Hz median — and per-adapter spread ranges 14.4 to 71.9 Hz.

**The problem.** All 75 declared values were profiled before the seed fix, so
each is a mean over independent draws. They are close enough to rank voices and
too coarse to *separate* them by number, which is exactly what a casting
constraint would do.

**Change.** Either re-profile seeded, or mark the field as indicative in the
manifest so the next person does not compute pairwise distances from it. The
second costs nothing and prevents the error; the first costs ~3 GPU hours and
is only worth it if pitch will actually gate casting.

---

## 6. The Voice Lab pipeline's justification is not currently supported

**Measured**, on the reviewer's Stage 3 rerun with a proper ECAPA speaker
embedding, seeded, always regenerating: **mean LoRA − clone = −0.0236, adapter
ahead on 4 of 9 voices.** On under-trained adapters (<100 samples), +0.0069.

**What that questions.** Preparer → dedup → LoRA training → profiling → naming
is the most expensive subsystem in this repo, and it exists on the premise that
a trained adapter holds a voice better than zero-shot cloning from the same
reference. On this evidence it does not.

**Explicitly not a verdict.** n=9, one sentence, one seed. The old artifact is
gone from disk, so the old and new numbers cannot be compared directly — two
variables changed at once (metric *and* stale audio). This is the strongest
version of the measurement so far and still a first clean result.

**Change.** Do not rip anything out. Re-run at n≈30 with several sentences
before treating it as settled. But stop citing the pipeline's value as
established, because right now it is not.

---

## 7. Production outputs carry no provenance

Experiments now record script, commit, dirty tree, host, arguments and seed.
A generated audiobook records none of that — not the voice config, not the
seeds, not the commit, not which adapter attributed the script.

**Change.** Write a provenance block beside the export. `experiments/
provenance.py` is already the shape; the work is choosing what an *export*
needs, which is not identical to what an experiment needs.

This is the item that pays off when someone asks "why does chapter 9 sound
different" three weeks from now.

---

## What is already done, so it is not re-litigated

- one shared renderer with stale-output removal, return-value checking, and
  decodable/truncation rejection (`experiments/generation.py`, 17 tests)
- GPU serialisation behind a real `flock` gate, with deployment identity logged
  before every job (`gpu_job.sh`, 11 tests, mutation-checked)
- machine-specific paths removed from every tracked Python file, enforced by
  `test_no_machine_paths.py` over 283 files
- seed plumbing asserted at source level across all seven generation paths
- the results index now declares what it cannot represent (68 of 230 artifacts)
- corpora moved out of a session scratchpad whose path embedded a session UUID

---

## Order I would take them

1. **§2, `tts.py` validation** — smallest, and it is the one that can ship a
   defect to a listener.
2. **§1, adapter in config** — largest measured gain, and mostly configuration.
3. **§3, per-character seeds** — small change, directly addresses something a
   listener already noticed.
4. **§5, mark `mean_f0` indicative** — one line, prevents a repeat of an error
   already made once.
5. **§4 and §7** — real, but neither is blocking anything today.
6. **§6** — measure before deciding; the decision is expensive either way.
