# Speaker attribution: current evidence and decisions

Date: 2026-07-27

Repository: `alexandria-audiobook2.git`

Branch: `agent/model-comparison`

Audience: an external reviewer with repository access and no session history

This replaces the earlier chronological brief. Superseded interpretations and
reviewer dialogue were removed; material corrections and provenance caveats
remain.

## Executive summary

Speaker attribution remains Alexandria's largest measured quality bottleneck.
The work has improved the measurement system, ruled out many proposed fixes,
and shown that model choice matters. It has not produced a configuration near
unattended production quality.

The defensible conclusions are:

1. **Selection, not candidate recall, is the main measured failure.** On the
   Mushoku fixture, the full roster contains the correct speaker for 85.0% of
   lines, while the original shipped result was only 44/147 = 29.9%.
2. **The original qwen3.5-9b is weaker than the tested 14B-class alternatives
   on the controlled decomposition.** Changing models is the only intervention
   that currently survives the investigation as a useful direction.
3. **No prompt, roster, candidate, voting, or confidence intervention tested
   so far has demonstrated a shippable production gain.**
4. **The apparent `because` improvement does not transfer to the production
   path.** The production-path reversal is persuasive but still provisional
   because its run produced only an incomplete log, not a final JSON artifact.
5. **The evidence does not establish a 90% path, an intrinsic model ceiling, or
   that Gemma is significantly worse than Qwen end to end.**

The next high-value action is a clean, production-only baseline-versus-
`because` rerun that writes and validates an artifact. After that, settle the
shipping configuration before spending more GPU time on a full-book matrix.

## 1. Pipeline and constraints

The three LLM-assisted passes are:

1. segment prose into frozen `NARRATOR` and `SPOKEN` entries;
2. attribute each `SPOKEN` entry to a speaker;
3. add delivery direction.

Pass 2 receives batches of about 25 entries, each target's text and ±1
neighbours, a running character roster, and a `{n, speaker}` output contract.
Narration is deterministic; attribution may add only `speaker`.

Relevant locations:

- `app/three_pass_generate.py`
- `app/pass_quality.py`
- `app/attribution_accuracy.py`
- `app/fixtures/attribution_gold_random.json`
- `app/experiments/`
- `ab_test_runtime/experiments/`

Inference is local, runs on one consumer GPU, and is serialized. LM Studio
`parallel: 1`, VRAM headroom checks, retries, checkpointing, and the global GPU
lock are deliberate safety constraints and must not be weakened for testing.

## 2. Measurement foundation

The principal fixture contains 147 randomly sampled Mushoku 16 lines with
hand-judged speakers. Independent-reader agreement was 94% on 63 overlapping
Mushoku lines and 97% with alias credit on 35 Grimgar lines.

The scorer now:

- aligns by full normalized text rather than a 60-character prefix;
- honors fixture-declared aliases;
- rejects duplicate gold identities;
- excludes repeated-text cases that cannot be aligned uniquely;
- recomputes summaries from per-line rows.

Alias-aware scoring raised the original Mushoku baseline from 20.4% to 29.9%;
14 lines had previously lost credit solely because of `RUDEUS`/`RUDI`
spelling.

Temperature-zero attribution was deterministic on an idle GPU. Earlier
variation was caused by concurrent requests sharing LM Studio, not useful
sampling noise.

### Valid-artifact requirements

A result used for a decision should record:

- exact arms and identical expected gold-ID sets;
- expected denominator and no duplicate `(arm, gold_id)` pairs;
- summaries recomputed from rows;
- fixture, prompt, and harness hashes;
- decoding settings and actual loaded model;
- context length, parallel setting, and optimized state;
- clean commit provenance, or exact fingerprints of relevant dirty files.

“Validation: ok” proves internal consistency only. It does not by itself prove
that the harness represented the production path or that the source is
reconstructable from the recorded commit.

## 3. Baseline and error structure

| book/sample | measured result |
|---|---:|
| Mushoku 16 original random fixture | 44/147 = **29.9%** |
| Grimgar 03 early judged sample | approximately **54%** on 35 lines |

Of the 103 Mushoku baseline errors:

| error class | share |
|---|---:|
| wrong real character | 64% |
| invented name | 33% |
| `UNKNOWN` | 3% |

Nearby names were originally treated as stronger evidence than they are.
Reclassification found:

| nearby relation | share of errors |
|---|---:|
| name absent | 62.1% |
| bare mention | 18.4% |
| vocative/addressee | 12.6% |
| speech-verb tag | 6.8% |

Thirteen errors are explicit addressee/speaker inversions. A nearby name often
identifies the listener rather than the speaker.

## 4. Consolidated experiment ledger

Results below are scoped to their fixture, model, prompt, and harness.

| intervention or diagnostic | current conclusion |
|---|---|
| full-roster recall | correct speaker available on 85.0% of Mushoku lines |
| oracle small candidate set on tested 9B | 49.0% conditional selection; pruning alone is insufficient |
| explicit context | helpful in the tested 9B decomposition |
| scene-local candidates | no demonstrated gain over full roster |
| roster warm-up | early +5.1 result did not survive later paired/model checks as a production recommendation |
| candidate-ID output contract | worse than free-form speaker names in its experiment |
| deterministic speech tags | corrected recall 10.2%; too sparse to carry attribution |
| model ensemble unanimity | alias-normalized coverage 17.0% at 76.0% accuracy; not shippable |
| self-consistency voting | 69/139 versus baseline 69/139; null |
| narration included in batch | 48/139 versus 69/139; harmful, paired p≈0.001 |
| narrator hint | 72/139 versus 69/139; no significant gain, p≈0.720 |
| prose-passage representation | 66/139 versus 69/139; no significant effect, p≈0.771 |
| model swap from qwen3.5-9b | only direction that remains supported |

The four production-path rechecks are stored in
`ab_test_runtime/experiments/reexamine__qwen__qwen3-14b.json`. The row sets and
summaries validate, and the artifact contains a harness SHA-256. However, it
records `dirty: true` and says the harness was untracked at run time. It is
arithmetically inspectable but not a clean-commit experiment.

## 5. The reasoning experiment and its reversal

The simplified reasoning harness tested 139 unambiguous lines with
`qwen/qwen3-14b`:

| arm | correct | accuracy | paired result vs baseline |
|---|---:|---:|---:|
| baseline | 55/139 | 39.6% | — |
| `because` | 70/139 | 50.4% | +20/−5, p≈0.004 |
| scaffold | 57/139 | 41.0% | p≈0.885 |
| thinking | 58/139 | 41.7% | p≈0.690 |
| scaffold + thinking | 67/139 | 48.2% | p≈0.088 |

That artifact is
`ab_test_runtime/experiments/reasoning_arms__qwen__qwen3-14b.json`. It is
internally validated but records:

- `dirty: true`;
- a modified harness;
- `optimized: false`;
- a commit that does not itself contain the exact recorded source state.

The significant result therefore supported a hypothesis, not a production
decision. The intervention changed both the prompt and the required response
schema, so “output expressiveness” is also too narrow a causal label.

The production-path recheck used the shipping `attribute_batch` prompt:

| arm | simplified harness | production path |
|---|---:|---:|
| baseline | 55/139 = 39.6% | 69/139 = 49.6% |
| `because` | 70/139 = 50.4% | 59/139 = 42.4% |

The best interpretation is:

> A justification clause improved a weakened experimental baseline but showed
> no production benefit and likely harmed the shipping prompt configuration.

Do not say the intervention “was never helping.” It helped relative to the
simplified baseline; it did not transfer to the configuration that matters.

### Remaining evidence gap

The production-path numbers currently exist in
`ab_test_runtime/results/overnight_20260726-185022/because_production.log`.
That run continued into `scaffold_thinking` and ended before writing its final
JSON artifact. The reversal is strong evidence, but it is not yet
artifact-grade.

Required closure:

1. run only production baseline and production `because`;
2. start from a clean recorded commit;
3. freeze the 139 IDs, prompt, roster, model, and LM Studio settings;
4. write per-line rows and paired transitions;
5. validate the complete artifact before updating the ledger.

## 6. Model comparison: what is and is not known

The controlled closed-set decomposition supports moving off qwen3.5-9b and
testing 14B-class candidates. It does not prove that the task has a fixed model
ceiling.

The overnight full-book 2×2 matrix did **not** complete:

| cell | status |
|---|---|
| Gemma / Grimgar | partial; 15 failures |
| Gemma / Mushoku | partial; 9 failures |
| Qwen / Grimgar | partial; 1 failure |
| Qwen / Mushoku | complete |

On the current random fixture:

- Gemma/Mushoku: 42/146 = **28.8%**, with one unaligned row;
- Qwen/Mushoku: 56/147 = **38.1%**.

On the 146 rows shared by both outputs, Gemma has 42 correct and Qwen 55.
Their discordant counts are 15 Gemma-only and 28 Qwen-only, giving exact
McNemar p≈0.066. This favors Qwen numerically but does **not** establish a
statistically significant end-to-end difference at 0.05.

Do not describe the matrix as completed or claim that Gemma is measurably
worse from this run. Fixture choice also matters: an older 40-line fixture
scores both outputs at 13/40.

`mistralai/magistral-small` was not tested because its 13.51 GiB weights did
not safely fit the 15.92 GiB card. This was a deliberate VRAM-safety decision.

## 7. Human judging and second-book validation

Four hundred Grimgar 03 rows were independently judged in ten batch files.
Mechanical validation found:

- all 400 expected IDs present;
- no duplicated or missing IDs;
- 20 marked `AMBIGUOUS`;
- 3 marked `NARRATOR`;
- source batch files unchanged.

The expanded-window/rejudge tooling exists because some rows cannot be judged
fairly from a narrow excerpt. Human-listening review and attribution scoring
remain separate tasks: a scoring fixture can evaluate model ranking without
being sufficient to approve audiobook quality.

Finish and freeze one second-book fixture before committing to a larger judging
queue. Additional labels should buy a specific decision, not merely a tighter
aggregate.

## 8. Current decisions

### Supported

- Preserve the repaired scorer, fixture identity rules, validators, and
  per-line artifacts.
- Treat speaker selection as the main measured bottleneck.
- Prefer a stronger tested model over qwen3.5-9b for subsequent work.
- Keep narration deterministic.
- Retain unattested-speaker rejection and name-attestation repairs.
- Evaluate on at least two books with different narrative structure.

### Not established

- unattended 90%+ attribution;
- an intrinsic ceiling for any model or for the task;
- a useful confidence/coverage operating point;
- a production benefit from warm roster, reasoning fields, scaffolded
  questions, thinking tokens, voting, or candidate IDs;
- a significant end-to-end Gemma/Qwen difference from the partial matrix;
- generalization from Mushoku to Grimgar.

### Do not do yet

- do not ship the `because` field;
- do not resume the full matrix against a changing pass-2 configuration;
- do not build a two-model “writer then JSON converter” pipeline merely to
  repair formatting—the attribution model already returns parseable JSON, and
  a formatter cannot repair a wrong speaker choice;
- do not weaken retries, VRAM guards, checkpointing, or the global GPU lock.

## 9. Recommended next steps

1. **Close the `because` question.** Run the clean two-arm production test
   described above.
2. **Choose a settled attribution model/configuration.** Use the controlled
   decomposition, latency, memory fit, and completion reliability—not the
   incomplete full-book matrix alone.
3. **Freeze the Grimgar scoring fixture and policy.** Report ambiguous and
   unaligned rows separately.
4. **Run a compact two-book production comparison.** Preserve segmentation and
   score paired per-line transitions.
5. **Only then consider routing.** Report accepted accuracy against coverage;
   raw model agreement is not enough.

If a two-model design is revisited, split by semantic responsibility, not by
serialization:

- model A proposes the speaker plus compact evidence;
- model B independently verifies or challenges uncertain cases;
- deterministic code validates and converts the final result to JSON.

This costs more than one pass and is justified only if the verifier produces a
useful high-precision subset or repairs enough errors to beat a stronger single
model.

## 10. Reviewer assessment

My reading is that the investigation produced durable measurement
infrastructure and a useful model-selection result, but no architectural
breakthrough. That is still valuable: several attractive ideas now have paired
evidence against them, and multiple instrument defects were found before those
results became product changes.

The most important methodological correction is symmetrical:

- a negative can be caused by a broken instrument;
- a positive can be caused by a weak comparison.

Every intervention should therefore be tested against the exact configuration
that could ship. Simplified harnesses are excellent for generating hypotheses,
not for declaring production wins.

The evidence for the production `because` reversal is compelling enough to
stop pursuing it casually, but not complete enough to close the record. A
short clean rerun is cheaper and more informative than debating the incomplete
log.

The model result should also be stated narrowly. The evidence supports moving
off the original 9B; it does not yet select a universal winner, prove that
Gemma is inferior end to end, or show that model scaling alone will reach the
quality target.

Finally, the product may need a human-assisted success criterion. If no model
approaches unattended accuracy, the relevant question becomes whether the
system can automatically accept a large, high-precision subset and route the
rest for efficient correction. No tested confidence signal has yet
demonstrated that operating point.

## 11. Handoff checklist

Before accepting any new headline result, verify:

- the run used the production call path or is labeled exploratory;
- all expected arms and IDs are present;
- aggregates recompute from rows;
- the actual loaded model and LM Studio state match the declaration;
- source provenance is reconstructable;
- the comparison is paired where possible;
- partial runs are not presented as completed;
- statistical non-significance is not described as equivalence;
- conclusions remain scoped to the tested books and fixtures.

The earlier 34-section discussion is intentionally not retained here. Git
history preserves it at commit `2ce90a9` if the full chronology is needed.
