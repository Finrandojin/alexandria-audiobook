# Speaker attribution: current evidence and decisions

Date: 2026-07-27

Repository: `alexandria-audiobook2.git`

Branch: `agent/three-pass-generation`

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
   path.** This is now closed: the clean production-only rerun requested in the
   previous revision completed and validated (§5).
5. **The evidence does not establish a 90% path, an intrinsic model ceiling, or
   that Gemma is significantly worse than Qwen end to end.**
6. **Two independent generalization failures are now measured, and they are the
   most important new result.** The same comparison changes answer when the
   book changes, and again when the harness changes (§6). Both are documented
   below with paired statistics. Every single-book, single-harness conclusion in
   this brief — which is nearly all of them — should be read against that.

Consequently the highest-value action is no longer another intervention. It is
to establish how far any of the existing results generalize, because the
project has now been surprised twice in one day by results that were treated as
settled.

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

### Evidence gap: now closed

The previous revision recorded that the production-path numbers existed only as
an incomplete log. That gap is closed. The rerun completed and wrote a
validated artifact,
`ab_test_runtime/experiments/because_production__qwen__qwen3-14b.json`
(qwen/qwen3-14b, 139 frozen IDs, all three arms present):

| arm | production path |
|---|---:|
| baseline | 69/139 = **49.6%** |
| `because` | 59/139 = **42.4%** |
| `scaffold_thinking` | 60/139 = **43.2%** |

The reversal is therefore artifact-grade on this book. `because` costs 7.2
points against the configuration that could actually ship, having appeared to
gain 10.8 points against a weaker harness baseline of 39.6%.

The mechanism is worth stating because it recurred later the same day: the
exploratory harness used a simplified prompt, which depressed *its* baseline
rather than lifting `because`. A positive result measured against a weakened
control is not a positive result. See §6 for the second instance of the same
class of error, which was not caught by this rule because the harness there was
not simplified — it was merely a different harness.

Caveat retained: this is one book. The five reasoning arms have never been run
on a second book, which is why the run described in §7.1 is in flight.

## 6. Model comparison, and two generalization failures

The controlled closed-set decomposition supports moving off qwen3.5-9b and
testing 14B-class candidates. It does not prove that the task has a fixed model
ceiling. What has changed since the previous revision is that the decomposition
has now been run on a second book, and the full pipeline has been scored on
both — and the three views disagree with each other.

### 6.1 The decomposition on both books

Frozen segmentation from the qwen3.5-9b run; only pass 2 varies. Mushoku 16 is
147 gold lines, Grimgar 03 is 400 (provisional fixture, see §7).

| model | Mushoku open | Grimgar open | Mushoku oracle | Grimgar oracle |
|---|---:|---:|---:|---:|
| qwen/qwen3-14b | 48.3% | **60.8%** | 66.0% | 72.5% |
| gemma-4-e4b | 39.5% | 57.2% | 49.7% | 65.5% |
| ministral-3-14b | 47.6% | 51.5% | 61.2% | 58.2% |
| microsoft/phi-4 | 45.6% | not run | 59.2% | not run |
| qwen3.5-9b (shipped) | 35.4% | not run | 49.0% | not run |

Grimgar open-arm pairwise, exact McNemar:

| comparison | discordant | p |
|---|---|---:|
| qwen3-14b vs ministral | 72 / 35 | **0.00045** |
| gemma vs ministral | 70 / 47 | **0.04150** |
| qwen3-14b vs gemma | 56 / 42 | 0.18885 (unresolved) |

Both books agree that qwen3-14b is at or near the top and that the shipped 9B is
not. They do **not** agree on second place: ministral is competitive on Mushoku
(47.6%, within noise of qwen3-14b) and clearly last on Grimgar (p=0.042 behind
gemma). Do not carry a full ranking across books; carry only "qwen3-14b ≥ the
others, and the 9B is worse than all of them."

### 6.2 Generalization failure 1 — `closed-6` across books

`closed-6` restricts the model to six scene-derived candidates. Paired against
the open arm, same model, same lines:

| book / model | effect | discordant | p |
|---|---:|---|---:|
| Mushoku / qwen3-14b | **−11.6pt** | 8 / 25 | **0.0046** |
| Mushoku / phi-4 | **−12.9pt** | 9 / 28 | **0.0026** |
| Mushoku / ministral | −6.1pt | 9 / 18 | 0.1221 |
| Mushoku / gemma | −0.7pt | 12 / 13 | 1.0000 |
| Mushoku / qwen3.5-9b | −0.7pt | 11 / 12 | 1.0000 |
| Grimgar / gemma | +2.5pt | 29 / 19 | 0.1934 |
| Grimgar / ministral | +1.0pt | 29 / 25 | 0.6835 |
| Grimgar / qwen3-14b | −0.2pt | 25 / 26 | 1.0000 |

Stated precisely, and narrower than an earlier session note claiming a
"reversal": candidate pruning is **significantly harmful on Mushoku for two of
five models, and indistinguishable from no effect on Grimgar for all three**.
The sign flips numerically on Grimgar but never significantly. The honest
summary is that `closed-6`'s measured harm is specific to one book, not that it
helps on the other.

This still matters, because `closed-6` was the most-replicated negative in the
ledger — five models agreeing. Five models on one book is one book.

### 6.3 Generalization failure 2 — decomposition vs. whole pipeline

The overnight full-book 2×2 produced complete three-pass output for both models
on both books. Scored against the same fixtures, excluding repeated-text lines
and applying fixture aliases:

| model | Mushoku end-to-end | Grimgar end-to-end |
|---|---:|---:|
| gemma-4-e4b | 39/138 = 28.3% | 237/399 = **59.4%** |
| qwen/qwen3-14b | 53/139 = **38.1%** | 215/399 = 53.9% |

Paired exact McNemar: Mushoku 27/14 discordant, p=0.0596; Grimgar 51/73
discordant, p=0.0589. **Neither difference is significant**, and neither model
should be described as the end-to-end winner.

The finding is not in either column, it is in the comparison between §6.1 and
§6.3 on the same book:

- decomposition, Grimgar: qwen3-14b 60.8% > gemma 57.2%
- end-to-end, Grimgar: gemma 59.4% > qwen3-14b 53.9%

The two instruments rank the two models in opposite orders on identical source
text and an identical fixture. The decomposition freezes segmentation from the
9B run and varies only pass 2; end-to-end runs each model's own pass 1 *and*
pass 2. So the decomposition measures a component under a fixed input that no
shipping configuration would ever present it with, and the pipeline measures the
product.

Every model conclusion in §6.1 comes from the decomposition. This does not
invalidate it as a pass-2 probe — that is what it was built to be — but it does
mean **the decomposition has not been shown to predict pipeline behavior**, and
it is the pipeline that ships. Related: gemma's 59.4% is the highest end-to-end
figure this project has produced, from a 7.5B model that the decomposition
ranks second.

Also note this defeats a rule the project adopted after the `because` reversal.
"Test against the exact configuration that could ship" catches a *simplified*
control; it does not catch a faithful-but-different harness. Both failures are
the same underlying error — the measured thing was not the shipped thing — and
only the first was detectable by inspecting the prompt.

`mistralai/magistral-small` was not tested because its 13.51 GiB weights did
not safely fit the 15.92 GiB card. This was a deliberate VRAM-safety decision.

## 7. Runs in flight at the time of writing

Two runs are executing and are not yet reflected in any table above. Both are
recorded here so a reviewer can tell missing results from suppressed ones.

### 7.1 Reasoning arms on Grimgar 03 (local, running)

`app/experiments/reasoning_arms.py`, qwen/qwen3-14b, five arms (baseline,
`because`, `scaffold`, `thinking`, `scaffold_thinking`), 400 Grimgar gold lines
across 98 batch windows. Log:
`ab_test_runtime/results/overnight_20260726-185022/day/reasoning_g03.log`.
Expected artifact:
`ab_test_runtime/experiments/reasoning_arms__grimgar03__qwen__qwen3-14b.json`.

Sizing: 98 gold-bearing windows against Mushoku's 60 (1.63×); Mushoku took 84
minutes, of which `thinking` and `scaffold_thinking` were 59. Estimated 2h15m.

Purpose: four of the five arms were negatives, and **all four are single-book
results**. Given §6.2 and §6.3, a single-book negative is not a retired idea.

The harness required a change to run at all: book, source text, checkpoint,
gold fixture and *output filename* were hardcoded to Mushoku. Run as-was it
would have loaded Mushoku data under a Grimgar label and overwritten the
existing Mushoku artifact in place. It is now parameterized by
`EXPERIMENT_BOOK`/`EXPERIMENT_GOLD` with the book in the filename, matching
`closed_set.py`. Consequence for provenance: the existing Mushoku artifact keeps
its old name `reasoning_arms__qwen__qwen3-14b.json`, while a future Mushoku
rerun would write `reasoning_arms__mushoku16__…`.

### 7.2 Thunder A6000 context control (remote, downloading)

Instance 0 (`lho3lk5l`, A6000 48 GB) is fetching `qwen/qwen3-14b@q4_k_m` then
`qwen/qwen3-32b@q4_k_m`; 4.2 GB of the first model retrieved so far, no
inference started, `thunder_run.sh` not yet launched.

The control is the point. Locally qwen/qwen3-14b is capped at 16384 context by
VRAM, so every "bigger model wins" result in §6.1 is confounded with "bigger
model was also given more context". Running the *same weights* at 98304 on
rented hardware separates the two factors. qwen3-32b then tests scale with the
family held constant.

Two risks a reviewer should know: the instance bills continuously until
**deleted** (stopping is not enough), and `tnr connect` timed out from this
machine during a status check, so the orchestration script's remote-command
path is not yet proven end to end. If the context control turns out not to
matter, this instance should be deleted rather than repurposed on the spot.

## 8. Human judging and second-book validation

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

## 9. Current decisions

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
- a significant end-to-end Gemma/Qwen difference — the 2×2 now completes on both
  books and is non-significant on both (p≈0.06), in *opposite* directions;
- generalization from Mushoku to Grimgar — now actively contradicted, not merely
  untested, for `closed-6` (§6.2) and for the model ranking (§6.3);
- that the closed-set decomposition predicts pipeline behavior.

### Do not do yet

- do not ship the `because` field — now settled by artifact, not provisional;
- do not treat any decomposition result as a shipping decision until §6.3 is
  understood; it is a pass-2 probe whose ranking disagreed with the pipeline's;
- do not resume the full matrix against a changing pass-2 configuration;
- do not build a two-model “writer then JSON converter” pipeline merely to
  repair formatting—the attribution model already returns parseable JSON, and
  a formatter cannot repair a wrong speaker choice;
- do not weaken retries, VRAM guards, checkpointing, or the global GPU lock.

## 10. Recommended next steps

Reordered from the previous revision. `because` is closed, and the two
generalization failures displace the remaining intervention work.

1. **Finish the two runs in §7** and record both outcomes, including null ones.
2. **Resolve which instrument to trust (§6.3).** This gates everything else: the
   ledger is almost entirely decomposition results, and the decomposition just
   disagreed with the pipeline about model ranking. The cheap discriminating
   test is to re-run the decomposition on gemma's *own* segmentation instead of
   the frozen 9B segmentation. If the ranking flips to match end-to-end, the
   frozen input is the cause and every §6.1 number is conditional on a
   segmentation no shipping configuration produces.
3. **Freeze the Grimgar scoring fixture and policy.** Report ambiguous and
   unaligned rows separately. The 400-line fixture is currently single-judge and
   provisional; it is adequate for the paired rankings above and not for any
   absolute accuracy claim.
4. **Only then choose a settled attribution model/configuration**, on
   end-to-end evidence from both books plus latency, memory fit, and completion
   reliability.
5. **Only after that, consider routing.** Report accepted accuracy against
   coverage; raw model agreement is not enough.

Deliberately *not* recommended: more single-book interventions. Ten have been
measured and one survives, and today's results show the measurement was not
sensitive enough to justify that conclusion book-wide.

If a two-model design is revisited, split by semantic responsibility, not by
serialization:

- model A proposes the speaker plus compact evidence;
- model B independently verifies or challenges uncertain cases;
- deterministic code validates and converts the final result to JSON.

This costs more than one pass and is justified only if the verifier produces a
useful high-precision subset or repairs enough errors to beat a stronger single
model.

## 11. Reviewer assessment

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

The `because` reversal is now closed by a validated artifact and needs no
further defense.

### Added this revision

That symmetry needs a third line, because today produced a failure neither
existing line predicts:

- a negative can be caused by a broken instrument;
- a positive can be caused by a weak comparison;
- **either can be caused by a sound instrument measuring something adjacent to
  what ships.**

The `because` reversal was caught by comparing prompts — the exploratory
baseline was visibly weaker. Nothing analogous was available for §6.3. The
closed-set decomposition has no defect: correct prompt, production call path,
validated artifact, paired statistics. It simply holds segmentation fixed, and
that single design choice was enough to invert a model ranking on the same book
and fixture. There was no way to see that by reading the harness; it took
scoring the whole pipeline and finding the opposite answer.

The practical consequence is uncomfortable. Most of §4's ledger rests on the
decomposition or on similarly frozen-input harnesses, so the confidence in each
of those rows should drop until §10 step 2 resolves whether the frozen input is
responsible. I do not think the ledger is *wrong* — several entries have large
paired effects that a segmentation difference is unlikely to reverse — but
"measured under frozen 9B segmentation" now belongs on every one of them.

The model result should also be stated narrowly, and more narrowly than the
previous revision put it. The evidence supports moving off the original 9B. It
does not select a winner among the 14B-class candidates: the two harnesses
disagree on Grimgar, both end-to-end comparisons are non-significant, and the
second-place model differs by book. The context confound (§7.2) is still
unresolved, so even "bigger model wins" is not yet separated from "bigger model
got more context".

Finally, the product may need a human-assisted success criterion. If no model
approaches unattended accuracy, the relevant question becomes whether the
system can automatically accept a large, high-precision subset and route the
rest for efficient correction. No tested confidence signal has yet
demonstrated that operating point.

## 12. Handoff checklist

Before accepting any new headline result, verify:

- the run used the production call path or is labeled exploratory;
- all expected arms and IDs are present;
- aggregates recompute from rows;
- the actual loaded model and LM Studio state match the declaration;
- source provenance is reconstructable;
- the comparison is paired where possible;
- partial runs are not presented as completed;
- statistical non-significance is not described as equivalence;
- conclusions remain scoped to the tested books and fixtures;
- the result holds on both books, or says which one it was measured on;
- if the harness freezes any pipeline stage, that stage is named in the claim.

The earlier 34-section discussion is intentionally not retained here. Git
history preserves it at commit `2ce90a9` if the full chronology is needed.
