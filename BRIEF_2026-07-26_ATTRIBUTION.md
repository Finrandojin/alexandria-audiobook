# Speaker attribution: evidence, current conclusions, and next decisions

Date: 2026-07-26  
Repository: `alexandria-audiobook2.git`  
Audience: an external reviewer with repository access and no session history

## Executive summary

Alexandria converts novels into multi-voice audiobooks. Its speaker-attribution
step is currently the largest quality bottleneck:

- **mushoku16:** 29.9% correct (44/147) on a random, alias-aware gold set;
- **grimgar03:** approximately 54% on a smaller 35-line judged set.

The strongest current diagnosis is:

1. **Speaker selection is harder than candidate recall.** The full roster
   contains the correct speaker on 85.0% of mushoku16 gold lines, but the
   shipped pipeline selects correctly on only 29.9%.
2. **Context is essential.** Explicit ±1 context improves both single-target
   and batched attribution for the tested 9B model.
3. **The flat, incrementally built roster has a measurable cost in the current
   harness.** A complete attested roster improved 27.3% to 32.4% (+5.1 points)
   over 139 unambiguous lines.
4. **Candidate pruning alone is insufficient.** Even when the true speaker is
   guaranteed among a small oracle set, the tested 9B selects correctly only
   49.0% of the time under that experiment's prompt and decoding setup.
5. **Confidence routing and a stronger pass-2 model remain open.** There is no
   demonstrated path to unattended 90%+ attribution.

The +5.1 warm-roster result is a real positive result inside its harness, but
it is **not yet a validated production change**:

- it has only been measured on mushoku16;
- a final clean, contract-validated reproduction was still running when this
  brief was rewritten;
- the production pipeline A/B and second-book validation have not happened.

## 1. System and constraints

The pipeline has three LLM-assisted passes:

1. **Segment:** convert prose into frozen `NARRATOR` and `SPOKEN` entries.
2. **Attribute:** assign a speaker to each `SPOKEN` entry.
3. **Instruct:** add a delivery direction.

Pass 2 currently receives:

- batches of roughly 25 entries;
- each target's text and ±1 neighbour objects;
- a running roster of established character names;
- a requirement to return `{n, speaker}` without changing text.

Narration is resolved deterministically. Attribution may add only `speaker`.

Relevant files:

- `app/three_pass_generate.py`
- `app/pass_quality.py`
- `app/attribution_accuracy.py`
- `app/fixtures/attribution_gold_random.json`
- `app/experiments/`
- `ab_test_runtime/experiments/`

Constraints:

- local inference only;
- primary model is
  `qwen3.5-9b-uncensored-hauhaucs-aggressive`;
- one consumer GPU;
- LM Studio `parallel: 1` is a deliberate VRAM safety constraint;
- books contain roughly 2,000–3,300 segmented entries;
- long GPU work is serialized by the application's global lock;
- safety checks, headroom guards, retries, and checkpointing are load-bearing.

## 2. Measurement

### Gold data

`app/fixtures/attribution_gold_random.json` contains 147 randomly sampled
mushoku16 lines with hand-judged speakers.

Independent-reader agreement:

| set | overlap | agreement |
|---|---:|---:|
| mushoku16 | 63 lines | 94% |
| grimgar03 | 35 lines | 97% with alias credit |

Reader disagreements concerned non-character or genuinely ambiguous lines, not
which named character spoke.

Alias equivalence is declared in the fixture, not inferred globally in code.
For example, `RUDEUS` and `RUDI` receive credit as the same character, while
`FUTURE_ME` remains an invented and incorrect label.

### Scorer repairs

The evaluation tools now:

- align using full normalized text rather than a 60-character prefix;
- score fixture-declared aliases as equivalent;
- reject duplicate gold identities;
- exclude ambiguous repeated-text alignments where a unique source occurrence
  cannot be established.

The old 60-character identity did not change the current mushoku16 score, but
distinct shared prefixes exist elsewhere in the corpus, so the fix was real.

Alias-aware scoring raised the canonical mushoku16 baseline from 20.4% to
29.9%; 14 of 147 lines were previously lost solely to `RUDEUS`/`RUDI`
spelling.

### Determinism

Temperature-0 attribution on an idle GPU was exactly reproducible in repeated
arms. Previously observed variation came from a concurrent model run sharing
LM Studio, not useful sampling noise.

Rules for comparable results:

- idle GPU;
- no competing LM Studio request stream;
- exact model/load settings captured;
- frozen inputs and prompt bytes;
- per-line artifacts, not aggregate tables alone;
- identical scorer and fixture hashes.

## 3. Baseline and error structure

| book | measured accuracy |
|---|---:|
| mushoku16 | 44/147 = **29.9%** |
| grimgar03 | approximately **54%** on 35 judged lines |

mushoku16 is a difficult first-person book whose narrator is rarely named in
narration. The 24-point book spread is a warning against validating an
architecture on mushoku16 alone.

Among mushoku16's 103 baseline errors:

| class | share |
|---|---:|
| wrong real character | 64% |
| invented name | 33% |
| `UNKNOWN` | 3% |

An earlier statement that 31% of errors had the true name within ±2 lines was
technically true but operationally misleading. Relabelling the nearby name's
grammatical role produced:

| relation | share of errors | usable speaker evidence? |
|---|---:|---|
| name absent nearby | 62.1% | — |
| bare mention | 18.4% | no |
| vocative/addressee | 12.6% | usually identifies listener |
| speech-verb tag | 6.8% | yes |

Thirteen measured errors are addressee/speaker inversions: the model assigns a
line to the person being addressed.

This correction explains why context experiments justified by the original
31% statistic were poorly targeted.

## 4. What has been measured

### Production/prompt experiments

| change | result |
|---|---:|
| first-person narrator hint | no effect |
| interleaved prose with `[n]` markers | −5.4 points; 50% more format retries |
| narration retained as answerable batch rows | −2.1 points; roughly 2× slower |
| three-sample seeded voting | inconclusive; greedy approximately equal |
| temperature 0 | removed contention/sampling confusion |
| unattested-speaker rejection | removed 279 invented assignments in one book |
| honorific-name attestation fix | recovered roughly 70 real names per affected book |

These results should not be generalized beyond their tested model, prompt,
book, and decoding configuration unless the measurement itself is
model-independent.

### Candidate recall

Measured on mushoku16:

| candidate source | recall | median set size |
|---|---:|---:|
| speech tags | 7.5% | 0 |
| recent speakers | 53.7% | 3 |
| scene names | 71.4% | 4 |
| tag + recent + scene | 74.8% | 6 |
| full production roster | 85.0% | 17 |

These are text/candidate-generator measurements, not model-selection scores.

Consequences:

- deterministic tag extraction can be a precise fast path but cannot solve
  most lines on this book;
- the tested scene candidate generator sacrifices too much recall;
- 15% of gold speakers are absent even from the full roster.

### Closed-set selection

All arms below are single-target calls:

| arm | candidate recall | total accuracy | conditional accuracy |
|---|---:|---:|---:|
| open full roster | 85.0% | 35.4% | 41.6% |
| tested scene candidate set | 73.5% | 34.7% | 48.9% |
| oracle true speaker + distractors | 100% | 49.0% | 49.0% |

The correct interpretation is:

> With the true speaker guaranteed among a small supplied set, this 9B model
> using this prompt and decoding configuration selects correctly on 49.0% of
> the gold lines.

This is a configuration ceiling, not an intrinsic model ceiling.

The tested scene generator is rejected for this 9B because its conditional
gain does not recover the lost recall. That does not prove all possible scene
models are ineffective.

### Explicit context × batch-size experiment

| | batch of consecutive targets | single target |
|---|---:|---:|
| no explicit neighbour objects | 19.4% | 2.2% |
| ±1 neighbour objects | 34.5% | 18.7% |

Explicit context helps in both rows.

This is not a clean independent estimate of batching: a batch of consecutive
lines is itself conversational context. The supported conclusion is that
context is useful, while the previously tested prose/narration formats were
harmful.

### Roster warm-up

The corrected per-line artifact contains 139 unambiguous gold lines per arm:

| arm | roster | correct | accuracy |
|---|---|---:|---:|
| incremental | grows during attribution | 38/139 | 27.3% |
| warm | complete 17-name attested roster from start | 45/139 | 32.4% |
| oracle diagnostic | discovered roster + every gold answer | 49/139 | 35.3% |

Warm roster improves the harness by seven lines, or **+5.1 points**.

By book quartile:

| quartile | incremental | warm | gain |
|---|---:|---:|---:|
| Q1 | 19.0% | 26.2% | +7.2 |
| Q2 | 29.3% | 34.1% | +4.8 |
| Q3 | 28.0% | 28.0% | 0.0 |
| Q4 | 35.5% | 41.9% | +6.4 |

Q1 has the largest gain, but Q4 is only 0.8 points behind. The result does not
support a simple “missing names only hurt early” explanation. Warm roster state
changes selection throughout the book.

## 5. Measurement failures and reversals

The experiment history contains important invalid results:

1. The first roster run scored repeated text at every occurrence, turning 146
   judged lines into 155 rows. All arms appeared to score 55/155.
2. A post-hoc filtered interpretation then reported the roster change as flat.
3. The committed JSON still contained the stale 55/155 artifact.
4. A real rerun after unique-identity filtering reversed the conclusion:
   warm roster gained +5.1.
5. Early artifacts failed to capture LM Studio state because the metadata
   helper was called with the wrong signature and the exception was swallowed.

The lesson is not that architectural reasoning is useless. It is that
unverified measurement will confidently support a false conclusion.

Every experiment must preserve per-line rows and validate identity,
denominators, environment, and summaries before its aggregate is used.

## 6. Artifact status

Committed harnesses:

- `app/experiments/closed_set.py`
- `app/experiments/two_by_two.py`
- `app/experiments/roster_warmup.py`
- `app/experiments/manifest.py`
- `app/experiments/candidates.py`

Committed artifacts:

- `ab_test_runtime/experiments/closed_set.json`
- `ab_test_runtime/experiments/two_by_two.json`
- `ab_test_runtime/experiments/roster_warmup.json`

Verified properties of the corrected roster artifact:

- 417 total rows;
- 139 rows per arm;
- no duplicate `(arm, gold_id)` identities;
- summaries recompute from the rows;
- LM Studio recorded as loaded;
- context length 32768;
- parallel 1;
- optimized true.

Limitation of that artifact:

- it records `dirty: true`;
- its `meta.git` block does not contain the later-added harness fingerprint or
  modified-tracked-file list.

Therefore its arithmetic and model-load settings are inspectable, but the exact
dirty harness source is not reconstructable from that JSON alone.

The harness now supports stronger code identity and contract validation. A
final clean rerun from commit `af6ded1` was in progress when this brief was
rewritten. Its result must be checked before replacing the status above.

## 7. Required artifact contract

A valid experiment artifact should require:

1. exact expected arm names;
2. identical expected gold-ID sets across arms;
3. expected denominator after ambiguity filtering;
4. every ID belonging to the declared fixture;
5. no duplicate `(arm, gold_id)` identity;
6. summaries recomputed from rows;
7. non-null context length and parallel setting;
8. optimized LM Studio state;
9. intended model matching the actually loaded model;
10. fixture hash;
11. prompt hashes and decoding settings;
12. harness-source fingerprint;
13. clean tracked tree, or explicit fingerprints for relevant modifications.

An artifact can be internally consistent and still incomplete. A validator that
only checks that its summary describes its rows cannot detect a missing arm or
half a fixture.

## 8. Current conclusions

### Supported

- Selection is the primary measured bottleneck for the tested 9B.
- Context is materially helpful.
- Candidate pruning alone cannot make the tested 9B reliable.
- The tested scene candidate generator does not beat the full roster.
- Speech tags have too little recall to be the main architecture on mushoku16.
- Warm roster is the leading candidate production change at +5.1 in its
  corrected harness.
- Some review/routing path is required for lines with no available true
  speaker.

### Model-independent or mostly model-independent

- candidate recall for a fixed candidate generator;
- roster recall;
- speech-tag coverage;
- fixture identity and scorer correctness.

### Specific to the tested 9B/configuration

- 49.0% oracle conditional selection;
- scene-set versus full-roster selection;
- response to explicit context representation;
- warm-roster selection gain until reproduced on another model/book.

### Not established

- that 49% is the model's intrinsic ceiling;
- that every scene-cast architecture fails;
- that warm roster improves the production pipeline;
- that warm roster generalizes to grimgar03;
- that confidence features produce a useful risk/coverage curve;
- that the 14B materially improves conditional selection;
- why the harness's context arm previously exceeded the shipped 29.9%
  baseline.

## 9. Next experiments

### A. Finish and verify the clean roster reproduction

Before further interpretation:

- inspect the final clean artifact;
- require `dirty: false`;
- require `harness_sha256`;
- require the declared roster contract;
- recompute every aggregate from rows;
- verify the exact 139 IDs in all three arms;
- confirm actual loaded model, context 32768, parallel 1, optimized true.

If the clean run does not reproduce the direction and approximate magnitude,
the +5.1 claim returns to unresolved.

### B. Production warm-roster A/B

Implement warm roster behind an experimental switch. Do not replace the
incremental path.

Use frozen segmentation and report per line:

- gold speaker;
- incremental and warm predictions;
- whether gold was available in each roster;
- roster sizes and added names;
- transition:
  - wrong → correct;
  - correct → wrong;
  - wrong → different wrong;
  - unchanged;
- book position;
- retries, latency, and token cost.

Validate on both mushoku16 and grimgar03 before shipping.

The experiment must distinguish:

1. **availability repair:** warm roster adds a previously unavailable true
   speaker;
2. **choice perturbation:** both rosters contain the true speaker, but the
   larger roster changes the model's selection.

### C. Run the same decomposition on the 14B

For a meaningful comparison, report:

- roster/candidate recall;
- conditional selection;
- oracle closed-set score;
- addressee-inversion count;
- accuracy by candidate-set size;
- risk/coverage curve;
- latency and completion-token cost.

This distinguishes “the task is intrinsically difficult” from “this 9B is
weak at the task.”

### D. Confidence routing

Temperature-0 repeats are deterministic, so identical-repeat agreement is not
a confidence feature.

Evaluate candidate signals such as:

- explicit speech-tag support;
- vocative/addressee conflict;
- incremental/warm prediction agreement;
- batch/single-target agreement;
- prediction stability after irrelevant roster removal;
- top-two probability margin if available;
- candidate provenance;
- parse/retry history.

Report a risk/coverage curve:

| threshold | auto-accepted lines | coverage | accepted accuracy | routed lines |
|---|---:|---:|---:|---:|

The product question is whether a sufficiently large subset can be accepted at
a useful accuracy target—not merely whether confidence correlates with
correctness.

### E. Relation annotation, only if still justified

Thirteen baseline errors are addressee/speaker inversions. A narrow experiment
may annotate neighbouring names as:

- `SPEECH_TAG`;
- `VOCATIVE/ADDRESSEE`;
- `MENTION`.

Measure whether explicit relation labels repair those inversions without
damaging other lines. This remains an untested diagnostic, not a production
proposal.

## 10. Production decision gate

Warm roster should be described as:

> **Promising and ready for a production A/B, but not validated as a production
> change.**

It should ship only if:

1. the clean reproduction confirms the harness result;
2. mushoku16 production A/B shows a meaningful gain;
3. grimgar03 confirms direction without a new failure class;
4. transition analysis shows the gain is not a near-cancellation of unrelated
   regressions;
5. added latency and cost are acceptable;
6. validators and safety gates remain intact.

## 11. Questions for the next reviewer

1. What confidence signals are available from LM Studio or the model that do
   not require stochastic repeats?
2. Is there a better way to distinguish speaker from addressee without asking
   the same 9B to solve another open-ended generation task?
3. Which second book and sample size are sufficient to validate warm roster
   without overfitting mushoku16?
4. What minimum accepted-accuracy/coverage trade-off would make human-assisted
   attribution useful as a product?
5. Should the 14B decomposition precede production warm-roster work, or run in
   parallel after the current matrix finishes?

## Operational status

Current as of the six-model benchmark.

**Complete and artifact-backed** (`ab_test_runtime/experiments/`):

- closed-set decomposition on six models, all contract-validated;
- roster warm-up on qwen3.5-9b, clean commit, `dirty: false`;
- two-by-two context/batch grid on qwen3.5-9b;
- VRAM profiles for phi-4 and qwen3-14b, measured on an idle card.

**Incomplete, and not to be quoted as results:**

- **Roster warm-up on ministral-3-14b.** The oracle arm hung and the harness
  writes its artifact only at the end, so nothing was produced. The log shows
  incremental 41.0% and warm 44.6% (+3.6) but there is no per-line record
  behind either figure. Treat as unverified until re-run.

**Paused:**

- The model matrix, mid-ministral/grimgar03, checkpointed. Note it is currently
  measuring end-to-end quality using pass 2 on a model the decomposition ranks
  in the weakest tier, so its absolute numbers will not reflect a
  14B-attribution pipeline.

**Not fitted:** `mistralai/magistral-small` - 13.51 GiB of weights on a 15.92
GiB card. No profile; falls back to the conservative default by design.

Experiment branch `agent/experiment-artifacts`, PR #236. Release suite 990 tests.

Treat this section as transient. Verify live state and output files rather than
relying on this snapshot.

## 12. Reviewer assessment

This section is opinion informed by the evidence above. It is deliberately
separate from the measured record.

### The project has found a real lead, not a solution

Warm roster is the first architectural change in this investigation with a
credible positive result:

- the corrected artifact has unique per-line identities;
- its aggregate recomputes from those rows;
- the model load settings are captured;
- the gain is seven additional correct lines over the incremental arm.

That makes it worth a production A/B. It does not make it ready to ship.

A five-point gain from a 27–30% baseline still leaves attribution wrong on
roughly two of every three lines in the hard book. Warm roster may improve the
system, but it cannot by itself change the product from human-dependent to
unattended.

### The most important next result is not another aggregate accuracy

The warm roster changes many predictions, not only predictions whose correct
speaker was previously unavailable. That means the gain may be a balance of:

- newly solvable lines;
- wrong answers repaired through better global cast awareness;
- previously correct answers broken by added distractors;
- wrong answers changed to different wrong answers.

The production A/B should be considered successful only if its transition
table is healthy. A net gain of seven produced by twenty repairs and thirteen
regressions is less stable and less likely to generalize than seven repairs
with no regressions.

The transition analysis is therefore more informative than whether the total
score happens to rise by five points again.

### Warm roster should be evaluated as state, not merely a longer prompt

The result does not prove that “more names are better.” The oracle roster is
larger than the warm roster and gains only another 2.9 points. The scene set is
smaller and loses recall. Candidate count alone does not explain the curve.

The roster carries several kinds of information:

- which characters exist;
- canonical spelling;
- which characters have appeared by this point;
- accidental signals from ordering;
- distractors.

Future experiments should record roster order and provenance, not only roster
membership. If the warm pass orders names by first appearance, frequency, or
discovery confidence, that ordering may influence selection independently of
completeness.

One cheap ablation is to run the same warm roster in:

- first-appearance order;
- alphabetical order;
- frequency order;
- a fixed shuffled order.

If accuracy moves, the model is using roster position as a latent prior. That
would be important production behavior and a possible confidence feature.
Do this only after the clean reproduction and production A/B; it is a
diagnostic, not the next priority.

### The 14B experiment has higher information value than more 9B prompt work

The oracle result shows that this 9B/configuration reaches 49% even when
candidate recall is perfect. That leaves too much selection error for candidate
engineering alone.

Running the same closed-set and warm-roster decomposition on the 14B answers a
more fundamental question:

- if conditional selection rises substantially, model capacity is the current
  bottleneck;
- if it remains near 49%, the prompt/task representation is the stronger
  suspect;
- if total accuracy rises but conditional selection does not, the gain comes
  from recall, abstention, aliases, or output behavior rather than better
  reasoning.

I would prioritize this over another general attribution prompt rewrite.

### Confidence routing is not optional, but useful routing is unproven

At least 15% of mushoku16 gold speakers are absent from the production roster.
No selector constrained to that roster can answer those lines correctly.
Other lines are genuinely unnamed, non-speech, or ambiguous.

The system therefore needs a way to abstain or route work even if the model
improves substantially.

What remains unknown is whether available signals can isolate a large,
high-accuracy automatic subset. The desired deliverable is a table such as:

| accepted accuracy | maximum coverage | routed share |
|---:|---:|---:|
| 80% | ? | ? |
| 90% | ? | ? |
| 95% | ? | ? |

If 90% accepted accuracy requires routing 85% of lines, confidence exists but
does not create a useful product. If it retains most lines, human-assisted
attribution becomes plausible even before raw accuracy reaches 90%.

### The product may need a different success criterion

Unattended multi-voice generation probably requires attribution accuracy far
above the current range. A wrong voice is salient and can make dialogue harder
to follow, so errors are not evenly tolerable.

A human-assisted product could still be valuable if it:

- automatically resolves a high-confidence subset;
- groups uncertain lines by scene;
- offers likely candidates rather than a blank field;
- highlights speaker/addressee conflicts;
- lets one correction propagate safely through a local exchange;
- never hides low confidence behind a polished final export.

This suggests measuring editing effort, not only line accuracy:

- lines requiring review;
- corrections per thousand words;
- time to resolve one scene;
- percentage of corrections suggested in the top two;
- regressions caused by propagation;
- final audiobook error rate after a bounded review session.

Raw accuracy remains necessary for model comparison, but review-time reduction
may be the better product metric.

### Measurement infrastructure is now part of the architecture

This investigation produced several confident but wrong conclusions from:

- contention;
- alias-blind scoring;
- prefix alignment;
- duplicate repeated-text scoring;
- stale committed artifacts;
- missing environment capture.

Those were not cosmetic reporting issues. They changed which architecture
appeared to win.

The shared artifact contract, per-line records, environment capture, and
fixture identity checks should be treated like production safety nets. New
experiments should use the shared framework by default; bespoke scripts should
not be trusted until their outputs pass the same validator.

### Recommended priority

My preferred order is:

1. finish and inspect the clean roster reproduction;
2. merge the experiment framework only after its contract tests pass;
3. run the production incremental-versus-warm A/B on frozen mushoku16
   segmentation;
4. run the identical production A/B on grimgar03;
5. run the closed-set decomposition on the 14B;
6. build risk/coverage curves from the predictions already collected;
7. decide whether the product target is unattended generation or
   human-assisted correction;
8. only then consider relation annotations, roster-order ablations, or another
   scene representation.

### What would change this assessment

I would recommend shipping warm roster if:

- the clean run reproduces;
- both production books improve;
- correct→wrong regressions are limited;
- the gain survives alias-aware, unique-identity scoring;
- latency and GPU cost remain acceptable.

I would deprioritize warm roster if:

- the clean run loses the gain;
- grimgar03 is flat or negative;
- the net gain comes from large cancelling prediction churn;
- the production implementation requires weakening attestation or other safety
  gates.

I would shift most attribution effort toward the stronger model if the 14B
substantially raises oracle conditional selection. I would shift toward
human-review tooling if neither model produces a useful risk/coverage curve.

### Bottom line

The evidence no longer supports “nothing helps.” It supports a narrower and
more useful statement:

> Warm roster is a credible five-point lead that deserves production and
> second-book validation. It is not large enough to solve attribution, and the
> larger decision is whether a stronger model plus confidence routing can
> produce a useful automatic subset—or whether Alexandria should explicitly
> become a human-assisted attribution system.

---

## 13. Answers to §11, from the repository side

Answering the five questions where evidence exists, and saying plainly where it
does not.

**1. Confidence signals that need no stochastic repeats.**
Temperature-0 repeats are useless here, confirmed twice: two independent clean
reruns of the roster experiment produced byte-identical arm totals. Signals
available without perturbation:

- whether the selected speaker appeared in a speech-verb tag within ±2 entries
  (a deterministic text feature, and only 6.8% of errors have one, so it is a
  high-precision/low-coverage gate);
- whether the answer is in the scene-local candidate set, the recent-speaker
  set, or only in the global roster - already computed by `app/candidates.py`
  and recorded as `candidate_provenance` in every artifact;
- agreement between the batched and single-target predictions, which differ in
  context supply rather than in sampling, so they disagree informatively;
- agreement between incremental and warm roster passes - free once warm roster
  ships, and directly measurable from the existing artifact.

Token probabilities would be the strongest signal and are not currently
retrieved from LM Studio. Whether `logprobs` is available through this endpoint
is unverified and worth ten minutes to check.

**2. Distinguishing speaker from addressee without another open-ended task.**
The 13 measured inversions all involve a name inside the line being judged
("Luci, how is she?" → answered SYLPHY, gold RUDEUS). A vocative is
deterministically detectable - a capitalised roster name adjacent to a comma or
sentence boundary *inside the quoted span*. That does not require the model at
all; it requires excluding those names as candidates, or annotating them. The
review's `SPEECH_TAG` / `VOCATIVE` / `MENTION` annotation experiment is the
right test and is still untested.

**3. Second book and sample size.**
`grimgar03` is the natural second book: it already has 35 hand-judged lines at
97% two-reader concordance, a checkpoint from the same matrix, and it is
materially easier (~54% vs ~30%), so it tests generalisation rather than
repeating the hard case. 35 lines is too few to resolve a 5-point effect;
expanding it to ~100 randomly sampled lines is the prerequisite, and costs one
Gemini pass plus one Claude pass at the concordance protocol already used.

**4. Minimum accepted-accuracy/coverage for a useful product.**
Not answerable from this repository. It depends on what the owner will tolerate
in a finished audiobook, and a wrong speaker is not a subtle defect - it is a
character speaking in another character's voice. My own estimate, offered as
opinion: below ~95% accuracy on the auto-accepted subset the listener will
notice, so the useful question is what coverage survives at 95%, not what
accuracy survives at high coverage.

**5. 14B decomposition before or after production warm-roster work.**
Before, and it is cheap: ~25 minutes per model against ~a day for a production
A/B. It also changes what the warm-roster A/B means. If the 14B's conditional
selection is much higher, warm roster on the 9B is optimising a component that
is about to be replaced. Note the 14Bs are profiled at **16384 context, half
the 9B's**, so the comparison must report context alongside accuracy.

### One caution on the roster ablation proposal

The order ablation (first-appearance / alphabetical / frequency / shuffled) is
well-judged and should be run *after* the production A/B, as stated. Worth
adding: the current roster is built in **first-appearance order** by
construction, since `build_roster` appends as it encounters names. So the
production incremental arm and the warm arm differ in *both* completeness and
ordering stability - the warm roster's order is fixed from the start, while the
incremental one's is the same sequence revealed progressively. That is a
smaller confound than it first appears, but it is not zero.

---

## 14. Four-model decomposition — the ceiling was the model

The closed-set decomposition was run against every model with a verified VRAM
profile. Frozen inputs held constant: the same segmentation, the same roster,
the same 147 gold lines, temperature 0, idle GPU. Only the attributing model
changes. All four artifacts are contract-validated and committed as
`ab_test_runtime/experiments/closed_set__<model>.json`.

| model | context | open | closed-6 | oracle |
|---|---:|---:|---:|---:|
| qwen3.5-9b | 32768 | 35.4% | 34.7% | 49.0% |
| gemma-4-e4b | 32768 | 39.5% | 38.8% | 49.7% |
| **ministral-3-14b** | **16384** | **47.6%** | 41.5% | **61.2%** |
| ministral-3-14b-heresy | 16384 | 46.9% | 40.8% | 59.2% |

Paired significance on the same 147 lines (exact McNemar, so this tests whether
the models differ on this sampled set of lines rather than merely comparing two
aggregate percentages; temperature-0 runs here are deterministic and reproduce
byte-identically):

```
open arm
  qwen-9b vs gemma-e4b               15/21   p=0.405
  qwen-9b vs ministral-14b           15/33   p=0.013  significant
  qwen-9b vs ministral-14b-heresy    18/35   p=0.027  significant
  gemma-e4b vs ministral-14b         16/28   p=0.096
  ministral-14b vs heresy             8/7    p=1.000

oracle arm
  qwen-9b vs ministral-14b           16/34   p=0.015  significant
  gemma-e4b vs ministral-14b         19/36   p=0.030  significant
  ministral-14b vs heresy             9/6    p=0.607
```

### What this settles

**Model capacity was a major part of the observed 49% conditional ceiling.**
The 14B reaches **61.2%** with the same oracle candidate set, +12.2 points,
p=0.015. Every statement in earlier sections about a 49% "ceiling" should be
read as a property of qwen3.5-9b. The remaining 38.8% oracle error also shows
that model choice is not the whole problem.

**And it does so at half the configured context** - 16384 against 32768. This
rules out extra configured context as the explanation for its advantage. It
does not prove that the advantage is understated: if the evaluated prompts fit
inside 16384 tokens, a 32768 setting may not help. Whether a longer context
improves the 14B is unmeasured, and its verified VRAM profile currently caps it
at 16384 on this card.

**The two 14B variants are indistinguishable** (8/7 discordant, p=1.000). The
"absolute-heresy" fine-tune neither helps nor harms attribution, so the choice
between them can be made on other grounds.

**gemma-4-e4b is not significantly better than the 9B** (p=0.405 open, p=1.000
oracle) despite scoring 4 points higher on the open arm. On these 147 lines that
difference does not separate from chance.

**closed-6 remains a numerical loss for every model.** Scene-local candidates
cost points against simply supplying the full roster in all four runs. The
direction is consistent across model families and sizes, but the within-model
open-versus-closed differences are not individually significant here (exact
McNemar: qwen p=1.000, gemma p=1.000, ministral p≈0.122, heresy p≈0.093).
This is strong reason not to adopt closed-6, not proof that every possible
scene-local candidate method must fail.

### What it does not settle

The rejections of context supply and the earlier prompt-format experiments were
all measured on the 9B only. They have not been repeated on the 14B, and a model
with materially better selection could plausibly use context the 9B could not.
Those remain provisional.

Nor does this change the product arithmetic much. **61.2% under a perfect
oracle candidate set is still four wrong speakers in every ten lines**, and the
realistic open-arm figure is 47.6%. A better model moved the ceiling
substantially; it did not move it into unattended-audiobook territory.

### Practical consequence

Ministral-3-14b is the leading pass-2 candidate and should replace qwen3.5-9b
for the next validation run. It produced the single largest measured
improvement in this investigation - **+12.2 points on the open arm, larger than
warm roster's +5.1** - and requires no architectural change for an experiment,
only a model swap. It costs roughly 1.6x the wall-clock per book, from the
matrix timings. The repository's stated two-book validation rule means it
should not yet be called the production winner from one book alone.

The warm-roster A/B should be re-run on the 14B before it is implemented, since
its +5.1 was measured against a selector we now know to be the weaker one.

## 15. Model and responsibility split: recommended next design

### What the current model is being asked to do

Yes: the current attribution call combines two different responsibilities.
The model must first reason about who spoke a line, then serialize that answer
into the required JSON shape. Those are not equally valuable uses of model
capacity. Speaker/coreference inference is the hard semantic task; exact JSON
construction is a deterministic formatting task.

This distinction does **not** justify adding a second LLM whose only job is to
rewrite the first model's answer as JSON. A formatting model would add latency,
another failure surface, and an opportunity to alter a correct answer. The
simpler split is:

1. give each allowed speaker an opaque stable ID such as `C01`, `C02`, ...;
2. ask the attribution model to return only the selected ID (and any separately
   tested confidence/evidence fields);
3. validate that the returned ID belongs to the supplied set;
4. have Python map the ID to the canonical character name and construct the
   final JSON deterministically.

This removes spelling drift, aliases, invented names, quoting/escaping errors,
and most JSON-repair work without asking another model to reinterpret the
decision. It may improve reliability and speed; it should not be assumed to
produce a large raw attribution-accuracy gain until measured.

### Where separate models do make sense

A multi-model pipeline is reasonable when the stages require genuinely
different capabilities:

| stage | preferred responsibility |
|---|---|
| segmentation / quote extraction | deterministic code where possible; otherwise a small structured-output model |
| speaker attribution | strongest locally viable coreference/reasoning model |
| prose or instruction rewriting | an instruction-following model selected for that task |
| canonical names and final JSON | deterministic Python, not an LLM |

The important boundary is between semantic decisions and serialization, not
between "one model that writes prose" and "another model that adds braces."

### Hugging Face candidates worth testing

Test these on the frozen gold harness rather than selecting from general
benchmarks. Recommended order:

1. **Current Ministral-3-14B with candidate-ID output.** This isolates the
   output-contract change while retaining the strongest measured local model.
2. **[Microsoft Phi-4](https://huggingface.co/microsoft/phi-4).** A dense 14B,
   16K-context, MIT-licensed instruction/reasoning model. It is the most useful
   same-size challenger to the current leader, subject to a clean VRAM profile.
3. **[Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)** using the official
   checkpoint/quantization rather than an uncensored derivative. Test
   non-thinking first, then thinking mode as a separate arm because the latter
   changes latency and output behavior. Official GGUF quantizations are
   [available here](https://huggingface.co/Qwen/Qwen3-14B-GGUF).
4. **[Mistral Small 3.1 24B Instruct](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)**
   only if a quantization passes the existing VRAM-headroom checks. Its official
   card describes 24B parameters and 128K context; it is likely tight on this
   machine and must not be made to fit by weakening safety limits.
5. **[Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)** only as an
   explicit offload experiment. Although it activates about 3.3B parameters per
   token, all MoE weights still have to be stored, so low active-parameter count
   does not imply a comfortable 16GB load.

[Gemma 3 27B](https://huggingface.co/google/gemma-3-27b-it) is another plausible
large-model comparison, but its size, gated access, and likely offload cost make
it a lower-priority local candidate than the tests above.

These are candidates, not claims that any will beat Ministral-3-14B on this
task. Model-card capabilities and broad benchmarks do not substitute for
speaker-attribution results on the project's own fiction data.

### Exact experiment matrix

Keep inputs, segmentation, roster, temperature, load settings, and scorer
fixed. Run:

1. Ministral-3-14B, current JSON contract;
2. Ministral-3-14B, candidate-ID contract;
3. Phi-4 14B, candidate-ID contract;
4. official Qwen3-14B non-thinking, candidate-ID contract;
5. official Qwen3-14B thinking, candidate-ID contract;
6. Mistral Small 3.1 24B only after a safe load profile.

For every arm record:

- raw and oracle speaker accuracy;
- invalid-ID / parse-failure rate;
- correction or retry rate;
- latency and tokens per line/book;
- peak VRAM and actual loaded settings;
- exact paired disagreements and McNemar result;
- confidence risk/coverage if a usable confidence signal is emitted.

The candidate-ID comparison must use the same model on the same lines first.
Otherwise a simultaneous model swap would make it impossible to attribute any
gain to the cleaner contract.

### Decision rule

The expected ranking of effects is:

- deterministic ID-to-JSON conversion should primarily improve validity,
  consistency, and speed;
- changing the attribution model is more likely to move semantic accuracy;
- combining the strongest validated model with deterministic serialization is
  the most promising design;
- no model becomes the production default until it clears the second-book
  validation and produces a useful confidence/routing curve.

The immediate next test should therefore be **Ministral-3-14B current JSON
versus Ministral-3-14B candidate-ID output on the frozen harness**, followed by
Phi-4 and official Qwen3-14B under the winning output contract.

---

## 16. Six-model decomposition, and what separates them

Three further models were profiled and benchmarked on the frozen harness: same
segmentation, roster, 147 gold lines, temperature 0, idle GPU, thinking off.
All artifacts committed under `ab_test_runtime/experiments/`.

| model | context | open | closed-6 | oracle |
|---|---:|---:|---:|---:|
| **qwen/qwen3-14b** | 16384 | **48.3%** | 36.7% | **66.0%** |
| ministral-3-14b | 16384 | 47.6% | 41.5% | 61.2% |
| ministral-14b-heresy | 16384 | 46.9% | 40.8% | 59.2% |
| microsoft/phi-4 | 16384 | 45.6% | 32.7% | 59.2% |
| gemma-4-e4b | 32768 | 39.5% | 38.8% | 49.7% |
| qwen3.5-9b | 32768 | 35.4% | 34.7% | 49.0% |

Paired exact McNemar against the leader, oracle arm:

```
qwen3-14b vs qwen3.5-9b            31/ 6   p=0.0000  significant
qwen3-14b vs gemma-4-e4b           38/14   p=0.0012  significant
qwen3-14b vs phi-4                 20/10   p=0.0987
qwen3-14b vs ministral-14b-heresy  25/15   p=0.1539
qwen3-14b vs ministral-14b         25/18   p=0.3604
```

### What separates, and what does not

**The 9B/e4b tier is clearly beaten.** Every 14B-class model is significantly
better than both, and the gap is large: qwen3-14b repairs 31 lines the 9B gets
wrong while breaking 6.

**The four 14B-class models do not separate from each other.** qwen3-14b leads
on every arm but none of the pairwise comparisons reach significance (p=0.099 to
p=0.360). On 147 lines, a 5-7 point spread among them is not resolvable. Ranking
them beyond "all clearly better than the 9B tier" would be reading noise.

**Thinking was verified off, not assumed.** qwen3-14b is a reasoning model; zero
reasoning tokens were recorded across the run, so `reasoning_effort: none` is
honoured and the result is not inflated or penalised by hidden thinking.

**closed-6 is a loss for all six.** Scene-local candidates cost 5-13 points
against the full roster in every model tested, across four architectures and two
size tiers. This is now the most robustly replicated negative result in the
investigation.

### magistral-small does not fit and was refused

Weights alone occupy 13.51 GiB of a 15.92 GiB card, leaving 0.16 GiB at the
8192 minimum. It gets no verified profile and falls back to the conservative
default rather than being made to fit by weakening the VRAM guard, per this
brief's own ground rules.

### A measurement note worth keeping

The first profiling run of these three models was contaminated and discarded: a
hung experiment still held 9.75 GiB, so both new models measured a *negative*
context cost. Two lessons already recorded elsewhere applied again - a frozen
log does not mean a dead process, and an idle card must be verified rather than
assumed. `app/experiments/profile_vram.py` now exists so profiling a new model
is a command rather than a scratchpad script plus a hand-edited dict.

Also of note for capacity planning: the two new dense models cost **~160 KiB per
context token, ten times** the compressed-KV models already in use. That, not
parameter count, is what caps them at 16384 here.

### Revised recommendation

Pass 2 should move off qwen3.5-9b to a 14B-class model. **The evidence does not
support picking a specific one of the four** - choose on cost, licence and
stability, not on these scores. qwen3-14b and ministral-3-14b are the two
sensible defaults, and ministral is already validated in the model matrix.

The gain is real but bounded: **66.0% under a perfect oracle candidate set, and
48.3% realistically.** Better model selection has roughly doubled the 9B's
accuracy on the open arm and still leaves one line in two wrong.

---

## 17. Where this stands, and what should happen next

### The measured picture, in one place

| question | answer | evidence |
|---|---|---|
| Is the true speaker available? | 85% in the roster | text measurement |
| Does the pipeline pick it? | 29.9% (9B, shipped) | gold set |
| Is that a recall or selection problem? | **selection**, 55-point gap | decomposition |
| Can candidate pruning close it? | **no** - loses for all six models | closed-6, six models |
| Can deterministic tag rules? | **no** - 7.5% recall | text measurement |
| Can more context? | **no** - the model already has it | 2x2 + decomposition |
| Does roster warm-up help? | yes, +5.1 on the 9B | artifact-backed |
| Does a better model help? | **yes, roughly doubles open accuracy** | six-model benchmark |
| Is it enough? | **no** - 48.3% realistic, 66.0% oracle | six-model benchmark |

### The honest summary

A day of work moved the best measured configuration from **29.9% to ~48%**, and
established a ceiling of **66%** even when the correct answer is handed to the
model among five candidates. Every architectural idea tried - candidate
generation, tag extraction, context reformatting, prose passages - failed. The
two things that worked were a better model and a warmer roster, and neither is
architectural.

That leaves the project with a clear but uncomfortable position: **unattended
multi-voice attribution is not reachable on this hardware with these methods.**
One line in two is wrong at realistic settings. The remaining paths are:

1. **Confidence routing.** Find the subset that can ship unreviewed and route
   the rest to a human. Nothing measured so far tells us how large that subset
   is; the risk/coverage curve is the missing number and it has never been
   computed.
2. **The candidate-ID output contract (§15).** Structurally eliminates the 33%
   of errors that are invented names and misspellings, and retires the
   attestation gate and its retry cost. Cheap, well-motivated, untested. It must
   include an explicit not-listed option, because 15% of gold lines have a true
   speaker absent from the roster and forcing a choice there would convert
   honest abstention into confident error.
3. **Accepting a human-in-the-loop product**, and optimising for review speed
   rather than raw accuracy.

### What should not happen next

- Another prompt or context format experiment on the 9B. That model is no longer
  the intended production selector and every context idea has been tested twice.
- Ranking the four 14B-class models on the current gold set. They do not
  separate at n=147; more precision requires more judged lines, not more runs.
- Building scene-cast candidate generation. Rejected across six models.

### The methodological record

Worth stating because it shaped everything above. Across this investigation,
**every architecture reasoned about in advance failed, and every result that
survived came from inspecting data nobody had looked at**: shipped output
revealed 279 invented speakers, a second book revealed every honorific name
being rejected, an external audit revealed every scene break losing its silence,
per-line artifacts revealed a counting bug masquerading as a finding, and a
paired significance test revealed that a 4-point model difference was noise.

Two conclusions were reversed by better measurement *during* the same day. Both
reversals were caught by external review demanding artifacts rather than
aggregates. That is the single practice most worth carrying forward.
