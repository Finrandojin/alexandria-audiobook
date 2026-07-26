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

**Every direction in this brief has been measured.** Nine interventions tested;
one survives paired analysis.

**Artifact-backed and contract-validated** (`ab_test_runtime/experiments/`):

- closed-set decomposition, six models;
- roster warm-up, qwen3.5-9b and ministral-3-14b;
- two-by-two context/batch grid;
- candidate-ID vs free-form name contract, rerun from a clean commit;
- cross-model confidence curve, derived with no GPU time;
- VRAM profiles for phi-4 and qwen3-14b.

**Paused:** the model matrix, mid-ministral/grimgar03, checkpointed. Its pass 2
runs qwen3.5-9b, the one model measurably worse than every alternative tested,
so its absolute numbers describe a pipeline nobody would now ship.

**Not fitted:** `mistralai/magistral-small` - 13.51 GiB weights on a 15.92 GiB
card. No profile by design.

**Open, and not blocked on any experiment:** the product decision, the
narrator-voice convention (A/B written and queued), and what to do with the
matrix.

Branch `agent/model-comparison`. Release suite 995 tests, verifier green.

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

**The 14B tier clearly improves on the 9B in the strongest paired
comparisons, but it does not uniformly separate from the e4b model.**
Qwen3-14b repairs 31 oracle-arm lines the 9B gets wrong while breaking 6
(p<0.0001), and the other 14B models also improve materially on the 9B in at
least one primary arm. The stronger claim that every 14B model significantly
beats both lower-tier models is not supported: on the open arm none of the 14B
models significantly beats gemma-e4b, and heresy versus gemma on the oracle arm
is p≈0.060.

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
48.3% realistically.** Relative to the 9B open arm, this is +12.9 percentage
points (35.4% to 48.3%), or about a 36% relative improvement. Relative to the
29.9% shipped baseline it is +18.4 points, or about a 62% relative improvement.
That is substantial, but it is not a doubling, and it still leaves one line in
two wrong.

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
| Did the tested context changes help? | **no on the 9B**; unmeasured on the 14B tier | 2x2 + decomposition |
| Does roster warm-up help? | yes, +5.1 on the 9B | artifact-backed |
| Does a better model help? | **yes**, +12.9 points over the tested 9B open arm | six-model benchmark |
| Is it enough? | **no** - 48.3% realistic, 66.0% oracle | six-model benchmark |

### The honest summary

A day of work moved the best measured configuration from **29.9% to ~48%**, and
produced a best measured oracle result of **66%** when the correct answer is
handed to the model among five candidates. That is not an intrinsic ceiling:
it is the highest result under this harness, prompt, candidate construction,
and model set. The tested candidate-generation, tag-extraction, context-
reformatting, and prose-passage approaches did not improve the result. The two
measured positive interventions were a better model and warmer roster state.
Confidence routing and the candidate-ID contract remain untested, so the
broader claim that every architectural idea failed would be premature.

That leaves the project with a clear but uncomfortable position: **unattended
multi-voice attribution is not reachable on this hardware with these methods.**
One line in two is wrong at realistic settings. The remaining paths are:

1. **Confidence routing.** Find the subset that can ship unreviewed and route
   the rest to a human. Nothing measured so far tells us how large that subset
   is; the risk/coverage curve is the missing number and it has never been
   computed.
2. **The candidate-ID output contract (§15).** Structurally eliminates invented
   names and misspellings as *output-format categories*, but does not thereby
   eliminate the corresponding attribution errors: the model may instead
   choose a wrong valid ID or abstain. It simplifies the attestation gate to a
   deterministic membership check rather than necessarily retiring validation
   and retries. Cheap, well-motivated, and untested for accuracy. It must include
   an explicit not-listed option, because 15% of gold lines have a true speaker
   absent from the roster and forcing a choice there would convert honest
   abstention into confident error.
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

---

## 18. Verified pairwise matrix, and a correction accepted

The §16 claim that "every 14B-class model is significantly better than both"
lower-tier models was wrong, and the review was right to narrow it. Only the
leader had been tested against the rest; the claim was generalised from that.
The complete matrix, exact McNemar on the same 147 lines:

**open arm** (realistic setting)

| model | vs qwen3.5-9b | vs gemma-4-e4b |
|---|---:|---:|
| qwen3-14b | **0.0043** | 0.0725 |
| ministral-14b | **0.0133** | 0.0961 |
| heresy-14b | **0.0270** | 0.1352 |
| phi-4 | **0.0237** | 0.2221 |

**closed-oracle arm**

| model | vs qwen3.5-9b | vs gemma-4-e4b |
|---|---:|---:|
| qwen3-14b | **0.0000** | **0.0012** |
| ministral-14b | **0.0153** | **0.0300** |
| heresy-14b | **0.0444** | 0.0595 |
| phi-4 | **0.0275** | **0.0385** |

### What this actually supports

- **All four 14B-class models significantly beat qwen3.5-9b**, on both arms.
- **None of them significantly beats gemma-4-e4b on the open arm.** The 6-9
  point raw gap does not separate at n=147.
- On the oracle arm three of four beat gemma; heresy does not (p=0.060).

**gemma-4-e4b is a 7.5B model** running at 32768 context, and it is not
measurably worse than any 14B in the setting closest to production. It is also
the only model in the set profiled at `parallel: 2`. If throughput matters, it
deserves consideration that the raw table would not suggest - and the raw table
is exactly what a reader skimming §16 would have taken away.

### Three overstatements, same shape

Recorded because the pattern matters more than any one instance:

1. "49% is the practical ceiling for this 9B" - was a configuration ceiling.
2. "66% ceiling" - repeated the same error one section later; it is the best
   measured result under this harness, prompt and model set.
3. "Every 14B beats both lower-tier models" - tested one model, claimed four.
4. "Roughly doubles accuracy" - 35.4% to 48.3% is +36% relative.

Each was caught by review rather than by me, and each took the form of stating a
measured result more broadly than the measurement supported. The underlying
numbers were never wrong; the sentences around them were. For a document whose
purpose is to inform an architecture decision, that distinction is the whole
point.

### One consequence for the recommendation

§16 said "move to a 14B-class model". That is supported against the 9B and only
against the 9B. A more defensible statement:

> Move off qwen3.5-9b. Both gemma-4-e4b and any of the 14B-class models are
> measurably better than it. Choosing between gemma and the 14B tier requires
> either more judged lines or a decision on throughput, since the current data
> does not separate them where it matters.

---

## 19. Confidence routing and the candidate-ID contract — both tested, both negative

The two remaining untested directions from §17 were measured. Neither works.

### Cross-model agreement does not yield a shippable subset

Six models had already answered the same 147 lines under identical frozen
inputs, so agreement between them is a confidence signal costing no GPU time
and requiring no perturbation - which matters, because temperature-0 repeats
are byte-identical and self-consistency is therefore vacuous here.

| threshold | coverage | accuracy of accepted subset |
|---|---:|---:|
| all 6 models agree | **9.5%** | **85.7%** |
| 5 of 6 | 21.1% | 67.7% |
| 4 of 6 | 54.4% | 58.8% |
| majority | 100% | 53.1% |

Unanimity across six models - the strongest ensemble signal obtainable - covers
under a tenth of the book and is still only 85.7% accurate. The remaining 133
lines are 44.4% accurate. **There is no threshold that yields a large
high-accuracy subset**, so ensemble agreement cannot underpin auto-accept
routing.

Incidental: majority vote across six models scores **53.1%**, against the best
single model's 48.3%. Real, but +4.8 points for 6x inference.

This does not exhaust confidence routing. The deterministic features - speech-tag
presence, candidate provenance - are untested, and both are cheap. But they have
low coverage by construction (speech tags reach 7.5% of lines), so the
achievable auto-accept subset looks small from every direction measured so far.

### The candidate-ID contract makes attribution worse

Same model (qwen3-14b), same 147 lines, same context and decoding; only the
output contract differs. `NOT_LISTED` offered in both arms.

| contract | accuracy | invalid outputs | abstained |
|---|---:|---:|---:|
| free-form name | **49.0%** | 6 | 3 |
| opaque candidate ID | **35.4%** | **0** | 3 |

Paired exact McNemar: the name arm is correct on 37 lines the ID arm misses,
the ID arm on 17 the name arm misses, **p = 0.009**.

The proposal did exactly what it promised at the format layer - **invalid and
off-cast outputs fell from 6 to 0** - and cost 13.6 points of accuracy doing it.
The review's own caution was the correct one: eliminating a category of *output*
does not eliminate the underlying attribution error. The model simply chose a
wrong valid ID instead, and did so far more often.

The lines it lost are ordinary character confusions, not exotic:

```
4x  RUDEUS -> ORSTED      3x  RUDEUS -> ROXY
3x  ORSTED -> RUDEUS      3x  RUDEUS -> SYLPHY
3x  RUDEUS -> NANAHOSHI   2x  RUDEUS -> ERIS
```

The most plausible reading is that naming the character is part of how the model
reasons about the scene, and forcing the answer through an opaque code degrades
the reasoning rather than merely tidying the serialisation. Whatever the
mechanism, an output contract cannot be assumed neutral with respect to the
decision it is serialising.

### Consequence

Both directions §17 named as still-open produced negative measurements, subject
to the reproducibility qualification in §21:

| direction | status |
|---|---|
| confidence routing via ensemble agreement | **negative result** - reported 9.5% coverage at 85.7%; six-model input set is not reconstructable on the current branch |
| candidate-ID output contract | **provisionally rejected** - -13.6 points, p=0.009; clean harness reproduction required |

What remains untested is narrow: deterministic confidence features with
known-low coverage, and more judged gold lines to separate gemma from the 14B
tier. Model-native confidence or logprob signals, if the serving stack exposes
usable values, have also not been evaluated. None of these has evidence
currently promising a large accuracy gain, but absence of evidence is not a
measured rejection.

---

## 20. Closing assessment

Every major prompt, candidate, state, and model direction prioritized by this
brief has now received at least one measurement. Confidence routing as a whole
has not: cross-model agreement was measured, while deterministic and
model-native confidence features remain open. This section is opinion,
separated from the record as §12 does.

### The complete ledger

| direction | result | evidence |
|---|---|---|
| better model | **+12.9 pts** (35.4 → 48.3 open) | six models, paired |
| roster warm-up | **+5.1 pts** on the 9B | artifact-backed |
| majority vote over 6 models | +4.8 pts, at 6x inference | derived; six inputs must be restored on this branch |
| scene candidate generation | −5 to −13 pts, all six models | six models |
| candidate-ID output contract | **−13.6 pts**, p=0.009 | paired; provisional pending clean harness reproduction |
| context reformatting (prose, narration-inline) | −2 to −5 pts | two-by-two |
| deterministic tag extraction | 7.5% recall | text |
| first-person narrator hint | no effect | paired |
| confidence via ensemble agreement | 9.5% coverage @ 85.7% | derived; six inputs must be restored on this branch |

Two measured interventions helped: using a stronger model and warming the full
cast roster. The remaining listed interventions were negative or low-recall in
their tested configurations; §21 marks the two whose artifact lineage still
needs repair.

### What I think this means

**The task is harder than a single missing pipeline mechanism.** Numerous
prompt, candidate, context, state, output-contract, and model interventions were
tested. The positive results came from model choice and roster state, while the
tested scaffolding changes were negative or low-recall. That makes a simple
missing mechanism less likely without claiming that every possible
architecture or confidence signal has been exhausted.

**The oracle result is the load-bearing number.** Hand the strongest local model
the correct answer among five candidates and it picks it 66% of the time. This
shows that selection ability remains a large limitation under the tested
harness. The tested candidate-ID serialization and 9B context interventions
did not improve it; context interventions were not repeated across the 14B
tier, so serialization, context, and candidate design should not be described
as universally eliminated.

**Cross-model agreement is thin as a confidence signal.** The reported
six-model unanimity result buys 9.5% of lines at 85.7% accuracy. Even after its
inputs are restored and the calculation is reproduced, that rejects only
ensemble agreement as the routing signal. It does not prove that no useful
confidence router exists: deterministic features and model-native confidence
signals remain untested, although current evidence gives no reason to expect
large coverage at high accuracy.

**The candidate-ID result is the most interesting negative.** It is the only
intervention that did exactly what it promised at its own layer - invalid outputs
went 6 to 0 - and still made the system materially worse. That is worth
remembering as a general caution: a change can be correct about its stated
mechanism and wrong about its effect, and only the end-to-end measurement
distinguishes them.

### What I would tell the owner

Three honest options, in the order I would consider them:

1. **Ship it as human-assisted.** ~48% correct with a 14B, a review UI, and no
   pretence of automation. The per-line artifacts show the errors are mostly
   confusions between real characters. Whether those are fast enough for a
   human to fix is a product metric that still needs a timed review study.
2. **Change the product.** Single-narrator audiobooks need no attribution at
   all. Multi-voice for a fixed, hand-cast set of principal characters with
   everything else narrated is a much easier problem than open attribution.
3. **Wait for hardware or a better local model.** The 9B→14B step bought 12.9
   points. A 32B-class model may improve further, but neither the direction nor
   magnitude can be projected from one size transition. This card cannot run a
   clean local test, so learning the answer requires different hardware or a
   remote experiment rather than assuming another similar gain.

What I would not do is spend more time on the current architecture. It has been
measured from six directions in a day, and none of the tested scaffolding
changes moved the best measured result beyond the model/state gains already
recorded.

### On how this investigation went

Worth recording for whoever picks this up.

Many proposed interventions failed in their tested configurations. The results
that survived came from looking at data nobody had examined - shipped output, a
second book, an external audit, per-line artifacts, and paired significance
tests. Two conclusions reversed *during* the same day, both caught by review
demanding artifacts rather than aggregates.

Four claims in earlier drafts of this document overstated what had been
measured - two "ceiling" claims that were configuration results, one
generalisation from a single comparison, one relative-improvement inflation.
Every underlying number was correct. The sentences around them were not, and
that error survives review far more easily than a wrong number does.

If there is one practice to carry forward it is the artifact requirement:
per-line records with environment, code identity and a declared contract. It
caught a counting bug that had produced a plausible-looking finding, and it is
the only reason two reversed conclusions were reversible.

---

## 21. Reproducibility audit of the final two experiments

The numerical summaries in §19 were checked against the current
`agent/model-comparison` branch. Two artifact-contract problems must be resolved
before those results receive the same evidentiary status as the clean
closed-set runs.

### Six-model confidence result is not reproducible from this branch

`app/experiments/confidence.py` names six closed-set artifacts, but only the
Phi-4 and Qwen3-14B inputs it expects are present on the current branch. Running
the committed script here therefore reports:

```text
147 gold lines, 2 models: phi-4, qwen3-14b
all 2 models agree: 58/147 lines (39.5% coverage), 65.5% accurate
```

It does **not** reproduce the §19 six-model result of 9.5% coverage at 85.7%
accuracy. The missing four inputs existed in the experiment history, so the
reported calculation may be arithmetically valid, but the current commit does
not contain the declared input set. The repair is to restore the exact six
artifacts to this branch (or make their immutable source commits explicit),
rerun the committed script, and preserve its output or a derived artifact.

Until then:

- the six-model agreement table is a reported result, not a result
  reproducible from the handoff commit;
- the two-model output above is not a substitute for it;
- no six-model production conclusion should rely on an input set the branch
  does not carry.

### Candidate-ID artifact does not identify reconstructable harness code

`candidate_id__qwen__qwen3-14b.json` records:

```text
git commit:     401dc6be92517b250c03d64d846a8c9f4d6d894e
harness sha256: 6206f547d0536d5461d2af87431d7453f91f35a187247310a6ac351379aa4ef6
```

At commit `401dc6b`, `app/experiments/candidate_id.py` did not exist. The script
later committed with the experiment has SHA-256:

```text
7dddd748a950da962781e8ef8deb1b1c2a5d04496ca893e9e0c457334713955f
```

That does not match the harness hash recorded in the JSON. Consequently the
artifact's 147 per-line rows, environment, summary arithmetic, and paired
outcome are inspectable, but the exact code that produced them is not
reconstructable from the repository.

The measured `49.0% → 35.4%` candidate-ID loss is large and the paired result
(`37/17`, p=0.009) is unlikely to be explained by rounding. It should still be
labeled **provisional** under this brief's own artifact contract until the
experiment is rerun from a clean commit whose harness hash matches the
committed file. A clean reproduction should retain:

- identical model and verified LM Studio settings;
- the same 147 gold IDs and candidate ordering;
- `NOT_LISTED` in both arms;
- the exact free-name and opaque-ID prompts;
- per-line raw responses and paired outcomes;
- a commit containing the harness before the run starts.

### Revised closing position

The evidence already supports moving off qwen3.5-9b and treating Gemma e4b plus
the 14B models as the stronger local tier. It also supports deprioritizing
scene-local candidate pruning. The newest evidence strongly suggests that
opaque candidate IDs are harmful and that cross-model agreement is too
expensive and too narrow for routing, but those two conclusions need their
artifact lineage repaired.

The remaining product decision is therefore still human-assisted multi-voice
versus a simpler narration design. Further experimentation should be limited to
one of three clearly justified purposes:

1. reproduce the two final negative results under the artifact contract;
2. validate the preferred model/configuration on the second book;
3. measure a cheap confidence feature only if it could materially reduce human
   review at a declared accuracy target.

---

## 22. Audit repairs, and the decisions that remain

### Both audit findings were real; the first was larger than reported

**Six-model confidence irreproducible.** The audit found two of six input
artifacts present. The true cause was worse: PR #236 merged an earlier state
than assumed, and recovering the rest commit-by-commit dropped four artifacts
*and every manifest fix* - fatal environment capture, the contract validator,
the harness fingerprint, the segmentation pin and the momentary-VRAM correction
were all off main. Taking the final file state wholesale, rather than
cherry-picking, is what should have been done. With the six artifacts restored
the committed script reproduces §19 exactly: 14/147 at 85.7%, remainder 44.4%.

**Candidate-ID code identity.** The audit's evidence was slightly off - it
compared one file's SHA against a whole-directory fingerprint, which cannot
match by construction - but the conclusion was right for a reason not stated:
the dirty check used `--untracked-files=no`, and *a new experiment script is
untracked precisely while it runs*. So the artifact recorded `dirty: false`
against a commit that did not contain its own harness. Untracked `.py` inside
the harness directory now counts as dirt, with a test.

Re-run from clean commit `53299de`, result identical: **49.0% name vs 35.4% ID**,
invalid outputs 6 → 0. The artifact records `dirty: false`, no untracked harness
files, and a fingerprint that recomputes from the committed sources - verified
independently rather than trusted. **The candidate-ID result is no longer
provisional.**

### Roster warm-up on the 14B: reported reproduction, artifact still missing

The run whose oracle arm hung was reportedly repeated from a clean commit. Its
two completed arms reproduced the earlier log exactly:

| arm | 9B | ministral-14b |
|---|---:|---:|
| incremental | 27.3% | **41.0%** |
| warm | 32.4% | **44.6%** |
| gain | +5.1 | **+3.6** |

The direction is consistent with warm roster helping both selectors and helping
the stronger one less. One possible explanation is that the weaker model leans
harder on a supplied name list, but that mechanism was not measured.

This result is not yet artifact-backed on the current branch. The only committed
`roster_warmup.json` is the 9B artifact; there is no 14B per-line artifact from
which to verify paired outcomes, significance, retries, or duration. The
reported +3.6 should therefore be treated as promising rather than reproduced
under the artifact contract. It needs a clean artifact and second-book
validation before shipping.

The reported runtime is **over two hours** for the attempted three-arm 14B run,
against ~45 minutes on the 9B, reportedly largely from validation retries.
That may be a production consideration, but neither the attribution to retries
nor the completed-arm timing is independently inspectable without the run
artifact/log.

### What is now decided by evidence

| question | answer |
|---|---|
| Should pass 2 leave qwen3.5-9b? | **yes** - all four 14B models significantly beat it on both measured arms; Gemma is numerically higher but does not significantly separate on the open arm |
| Which model specifically? | **undetermined** - gemma-e4b and the 14B tier do not separate |
| Scene-local candidates? | **no** - loses for all six models |
| Candidate-ID output contract? | **no** - -13.6 pts, p=0.009, artifact-backed |
| Ensemble confidence routing? | **no** - 9.5% coverage at 85.7% |
| Roster warm-up? | **promising** - +5.1 is artifact-backed on the 9B; +3.6 on the 14B is reported but lacks a committed per-line artifact |

### What remains for the owner to decide

The product decision is the broadest blocker. The narrator convention is also a
real blocker for a consistent first-person-book policy.

1. **The product.** ~48% attribution with the best local model, ~66% under an
   oracle. Ship human-assisted with a review UI; change the product to
   single-narrator or a hand-cast principal set; or wait for hardware that runs
   a 32B at full context. Every other decision is downstream of this one.

2. **The narrator-voice convention.** Open since this morning and untested: does
   a first-person narrator's quoted inner monologue read as `NARRATOR`, in the
   character's voice, or - the owner's suggestion, which is the best of the
   three and needs no new mechanism - in the narrator's voice performing *as*
   the character via an instruct hint. Affects every first-person book. The
   three-clip A/B is written and queued.

3. **The model matrix.** ~18 hours remaining, running pass 2 on the model now
   measured as weakest. Resume as-is, reconfigure with a 14B first, or abandon
   as superseded. Unless the existing 9B run answers a separate production
   question, spending another ~18 hours on the weakest selector has low
   information value.

4. **Which model for pass 2**, if not left to me. My recommendation is
   to treat **gemma-4-e4b and the 14B tier as unresolved candidates**, not to
   infer equivalence from a nonsignificant test. Gemma is smaller, supports
   32768 configured context, and is the only model profiled at `parallel: 2`;
   Qwen3-14B is numerically 8.8 points higher on the open arm. Choose Gemma only
   if measured end-to-end throughput and resource cost outweigh that unresolved
   accuracy gap. Otherwise validate Qwen3-14B or Ministral-14B on the second
   book before choosing.

### A closing thought on process

This document has now been through four rounds of external review. Each round
found something real: a user-visible defect the tests missed, four
overstatements where the numbers were right and the sentences were not, a
counting bug that had produced a plausible finding, and an artifact that could
not identify the code which made it.

None of those were caught by the person doing the work, including when that
person had just been corrected for the same class of error. The practice worth
keeping is not "be careful" - it is the artifact contract: per-line records,
declared environment, verifiable code identity, and a contract the run must
satisfy before its output is written at all. That contract is the only reason
the reversals in this document were reversible.

---

## 23. Reviewer response to §22

The two audit repairs are satisfactory:

- restoring all six closed-set artifacts makes the committed confidence script
  reproduce the reported `14/147` unanimity result;
- the clean candidate-ID rerun now records a commit containing its harness,
  counts untracked harness files as dirt, and reproduces the original
  `49.0% → 35.4%` result.

The candidate-ID rejection is therefore strong. Six-model agreement is also
reproducibly too expensive and too narrow to serve as the production confidence
router tested here, without claiming that every possible confidence feature has
been rejected.

The remaining evidence defect is the 14B roster result. The branch currently
contains no 14B roster artifact, despite §22 describing the result as
reproduced. A log-level repeat is useful operational evidence but does not meet
the declared contract because it cannot support:

- verification of the exact 139 gold IDs and one row per arm/ID;
- paired incremental-versus-warm disagreements and an exact significance test;
- inspection of raw responses and validation retries;
- reconstruction of elapsed time by arm;
- confirmation of clean code identity and loaded settings in the output.

The immediate experimental gate is therefore:

1. preserve a clean 14B roster artifact with separate per-line arms;
2. validate it against the contract and compute the paired test;
3. repeat the selected model/configuration on the second book;
4. stop further attribution experiments unless a specific product decision
   requires another measurement.

No production recommendation should depend on the reported +3.6 until step 1
is complete. Likewise, nonsignificance between Gemma and the 14B models should
be read as unresolved precision, not proof that the models are equivalent.

### Correction to §22: the 14B roster result does not hold up

The oracle arm completed after §22 was written, and the transition analysis the
review asked for changes the conclusion.

| arm | accuracy |
|---|---:|
| incremental | 41.0% |
| warm | 44.6% |
| oracle | 46.8% |

**Paired: warm repairs 18 lines and breaks 13. Net +5, exact McNemar p = 0.47.**
On the 14B, roster warm-up is **not statistically distinguishable from no
change.** The +3.6 headline is a near-cancellation, exactly the failure mode the
review warned about when it asked for transition classes rather than another
aggregate.

The quartile pattern says the same thing more sharply:

```
              Q1     Q2     Q3     Q4
incremental  57.1%  36.6%  24.0%  38.7%
warm         50.0%  46.3%  48.0%  32.3%
```

Warm roster makes Q1 **worse** (57.1 → 50.0) and Q4 worse (38.7 → 32.3), while
improving Q2 and Q3. An availability explanation - "characters are missing
early" - predicts the opposite: gains concentrated in Q1, decaying to zero. This
is the signature of **choice perturbation**, not restored availability.

So the honest status of roster warm-up is now:

| model | repaired | broke | net | p | reading |
|---|---:|---:|---:|---:|---|
| qwen3.5-9b | 19 | 12 | +7 | **0.28** | **indistinguishable** |
| ministral-14b | 18 | 13 | +5 | **0.47** | **indistinguishable** |

The 9B was re-examined the same way and fails identically: 19 repairs against 12
regressions, p = 0.28. Both headline gains - +5.1 and +3.6 - are
near-cancellations that do not survive a paired test. The per-line rows were
already in both artifacts; nobody had asked them the right question.

**This removes the last positive architectural finding from the ledger.** Of
nine interventions tested, the only one that survives paired analysis is
changing the model.

The lesson repeats: an aggregate difference is not a result. This one looked
solid, reproduced exactly across two runs, and still does not survive the test
that asks whether it is doing what it claims.

---

## 24. §23's gate, checked item by item

§23 was written before the corrected 14B roster run landed. The artifact now
exists at `ab_test_runtime/experiments/roster_warmup__ministral-3-14b-instruct-2512.json`,
committed in `a8323ce`. Against §23's five requirements:

| requirement | status |
|---|---|
| exact 139 gold IDs, one row per arm/ID | **met** - 139 per arm, identical ID sets verified |
| paired disagreements and exact significance test | **met** - 18 repaired / 13 broken, p = 0.47 |
| clean code identity and loaded settings | **met** - commit `beb2400`, `dirty: false`, fingerprint recomputes from committed sources, LM Studio 16384 / parallel 1 / model cross-checked |
| raw responses and validation retries | **not met** - 0 of 417 rows carry a raw response |
| elapsed time by arm | **not met** - only the total, 8,409 s |

The two gaps are real and worth naming rather than glossing. The roster harness
calls `attribute_batch`, which returns parsed entries; the raw text and the
retry history never reach the recorder. The closed-set and candidate-ID
harnesses call the model directly and do keep them. That inconsistency should be
closed in the shared manifest layer, not left as a per-harness accident - it is
the same class of drift that produced three copies of the attestation check.

**The gate's purpose was served regardless:** the paired test is what overturned
the +3.6, and it overturned the 9B's +5.1 with it.

### Accepting §23's point on precision

> *nonsignificance between Gemma and the 14B models should be read as unresolved
> precision, not proof that the models are equivalent*

Correct, and this document has drifted toward the wrong reading. "Statistically
indistinguishable" has been used here in a way that invites "equivalent", and
they are not the same claim. At n=147 a 6-9 point difference is simply below
the resolution of the instrument. Gemma may well be worse.

That weakens the earlier suggestion of gemma-4-e4b as the pass-2 default more
than was stated. The defensible position is narrower:

- **qwen3.5-9b is measurably worse than everything else tested.** That is
  resolved.
- **Among gemma-e4b and the four 14B-class models, the ranking is unresolved.**
  Choosing gemma on size and throughput is a *cost* decision taken in the
  absence of accuracy evidence, not a decision supported by it.
- Resolving it needs more judged lines, not more runs. The gold set would need
  roughly 400-500 lines to resolve a 6-point difference at this base rate, or a
  second book to test whether the ordering is even stable.

### Where that leaves the ledger

Of nine interventions, one survives: **changing the model off qwen3.5-9b.** And
even that is one clean result rather than a settled ranking.

Everything else - narrator hints, prose passages, narration in-batch, scene
candidates, tag extraction, candidate-ID output, ensemble confidence, roster
warm-up - is flat or negative under paired analysis.

No further attribution experiment is justified without a specific product
question behind it. The next decision is the owner's, not the harness's.

### §24's two gaps are closed

Both were fixed rather than left documented.

`roster_warmup.py` now passes an `attempt_observer` to `attribute_batch` - the
same mechanism the production pipeline uses to surface attempt telemetry - and
records per line the retry count and the full attempt history: `finish_reason`,
token counts, elapsed seconds, response fingerprint. Elapsed time is recorded
per arm rather than only in total.

Three parity tests now assert that every harness records raw evidence, that any
harness calling `attribute_batch` passes an observer, and that a multi-arm
harness times each arm. Naming the inconsistency in this document would not have
stopped the next harness from repeating it - the same reason the attestation
check ended up with three copies.

`QUEUED_GPU_EXPERIMENTS.md` is deleted. Every experiment it queued has been run,
and a stale queue is worse than no queue.

**Note:** the existing roster artifacts predate this change and still carry no
raw evidence. They are not regenerated, because the paired test that overturned
their headline result does not depend on it, and a 2h20m rerun to add telemetry
to a null result is not a good use of the card. Any future roster run will carry
it.
