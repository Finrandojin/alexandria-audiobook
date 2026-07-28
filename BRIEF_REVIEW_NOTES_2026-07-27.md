# Review notes for the current attribution brief

Date: 2026-07-27

Status: untracked while clean-tree experiments are active. Merge these points
into `BRIEF_ADDENDUM_2026-07-27_CLOUD.md` after the current reasoning-arms and
pipeline-repeat runs finish.

## Findings

### 1. Production thinking is the strongest intervention result, but not yet a shipping choice

The clean production artifacts support:

| book | baseline | thinking | effect | p |
|---|---:|---:|---:|---:|
| Grimgar 03 | 55.5% | 63.7% | **+8.2** | **0.0022** |
| Mushoku 16 | 49.6% | 52.5% | +2.9 | 0.627 |

The brief correctly calls Mushoku a failure to confirm rather than a
contradiction: its interval includes both zero and the Grimgar effect. The
evidence is one significant production result plus one underpowered,
directionally positive result.

The “headroom behaves as predicted” paragraph is wrong. The prediction in §6a
was that the effect would shrink as the baseline rises. Instead:

- lower baseline, Mushoku 49.6%: +2.9;
- higher baseline, Grimgar 55.5%: +8.2.

That is the opposite pattern. Headroom does not explain the cross-book effect
difference on these data and should be removed as a mechanism.

More Mushoku labels would improve precision on that book. A third large book
would add both power and generalization evidence; it should not be dismissed as
inferior to extending Mushoku because the two investments answer different
questions.

### 2. The crossover supports uncertainty, not a causal story

The corrected crossover interpretation is sound: none of the four held-fixed
effects is significant, and the pooled segmentation p-value is invalid because
it double-counts the same lines.

The one cross-reload temperature-zero change is observed, but its mechanism is
unknown. GPU reduction order or memory layout is a hypothesis, not a finding.
The safe conclusion is simply that this configuration was perfectly stable
within a loaded instance and differed on 1/399 rows after reload.

Three temperature-0.6 repeats establish an observed 4.5-point range in one
cell. They do not establish a general variance distribution. The predeclared
`2 × SD` rule is useful screening but is not an inferential test with three
runs.

### 3. Qwen3-32B does not establish a general model-size plateau

Qwen3-32B does not significantly improve on Qwen3-14B in the Mushoku
closed-set harness. That is a valid within-family negative result.

The broader statement that accuracy “plateaus above about 7B” and therefore
points to task representation is too wide. The experiment:

- uses one fixture and a component harness;
- has limited power among the clustered models;
- mixes architectures, training recipes, quantizations, and serving stacks;
- does not show that all larger models or production paths share a plateau.

The defensible conclusion is narrower: within this Mushoku closed-set
measurement, increasing Qwen from 14B to 32B did not produce a detectable gain,
and model size alone has not explained the remaining error.

### 4. Grammar-constrained decoding produced a new, scoped positive result

A clean artifact landed after the brief update:
`grammar_constraint__mushoku16__mistralai__magistral-small__local-llamacpp.json`
(`validation: ok`, `dirty: false`).

| arm | free | grammar | paired transitions |
|---|---:|---:|---:|
| open roster | 74/139 = 53.2% | 72/139 = 51.8% | 0 repairs / 2 regressions |
| oracle five-name set | 81/139 = 58.3% | 92/139 = 66.2% | 12 repairs / 1 regression |

The oracle gain is +7.9 points with exact paired p≈0.0034. The open-roster arm
does not improve.

Interpretation:

- constrained decoding can repair canonical-name/off-list failures when the
  correct answer is guaranteed inside a small candidate set;
- it does not improve attribution with the full roster in this experiment;
- it therefore repairs an output-contract failure, not the central
  open-roster speaker-selection problem;
- the experiment is single-line closed-set decoding, not the production
  batched JSON path.

This nominates a production-path grammar experiment only if a safe dynamic
grammar can preserve the JSON schema and restrict each speaker value to
canonical roster names. It does not justify fuzzy identity matching or a
candidate-pruning architecture by itself.

### 5. Compare the actual shipping candidates

The next model decision should compare:

- Gemma-3-27B without reasoning;
- Qwen3-14B with reasoning;

through the same production path, serving stack, segmentation policy, books,
and paired IDs. Report accuracy together with wall time, reasoning tokens,
VRAM, retries, parse failures, and completion reliability.

Gemma's exploratory 71.5% and Qwen's production 63.7% are not directly
comparable. Qwen thinking is roughly 5–7× the baseline pass-2 cost, so a cheaper
model that matches it in production could be the better product even without a
higher component score.

## Recommendations

1. Correct the headroom paragraph before the addendum is merged.
2. Add the grammar result as an output-contract result, not a general
   attribution breakthrough.
3. Keep Qwen3-32B's conclusion fixture- and harness-scoped.
4. Let the active Qwen3-32B reasoning run finish, but interpret it only after
   confirming nonzero reasoning tokens and production-path transfer.
5. Finish the current production repeat before editing tracked files.
6. After the queue drains, replace stale status sections and merge the main
   brief plus cloud addendum into one compact current-state handoff. At more
   than 1,700 combined lines, the documents are again becoming a chronological
   lab notebook rather than a decision brief.

## Suggested test programme

The addendum's §13.7 identifies the right broad areas. The experiments below
would make them answerable rather than merely generate another ranking.

### Priority 1 — Find the cheapest useful reasoning policy

Production thinking is the only prompt/runtime intervention with a significant
production result. Its problem is cost. Before testing more models, measure an
accuracy–cost curve on Qwen3-14B through the production path:

| arm | purpose |
|---|---|
| reasoning off | shipping baseline |
| full thinking | established upper point |
| small reasoning budget | cheap point |
| medium reasoning budget | middle point |
| selective full thinking | route only predicted-hard rows |

Use whatever reasoning-budget control the serving API actually supports; do not
pretend `max_tokens` is a clean reasoning budget if it also truncates the JSON
answer. Record reasoning tokens, wall time, truncations, retries, repairs, and
regressions per line.

For selective thinking, derive labels from existing paired rows:

- `RESCUE`: baseline wrong, thinking right;
- `HARM`: baseline right, thinking wrong;
- `NEUTRAL`: both agree in correctness.

Test cheap routing features offline first: line length, vocative, speech tag,
candidate availability, prior-speaker continuity, roster size, baseline answer
attestation, parse/retry history, and disagreement with one cheap alternative.
Cross-validate by scene or book, not random rows, to avoid adjacent-dialogue
leakage.

Decision metric:

> Net repaired lines and accepted accuracy as a function of the percentage of
> rows sent to thinking, with actual wall-time cost.

Stop if no router beats random routing at the same coverage on held-out scenes.

### Priority 2 — Separate useful history from error propagation

“Give the model the previous speaker” needs three arms, not two:

1. no committed history;
2. **oracle** previous-speaker history;
3. **predicted** previous-speaker history.

Reset history at defensible scene boundaries and report accuracy by dialogue
turn distance. This separates:

- whether previous-speaker state contains useful information;
- whether the current system predicts it accurately enough to use;
- how quickly a wrong assignment propagates.

If oracle history helps but predicted history does not, the representation is
useful and the state source is not. If both help, it is a production candidate.
If neither helps, retire simple sequential history.

A second, distinct test should attribute an entire dialogue scene jointly with
canonical speaker IDs. Joint scene decoding can exploit turn-taking without
committing an early error as immutable state. Do not conflate it with the
sequential-history experiment.

### Priority 3 — Context width, with evidence-distance analysis

The running w1/w4/w15/w40 sweep is worthwhile, but “62.1% of errors have no
nearby name” does not itself imply that a wider window contains the answer.
Names may remain absent, and irrelevant prose may dilute the signal.

For each paired row, record:

- distance to nearest true-speaker mention;
- distance to nearest explicit speech tag;
- whether a vocative names the addressee;
- prompt tokens and latency;
- repairs and regressions at each width.

Report effects stratified by evidence distance. A global average can hide “wide
context helps distant-evidence rows and harms local dialogue.” If that pattern
appears, use retrieval or adaptive width rather than shipping w40 everywhere.

The production decision requires a production-path confirmation at the winning
width. A diagnostic-harness gain is only a nomination.

### Priority 4 — Test recall and selection as separate axes

The oracle candidate arm does not show that a 97%-recall scene cast will gain
17 points. It changes both answer availability and the candidate composition,
and selection still fails frequently with the answer present.

Build a recall/size frontier using real candidate-generation policies:

| policy | mean set size | speaker recall | conditional selection | total accuracy |
|---|---:|---:|---:|---:|
| full roster | | | | |
| current scene cast | | | | |
| high-recall scene cast | | | | |
| high-recall + history | | | | |

Predeclare a recall floor, preferably at least 97%, before comparing selection.
Never interpret a smaller candidate set as better if it silently removes true
speakers.

### Priority 5 — Use grammar as targeted recovery

Grammar constraints help when off-list output is common and do nothing when it
is rare. Test them as a targeted recovery path:

1. run the normal production call;
2. if the predicted name is unattested or the response is invalid, retry once
   with a grammar restricted to canonical roster names;
3. compare that with the current reject/fallback behavior.

Report:

- number of rows routed;
- correct recoveries;
- wrong forced labels;
- canonical-name repairs;
- added latency.

This is likely cheaper and safer than constraining every production response.
The grammar must preserve the complete batched JSON contract; a single-name
diagnostic grammar is not enough.

### Priority 6 — Audit the oracle errors

The 24–34% failure rate when the true speaker is among five candidates is now
more informative than another model ranking. Blindly adjudicate a stratified
sample of oracle errors into:

- insufficient supplied context;
- addressee inversion;
- turn-taking/history failure;
- narration/quotation-boundary error;
- alias/canonical-name error;
- genuinely ambiguous or questionable gold;
- unexplained selection failure.

Sample both model-consensus errors and model-disagreement errors. If multiple
strong models fail the same line, inspect the input and gold before calling it
a capability ceiling.

### Priority 7 — Verify fixture representativeness

The fixture is random at construction, but later ambiguity and unique-text
filters can change the evaluated population. Compare the final scored rows with
all spoken lines on:

- line length;
- character frequency;
- dialogue density;
- explicit-tag frequency;
- vocative frequency;
- repeated-text frequency;
- book position and scene type.

This determines whether the measured ~50–70% range describes representative
dialogue or a filtered subset enriched for hard lines.

## Recommended order while the owner is away

1. Let the already-running Qwen3-32B reasoning and production-repeat jobs
   finish without tracked edits.
2. Finish and score the context-width sweep already described in the brief.
3. Run offline analyses that consume existing artifacts only:
   selective-thinking routing features, oracle-error strata, and fixture
   representativeness.
4. Run the oracle-history versus predicted-history factorial locally.
5. Run targeted production grammar recovery.
6. Run the production Gemma-3-27B versus Qwen3-14B-thinking comparison only
   after serving stack and segmentation policy are matched.

Do not restart the broad cloud model sweep merely because time is available.
The next useful information comes from decomposing state, context, routing, and
output constraints—not from adding another row to an unresolved model table.

## Overnight stop conditions

- Stop an arm after repeated endpoint failure according to the fixed retry
  policy; resume only from a fingerprint-matching checkpoint.
- Do not write a headline from an incomplete artifact.
- Do not extend a test after its predeclared question is answered.
- Do not run reasoning arms on a model until one probe confirms nonzero
  reasoning tokens.
- Do not keep the rented instance alive while the queue is paused.
- Preserve per-line transitions and paired IDs for every comparison.

## Additional idea bank

These are deliberately broader than the immediate queue. Each should first run
as a small paired diagnostic on existing gold rows.

### A. Score candidates instead of generating a name

The model currently generates one speaker name. An alternative is to compute a
score for each candidate under a statement such as:

> “The speaker of this line is HARUHIRO.”

Use forced-choice conditional log likelihood, token log probabilities, or a
binary entailment score, then select the highest candidate. This turns open
generation into ranking and makes a top-two margin available for routing.

Test on:

- oracle five-name sets first, where candidate recall is fixed at 100%;
- then high-recall scene sets;
- finally the full roster.

Compare accuracy, top-two margin calibration, latency, and character-frequency
bias. Stop if per-candidate calls make it slower than full thinking without
improving oracle selection.

### B. Pairwise candidate tournament

If multiway choice is the failure, ask pairwise:

> “Which is more likely to speak this line: RANTA or HARUHIRO?”

Use a bracket or all-pairs voting within a small high-recall cast. Pairwise
judgment may be easier than choosing among 17 names, but it can be
non-transitive and expensive. Measure cycles explicitly. This is only sensible
after a candidate generator reaches the recall floor.

### C. Scene-level joint attribution

Give the model a complete dialogue scene and ask it to label all quotes at
once, using canonical IDs and grammar-constrained output. This exposes
turn-taking, question/answer pairs, interruptions, and stable speaker runs that
independent batches cannot use.

Controls:

- same scene with lines independently attributed;
- same scene jointly attributed;
- joint attribution with line order shuffled as a negative control.

If joint decoding helps but shuffled decoding does not, the gain comes from
sequence structure rather than merely more tokens.

### D. Global sequence decoding

Treat each line's candidate scores as emissions and choose the best sequence
with a lightweight dynamic programme. Possible transition features:

- speaker usually changes after an explicit response;
- consecutive fragments may retain a speaker;
- vocatives favor the named person as addressee, not speaker;
- explicit tags override transition priors;
- scene entry/exit constrains the active cast.

Learn no transition weights from the test fixture. Set them on a development
book or use transparent hand-set values, then validate cross-book. Compare
against greedy per-line attribution and report whether errors propagate.

### E. Retrieval instead of a uniformly wider window

Fixed w40 may add mostly irrelevant prose. Retrieve compact evidence:

- last explicit speech tag for each active character;
- last action involving the character;
- last line attributed to the character;
- most recent entrance/exit;
- nearest vocative and its likely addressee;
- scene synopsis and active cast.

Give the model w4 plus the retrieved evidence. Compare with w15/w40 at similar
token counts. If retrieval wins, the bottleneck is evidence selection rather
than raw context width.

### F. Oracle evidence test

Before building a retriever, hand the model the exact sentence or prior turn
that a human used to resolve a stratified sample of errors. If accuracy remains
poor with oracle evidence, retrieval cannot solve the selection problem. If it
jumps, building retrieval is justified.

This is cheaper than implementing an elaborate memory system before proving
that the missing evidence is sufficient.

### G. Character-language fingerprints

Some characters have strong idiolect:

- third-person self-reference;
- dialect or contractions;
- honorific choices;
- recurring phrases;
- formality and pronoun patterns.

Build profiles from text outside the evaluated scenes, then add one compact
profile sentence per active candidate or use a separate style-similarity
reranker. Split by scene or chapter so the evaluated line never contributes to
its own profile.

Report gains by frequent versus rare character. Stop if it merely amplifies the
most common speaker.

### H. Confusion-directed specialist rules

Build paired confusion matrices, not only aggregate accuracy. If a small number
of recurring confusions dominate—speaker/addressee, protagonist/narrator,
Ranta/Haruhiro—test narrow resolvers only on those pairs.

Examples:

- a vocative resolver for addressee inversion;
- a continuation resolver for split quote fragments;
- a narrator/protagonist convention;
- a two-character dialogue alternation check.

Each rule must report repairs and regressions on all lines it touches. Avoid a
general rule justified by one dramatic example.

### I. Quote-fragment grouping

Pass 1 may split one utterance into multiple `SPOKEN` entries. Group adjacent
fragments that share quotation continuity, punctuation, or no intervening
speaker cue, then attribute the group once.

Measure:

- how many gold lines belong to split utterances;
- whether fragment-level predictions disagree within the same utterance;
- accuracy when the group receives one shared speaker.

This could improve attribution without changing the model.

### J. Scene-boundary quality

Many proposed state and retrieval mechanisms depend on correct scene
boundaries. Audit boundary precision before relying on them:

- false splits lose useful history;
- missed boundaries carry stale characters and speaker state forward.

Compare fixed chapter windows, heuristic scene boundaries, and model-generated
boundaries on a small hand-checked set. Use the simplest boundary method that
supports the downstream gain.

### K. Batch-position and neighbor contamination

Test whether attribution changes with:

- target at the start, middle, or end of a batch;
- unrelated rows inserted before the target;
- targets ordered chronologically versus grouped arbitrarily;
- neighboring answerable rows included versus context-only rows.

Temperature zero makes this a cheap deterministic robustness check. If
irrelevant batch composition changes many answers, batch coupling is a hidden
source of instability and scene-coherent batches may help.

### L. Context perturbation as a routing signal

Earlier confidence methods failed as acceptance criteria. A lower bar is
routing. Compare a cheap baseline under two meaningful perturbations:

- w4 versus retrieved context;
- full roster versus removal of clearly inactive names.

Send only disagreements to thinking. Evaluate whether disagreement enriches
`RESCUE` rows more than random routing at the same coverage.

### M. Cheap-model cascade

Run two inexpensive, non-thinking models first:

- if they agree on a canonical name and the line has explicit support, accept;
- if they disagree, route to Qwen thinking;
- if thinking remains unsupported or off-roster, route to human review.

The existing ensemble-confidence result does not settle this design because it
tested agreement as an acceptance solution by itself. Here agreement is one
feature in a cost-aware cascade. Report coverage, accepted accuracy, thinking
rate, human-review rate, and total time.

### N. Baseline-versus-thinking error anatomy

Audit Grimgar's 72 thinking repairs and 39 regressions separately. For each,
label:

- evidence distance;
- dialogue length;
- explicit tag;
- vocative;
- prior-speaker dependency;
- number of plausible candidates;
- baseline and thinking reasoning behavior.

The goal is not another accuracy number. It is to discover whether thinking's
benefit is concentrated enough to route cheaply and whether its regressions
share a preventable pattern.

### O. Reasoning-output faithfulness

On a sample, compare the hidden/returned reasoning with the evidence actually
supporting the gold label. Categorize:

- correct conclusion, sound evidence;
- correct conclusion, invented evidence;
- wrong conclusion, locally coherent evidence;
- wrong conclusion caused by addressee inversion;
- reasoning identifies the right speaker but final JSON names another.

If reasoning frequently reaches the right evidence but emits the wrong final
name, output constraints or a verifier may help. If the evidence itself is
wrong, formatting work will not.

### P. Reasoning distillation

Use successful thinking traces only as training or prompt-development data,
not as trusted labels. Extract recurring compact evidence patterns and test
whether a non-thinking model can use those patterns at lower cost.

Avoid training on the evaluation fixture. Develop on one book/chapter and test
on another. The key question is whether the +8.2 gain can be retained without
the 5–7× inference cost.

### Q. Few-shot examples selected by error type

Generic few-shot prompting may waste context. Test a small fixed set containing:

- an addressee inversion;
- an alternating dialogue;
- a split utterance;
- an absent-name inference;
- an explicit speech tag.

Use examples from a different book than the test fixture. Compare against no
examples and randomly selected examples. If curated examples help only their
matching error class, adaptive example retrieval may be worthwhile.

### R. Character-frequency priors

Measure whether models overpredict protagonists or frequent speakers.
Report macro accuracy by character alongside micro accuracy. Then test:

- no prior;
- scene-frequency prior;
- book-frequency prior;
- debiased candidate scoring.

A prior may raise aggregate accuracy while making rare characters worse, so it
must be judged per character and not by one headline percentage.

### S. Gold-disagreement adjudication

Select rows where:

- most strong models agree against gold;
- thinking repairs one model but contradicts all others;
- exact and phonetic scoring disagree;
- predictions flip across context widths.

Blindly re-adjudicate those rows with wider source context. This is the most
efficient way to find label errors or genuine ambiguity because it targets
high-information disagreements rather than random rows.

### T. Product-level human correction study

If unattended accuracy remains below target, measure the real alternative:

- time to correct raw baseline output;
- time to correct selective-thinking output;
- time to assign speakers from scratch;
- error rate after correction.

The best technical score may not produce the fastest editing workflow. A model
that groups uncertain lines and exposes evidence could beat a more accurate but
opaque output in actual audiobook-production time.

## Additional prioritization

Highest information per GPU hour:

1. baseline-versus-thinking error anatomy;
2. offline selective-routing evaluation;
3. oracle-history versus predicted-history;
4. oracle-evidence sufficiency;
5. batch-position robustness;
6. fixture representativeness and gold-disagreement adjudication.

Best architectural bets if those diagnostics support them:

1. retrieved evidence plus compact state;
2. scene-level joint attribution;
3. candidate likelihood scoring with calibrated margin;
4. global sequence decoding;
5. targeted reasoning cascade.

Defer until evidence supports them:

- LoRA/fine-tuning;
- another broad model sweep;
- a complex learned scene-cast generator;
- global fuzzy-name normalization;
- a second model used only to convert text into JSON.

## Review of addendum §§13.7–13.8

### The input-limit diagnosis remains a hypothesis

Model clustering and failed prompt interventions are consistent with an input
or representation limit, but they do not establish one. Other explanations
remain:

- the fixture lacks power to separate clustered models;
- the component harness compresses model differences;
- the task has irreducible ambiguity on some rows;
- the prompts expose evidence poorly even when it is technically present;
- the tested models share training or decoding weaknesses.

The context-width sweep is a good discriminator. Its conclusion must remain
paired and production-scoped. The fact that 62.1% of errors lack a nearby name
does not show that the needed evidence exists farther away.

### The multi-model oracle union is not “mostly model variance”

The new analysis establishes:

- 27/400 Grimgar rows and 16/139 Mushoku rows were wrong in every included
  oracle-arm run;
- many other rows were answered correctly by at least one run.

That is a useful **union upper bound** and identifies a consensus-hard
adjudication set. It does not show that the remaining failures are mostly
random model variance or individually recoverable by the product:

- several “runs” may share models, prompts, weights, or environments and are
  not independent judges;
- knowing after the fact that one model was correct does not provide a routing
  rule for choosing it;
- disagreement may reflect systematic model-specific biases rather than noise.

Use “consensus-hard core” and “multi-run oracle coverage,” not “the ceiling is
mostly variance.” The next useful offline calculation is an ensemble-selection
bound:

1. majority vote;
2. agreement-only acceptance;
3. leave-one-book-out router;
4. unattainable oracle-best-model upper bound.

The distance between the deployable rules and the oracle union measures how
much of that apparent recoverability can actually be selected.

### Fixture length skew does not yet establish optimistic accuracy bias

The scored fixtures have longer median lines than the full spoken-line
population. That proves a length-distribution mismatch.

It does **not yet** prove that reported accuracy is too high. “Short lines are
harder” must be measured in these artifacts rather than assumed. Required
analysis:

| length bin | population share | fixture share | baseline accuracy | thinking accuracy |
|---|---:|---:|---:|---:|
| 1–15 chars | | | | |
| 16–30 | | | | |
| 31–60 | | | | |
| 61+ | | | | |

Then reweight fixture accuracy to the full spoken-line length distribution,
with uncertainty. Also stratify by dialogue density and explicit evidence,
because length may be a proxy for those variables. Eight excluded repeated
Mushoku rows are too few to determine the bias direction alone.

Until that analysis lands, say:

> The fixture is not representative on line length; the direction and size of
> resulting accuracy bias are unknown.

### The routing result rejects two features, not selective thinking

Line length and nearby speech-tag presence do not separate `RESCUE` from
`HARM`; speech tags even point in the wrong direction. This correctly retires
those two features as standalone routers.

It does not reject:

- model disagreement;
- context perturbation;
- prediction-margin features;
- roster perturbation;
- sequence inconsistency;
- a learned cross-book combination of weak features.

Any learned router must split by scene/book and report benefit over random
routing at identical coverage. Because there are only 72 Grimgar rescues and
39 harms, keep the feature set small and use nested or leave-one-scene-out
validation to avoid fitting noise.

### Candidate recall and oracle gain need separate wording

The closed-oracle arm does more than place the answer in the set. It also
changes distractor identity, candidate count/composition, and the prompt. The
rough +17-point gain cannot be assigned entirely to recall.

The proposed high-recall scene cast is still worth testing, but it must report:

- availability gain;
- conditional selection with the answer present;
- errors introduced by changed distractors;
- total accuracy and mean set size.

An oracle candidate set is an upper-bound diagnostic, not an estimate of what a
97%-recall generator will achieve.

### Thinking has two blockers

Cost is not the only blocker. The production effect is significant on Grimgar
and unresolved on Mushoku. Therefore the blockers are:

1. 4.8–7.3× pass-2 cost;
2. incomplete cross-book confirmation.

Selective routing addresses the first. More representative labels or another
large production-path fixture address the second.
