# Attribution: what survived, 2026-07-29/30

Two days of experiments across four books and two model scales. The short
version: **every intervention that changes the prompt or the candidate list
has now failed to replicate. The only thing that survives is spending more
capability on the hard rows.**

## The corpus is now four books of adopted gold

793 rows, two independent frontier judges at 12 segments, every disagreement
adjudicated, four conventions ruled. Second-judge agreement **96.5%**
[94.9-97.6]. Details in the fixtures' own `status` block and
`app/fixtures/README.md`.

Two facts from building it that bound everything else:

- **UNKNOWN is 0-1.1% per book.** Two genuinely undeterminable lines in 839.
  Attribution is essentially never ambiguous, so the gap between models
  (45-83%) and gold is real headroom, not label noise.
- **Segmenter error runs 1.0% to 17.5% by book** and neither label-free proxy
  predicts it. On index18 one in six "spoken" lines is not dialogue and is
  voiced as character speech. This is the largest correctness gap in the
  pipeline and it is upstream of all attribution work.

## What survives

**The disagreement cascade.** Run the cheap model twice, send the rows where
it disagrees with itself to a 70B.

| book | cheap w1 | cascade | delta | p |
|---|---|---|---|---|
| grimgar03 | 55.8% | 77.8% | +22.0 | 1.4e-23 |
| mushoku16 | 47.5% | 64.0% | +16.5 | 1.2e-4 |
| index18 | 62.6% | 73.7% | +11.1 | 0.007 |
| owarimonogatari3 | 42.0% | 56.2% | +14.2 | 6.1e-4 |

Four books, two of which the design never saw. Cost is ~70% of the book's
batches, not the 40-60% of rows routed - the window around each routed line
goes too.

**It needs a 70B specifically.** The cost curve is the important negative:

| expensive arm | result |
|---|---|
| llama-3.3-70b | +11.1 to +22.0 |
| qwen3-32b (mushoku16) | **-2.2**, p=0.71 |
| gemma-3-27b (index18, owari) | **+3.0 / +3.7**, both ns |

There is no cheap-hardware version of this design. Escalating to "something
bigger" does not work; escalating to a 70B does.

## What did not survive

Every one of these was measured on at least two books, most on four:

- **w4 context** - +10.5 on the 14B, **-2.5 on the 70B**. A crutch. Retracted.
- **tag-priority** - +6.5 once (p=0.001), then **-1.8 on a repeat of the same
  book and model**, null on two new books, and **+0.3 on the 70B**. The
  original was an anecdote with a p-value. Do not ship.
- **candidate-list interventions, all of them**: closed-6 (-9.4), a gold
  roster (+2.2, ns), scene-cast narrowing (null on four books). A perfect
  26-name roster buys 2.2 points and padding it to 46 costs 1.1. **The list is
  not the constraint.**
- **joint scene decoding** - loses to per-line on both models. Order *within*
  a joint prompt is worth +15.4 on the 70B, but batching costs more than
  ordering returns.
- **voting** (+2.2, below its 3x cost), **committed history** (zero rows moved
  with the true previous speaker supplied), **reasoning-consistency routing**
  (sign reverses between books), **adaptive width** (LOO router captured 0.0).

## Routing is per book, not per section

Four interventions split hard by book - w4 (+10.5/-5.0), batch-size peak
(10/25/50/50), tag-priority (+6.5/-5.8). But two candidate section-level
features were tested and **neither flips the sign inside a book**: local tag
density is uniformly positive in one book and uniformly negative in another at
every band, and local first-person density has the books behaving *oppositely*
in third-person windows.

The one section-level rule supported is negative: tag-priority does nothing in
first-person passages either way.

Open hypothesis, untested: mushoku16's -19.4 cell is third-person windows
inside a heavily first-person book, most likely the letter and diary passages.
If so the routing feature is **passage type**, not tag density or narrative
person. Needs the epistolary sections marked.

## Why the model gets rows wrong

`roster_quality` ruled out recall, so the error taxonomy across 30,727 wrong
answers asked which wrong name is chosen:

| | share |
|---|---|
| someone else entirely | **44.7%** |
| named in adjacent narration (addressee or actor) | 20.4% |
| the book's most frequent speaker | 16.0% |
| abstained | 14.5% |
| the previous speaker | 4.4% |

**The taxonomy fails on the plurality** and should not be read as an
explanation. Two things it does establish: previous-speaker confusion is only
4.4%, which independently explains why committed-history was null; and
owarimonogatari3 collapses to the frequency prior at **36.9%** against 1.2-17.3%
elsewhere.

## Instrument failures worth remembering

- **Three artifacts validated while measuring nothing** - a missing prompt file
  (0.0%), a renumbered id scheme (0.0%), and a name parsed as `**33**: RUDI`
  (9.4%, which *looks* like a result). Guards added for unanswered rows and for
  answers that are not roster names.
- **Alias gaps five times in one day**, each silently turning correct answers
  into errors, including 162 rows across the ledger differing from gold by a
  full stop. `experiments/scoring.py` is now the single comparison.
- **`finalise_fixture` twice destroyed decided metadata**, once erasing
  convention rulings minutes after they were set.
- **A syntax error shipped into a running queue** cost six cost-curve runs, and
  3-hour wait loops then orphaned three more stages for eight idle GPU hours.
  Waiters are now 24 hours.

## What I would do next

1. **Segmentation.** Largest correctness gap, upstream of everything, and the
   judges' labels make it self-scoring. Rule-based filtering failed at a 6%
   false-positive rate; a classifier was the remaining approach and is now
   measured - see the correction below.
2. **More books.** Every routing claim is a line through four points. The gold
   pipeline is solid; the cost is judging time, not GPU.
3. **The realizable router** (offline) - whether per-passage adaptation is real
   or per-book is the end of it.

## Distillation: built, not yet run (2026-07-30)

The cost curve made the cascade a **70B-class commitment** — a 32B scored -2.2
on routed rows and a 27B +3.0/+3.7, against the 70B's +11.1 to +22.0. So
"escalate to something bigger" is false and "escalate to a 70B" is true. The
attempt to move that capability into the 14B rather than rent it per book now
has all three pieces written and committed:

- **`experiments/distill_collect.py`** — done, and it produced the data.
  **1,091 rows** from two books with no gold, grimgar06 (498) and mushoku18
  (593), each a line where two cheap passes disagreed and the 70B answered.
  The teacher supplies an answer *neither* cheap pass produced on 26% of
  grimgar06's rows and 45% of mushoku18's, so there is something to learn
  rather than a re-weighting of existing guesses.
- **`experiments/distill_train.py`** — written, dry-run verified (1,091
  examples, 61 distinct teacher labels, prompts median 742 / max 3,493 chars).
  Holds nothing back: the four gold books are the evaluation.
- **`experiments/distill_eval.py`** — written and tested, never run.

**Nothing has been trained.** There is no adapter and therefore no result. The
training needs a GPU that fits a 14B LoRA in bf16 (~30-40 GB), which the local
16 GB card does not, so it waits on an instance restored from the snapshot.

**The known risk, recorded before running rather than discovered after:**
training is one entry per example, because a per-row teacher label cannot
supervise a 25-entry batch response — but inference batches 25. That mismatch
is the most likely reason this fails. `distill_eval` prints per-arm unanswered
and distinct-speaker counts specifically to separate "learned nothing" from
"can no longer follow the batch format", which are different failures.

Both arms run through **one loaded model**, separated only by peft's
`disable_adapter()`, so the adapter is provably the only difference; and both
go through the production `attribute_batch`, so batching, JSON repair, the text
freeze and the retry policy stay inside the comparison. The shim standing in
for the OpenAI client is covered by `app/test_distill_eval_shim.py` (4 tests,
no GPU) — if it drifted, every row would become a failed batch and the adapter
would take the blame.

**Read the result against the cascade's gains on these same books, not against
zero.** A tuned 14B that beats base but falls well short of +11.1 has not
replaced the 70B.

## Cloud state

Instance 0 (A6000) **deleted 2026-07-30**, billing stopped, after snapshot
`alexandria-attribution-2026-07-31` (id `MRqS2nKqYE0DEGyDu4gM`, 300 GB) reached
READY. The snapshot holds the CUDA llama.cpp build and the 70B weights — about
four hours to reconstruct from scratch. Restore from it rather than rebuilding.

## Correction: "839 NOT_DIALOGUE labels" was wrong (2026-07-30)

Earlier text in this brief and in `segmentation_filter`'s docstring described
"the 839 NOT_DIALOGUE labels". **839 is the number of judged rows.** Only **46**
of them are NOT_DIALOGUE; 793 are real speech. The positives are also
concentrated — index18 has 21 and owarimonogatari3 18, while grimgar06 and
mushoku18 have none at all.

That correction changes the segmentation plan. `experiments/segmentation_classifier.py`
trains a logistic model leave-one-BOOK-out, with the operating threshold fixed
at a 1% false-positive rate on the *training* books:

    pooled recall     10/46 = 21.7%  [10.9-36.4]
    pooled false pos  14/1033 = 1.36%  [0.74-2.26]

Against the rule baseline (`cut`: 39.1% recall at 3.66% false positives) this is
**not a demonstrated improvement**. The recall interval spans 25 points, so the
labels cannot resolve whether a classifier beats the rules either way.

**The binding constraint is the label count, not the model.** More
NOT_DIALOGUE labels is the prerequisite for any further segmentation work; a
better classifier on 46 positives is not.

## Two corrections from the baseline work (2026-07-31)

### Book scores were never comparable, and owarimonogatari3 is below free

`experiments/trivial_baselines.py` computes what each book scores with no model
at all, on the same rows the harnesses score:

    book               floor  (which)             best arm   arms below floor
    grimgar03          35.3%  previous-speaker      86.8%      0/148
    index18            39.1%  previous-speaker      82.6%      0/63
    mushoku16          37.6%  majority              70.7%      3/87
    owarimonogatari3   50.0%  previous-speaker      69.8%     50/63

**Fifty of owarimonogatari3's 63 measured arms score below a baseline that just
repeats the previous line's speaker.** The book has been called hard; it is
worse than that — most interventions measured on it are worse than free. The
floors differ by more than 20 points, so two books with equal accuracy have
never meant the same thing, and any claim resting on owari needs re-reading
against 50.0%.

### `committed_history` was reported null. That was a pooling artifact.

                    none    oracle   predicted   floor
    grimgar03       63.5%   63.5%     62.3%      35.3%
    index18         63.6%   60.6%     63.6%      39.1%
    mushoku16       50.7%   54.4%     47.8%      37.6%
    owarimonogatari3 50.0%  59.3%     46.9%      50.0%

The TRUE previous speaker is worth **+9.3 points on owarimonogatari3** and +3.7
on mushoku16, while the model's OWN previous answer costs 3.1 and 2.9. That is
exactly the "oracle helps, predicted does not — work on the state source"
reading `committed_history` fixed in advance, and averaging four books hid it.

Note also that owari's `none` arm scores 50.0%, identical to its
previous-speaker floor.

**What this changes.** Sequential history is not retired. The representation
works where turn-taking carries the evidence; what fails is the state source,
because feeding back predictions that are wrong about half the time compounds
the error. The open question is whether a confidence-gated history — supply the
previous speaker only when it is likely right — beats supplying it always or
never. That is a real experiment, distinct from the one already run.

### Name-binding is worth ~10 oracle points, not the whole gap

`experiments/cluster_vs_name.py` scores each arm's partition of lines by
speaker, names discarded: mean ARI **0.416**, mean gain from an oracle
relabelling **+9.9 points**, with predicted cluster counts tracking gold
(21/22, 20/20) so the gain is structure rather than collapse. The model
partially tracks who is speaking. Fixing name-binding alone is worth less than
the 70B cascade's +11.1 to +22.0, so it is not the missing piece.

## Distillation works: +11.7 points, p=3.6e-11 (2026-07-31)

A LoRA trained on 1,091 rows the 70B answered on two books with NO gold
(grimgar06, mushoku18), evaluated on the four gold books it never saw. Both
arms ran through one loaded model, separated only by peft's
`disable_adapter()`, and through the production `attribute_batch`.

    book               base    tuned    delta
    grimgar03          68.8%   78.4%    +9.6
    index18            71.7%   75.0%    +3.3
    mushoku16          50.4%   62.4%   +12.0
    owarimonogatari3   40.1%   61.1%   +21.0

    pooled   base 463/772 = 60.0%   tuned 553/772 = 71.6%
    paired   +11.7 points   +139/-49 of 772   p=3.588e-11

Every book improves, and the effect is strongest exactly where the base model
was worst. **owarimonogatari3 moves from 40.1% - BELOW its 50.0%
previous-speaker floor - to 61.1%, above it.** Its unanswered rows fall from 15
to 1.

THE PREDICTED FAILURE DID NOT HAPPEN. Training was one entry per example and
inference sends 25; the concern was batch-format collapse. Instead the tuned
arm answers MORE rows (blank 4.0% vs 6.3%) with no name collapse (top
prediction 15.5% of rows). It learned attribution, not a shortcut.

DENOMINATOR WARNING. The cascade's headline +11.1 to +22.0 was measured on
ROUTED ROWS - the subset where two cheap passes disagreed. The +11.7 here is
over ALL scored rows. These are not the same denominator and the two numbers
must not be quoted as if they were. A like-for-like comparison needs the
cascade's whole-book effect, which has not been computed.

WHAT IS NOT YET KNOWN. Whether the 70B was necessary. The adapter also learned
the task's prompt format, and `--label_field cheap_a` trains the identical
adapter on the student's own answers to separate the two. Until that runs, the
claim is "distillation on these labels works", not "the 70B's knowledge
transferred".

Cost: ~1 hour training, ~14 hours evaluation on a rented A6000.

## Correction: the cascade denominator warning was wrong (2026-08-01)

The previous section warned that the cascade's +11.1 to +22.0 was measured on
routed rows and could not be compared to distillation's +11.7 over all rows.
**That was wrong.** Every cascade artifact scores `cheap-w1` and `cascade` over
the SAME full row set, so the cascade's whole-book effect was available the
whole time. `experiments/cascade_vs_distill.py` computes both.

The real incomparability is different, and it runs the other way:

    book               cascade end   adapter end   diff
    grimgar03             77.8%        78.4%       +0.7
    index18               73.7%        75.0%       +1.3
    mushoku16             64.0%        62.4%       -1.6
    owarimonogatari3      54.9%        61.1%       +6.2

**The cascade's deltas are inflated by a weak baseline.** Its cheap arm is
`cheap-w1` — context width ONE, a deliberately narrowed configuration — at Q4
through llama.cpp, scoring 55.8% on grimgar03. The adapter's base is the same
model in bf16 with production neighbour contexts at 68.8%. A lower starting
point produces a bigger delta, so +22.0 and +11.7 were never measuring the same
thing, and the END POINT is the fairer comparison.

On end points the distilled 14B **matches or beats the 70B cascade on three of
four books**, and loses one by 1.6. It does so at 14B inference cost with no
70B in the loop at run time, against a cascade that rents a 70B on every book
forever.

Still not established: whether the 70B was needed to CREATE the adapter. The
`--label_field cheap_a` ablation is what answers that.

## Why batch size works: conversation, not context (2026-08-01)

Batch size was the largest lever measured here - 60.5% at b1 to 79.2% at b25
for the 70B on grimgar03 - and the mechanism was untested.
`experiments/batch_contiguity.py` sends the same 25 entries with the same
per-entry neighbour contexts in both arms, changing only the COMPANIONS: the
line's own conversation, or 24 strangers drawn from >200 segments away.

    contiguous  287/385 = 74.5%  [69.9-78.8]   mean batch 1201 chars   3 exhausted
    scattered   223/385 = 57.9%  [52.8-62.9]   mean batch 1196 chars  33 exhausted

    scattered - contiguous  -16.6 points  +42/-106 of 385  p=1.446e-07

Prompt sizes are within 0.4% of each other, so this is not amortised context.
**The gain is conversational structure**, and scattered batches also break the
output format eleven times more often.

REPLICATED on a second book, on different hardware and a different inference
stack (local ROCm llama.cpp vs the cloud CUDA run):

    owarimonogatari3   contiguous 53.7%   scattered 35.2%
                       -18.5 points  +21/-51 of 162  p=0.000535

The effect is LARGER there than on grimgar03, and owarimonogatari3 is the book
with by far the most same-speaker structure (gold continuation 56.9% against
31.7%). That is the direction the "model exploits conversational runs"
explanation predicts, which is the first thing in this investigation to
survive a prediction rather than only fit after the fact.

WHAT THIS OPENS. Production cuts batches every 25 segments regardless of where
conversations begin and end, so some fraction of batches straddle a boundary
and get the scattered condition by accident. Aligning batch boundaries to
scene or turn runs is a new lever, and this is the first evidence that it
should be worth anything.

A first attempt scored 821 rows in the scattered arm against 385 in
contiguous - companions that happened to be gold lines were scored too, and
repeatedly. `ExperimentRecord.validate` refused to write the artifact, naming
238 duplicate identities, so no number entered the ledger. Each scattered
batch now scores exactly its target and companions are context only.

## The 70B was necessary: self-training buys nothing (2026-08-01)

The distillation gain could have been the adapter learning the task's output
format rather than anything the teacher knew. `--label_field cheap_a` trains an
identical adapter on the STUDENT's own b25 answers to the same routed rows -
same prompts, same count, same hyperparameters, only whose answer is learned.

    book               base    own-labels    70B-labels
    grimgar03         68.8%   68.3%  (-0.5)  78.4%  (+9.6)
    owarimonogatari3  40.1%   45.7%  (+5.6)  61.1% (+21.0)
    pooled            60.3%   61.6%  (+1.3)  71.6% (+11.7)
                              p=0.576        p=3.588e-11

**Self-training is indistinguishable from zero.** The teacher's labels are what
transferred.

FORMAT LEARNING HAPPENED WITHOUT ACCURACY. The self-trained adapter cut
grimgar03's unanswered rows from 25 to 12 - it plainly learned the output shape
- while accuracy moved -0.5. So the +11.7 is not an artifact of producing
better-formed batches, which was the main alternative explanation.

DETERMINISM CONTROL, obtained by re-running `base` rather than reusing the
earlier numbers: grimgar03 base reproduced EXACTLY - 265/385, unanswered 25,
distinct names 19 - across two runs fourteen hours and two adapter loads apart.
Every cross-run comparison in this brief rests on that.

THE SHAPE OF THE RESULT. Rent a 70B once for ~1,000 labels on books with no
gold, then serve a 14B forever. No 70B at run time, and end-to-end it matches
or beats the live cascade on three of four books.

## The gated-history test was designed wrong and is UNTESTED (2026-08-01)

The `gated` arm supplies the previous speaker only where a confirming pass
agrees. On owarimonogatari3:

    none       81/162 = 50.0%
    oracle     96/162 = 59.3%   (+9.3, reproduces the earlier run exactly)
    predicted  78/162 = 48.1%   (-1.9)
    gated      79/162 = 48.8%   (-1.2)

    gate supplied history on 161/162 rows = 99.4%

**The gate is not a gate.** At 99.4% coverage it is `predicted` under another
name, and the two scores match. The hypothesis is untested, not refuted.

The cause was a shortcut. The confidence signal was specified as agreement
between two INDEPENDENT passes at different batch sizes - the cascade's routing
signal, which disagrees on roughly 40% of rows. It was then changed to reuse
the `none` arm as the confirming pass because that costs no extra inference.
But `none` and the gated run are the same model at temperature 0 on nearly
identical prompts, so they agree almost always: the change removed exactly the
independence that made the signal informative.

Testing it properly needs the separate b25/b50 sweep, about one more GPU hour.

The oracle arm reproducing +9.3 exactly is the third independent determinism
check of the day, after grimgar03's base arm reproducing 265/385 across
fourteen hours and two adapter loads.

### Second attempt, same failure — and the cause is structural

Re-run with a genuinely independent confirming pass (no history, context width
12 against the arms' 4):

    none 50.0%   oracle 59.3% (+9.3)   predicted 48.1%   gated 48.8%
    gate supplied history on 161/162 rows = 99.4%

Identical coverage to the first attempt. The confirming pass was not the
problem.

**`prior_speakers` cannot express "supply nothing".** It walks backwards
collecting three speakers and CONTINUES past any entry it rejects, so when the
gate blocks a line the loop simply backfills from further back. The arm always
ends up with three names; the gate only changes WHICH ones. That is not the
intervention, and no confirming signal fixes it.

Testing the real thing needs `prior_speakers` to stop at the first blocked
entry, or a variant supplying only the immediately-previous line. That is a
third design on the same experiment, and it is not being attempted here:
two wrong designs in a row is a signal to stop and think rather than iterate
against a rented GPU.

**Left as an open, well-specified experiment**, with the +9.3 oracle showing
there is something to win and the mechanism now understood.

## Batch boundaries do NOT matter, which closes the lever (2026-08-01)

`batch_contiguity` showed contiguity is worth 16.6 points, so the obvious next
question was whether production's arbitrary cut every 25 segments costs
anything. `experiments/batch_alignment.py` packs whole runs of consecutive
spoken segments instead, never splitting one:

    fixed     275/385 = 71.4%  [66.6-75.9]   mean 24.8 entries
    aligned   273/385 = 70.9%  [66.1-75.4]   mean 24.4 entries
    aligned - fixed  -0.5 points  +52/-54 of 385  p=0.9227

A clean null, and the batch sizes match (24.8 vs 24.4) so it is not confounded
by size.

**Why both results are consistent.** grimgar03 has 940 spoken runs across 1,463
sendable segments — a mean run length of about 1.6. Runs that short are almost
never split by a 25-segment cut, so a fixed window already contains many
complete conversations. Contiguity is what the model exploits; there is simply
no misalignment left to fix.

This is the pre-registered "aligned ~ fixed" reading: the lever
`batch_contiguity` opened is closed, and the 16.6-point contiguity effect is
already being captured by production.

## The adapter and the cascade are COMPLEMENTARY (2026-08-01)

They reach the same place; they do not fix the same rows.
`experiments/adapter_vs_cascade_overlap.py`, offline over shared gold ids:

    book               both  adapter-only  cascade-only  neither   union
    grimgar03         69.6%      8.8%         10.9%       10.6%    89.4%
    index18           66.3%      8.7%         13.0%       12.0%    88.0%
    mushoku16         49.6%     12.8%         15.8%       21.8%    78.2%
    owarimonogatari3  46.9%     14.2%          8.0%       30.9%    69.1%

    pooled 772 rows: adapter alone 71.6%, cascade alone 72.4%,
    ORACLE union 83.0% (+10.6 over the better one)

**The cascade gets 11.4% of rows right that the adapter misses.** It is not
redundant, and stacking is worth testing: run the cascade with the TUNED 14B as
its cheap arm.

A DISTINCTION THAT MATTERS. `realizable_router` retired **book-level** method
selection (-0.96 against a fixed choice, oracle ceiling +2.42). This is a
**row-level** question, and the cascade already solves row-level routing with
two-pass disagreement - that is its design. So the +10.6 here is far more
reachable than the router's ceiling, and the two findings do not conflict.

The union remains an oracle until a stacked run measures what the disagreement
signal actually collects.

## Running the adapter on the local 16GB card (2026-08-01)

Merging the LoRA into the base and quantizing was not possible locally: the box
has **30GB RAM (24 available) and 14GB free disk**, against ~28GB RAM for a 14B
bf16 merge plus ~29GB base, ~29GB merged and ~9GB GGUF.

llama.cpp applies a LoRA at RUNTIME, so no merge is needed. The adapter
converts to a **128MB** GGUF and rides on top of the Q4_K_M base already on
disk.

Build it:

    cd /home/fakemitch/pinokio/api/alexandria-audiobook2.git
    ./app/env/bin/python \
      /home/fakemitch/.cache/yay/llama.cpp-hip/src/llama.cpp-b10121/convert_lora_to_gguf.py \
      ab_test_runtime/distill/adapter \
      --base-model-id Qwen/Qwen3-14B --outtype f16 \
      --outfile ab_test_runtime/distill/gguf/alexandria-attrib-lora-f16.gguf

Serve it (ROCm backend needs LM Studio's vendored libs on the path):

    B=/home/fakemitch/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-amd-rocm-avx2-2.21.0
    V=/home/fakemitch/.lmstudio/extensions/backends/vendor/linux-llama-rocm-vendor-v3
    G=~/.lmstudio/models/lmstudio-community/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_M.gguf
    LD_LIBRARY_PATH=$V:$B $B/llama-server -m $G \
      --lora ab_test_runtime/distill/gguf/alexandria-attrib-lora-f16.gguf \
      --port 8090 --host 127.0.0.1 -ngl 99 -c 32768 --parallel 1

**NOT YET VERIFIED FOR ACCURACY.** The +11.7 was measured on the bf16 model
through transformers. This path is Q4_K_M base plus an f16 LoRA through
llama.cpp, which is a different numeric stack, and quantisation could eat some
or all of the gain. The adapter loads and serves; whether it still scores +11.7
here has to be measured before the number is claimed for this configuration.

Both the adapter (257MB) and this GGUF (128MB) are gitignored and rebuildable.
