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
