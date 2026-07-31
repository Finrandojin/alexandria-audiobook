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
   839 NOT_DIALOGUE labels make it self-scoring. Rule-based filtering failed at
   a 6% false-positive rate; a classifier is the remaining approach.
2. **More books.** Every routing claim is a line through four points. The gold
   pipeline is solid; the cost is judging time, not GPU.
3. **The realizable router** (offline) - whether per-passage adaptation is real
   or per-book is the end of it.
