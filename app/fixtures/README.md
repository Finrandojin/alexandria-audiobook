# Attribution gold

Four books, 793 labelled lines, adopted as gold on 2026-07-28.

| file | rows | scored | second-judge agreement |
|---|---|---|---|
| `attribution_gold_grimgar03.json` | 396 | 385 | 96.2% |
| `attribution_gold_index18.json` | 99 | 92 | 96.0% |
| `attribution_gold_mushoku16.json` | 136 | 133 | 97.8% |
| `attribution_gold_owarimonogatari3.json` | 162 | 162 | 96.3% |

**96.5% overall** [94.9-97.6]. Every row was read by two independent frontier
judges at 12 segments of context, every disagreement was adjudicated, and four
conventions were ruled and are recorded in each file under `conventions`.

## What the number is and is not

It bounds the fixture's accuracy against a second model, not against truth. No
human read the passages. Where both judges share a blind spot - unmarked
turn-taking in rapid exchanges is the demonstrated one - agreement measures
correlation. At the 50-79% models currently score, that residual sits far below
the effects being measured.

## Reading a fixture

- `entries[].expected_speaker` - the label. May be `UNNAMED` (the text
  identifies the speaker but never names them) or `UNKNOWN` (the text does not
  determine who spoke). Both are dropped from scoring; only `UNKNOWN` bounds
  what a model could achieve. Rows the judges called `NOT_DIALOGUE` are removed
  entirely and their rate reported as the segmenter's error rate.
- `aliases` - groups of names for one character. **Load these.** Scoring
  without them silently marks correct answers wrong; that happened five times
  in one day building this set, once costing 10 points of apparent agreement.
- `roster_additions` - characters `build_roster` failed to find. Lines they
  speak are unwinnable for a roster-constrained model, so this is a roster bug
  rather than a model failure.
- `conventions` - rulings on cases the text underdetermines. Apply them to new
  books rather than re-deciding.

Compare speakers with `experiments/scoring.same_speaker`, which handles the
alias groups and punctuation. Do not write another comparison.

## Older files

- `attribution_gold_random.json` - mushoku16's original single-judge gold, what
  every result in the ledger was scored against. Kept unchanged for
  reproducibility; three labels are known wrong. Use the new file for new work.
- `*_provisional.json` - symlinks to the adopted files, so the 25 harnesses
  that still hardcode the old names keep working. Remove once those are updated.
