# Attribution scoring policy

Frozen before any model comparison, per §28 of `BRIEF_2026-07-26_ATTRIBUTION.md`.
Deciding these afterwards would let the rules be chosen to suit a result.

Status: **draft, awaiting owner approval.** Each item states a recommendation
and the reasoning; overrule any of them, but record the choice here before
scoring.

---

## 1. Genuinely anonymous speech

**Recommendation: excluded from the accuracy denominator, reported separately.**

A line spoken by an unnamed character - a battle shout, a crowd member, a
one-eyed man never named - has no correct answer in a name-based key. Scoring it
as `AMBIGUOUS` would reward abstention; scoring it as wrong would penalise a
model for the corpus.

Reported as a separate count so the size of the unattributable fraction stays
visible. Measured on the existing set it is around 3-5%.

## 2. `NARRATOR` segmentation errors

**Recommendation: excluded from attribution accuracy, reported as a pass-1
defect.**

When a judge marks a line `NARRATOR` it means pass 1 split narration, a sign or
a heading into a `SPOKEN` entry. One gold line turned out to be a poster on a
wall. That is a segmentation failure, and counting it against attribution
measures the wrong pass and hides a real defect behind an accuracy number.

Judge 1 found 3 in 400 grimgar03 rows - a ~0.75% pass-1 error rate worth
tracking in its own right.

## 3. Alias and honorific normalisation

**Recommendation: aliases declared per fixture, matched case-insensitively,
never inferred.**

A character named two ways is one character: `RUDI` and `RUDEUS` cost 14 of 147
lines before the scorer understood this. Aliases live in the fixture where a
reader can check them against the text, not in code.

Never `str.title()`. It capitalises after every non-letter, turning `BRI-CHAN`
into `Bri-Chan`, and has caused three separate defects in this repository in one
day. Comparison is case-insensitive throughout.

An invented name is never an alias. `FUTURE_ME` is a wrong answer, not a form of
`RUDEUS`, and a test asserts no shipped fixture declares it.

## 4. Accepting a `book`-supported answer

**Recommendation: two-judge agreement is sufficient; disagreement goes to
adjudication.**

An answer whose name appears in the book but not in the shown window is
inference from wider context, which is legitimate reading. Requiring explicit
human adjudication for every such row would cost more than it protects, given
two independent judges agreed at 94-97% on earlier rounds.

Where the judges disagree, a human decides - which is already the protocol.
`support_summary()` reports the window/book split so the size of this class is
visible per fixture.

## 5. Rows still unsupported at the expansion cap

**Recommendation: excluded, and counted.**

The window expands to 40 entries before and 20 after. A speaker still not
identifiable within roughly 60 entries of their line is not attributable from
local context by any method under test, so including such rows measures the
corpus rather than the model.

Counting them matters: this fraction is the floor on what any confidence-routing
scheme must send to a human, independent of model quality.

## 6. Comparison arm, significance, multiplicity

**Recommendation:**

- **Primary arm: `open`** - full roster, free-form name. It is the configuration
  closest to production. `closed-oracle` is a diagnostic ceiling and must not be
  quoted as an accuracy figure.
- **Per book first, pooled second, always both.** A pooled score can hide a
  model-by-book interaction, which is the main reason both books are being
  judged.
- **Paired exact McNemar** on identical lines, since every model sees the same
  frozen inputs. Report discordant counts, not just p.
- **Holm correction** across the pairwise model comparisons within a book. Six
  models give 15 pairs; uncorrected, roughly one spurious result is expected.
- **Nonsignificance means unresolved, never equivalent.** Claiming practical
  equivalence requires a pre-declared acceptable loss and a noninferiority
  design. Neither exists here.
- **A result on one book is not a library-wide ranking** unless its direction
  holds on the other.

---

## Denominator, stated once

```
scored = judged lines
       - anonymous speech            (item 1)
       - NARRATOR segmentation errors (item 2)
       - unsupported at cap           (item 5)
```

Every report states `scored` and each exclusion count alongside the accuracy, so
a number can never be read without knowing what it was computed over.
