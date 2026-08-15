# Alexandria Audiobook — Quality Goals

What "good" means for this app, as numbers a script can check.

Every current value below is a measurement with a source you can re-run. Every
target is a commitment. Where there is no baseline yet, the goal is *to take
the measurement*, and it says so — an unmeasured target is a wish, and this
document does not contain wishes.

**Last updated:** 2026-08-15

## How to read this

Each goal starts with a plain-language box explaining what it is, why it
matters to someone listening to the finished audiobook, and why the target is
believed reachable. Then comes the technical detail: the **metric** (what is
counted), the **probe** (the script that counts it), the **current** value with
its evidence, and the **target**.

Three status markers:

- **MET** — measured at or beyond target. Keep a test so it stays there.
- **OPEN** — measured, below target. The gap is the work.
- **NO BASELINE** — not measured. The first task is the measurement, not the fix.

A target is only listed when something in the measured record suggests it is
reachable — a better arm, a cloud model, a human ceiling. Where the ceiling
itself is unknown, the goal says so rather than inventing a number.

### A few words that repeat

- **A book's "gold" set** — a few hundred lines from that book where a human
  wrote down who really speaks each one. It is the answer key. Everything is
  scored against it.
- **An "arm"** — one complete attempt at a task using one particular method, so
  two arms can be compared fairly. Like running the same race twice with
  different shoes.
- **"Held-out"** — material deliberately kept away from the system while it was
  being built, so testing on it shows real ability rather than memory.
- **The "ceiling"** — the best score anything could plausibly get, measured by
  having a human compete against herself. Scores mean little without it.

---

## 1. Speaker attribution — who says which line

The core task. Everything downstream inherits its errors: a misattributed line
gets the wrong voice, and no amount of TTS quality repairs it.

### 1.1 Accuracy on the four annotated books

> **What this is.** The app reads a novel and decides, line by line, which
> character is speaking. This measures how often it gets that right.
>
> **Why it matters.** This is the decision the whole app rests on. If a line is
> credited to the wrong character, it gets read in the wrong character's voice.
> A listener hears the villain speaking in the heroine's voice and the scene
> falls apart — and no amount of beautiful narration fixes it, because the
> mistake happened before a single word was spoken aloud.
>
> **Why 75% is reachable.** The app can run on a small model on your own
> machine, or a very large one rented in the cloud. The big cloud model is
> better — but only by about two points. Two of the four books already clear
> 75% locally. We are asking the local model to do what it nearly does already,
> not to make a leap.

**Metric** — percent of gold-labelled lines assigned the correct speaker.
**Probe** — `app/experiments/` arms, aggregated in `results_index.csv`.
**Current** — best local arm per book, 616 scored arm rows:

| book | best local (qwen3-14b) | best cloud (llama-3.3-70b) | gap |
|---|---|---|---|
| grimgar03 | 84.4% | 86.8% | 2.4 |
| index18 | 81.5% | 82.6% | 1.1 |
| mushoku16 | 72.9% | 74.8% | 1.9 |
| owarimonogatari3 | 69.1% | 69.8% | 0.7 |

**Target — every book ≥ 75% on the local model.** Two of four already clear it;
owarimonogatari3 needs +5.9 and mushoku16 +2.1.

**Why not higher.** Setting it at 90% would be asking for something nothing has
reached on any book by any method.

**The honest caveat.** Median across all 616 arms is 46–67% depending on the
book. The best arm is not the shipped arm, and the spread between books (69.1
to 84.4 on the same method) is larger than the spread between most methods.
Book identity dominates — some novels are simply harder than others, and a
result from one book does not transfer to the next.

#### These numbers are measured on the HARD SUBSET, and understate real accuracy

The light-novel gold says how it was drawn: *"Sampled uniformly from spoken,
**non-deterministic**, textually unique segments."* Lines the deterministic
namer already resolves — the ordinary `"…," said Haruhiro` case — were
**excluded before sampling**. Every light-novel accuracy in this document is
therefore conditional on *the line being hard enough that the cheap path
failed*, not on a representative page of the book.

The PDNC evaluation does not filter that way: it takes `entries[:limit]`
straight off the fixture. Which is why the same base model, on human-annotated
gold, scores far higher there:

| set | gold labelled by | sampling | base model |
|---|---|---|---|
| PDNC Pride and Prejudice | humans (published corpus) | first N, unfiltered | **80.5%** |
| PDNC The Awakening | humans | first N, unfiltered | **86.0%** |
| PDNC The Sign of the Four | humans | first N, unfiltered | **80.5%** |
| four light novels | two frontier models | hard subset only | 46–67% median |

**Do not read that gap as genre difficulty, and do not read it as the
LLM-judged gold being wrong.** It is mostly the sampling. Comparing a
hard-subset score against a whole-population score and concluding anything
about the books, the judges, or the language is the exact error this table
exists to prevent.

Two consequences worth keeping straight:

- **Real-world accuracy on a whole book is higher than goal 1.1's numbers**,
  because most lines never reach the LLM at all. What 1.1 measures is the part
  that does.
- The one comparison that *is* clean: BookNLP, the field-standard tool, scores
  **54.2%** on PDNC Pride and Prejudice (n=1226) under this harness. That is a
  ruler from outside this project, on human gold.
- **The three PDNC books in the table above are the top third of the corpus**
  (ranks #2, #8 and #9 of 28 — see 1.3). Across the 25 novels nothing here has
  ever looked at, the same base model scores **71.0%**, not 80.5–86.0%. Quote
  those three as evidence of what PDNC can look like, never as PDNC's typical
  difficulty.

**Before any cross-set comparison, harmonise the sampling.** Running every
method on every book — which is worth doing — will produce nonsense if a hard
subset is scored against a full set.

### 1.2 Close the selection gap

> **What this is.** Before deciding who spoke, the app assembles a shortlist of
> plausible characters. Two separate things can go wrong: the right name might
> not be on the shortlist at all, or it might be there and get passed over.
> This measures the second.
>
> **Why it matters.** It tells us where the real problem is. The shortlist
> contains the correct name **85%** of the time, but the app only picks it
> **29.9%** of the time. So the information is already sitting in front of it in
> the overwhelming majority of cases — and it looks straight past it. That is a
> very different problem from not knowing, and it needs a very different fix.
>
> **Why 50% is reachable.** We are not asking for new information, new models,
> or more reading. The answer is already present. Getting from "picks it 3 times
> in 10" to "picks it 5 times in 10", when the answer is on the page 8.5 times
> in 10, is closing a gap rather than inventing an ability. This is the single
> biggest known opportunity in the app.

**Metric** — of lines where the correct speaker is present in the candidate
roster, the percent where the model picks it.
**Probe** — see memory `attribution_selection_not_recall`.
**Current** — re-measured 2026-08-08 on the shipped **qwen3-14b**
(`selection_gap_recheck.py`, closed_set OPEN arm, 793 rows across four books).
**MET.**

| | qwen3.5-9b (this goal's original basis) | qwen3-14b (shipped) |
|---|---|---|
| roster recall | 85.0% | **91.6%** |
| selection | **29.9%** | **62.9%** |

| book | recall | selection | n |
|---|---|---|---|
| grimgar03 | 95.2% | 65.3% | 396 |
| index18 | 89.9% | 73.0% | 99 |
| mushoku16 | 84.6% | 60.9% | 136 |
| owarimonogatari3 | 89.5% | 52.4% | 162 |

**Target — selection ≥ 50% with roster recall held at ≥ 85%. MET: 62.9% at
91.6% recall, and every individual book clears 50% as well.**

**The gap was the model, not the method.** 29.9% was measured on qwen3.5-9b,
which a later six-model comparison put ~17 points behind the shipped model on
this task; the true difference here is 33 points. Selection is stable across
backends — the same book varies by 3.1 points (grimgar03) and 5.0 (mushoku16)
across four and five runs — so this is not one lucky artifact.

Nothing was built to close this. The goal was written around a number from a
model that does not ship, and the warning to re-measure before spending
anything on it was correct: the entire 55-point gap it described was 29
points of weaker model.

The four rejected approaches in the table below were therefore tested against
a deficit that mostly was not there on the shipped model. That does not
resurrect them — they were measured and they failed — but it does mean the
premise they were attacking has changed, and owarimonogatari3 at 52.4% is now
the only book near the line.

> **CAUTION: these figures were measured on a model that is not the one that
> ships.** The 147-line random gold set behind them has `source_run`
> **qwen3.5-9b**, and a six-model comparison on the same frozen harness later
> showed that model to be the *weakest selector tested*:
>
> | model | open | closed-6 | oracle |
> |---|---|---|---|
> | **qwen/qwen3-14b** (production) | 48.3% | 36.7% | **66.0%** |
> | ministral-3-14b | 47.6% | 41.5% | 61.2% |
> | phi-4 | 45.6% | 32.7% | 59.2% |
> | **qwen3.5-9b** (source of 29.9%) | 35.4% | 34.7% | **49.0%** |
>
> The 49% "conditional ceiling" this goal was written against is a property of
> **qwen3.5-9b, not of the task**. The production model reaches 66.0% on the
> same measurement.
>
> To be exact: 29.9% is end-to-end pipeline selection and 66.0% is conditional
> selection given an oracle set, so they are not the same number and one does
> not replace the other. But the *framing* — "the answer is present and the
> model looks past it" — was calibrated on a model 17 points worse at exactly
> that.
>
> **That re-measurement was done on 2026-08-08 and closed the goal:** selection
> is 62.9% on qwen3-14b at 91.6% roster recall. The warning is kept because it
> was correct — the entire gap this goal was built around was a weaker model —
> but it is no longer an instruction. Nothing here needs re-measuring.

**Why this and not more context.** Two independent measurements say supply is
not the constraint. Feeding the model more of the book is treating the wrong
problem, and has been tried.

#### What has already been tried, with numbers

Recorded here because this knowledge lived in ~30 scattered artifacts, and on
2026-08-06 that cost a session in which three already-rejected ideas were
proposed as if new. **Read this before proposing a fix for the selection gap.**

| approach | result | verdict |
|---|---|---|
| widen attribution context (w1→w4) | +10.0 grimgar03 (4 repeats), +3.0 index18, **−5.0 mushoku16**, 0.0 owari | book-dependent, not a fix |
| route per book | leave-one-out router **56.5%** vs fixed **57.2%** | **worse than picking one setting** |
| constrain decoding to the roster (GBNF) | open arm: 0.0, −1.0, −1.4, −1.2 | no gain where it matters |
| shrink candidate set to 6 | +1.8 grimgar03, **−6.1 index18, −12.5 mushoku16, −4.9 owari** | loses the right name |
| oracle candidate set | +10 to +18 everywhere | not achievable; it needs the answer |

The shape of it: **`closed-oracle` wins big and `closed-6` loses**, and the only
difference is whether the shortlist contains the right name. Constraining the
model is not the lever. Whether the answer is *in the list* is.

**Routing deserves its own warning.** Every routing gain quoted before
`realizable_router` was fitted — the best arm per book read off the results
afterwards. When the choice must be made without seeing the held-out book, the
router wins 4 families, loses 5, ties 6, and lands **below** a fixed setting.
An oracle-routed number is not an achievable number.

#### The one lever positive on all four tested books — and only four

**Scope first, because the result is easy to overstate and was.** This ran on
grimgar03, index18, mushoku16 and owarimonogatari3. All four are Japanese light
novels **in English translation** — one genre, one language, one translation
pipeline. It has never run on the three PDNC public-domain English novels, on
the Chinese WP/JY sets, or on any Japanese-language text. Read every number
below as "four books of one kind", not "every book".

`roster_quality` varied the roster instead of the model. Adding the names
`build_roster` missed beats the generated roster on all four, and beats a
*perfect* roster too:

| book | generated | augmented | gold |
|---|---|---|---|
| grimgar03 | 59.7 | **63.6** | 61.6 |
| index18 | 67.4 | **71.7** | 69.6 |
| mushoku16 | 48.9 | **51.9** | 48.9 |
| owarimonogatari3 | 40.7 | **45.1** | 42.6 |

**+3.0 to +4.4, same direction every time** — the only intervention measured so
far with no book that it hurts. The experiment pre-registered this reading:
*"augmented >> generated → roster extraction is worth fixing, and the size of
the effect is the prize."*

The misses are not walk-on parts: ten characters across four books that
`build_roster` never found, including HITOGAMI with 9 lines in mushoku16 and
OUGI with 10 in owarimonogatari3.

The `inflated` arm — a gold roster plus twenty decoys — is the guard rail:
30.9% on owarimonogatari3 against 40.7% generated. **Adding names is only safe
when they are real.** A recall fix that pads the list will lose more than it
gains.

**So the next move on 1.2 is `build_roster` in `three_pass_generate.py`, not a
prompt, a constraint, or a router — but see the scope note first.**

#### Before acting on it: widen the book set

The cheapest way to find out whether this generalises is to run the same
experiment on PDNC. It is not blocked by missing data:

- PDNC ships its own curated roster (74 names for Pride and Prejudice), which
  is what the `gold` arm needs, and from a published annotated corpus rather
  than this project's own judging.
- Its gold sets are LARGER than the light novels': 1270 / 640 / 584 rows
  against 396 / 162 / 136 / 99.
- **PDNC contamination does not apply here.** That contamination concerns the
  distilled adapter's training set. `roster_quality` trains nothing; it runs
  the base model and varies only the roster at inference. Nothing is fitted, so
  held-out status is irrelevant to this particular experiment.

**What it actually costs** — corrected after reading the script's data
dependencies rather than guessing at them, having first written "an afternoon"
here without checking:

1. **A three-pass checkpoint per book. This is the real cost.**
   `roster_quality` reads `segmented` and `named` from a prior pipeline run
   (`matrix_20260725-115148/<model>/<book>/result.json.threepass_checkpoint.json`).
   Only six light novels have one. The PDNC books have never been through the
   pipeline, so each needs a **GPU segmentation run first** — mushoku16's single
   pass took 80 minutes at 45 chunks, and Pride and Prejudice is a longer book.
   Budget hours per novel, not minutes.
2. **Source text** must be placed where the script expects it; the matrix
   `inputs/` directory holds only the eight light novels.
3. **`roster_additions` does not exist in PDNC gold.** The light novels carry a
   hand-curated list of names the judges found missing; PDNC instead carries a
   `roster` of 74 curated names. So `additions` becomes `roster - generated`,
   which is arguably a cleaner definition — derived from a published corpus
   rather than from this project's own judging.
4. The hardcoded four-book decoy pool needs widening. This part *is* trivial;
   it was the only part visible without reading the data flow.

Chinese (WP/JY) would need more work: those sets use a different structure
(`dataset`/`results` rather than `entries`). No Japanese-language attribution
gold exists at all — the light novels are Japanese-origin but English text.

If roster augmentation holds on three English public-domain novels of a
different century and genre, it is a real finding and goal 1.3 gains evidence
at the same time. If it does not, then it was a property of translated light
novels and the whole recommendation changes.

#### Still open

`candidates.py` exists, states its own plan — *"an upper bound on recall;
ablate afterwards to find the smallest reliable set"* — and has no artifact.
The size-versus-recall curve it proposed was never run. Given `closed-6` fails
by losing the right name and `closed-oracle` wins by keeping it, that curve is
the one measurement that would say whether a small, honest candidate set is
reachable at all.

### 1.3 Generalisation beyond the four books

> **What this is.** Checking the app works on novels it has never encountered,
> rather than only the handful used while building it.
>
> **Why it matters.** A cook who can only make one dish is not a cook. If the
> app only performs well on the four books it was tuned against, it is not a
> product — it is a demo. Users will bring their own books.
>
> **Why this is reachable.** It is mostly a bookkeeping problem, not a
> capability problem. Public-domain novels with human-written answer keys
> already exist — 28 of them, freely available. The previous test was spoiled
> because 25 of those 28 had accidentally been used during development, which
> is like grading a student on questions they had already seen. When the truly
> unseen books were scored, the drop was **4.4 points** — real, but modest. The
> work is running a clean test, not building a new ability.

**Metric** — accuracy on held-out books never used in development.
**Probe** — PDNC gold sets (`attribution_gold_pdnc_*.json`, 1270 / 640 / 584
rows) plus `attribution_gold_random.json`.
**Current** — measured cleanly 2026-08-08 (`pdnc_generalisation.py`) on the
**base** arm, which has no PDNC training at all, so contamination does not
apply to it:

| group | books | rows | accuracy |
|---|---|---|---|
| never looked at by this project | 25 | 3000 | **71.0%** |
| the three quoted in 1.1 | 3 | 360 | **83.6%** |

**Gap −12.6 points against a target of 5. OPEN, and failing by more than
double.**

**The three books this project quotes are not typical books.** Ranked against
all 28: The Sign of the Four **#2**, The Awakening **#8**, Pride and Prejudice
**#9** — every one in the top third. Per-book accuracy runs from 50.0%
(Mansfield Park, The Gambler) to 91.7%, median 73.3%, IQR 62.3–80.6. The PDNC
figures cited in 1.1 are real measurements of favourably-placed books, and
generalise 12.6 points worse than they appear.

**Quote type does not explain it.** PDNC labels each quote Explicit, Implicit
or Anaphoric, and the obvious hypothesis is that hard books carry more implicit
attribution. They do not: books at ≥50% implicit quotes median **73.3%**, books
below 50% median **73.3%** — identical. Whatever drives a 41-point spread
between novels, it is not the quote-type mix.

**Failure telemetry, 2026-08-15.** Joining the saved base predictions to the
candidate rosters and context windows for the five weakest books explains all
272 errors (`pdnc_failure_telemetry.py`). Gold candidate recall is **100%**:
no error is caused by a missing gold speaker. The dominant class is choosing a
different valid candidate (**149, 54.8%**), followed by missing a gold alias
explicitly present in the supplied context (**59, 21.7%**), invalid/out-of-
roster answers (**39, 14.3%**), and one 25-row block with no saved predictions
(**9.2%**). Candidate expansion is therefore the wrong intervention. The next
attribution work should improve selection/context use and keep batch-failure
recovery measurable; constrained output alone can address at most the smaller
invalid-answer class.

**First-person narrator intervention, 2026-08-15.** A same-run controlled
comparison on 120 quotations from each of two weak first-person books found a
specific, correctable selection failure (`pdnc_narrator_prior__local-llamacpp.json`).
Supplying the narrator's exact character identity raised *The Gambler* from
**62/120 (51.7%) to 93/120 (77.5%)** and *The Sun Also Rises* from **65/120
(54.2%) to 91/120 (75.8%)**. On quotations owned by the narrator, the changes
were **0/56 to 38/56** for ALEXIS IVANOVITCH and **28/60 to 44/60** for JAKE
BARNES. The paired changes were +38/−7 and +29/−3 respectively.

The generic control, which told the model to infer a first-person narrator but
did not supply the name, was neutral on *The Gambler* and slightly worse on
*The Sun Also Rises* (`pdnc_narrator_prior__local-llamacpp-generic.json`). The
gain therefore comes from the explicit book-level identity, not generic prompt
wording. This supports the optional narrator hint now implemented in PR #299,
but does **not** close this goal: only two books and 120 rows per book were
tested, the audit classifies both artifacts as provisional because their
recorded worktree was dirty, and the target requires a clean held-out result on
at least three books.

**Target — a clean held-out number on ≥ 3 books, within 5 points of the
development books' figure.**

---

## 2. Voice — does it sound like the target speaker

### 2.1 Speaker similarity against a human ceiling

> **What this is.** The app can imitate a specific narrator's voice. This
> measures how close the imitation gets to the real person.
>
> **Why it matters.** It is the difference between "that sounds like a computer
> doing an impression" and "that sounds like her". But a similarity score on its
> own is meaningless — is 0.75 good? There is no way to know. So we also measure
> the same narrator against *herself*, reading different material. That is the
> ceiling: no imitation should beat a person being herself. Every score is read
> as a percentage of that ceiling.
>
> **Why 95% is reachable.** Japanese is already at 98% and English at 93%. This
> is not a hoped-for leap; it is bringing the weakest language up to where the
> strongest already sits.

**Metric** — ECAPA cosine similarity (a standard voice-fingerprint comparison),
generated vs the human reading, read against `human_vs_human` (same narrator,
different held-out line).
**Probe** — `app/experiments/ljspeech_score.py`.
**Current** — 2026-08-06, 150 held-out lines per language:

| language | ceiling | zero-shot clone | LoRA | clone as % of ceiling |
|---|---|---|---|---|
| English | 0.809 | 0.757 | 0.690 | 93% |
| Japanese | 0.796 | 0.779 | 0.755 | 98% |
| Chinese | *0.691 — anchor invalid* | 0.765 | 0.720 | — |

**Target — reach 95% of the ceiling in every language with a valid anchor.**

**A result worth stating plainly: the simple method beat the elaborate one.**
There are two ways to imitate a voice here. "Zero-shot cloning" just listens to
a short sample and mimics it. A "LoRA" is a small trained add-on, built from
many samples over hours of GPU time. The simple method won in all three
languages. That replicates an earlier finding and extends it across languages.

It is a measured comparison, not yet a recommendation — the LoRA is better at
matching the *melody* of speech (how pitch rises and falls) in English and
Chinese, while cloning is better at matching the *timbre* (what the voice
sounds like). The two disagree about which is better, so the question is not
closed.

### 2.2 Repair the Chinese anchor

> **What this is.** Making sure the measuring instrument works before trusting
> what it measures.
>
> **Why it matters.** In Chinese, the narrator scored **worse against herself**
> than the synthetic voices scored against her. Read that again: the real person
> was judged less like herself than a machine imitation was. That is impossible
> as a fact about voices, so it is a fact about the ruler. Any Chinese voice
> conclusion drawn from this data is unreliable — including the flattering ones.
>
> **Why this is reachable.** There is an obvious suspect. The Chinese clips are
> much shorter — about 3 seconds, against 7 for English — and this kind of
> voice-fingerprinting is known to get shaky on short audio. Testing it is
> cheap: chop the English clips down to 3 seconds and see whether its ceiling
> collapses too. If it does, we have the answer in an afternoon.

**Metric** — `human_vs_human` must exceed every arm it bounds.
**Probe** — `find_invalid_anchors` in `ljspeech_score.py`, tested in
`app/test_score_anchor.py`. Anchor construction: `build_anchor_side`.
**Current** — **CAUSE FOUND AND FIXED 2026-08-06.**

**Clip length was the whole cause**, established in both directions:

| direction | result |
|---|---|
| truncate ENGLISH clips to the Chinese median (3.17 s) | anchor **0.783 → 0.632**, below its own clone arm |
| join same-speaker CHINESE clips to 6.9 s | anchor **0.670 → 0.837**, clears both arms |
| join to 10.2 s / 13.6 s | 0.867 / 0.901 |

Shorten a good anchor and it breaks; lengthen a broken one and it repairs. Not
the corpus, not the narrator, not the language, and not ECAPA being unsuited to
Chinese — the clips were too short for a speaker embedding to be stable.

**The fix needed no new data.** All 150 Chinese clips are one speaker, and a
speaker embedding does not care about sentence continuity, only about quantity
of voiced material. `build_anchor_side` now joins consecutive same-speaker
clips until each side of the anchor carries `ANCHOR_MIN_SECONDS = 7.0`, chosen
from the knee of that curve.

**Target — every eval set's anchor above all of its arms. MET, confirmed in
evidence 2026-08-07.** All three sets re-scored, `anchor_invalid` empty in
each:

| set | anchor | clone | LoRA |
|---|---|---|---|
| English | 0.8328 | 0.7567 | 0.6899 |
| Japanese | 0.8355 | 0.7789 | 0.7551 |
| Chinese | 0.7655 | 0.7651 | 0.7197 |

Chinese clears its clone arm by **0.0004**. That satisfies the target and is
not a margin to lean on: a Chinese result that depends on the anchor sitting
above the clone should be treated as unresolved rather than passing.

**What this retroactively rescues.** The Chinese ARM numbers were always fine —
clone 0.765, LoRA 0.720. Only the yardstick was broken, so those measurements
become readable rather than being discarded.

**A note for other eval sets.** Any future set whose clips are short inherits
this. The guard is `find_invalid_anchors`, which now has a known cause to point
at rather than only a symptom.

### 2.3 Adapters that stop talking

> **What this is.** Making sure a trained voice knows when the sentence is
> over.
>
> **Why it matters.** This one already bit us, expensively. Two trained voices
> produced **163.8 seconds of audio for a 7-second line** — every single time.
> They never learned to stop, so they babbled until the system cut them off. The
> cruel part: the training reports looked completely normal. Nothing was wrong
> until you actually listened.
>
> **Why this is already met, and how it stays met.** The cause turned out to be
> one setting — the training speed dial — set five times too high. Turned down,
> three voices in three languages all came out correct. It is now checked
> automatically two ways: a short listening test before any voice is used, and a
> test that stops the wrong setting from creeping back in.

**Metric** — median generated duration ÷ human duration, per adapter.
**Probe** — `app/experiments/verify_adapter_stops.py`, gate at 3.0x.
**Current** — 1.01x / 0.87x / 0.94x across the three languages. **MET.**

**Target — hold median within 0.8–1.25x, and never ship an adapter above
3.0x.**

Training loss looked ordinary (2.9 and 3.4), so only generated output reveals
this. Protected by the gate plus `app/test_training_defaults.py`.

### 2.4 Duration fidelity in normal use

> **What this is.** Whether a spoken line lasts about as long as a human would
> take to say it.
>
> **Why it matters.** A line delivered in three-quarters of the natural time
> sounds rushed and clipped. It is the same *kind* of fault as the babbling
> voices above — wrong length, no error message, nobody told — but in the
> opposite direction and much subtler, which is exactly what makes it easy to
> ship by accident.
>
> **Why this is reachable.** Five of the six measured cases are already inside
> the target. Only Japanese zero-shot cloning sits outside it, so this is one
> specific case to investigate, not a broad weakness.

**Metric** — mean `dur_ratio` across held-out lines (1.00 = matches the human).
**Current** — 100 clips per language, both arms, 2026-08-08, plus the
narrator-controlled Japanese clone replication below (`duration_probe.py`).
**MET at the median; per-line spread remains a quality opportunity.**

Reproduced unchanged on 2026-08-12 in
`duration_probe_20260811_overnight.json`: Japanese clone median **0.7584**
with **94.0%** of clips outside the band. This confirms the baseline; it is
not an intervention or an improvement.

**Diagnosed 2026-08-15: the comparison does not isolate a duration defect.**
The Japanese clone's generated pace matches the pace implied by its reference
clip: generated duration / reference-rate-predicted duration has median
**1.023** across the same 100 lines. The reference reads 38 non-space
characters in 5.166 s (7.36 chars/s), while the held-out book's human is about
26% slower. Slowing production output to match that different reading would
erase the cloned speaker's pace.

The valid same-speaker measurement was completed 2026-08-15. Thirty held-out
clips and the excluded reference all come from the same *Kokoro* LibriVox
recording, whose 103 chapters are read by ekzemplaro. Japanese clone median is
**0.9269**, inside the 0.90–1.10 target (p10–p90 **0.78–1.05**, 43.3% outside).
The old 0.758 result was therefore dominated by reader/session pace mismatch,
not evidence for language-specific time stretching. Evidence:
`kokoro_same_speaker_generate.json` and
`duration_probe_same_speaker_20260815.json`.

| arm | median ratio | p10–p90 | clips outside band |
|---|---|---|---|
| English LoRA | 0.959 | 0.83–1.09 | 35% |
| English clone | 1.018 | 0.85–1.17 | 35% |
| Japanese LoRA | 0.929 | 0.78–1.15 | 57% |
| Japanese clone | **0.758** | 0.65–0.86 | **94%** |
| Japanese clone, same narrator/book | **0.927** | 0.78–1.05 | 43% |
| Chinese LoRA | 0.935 | 0.83–1.10 | 45% |
| Chinese clone | **0.896** | 0.78–1.03 | 57% |

**This is the twelve-clip finding that survived.** The Japanese clone arm was
0.76 at n=12 and is 0.7584 at n=100 — the same number to three decimals. Three
other twelve-clip findings dissolved under proper sampling on the same day
(2.5's two cells, 2.6's jitter), so the lesson is that sample size has to be
checked, not that small samples are always wrong.

The per-clip column is new and was hidden by the medians. Every arm, including
the four whose medians pass, puts a third to a half of its individual clips
outside 0.90–1.10. A median near 1.0 built from clips scattered on both sides
is a different defect from a uniformly rushed arm, and only the Japanese clone
is uniform: at 94% outside with a p90 of 0.86, nearly every clip is short, not
merely the average.

**Target — every arm within 0.90–1.10.**

The existing safety check catches runaway voices at 3.0x. It cannot see 0.76.

---

### 2.5 Pitch range, not just pitch shape

> **What this is.** Whether the generated voice uses as much of its pitch range
> as the human did — how far it moves between its lowest and highest notes in a
> line.
>
> **Why it matters.** This is the difference between a reading and a recital. A
> voice can follow every rise and fall of the original and still deliver them
> all within a narrow band, and the result sounds flat — the thing people mean
> when they say synthetic speech is monotone. A listener notices immediately.
>
> **Why nothing caught it until now.** `f0_corr` measures the *correlation* of
> the pitch contour, which stays high when the whole shape is squashed: the ups
> and downs happen in the right places, just smaller. Correlation is deliberately
> blind to scale. So the app measured pitch and missed the flatness.

**Metric** — generated f0 spread (p90 − p10) ÷ human f0 spread, on the same line.
**Probe** — `app/experiments/pitch_quality_probe.py` (calls `pitch_stats`; the
`voice_compare_view` CLI renders a view, it does not report numbers).
**Current** — 100 clips per language, both arms, 0 dropped, 2026-08-08. **MET.**

| set | clone | LoRA |
|---|---|---|
| English | 0.92x | 1.13x |
| Japanese | 0.91x | 1.02x |
| Chinese | 1.03x | 0.96x |

Every cell is inside 0.90–1.15x. At 12 clips this goal read OPEN on two cells —
English clone at 0.83x and Chinese LoRA at 0.81x — and both were sample size:
the same arms measure 0.92x and 0.96x over 100 clips. The 12-clip run's
artifact is not in the evidence tree, so the two runs cannot be reconciled
directly; what is claimed here is only what the n=100 artifact shows.

**Target — f0 spread within 0.90–1.15x of the human, and f0 median within
0.95–1.05x.**

English cloning also drops median pitch to **0.81x** — a voice pitched a fifth
of an octave low for the whole book. ECAPA rates that same clone *higher* than
the LoRA, because embedding similarity is a timbre measure and this is a pitch
failure.

**A caution, since this section could invite the wrong fix.** The app
deliberately refuses to classify gender from pitch
(`test_pitch_is_not_used_as_a_gender_classifier`), and that decision is right —
male and female f0 distributions overlap heavily. Nothing here labels a
speaker. This asks only whether an arm preserved *that speaker's own* range,
which is a comparison between two clips of one person.

### 2.6 Voice quality, the measures the field says matter most

> **What this is.** Three standard measures of how a voice actually behaves:
> **jitter** (cycle-to-cycle wobble in pitch), **shimmer** (wobble in
> loudness), and **HNR** (how much of the sound is clean tone versus noise).
>
> **Why it matters.** They catch a failure nothing else here can: a synthetic
> voice being *too clean*. Human phonation wobbles slightly on every cycle; a
> vocoder often does not. A voice with near-zero jitter sounds subtly
> unnatural even when every other measure looks good.
>
> **Why these three.** A study comparing speakers ranked what actually
> separates voices: jitter first, then F2, then shimmer, then HNR — and
> *duration* among the least discriminative. This project measured ECAPA, f0
> contour, MCD and duration, so it had none of the top four and one of the
> bottom.

**Metric** — jitter, shimmer and HNR of the generated line ÷ the human's, plus
vocal tract length from formant dispersion.
**Probe** — `voice_quality`, `vocal_tract_length` (Praat via parselmouth).
**Current** — all five measures on 100 clips per language, both arms, 0
dropped (jitter/shimmer/HNR 2026-08-08, tract length 2026-08-09). **MET on
jitter and shimmer. MET on HNR after the eval speaker was corrected. OPEN on
tract length, one cell: the Chinese clone arm at 1.064x.**

| set | jitter | shimmer | HNR |
|---|---|---|---|
| English LoRA | 0.99x | 0.92x | 0.98x |
| English clone | 0.94x | 0.88x | 1.02x |
| Japanese LoRA | 1.11x | 1.07x | 0.97x |
| Japanese clone | 1.09x | 1.07x | 0.96x |
| Chinese LoRA | 0.90x | 0.88x | **1.17x** |
| Chinese clone | 0.98x | 0.89x | 1.08x |

The Chinese LoRA was the "too clean" case at 12 clips — jitter 0.81x, HNR
1.20x. At 100 clips its jitter is 0.90x, inside the band: that half of the
finding was sample size. Its HNR is 1.17x, still outside 1.15 after a fivefold
sample increase.

**Clip length has been ruled out as the cause** (`hnr_length_probe.py`,
2026-08-08), using the two-direction test that settled 2.2:

| condition | clip length | HNR ratio |
|---|---|---|
| Chinese as-is | 3.1 s | 1.1793x |
| Chinese, clips joined | 8.9 s | 1.1818x |
| English as-is | 7.5 s | 0.9806x |
| English, truncated to Chinese length | 3.2 s | 0.9519x |

Nearly tripling the Chinese clip length moves the ratio by 0.0025, and
shortening English to Chinese length does not inflate it — it drifts 0.03 in
the opposite direction to the one a length artifact predicts. Short clips
destroyed the ECAPA anchor and do nothing to HNR, so the two are not the same
defect and 2.2 does not explain this.

**The corpus has been ruled out and the eval speaker has not**
(`corpus_hnr_baseline.py`, 2026-08-08; 40 AISHELL-3 speakers the adapter never
saw, 15 clips each, human recordings only):

| | median HNR |
|---|---|
| AISHELL-3, 40 unseen speakers | **12.02 dB** |
| LJSpeech (English) | 10.83 dB |
| Kokoro (Japanese) | 16.28 dB |
| SSB1585, the Chinese eval speaker | **9.39 dB** |

AISHELL-3 is *cleaner* than LJSpeech, which the English cell passes on, so
"recorded on consumer hardware" explains nothing. SSB1585 herself sits 2.63 dB
below her own corpus median at the **8th percentile** — 3 of 40 sampled
speakers are noisier than she is.

So the denominator is depressed, by the choice of eval speaker rather than by
the corpus. The generated side measures 11.17 dB against AISHELL-3's own 12.02
median: the adapter produces roughly corpus-typical phonation and looks too
clean only because it is divided by an atypically noisy narrator. Against a
median speaker the same audio would land near 0.93x — that number is
arithmetic on two medians, not a measurement.

**Status: CLOSED 2026-08-08.** The Chinese arm was re-run end to end against
**SSB0748**, the speaker closest to the corpus median (12.025 dB, +0.005 off),
with every other setting matched to the original: same trainer, lr 1e-6, 6
epochs, lora_r 64, seed 1234. New adapter, 150 lines, 300 clips, 0 dropped.
Artifacts and the chain script are on the `agent/asr-hybrid-cjk` branch.

| | SSB1585 (9.39 dB, 8th percentile) | SSB0748 (12.02 dB, median) |
|---|---|---|
| human HNR | 9.39 dB | **12.02 dB** |
| LoRA HNR ratio | **1.1669x** (out of band) | **1.0571x** (in band) |
| clone HNR ratio | 1.0839x | 1.1219x |
| jitter | 0.9041x | 0.9457x |
| shimmer | 0.8793x | 0.8885x |

**Every Chinese cell of 2.6 is now inside 0.85–1.15x.** The "too clean" Chinese
LoRA was an artefact of picking an unusually noisy narrator to compare against.

**Where the prediction was wrong, and it matters.** The chain script committed
beforehand predicted ~0.93x, on the arithmetic that the generated side would
stay at 11.17 dB while the denominator rose to 12.02. The generated side did
not stay: the new adapter produces **12.71 dB**. The conclusion held — in band,
speaker selection was the cause — but the specific number was not predicted
correctly, because an adapter trained on a different speaker synthesises
different phonation.

**The unavoidable confound, stated.** Changing the eval speaker necessarily
changes the adapter too; there is no adapter for a speaker without training one.
So this run does not isolate speaker identity from adapter instance on its own.
What makes speaker selection the better explanation is the corpus baseline: 40
speakers the adapter never saw median 12.02 dB, and SSB1585 sat at the 8th
percentile of them. The rerun is consistent with that, not independent proof of
it.

**Vocal tract length, now at 100 clips per language** (2026-08-09), both arms:

| arm | ratio | | arm | ratio |
|---|---|---|---|---|
| English LoRA | 1.004 | | English clone | 1.029 |
| Japanese LoRA | 1.016 | | Japanese clone | 1.042 |
| Chinese LoRA | 1.032 | | Chinese clone | **1.064** |

Five of six cells are inside 0.95–1.05. **The Chinese clone arm is not**, at
1.064 — every arm shifts the speaker's apparent vocal size slightly *upward*,
and that one shifts it past the target.

**The 12-clip entry claimed 0.98–1.08x and called this MET, but 1.08 was
already outside the 1.05 target.** It was recorded as met against its own
failing number. That is a bookkeeping error, not a measurement one, and it is
the reason this row is now stated per-arm rather than as a range: a range hides
which arm failed.

**Target — jitter, shimmer and HNR within 0.85–1.15x; tract length within
0.95–1.05x.**

### 2.7 Adapters must not be trained on their own test data

> **What this is.** Keeping some of a narrator's recordings away from training,
> so there is honest material to test the finished voice on.
>
> **Why it matters.** Otherwise there is no way to know whether a voice
> genuinely learned to sound like someone, or simply memorised the specific
> lines it was shown. A student who sees the exam paper in advance tells you
> nothing about what they know.
>
> **Why this is cheap to fix.** The split already exists and is already
> correct — the trainer just doesn't use it.

**Metric** — adapters whose training set includes their validation split.
**Current** — every dataset zip splits **180 train / 20 val with zero
overlap**, but `train_lora.py` loads the *root* `metadata.jsonl`, which is all
200. **OPEN — 60 of 75 shipped adapters are contaminated**, audited
2026-08-08 from each adapter's own `training_meta.json`.

| trained on | adapters |
|---|---|
| all 200 clips, including its own val split | **60** |
| 180, the train split only | 7 |
| some other count (24, 81, 88, 116, 130, 170, 188) | 8 |

The clean 7 are exactly the seven retrained and promoted on 2026-08-08; every
adapter predating that fix is contaminated. Their held-out scores are therefore
measured partly on clips they were trained on, and should be read as an upper
bound rather than a held-out result.

One trap for anyone re-running this audit: the retrained adapters record
`num_samples` while the older ones record `sample_count`. Two field names for
one concept - checking only one of them silently reports the wrong count, which
happened on the first pass of this audit.

**Target — train on `train/` only; 0 adapters trained on their own val split.**

Until then, every per-adapter fidelity number in the library is an **upper
bound**: it is measured on material the model has heard. It can still rank
adapters, because all are contaminated identically — a voice that scores badly
on its own training data is genuinely bad.

**Twenty clean retrains promoted 2026-08-15.** All 26 candidates that cleared
the retraining summary were independently regenerated and identity-gated on
six held-out lines. Twenty passed the 0.45 gate and beat the weights currently
shipped, so they were installed with receipt and rollback backup
`promotion_backups/20260815_152050.json`; five passed identity but did not beat
the current library and one failed identity at 0.393. Exact-source hash checks
confirmed every installed adapter matches the path recorded by its gate.

The live manifest now reports **47 of 75 adapters still trained on 200 clips**,
down from 67 before the seven earlier clean promotions and these twenty. The
goal remains **OPEN**, but deployment—not merely retraining evidence—has now
removed contamination from 20 additional shipped voices.

#### The reference clip is CAUSAL — established by intervention, not correlation

> **What this is.** Before training a voice, the pipeline picks one recording
> to define "who this speaker is". Every one of the 200 training samples is
> anchored to that single file. This measures what happens when it is the
> wrong person.
>
> **Why it matters.** It was chosen as "whatever clip happens to be first",
> with no check. When that clip belonged to someone else — which happens when
> the automatic speaker-splitting misfires — the whole voice was built on the
> wrong identity, and nothing reported a problem.

**Metric** — adapter speaker-similarity when only the reference clip changes.
**Probe** — `app/experiments/reference_intervention.py`.
**Current** — 2026-08-07. Same 180 training clips, same seed, same epochs;
**only the reference file differs**:

| reference | adapter |
|---|---|
| medoid (most representative clip) | **0.695** |
| least-typical clip of the same narrator | 0.634 |
| **a different narrator entirely** | **0.154** |

**A wrong-speaker reference costs 0.541.** That drops a working voice into
exactly the range the broken library adapters occupied (0.004–0.14), which is
enough to explain all of them.

The fine-grained contrast transfers almost perfectly: a **0.058** change in
reference quality produced a **0.061** change in the adapter — **transfer ratio
1.06**. Small reference differences matter proportionally; large ones are
catastrophic.

**This supersedes the correlational finding below.** The +0.76 correlation was
consistent with a third cause — a messy dataset yielding both a bad reference
and a hard learning problem. The intervention rules that out: nothing varied
except one file.

**It also explains the "training lottery" that never existed.** Retrained
adapters recovered from 0.027 to 0.685 on apparently identical settings. The
extracted dataset zips carry no `ref.wav`, so `train_lora.py` fell back to its
*first training clip* — a different, usually better reference. The recovery was
a reference change all along. Three runs at a fixed seed give 0.739 / 0.736 /
0.685, so training itself is deterministic and **a retry loop would be
useless**.

#### The correlational evidence that led here

`train_lora.py` extracts the speaker embedding from ONE reference clip and uses
it for all 200 training samples, so a single wrong file sets the voice identity
regardless of how clean the training audio is.

Swept across all 75 adapters on 2026-08-07 (`ref_clip_match.py`):

- **correlation(reference matches its narrator, adapter quality) = +0.59**
- **7 adapters have a reference that is not their own narrator** (<0.3)
- **4 of those 7 also failed** as adapters

+0.59 is a real effect and not a complete explanation. The clearest cases are
stark — `husky_baritone_20s_m_anime` has a reference scoring **−0.026** against
its own training data and an adapter at 0.027 — but `warm_tenor_20s_m` reaches
0.725 with a reference at 0.096. A mismatched reference does not doom an
adapter, and a good adapter does not prove a good reference.

**What this changes:** for those 4, "retrain" is the wrong prescription. The
*reference clip* needs fixing first, which is cheaper than either retraining or
rebuilding the dataset.

**Retraining the seven with a corrected reference**, measured on held-out clips:

| adapter | shipped | retrained |
|---|---|---|
| husky_baritone_20s_m_anime | 0.004 | **0.691** |
| warm_baritone_40s_m_fantasy | 0.062 | **0.694** |
| warm_tenor_20s_m | 0.090 | 0.411 |
| velvety_mezzo_30s_f_gothic | 0.084 | 0.187 |
| silky_baritone_45s_m | 0.079 | 0.088 |
| husky_baritone_40s_m_military | 0.141 | 0.149 |
| warm_baritone_50s_m_gothic | 0.544 | 0.468 |

Two dead voices became good ones. The three that barely moved are the ones
whose DATASETS are mixed-speaker (2.7 above): there the reference was never the
binding constraint, and rebuilding is still required.

### 2.8 A voice stays the same voice across a whole book

> **What this is.** Whether a voice slowly wanders into someone else over the
> course of a book, rather than staying recognisably one person.
>
> **Why it matters.** The product is ten hours of one consistent voice. Every
> other voice measurement here is a SINGLE LINE, and a slow drift is invisible
> to them: each line is scored against its own reference, so a voice that
> wandered steadily would score the same at the start and the end while
> sounding obviously wrong to someone who sat through it.
>
> **The answer is that it does not drift.** Measured at book length, a voice is
> as much itself on line 2000 as on line 1.

**Metric** — speaker similarity to an anchor built from the opening lines, as a
function of position through the run.
**Probe** — `app/experiments/voice_drift.py`.
**Current** — 2026-08-08, three adapters, **2000 consecutive lines each**, zero
failures. **MET.**

| adapter | first third | last third | fitted change |
|---|---|---|---|
| husky_tenor_30s_m_literary | 0.776 | 0.771 | −0.005 |
| warm_mezzo_30s_f_fantasy_2 | 0.739 | 0.738 | +0.009 |
| warm_baritone_40s_m_2 | 0.728 | 0.728 | +0.004 |

**Target — |drift| ≤ 0.03 across a book-length run.** All three clear it with
an order of magnitude to spare.

#### The 400-line result was noise, and this supersedes it

An earlier run over 400 lines reported drift of −0.018, −0.050 and −0.017 and
was written up here as a real defect, with pitch rising +1.9% to +6.6% offered
as its mechanism. At five times the length **every part of that reverses**:

| | 400 lines | 2000 lines |
|---|---|---|
| husky_tenor_30s_m_literary | −0.018 | −0.005 |
| warm_mezzo_30s_f_fantasy_2 | **−0.050** | **+0.009** |
| f0 across the run | **rises** 1.9–6.6% | **falls** 1.2–4.3% |

A trend that shrinks toward zero as the sample grows, and whose direction
flips, is line-to-line variation being fitted as a slope. Four hundred lines
was simply too short: normal variation over a few hundred lines looks like a
trend, and `polyfit` will always return one.

**What did not change:** vocal tract length is stable to within 1% over 2000
lines on all three adapters, and HNR within 1%. The speaker's physical voice
properties hold. Those were flat at 400 lines too, and they are the numbers
that were right both times.

**The lesson, since this is the second time it has bitten.** The original entry
carried the caveat — *"a 400-line drift of −0.05 does not license a claim about
5,000 lines; whether it is linear, plateaus, or accelerates is unmeasured"* —
and the goal was still written as though the drift were real. Stating a caveat
is not the same as heeding it. A measurement that cannot distinguish trend from
noise should be reported as **NO BASELINE**, not as a defect with a target
attached.

## 3. Reliability — does a run finish and produce the right thing

### 3.1 Chunk completion on script generation

> **What this is.** A novel is too long to process at once, so it is cut into
> chunks. This tracks how many chunks get through without the app giving up on
> them.
>
> **Why it matters.** A failed chunk is a hole in the audiobook. Runs take
> hours, so failures discovered at the end are expensive in wall-clock time and
> in patience.
>
> **Why 99% is reachable.** The failures have been studied and sort into two
> named groups: one is a near-miss against a threshold and is fixable by
> adjusting that threshold; the other is the model occasionally losing the plot
> for no reason connected to the text. Neither is mysterious. One book in the
> most recent run completed 9 chunks out of 9 cleanly, so clean runs plainly
> happen.

**Metric** — chunks completing without exhausting retries.
**Probe** — `logs/review_responses.log`, per-run logs.
**Current** — every saved book carries a `<name>.json.generation_quality.json`
recording `total_chunks`, `accepted_chunk_count` and, on newer runs,
`model_name`. Read across all 34 of them 2026-08-08
(`chunk_completion.py`), no inference required:

| model | books | chunks | completion | worst book |
|---|---|---|---|---|
| gemma-4-e4b-uncensored | 19 | 1313 | **100.00%** | 100% |
| *unrecorded* | 15 | 1586 | 25.28% | 1.0% |

**MET on the only model that can be attributed** — gemma-4-e4b completes every
chunk of every book, 1313 for 1313, against a 99% target.

**The development set is MET, but broader shipped-model reliability is OPEN.**
All four development books completed every chunk on qwen3-14b on 2026-08-10:

| book | chunks | completion |
|---|---|---|
| grimgar03 (run A) | 49/49 | 100% |
| grimgar03 (run B) | 49/49 | 100% |
| index18 | 81/81 | 100% |
| mushoku16 | 45/45 | 100% |
| owarimonogatari3 | 110/110 | 100% |

Two of these were ungeneratable 24 hours earlier. grimgar03 failed at chunk 1
of 49 because the repair logic refused a book for faithfully reproducing its
own repeated title; index18 was refused at the source gate over 6,662
replacement characters. Both are fixed, and grimgar03 was run twice because one
success does not distinguish a reliable book from a lucky one.

The unseen-book run on 2026-08-12 then found two new deterministic blockers:

| book | accepted chunks | result |
|---|---:|---|
| mushoku18 | 58/58 | written |
| mushoku23 | 120/120 | written, 11,387 entries |
| arc4_volume10wn | 132/154 | failed at chunk 133 |
| grimgar06 | 24/70 | failed at chunk 25 |

That is **334/402 = 83.1%** completion across the four unseen books, below the
99% target. Both failures exhausted the fixed retry policy and preserve valid
checkpoints. `arc4` repeatedly expanded a short repetitive passage until the
16,384-token ceiling; `grimgar06` repeatedly omitted parts of one passage even
after adaptive splitting. Retrying either unchanged is not a new measurement.
The current overall status is therefore **OPEN**: the development books are
100%, but the shipped model does not generalise that reliability to these
unseen formats.

**The qwen2.5-14b figures this goal used to quote are still not in the evidence
tree**, and the 15 historical failures remain unattributable - their manifests
record no model. That is now impossible for new runs, since both failure call
sites record `model_name`.

**The 15 failures cannot be attributed to any model, and that is the finding.**
All 15 have `status: failed` with `failure: chunk_failed_after_retries` after 8
to 33 attempts — genuine exhaustion, not interrupted runs. None records
`model_name`. All 15 response logs survive in `logs/responses/`, and none of
them names a model either: they record chunk, attempt, finish_reason and token
counts only. So the worst generation failures this app has ever produced are
permanently unattributable.

### 3.2 Every generated file is real audio

> **What this is.** Confirming that every audio file the app claims to have
> made actually exists and actually contains sound.
>
> **Why it matters.** The worst failures are the quiet ones. A missing or empty
> file that nothing complains about becomes a silent gap in the finished
> audiobook, discovered by a listener rather than by us.
>
> **Why this stays at zero.** All seven ways the app can produce audio were
> routed through a single checkpoint, so there is one place to verify rather
> than seven places to remember. A new generation path cannot bypass it without
> deliberately going around.

**Metric** — files that are absent, empty, or unreadable after generation.
**Probe** — `validate_generated_audio` in `app/audio_validation.py`, funnelled
through `_save_wav`.
**Current** — 0 known escapes since the funnel was added. **MET.**

**Target — 0. Any regression is a release blocker.**

### 3.3 One character, one voice

> **What this is.** Making sure two different characters never end up sharing
> the same voice by accident.
>
> **Why it matters.** This was a real bug with a memorable shape. The app
> stripped titles from names to help match them — so "Mr. Bennet" and
> "Mrs. Bennet" both became "Bennet", and a husband and wife were given one
> voice between them. It affected six books out of twenty-eight. Nothing
> errored. The audiobook was simply wrong, and only a listener would ever know.
>
> **Why this stays fixed.** Both directions are now tested, which matters
> because the first attempt at a fix broke the opposite case: "EMILIA" and
> "Emilia" are one character and *must* be merged, while Mr and Mrs Bennet are
> two and must not.

**Metric** — distinct roster entries sharing a voice through a name-matching
bug.
**Probe** — `app/test_generate_personas.py`.
**Current** — fixed. **MET.**

**Target — 0, with the couple case and the case-variant case both tested.**

### 3.4 Reproducible output

> **What this is.** Running the same job twice with the same settings should
> produce byte-for-byte identical audio.
>
> **Why it matters.** Without it, no comparison is trustworthy. If two runs
> differ on their own, there is no way to tell whether a change improved
> anything or the dice simply landed differently. Reproducibility is what makes
> every other number on this page mean something.
>
> **Why this is reachable.** It is already done for audio. Voices were being
> generated without a fixed starting seed for 70 of 71 characters; each now
> derives a stable one from its own name. The same discipline needs extending to
> the text side.

**Metric** — identical seed and inputs produce byte-identical audio.
**Probe** — waveform SHA-256 comparison.
**Current** — seeded generation confirmed deterministic; `character_voice_seed`
now derives a stable per-character seed. **MET for TTS.**

**Target — extend to the LLM path: same seed, same model, same script output.**

---

## 4. Speed and cost

### 4.1 Faster than real time

> **What this is.** How long the app takes to produce audio, compared with how
> long that audio lasts.
>
> **Why it matters.** It is the number a user actually feels. Right now a
> 10-hour audiobook costs roughly 10 hours of computer time — start it and come
> back tomorrow. Getting comfortably below 1.0 is the difference between
> "overnight" and "over lunch".
>
> **Why this is reachable.** Two of the three languages are already at 0.97–0.98
> and English at 0.91, so the target is a modest tightening rather than a
> redesign.

**Metric** — generation seconds ÷ audio seconds.
**Current** — median **0.91x / 0.98x / 0.97x**, slowest 1.21x (n=300 per
language, RX 9070 XT). **MET, barely.**

**Target — median ≤ 0.90x, worst case ≤ 1.5x.**

### 4.2 Local should not need the cloud

> **What this is.** Keeping the version that runs on your own machine roughly as
> good as the version that rents a much larger computer.
>
> **Why it matters.** Cloud runs cost money per hour and send your book to
> someone else's computer. If local is nearly as good, that is a real choice
> rather than a compromise — and the app has no dependency it cannot survive
> losing.
>
> **Why this is reachable.** It is already true: local is at 97–99% of cloud on
> all four books. This goal exists to *defend* a property already held, because
> properties like this are usually lost by accident rather than by decision.

**Metric** — best local accuracy ÷ best cloud accuracy, per book.
**Current** — 97.2% / 98.7% / 97.5% / 99.0%. **MET.**

**Target — hold local within 5% of cloud on every book.**

---

## 5. Text handling

### 5.1 Nothing unspeakable reaches the TTS

> **What this is.** Catching characters that have no spoken form before they
> reach the voice engine — things like `■`, `♪`, `∞`, or Chinese/Japanese
> characters embedded in English text.
>
> **Why it matters.** Nobody knows what the engine does with them. It might skip
> them, mangle them, or emit noise. There is currently no check at all, so
> whatever happens, happens silently.
>
> **What the first count found.** Counting the eight source books on 2026-08-06
> gave a partial answer. Chinese or Japanese characters appear inside otherwise
> English text in **five of eight books**, between 23 and 779 times each. And
> one book, index18, turned out to be **1.4% corrupt** — 6,662 "unknown
> character" marks left behind by a bad text conversion. The app already refuses
> that book at the door, which is correct, and is why it does not appear in the
> three-pass comparison.
>
> **What is still uncounted.** The source files are only the front door. Nobody
> has yet counted what reaches the *voice engine* after all processing, which is
> where the damage would actually occur. That is the measurement still owed.

**Metric** — characters passed to TTS with no spoken form.
**Probe** — source-level count is a script over the input `.txt` files; the
TTS-level count does not exist yet.
**Current** — **measured at the TTS boundary 2026-08-08**
(`tts_boundary_audit.py`, over `normalize_for_speech`'s output for all 48
saved books, 98,134 lines):

| | |
|---|---|
| unspeakable characters in raw script text | 56 |
| removed by normalization | 56 |
| **reaching the TTS engine** | **0** |

**Target — count what reaches the engine, then drive it to 0 with a
verbalization pass. MET, with one exclusion stated below.**

**The zero was not proof the code was sound.** Measured the same day, these
passed through `normalize_for_speech` unchanged: `♪` `∞` `★` `→` `♥` and
`U+FFFD`. Only `■` was handled, and only because it sits in `SPEECH_BREAKS`.
The audit read 0 because the 48 saved books do not happen to contain the
others — index18's 6,662 U+FFFD are refused by the *source gate* before the
book is ever saved. **The protection was the gate, not the normaliser**, and
anything reaching `scripts/` by another route had none.

`verbalize_symbols` now closes it: named symbols are spoken (`∞` → "infinity",
`→` → "to", `×` → "times"), and anything in an unspeakable Unicode category —
So, Sm, Sk, Co, Cn, Cs, plus U+FFFD — is dropped and **recorded** in the
transformation list rather than removed silently. Currency is deliberately
excluded, since `$` and `£` are speakable. 11 tests in
`test_speech_verbalization.py`.

**The exclusion: pictographic kana are not covered and cannot be, by this
method.** `へ` used as a drawing of a mouth is category Lo — a letter — and
identical to `へ` the particle. Any rule that dropped it would delete Japanese
text. Catching that needs context, not character class, and is not attempted
here.

### 5.2 Names pronounced consistently

> **What this is.** A dictionary telling the voice engine how to say unusual
> character names, so a name sounds the same on page 1 and page 300.
>
> **Why it matters.** Inconsistent pronunciation of a main character's name is
> the kind of flaw a listener cannot stop noticing once they have noticed it.
>
> **Why this is reachable, and the trap already avoided.** The machinery is
> built and tested; the dictionary just ships empty. Two traps were handled in
> advance. First, capitalisation matters: in one book "Felt" is a character and
> "felt" is an ordinary verb, appearing 242 and 65 times, and respelling the
> name must not touch the verb. Second, nicknames record *identity*, not
> *sound* — the app knows "Betty" is "Beatrice", but saying "Beatrice" aloud
> where the book wrote "Betty" would put a word in the audio that is not in the
> book. That is worse than mispronouncing it.

**Metric** — character names spoken the same way across a book.
**Probe** — `app/pronunciation.py`, `pronunciation.json` (ships empty).
**Current** — infrastructure exists, lexicon empty. **BASELINE TAKEN 2026-08-08** (`name_consistency.py`, 48 saved books):

| | |
|---|---|
| character names appearing in prose | 454 |
| spelled more than one way, beyond capitalisation | **23 (5.07%)** |
| of those, covered by the pronunciation lexicon | **0** |
| lexicon entries in total | 5 |

The dominant pattern is deliberate authorial syllabification — `Subaru` 1068
times against `Su-ba-ru` once, a stylistic stretch the engine will voice as
three separate syllables. Rare per book, but each instance is audible, and the
lexicon covers none of them.

**Capitalisation is excluded and that mattered.** Counting it gave 5x more
hits, all false: a character named Felt collides with every sentence-initial
"Felt", and the engine pronounces those identically. Only variation surviving
case-folding — an accent, a hyphen, a different letter — is counted.

**Measured on the prose, not the speaker labels.** The first version of this
counted variant `speaker` values and found 0 of 777, because those are
upper-cased and canonicalised upstream and cannot vary. That was a protected
surface, the same mistake as auditing TTS input rather than output (5.1).

**Target — a populated lexicon for the shipped demo book, and 0 substitutions
that alter a non-name word.**

### 5.3 Three-pass vs single-pass generation

> **What this is.** The app contains two different designs for reading a novel:
> the one that ships, and a more elaborate three-stage alternative that nothing
> currently uses.
>
> **Why it matters.** The second one has been carried along — with its own
> settings and instruction files — without anyone ever measuring whether it is
> better. It is either an unrealised improvement or dead weight, and right now
> nobody can say which.
>
> **Why this is reachable, and why either answer is fine.** It needs one fair
> comparison: both designs, same books, same settings, scored against the same
> answer key. Then it gets connected up or deleted. The goal is to *stop not
> knowing*. Carrying an unmeasured alternative forever is the only outcome that
> is not acceptable.

**Metric** — accuracy of `three_pass_generate.py` against the shipped single
pass, paired on line id.
**Probe** — `app/experiments/three_pass_vs_single.py`.
**Current** — **ANSWERED 2026-08-09.** Two books, both arms, qwen3-14b:

| book | single | three-pass | delta | comparable lines |
|---|---|---|---|---|
| mushoku16 | 45.5% | 40.3% | **−5.2** | 134 |
| owarimonogatari3 | 58.0% | 40.6% | **−17.5** | 143 |

**Three-pass loses on both.** Note the shape: three-pass sits at ~40% on both
books while single-pass ranges 45.5 to 58.0, which looks less like a method
that trails and more like one with a ceiling near 40% regardless of the book.

Three-pass is roughly **twice as fast** (40m against 76m on mushoku16, the one
book where both arms were timed in the same run). For an audiobook, where a
misattributed line is delivered in the wrong character's voice, 5 to 17 points
of accuracy is not worth halving the wall time. **Do not ship three-pass for
accuracy.**

**Getting the second book required a settings change, not a code fix.**
owarimonogatari3's three-pass arm aborted at 38m on one unattributable
one-entry batch, because `three_pass_generate` defaults to
`on_exhaustion='fail'` — correct for surfacing a failure rate, wrong for an
accuracy comparison. Re-run with `fallback` (production behaviour, unresolved
spans become UNKNOWN) it completed all 3929 entries in 63 minutes.

**Scope:** two Japanese light novels in translation. Goal 1.3 established that
this is the project's narrowest evidence base, and nothing here escapes it.

**Target — one clean comparison, then wire it in or delete it.**

---

### 5.4 Transcription and clip boundaries in the preparer

> **What this is.** Before any voice can be trained, an audiobook has to be
> transcribed and cut into clips. This measures how well that first step works.
>
> **Why it matters.** It is upstream of everything in section 2. A clip cut
> 30 ms early loses a consonant off *every* training sample; one cut 300 ms
> late drags in a word from the next line. Neither shows up as an error — it
> shows up later as a voice that never quite sounds right, with no way to trace
> the cause. Getting the words wrong is the more obvious failure and the less
> damaging one.
>
> **Why boundaries decide the choice.** Word accuracy is easy to compare and
> tempting to rank on. But the preparer needs the audio *sliced*, so a backend
> that hears perfectly and cannot place a boundary is useless to it.

**Metric** — WER (CER for CJK) against human transcripts, plus alignment error
against known boundaries.
**Probe** — `app/experiments/asr_backends.py`. The alignment probe concatenates
clips with 0.5 s gaps, so boundary truth is *arithmetic* — the only answer key
in this project that cannot itself be wrong.
**Current** — 50 clips per language, 2026-08-06:

| lang | backend | WER/CER | align median | within 300 ms |
|---|---|---|---|---|
| EN | whisper.cpp base | 3.0% | **80 ms** | **90%** |
| EN | whisper.cpp large-v3 | **1.8%** | 45 ms | 80% |
| EN | SenseVoice | 3.8% | 10158 ms | 20% |
| ZH | whisper.cpp base | 44.3% | **84 ms** | **100%** |
| ZH | whisper.cpp large-v3 | 14.1% | 826 ms | 20% |
| ZH | SenseVoice | **11.3%** | 17957 ms | 10% |

**MET for English and Chinese. OPEN for Japanese.**

**Chinese is solved by splitting the two jobs** (`whisper_cpp_hybrid`, added
2026-08-08). base decides the boundaries, large-v3 transcribes inside each one,
and each model is used only for the axis it wins:

| ZH arm | CER | align median | within 300 ms | segments |
|---|---|---|---|---|
| base | 44.3% | **84 ms** | **100%** | 10/10 |
| large-v3 | **14.1%** | 826 ms | 20% | 11/10 |
| **hybrid** | **14.5%** | **84 ms** | **100%** | 10/10 |

CER ≤20%, median ≤150 ms, ≥80% within 300 ms — all three met. The 0.4-point
CER cost against large-v3 alone is the price of transcribing inside windows the
model did not choose, and it is small: large-v3's advantage does *not* depend
on picking its own boundaries, which was the stated way this idea could have
failed.

**Japanese does not benefit, and the reason is upstream of the hybrid.** Both
checkpoints already fail to segment Japanese — base finds 5 boundaries out of
10 (median 2833 ms), large-v3 also 5 (4463 ms). The hybrid takes base's
segmentation, so it inherits that failure exactly: 25.2% CER, 2833 ms, 10%
within tolerance. Better transcription cannot repair boundaries that were never
found, and no combination of these two checkpoints will fix Japanese. That
needs a different segmenter, not a different pairing.

**A note on reproducing this.** The first run of the comparison omitted
`--build`, silently scored a different clip set, and produced base 58.0% /
large-v3 67.0% — which read as "the goal's table does not reproduce". It did
reproduce, exactly, once the same build was passed. Always pass
`--build ab_test_runtime/<corpus>_eval/build.json`.

**Target — CER ≤ 20% in every language, with alignment median ≤ 150 ms and
≥ 80% of boundaries within 300 ms.**

**The finding that shapes the choice: no single configuration wins both.**
`base` is the better aligner (84 ms, 100% in Chinese); `large-v3` is the better
transcriber (44.3% → 14.1% CER) and **ten times worse at boundaries**.
SenseVoice is the best Chinese transcriber of all and effectively cannot
segment — 1 of 10 boundaries, a 17-second median error.

So the reachable path is probably not to pick one. Words and boundaries can
come from different passes, and the failure modes are exactly complementary.
That hybrid is untested; nothing measured rules it out.

## 6. Measurement integrity

Goals about the instruments themselves. These earned their place by failing.

> **Why a whole section on this.** Every number above is only worth what the
> thing that produced it is worth. A broken ruler does not announce itself — it
> just quietly reports plausible numbers that are wrong, and those numbers get
> believed and acted on. Each goal here exists because a measurement was
> trusted that should not have been.

### 6.1 A ceiling must bound its arms

> **In short.** If the "best possible score" comes out lower than a score
> something actually achieved, the test is broken and must say so out loud
> instead of printing a tidy table. This is now automatic.

Covered at 2.2. Enforced by `find_invalid_anchors`, reported in every score
artifact as `anchor_invalid`.

**Target — 0 comparisons published from an eval set with an invalid anchor.
MET 2026-08-08**, now asserted rather than observed.

`test_score_anchor.py` pins the *detector* against constructed inputs.
`test_published_anchors.py` pins the *artifacts*: it walks every
`*_score.json` in the evidence tree and asserts each records
`anchor_invalid`, that it is empty, and — re-derived from the summary rather
than trusting the recorded flag — that every arm scores below its ceiling. A
correct detector nobody reads is exactly the 2026-08-06 failure, so the goal
needed a check over real files, not fixtures.

### 6.2 One source per decision

> **In short.** Any setting written down in more than one place will eventually
> disagree with itself, and the disagreement will be silent. The training-speed
> dial was written in four places; three said one thing, one said another — and
> the button in the interface used one of the wrong ones. That is how the
> babbling voices in 2.3 reached users' hands.
>
> **Why this is reachable.** Each case is small and permanent once fixed: write
> the value once, have everything else refer to it, and add a test that fails if
> a copy reappears. Two done, one known outstanding.

**Metric** — settings defined in more than one place.
**Current** — two found and fixed: the training learning rate and
`is_remote_llm`. One outstanding: `config["llm"]` versus `config["llm_local"]`,
which cost an hour on 2026-08-06 when a run dialled a dead endpoint while a
working server sat idle. **OPEN.**

**Target — 0 known parallel definitions; each new one gets a test that asserts
the copies agree.**

### 6.3 Indexes describe committed state

> **In short.** The record of results must be reproducible by someone else on a
> fresh copy. One index was quietly built partly from files that only existed on
> this machine, so it looked perfect here and was permanently wrong everywhere
> else.

**Metric** — index checks passing on a clean checkout.
**Current** — **MET**, after `collect_results.py` was found scoring gitignored
files and stamping rows with file mtimes.

**Target — every index check passes from a fresh clone with no untracked
files.**

### 6.4 No skipped tests

> **In short.** A test that skips is not a test that passes, and counting it as
> one is how a fault hides. Three tests here were skipping quietly. Rewriting
> them so they could not skip immediately exposed a genuine bug in the very
> thing they were meant to be checking — the skip had been covering it.

**Metric** — tests skipped in the release verifier.
**Current** — 0. **MET.**

**Target — 0. A skip is a failure.**

---

## Priority

If only three things get worked on:

1. **Generalisation (1.3)** — the app scores 71.0% on 25 PDNC novels it has
   never seen against 83.6% on the three it quotes, a −12.6 point gap against
   a target of 5. The three quoted books rank #2, #8 and #9 of 28. This is now
   the largest known overstatement in the document.
2. **Per-line duration spread (2.4)** — the narrator-controlled Japanese clone
   median is 0.927 and meets the goal, disproving the earlier cross-reader
   0.758 diagnosis. However, 43% of individual Japanese clips remain outside
   the band, similar to the other language arms.
3. **Train/val contamination (2.7)** — 60 of 75 shipped adapters trained on
   their own val split. The trainer is fixed; the library is not, and every
   held-out score from a contaminated adapter is an upper bound.

**Selection (1.2) was #1 on this list until 2026-08-08 and is now MET** — the
29.9% it was built on came from a model that does not ship. Re-measuring goals
before working on them has now twice been worth more than working on them.

Then: the three-pass baseline (5.3), and the CJK transcription gap (5.4) if
Voice Lab is ever pointed at a non-English audiobook. Reliability 3.1 is also
OPEN again on unseen books: diagnose the fixed failures at arc4 chunk 133 and
grimgar06 chunk 25 before spending another run on either unchanged.

## Rules for changing this file

- A current value moves only with an artifact and a date.
- A target moves only with a stated reason.
- Never delete an OPEN goal because it proved difficult. Convert it, or record
  why it was abandoned.
- Do not add a target without evidence that it is reachable. `NO BASELINE` is a
  respectable status; an invented number is not.
