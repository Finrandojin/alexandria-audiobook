# Alexandria Audiobook — Quality Goals

What "good" means for this app, as numbers a script can check.

Every current value below is a measurement with a source you can re-run. Every
target is a commitment. Where there is no baseline yet, the goal is *to take
the measurement*, and it says so — an unmeasured target is a wish, and this
document does not contain wishes.

**Last updated:** 2026-08-06

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
**Current** — roster contains the right name **85%** of the time; the model
picks it **29.9%** of the time. **OPEN.**

**Target — selection ≥ 50% with roster recall held at ≥ 85%.**

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
**Current** — the PDNC evaluation was **contaminated**: 25 of 28 books were in
training. Held-out performance measured **−4.4** against the contaminated
figure. **OPEN.**

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

**Target — every eval set's anchor above all of its arms.** Re-scoring of all
three sets is what confirms it; until those artifacts are regenerated this is
**fixed in code, unconfirmed in evidence**.

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
**Current** — LoRA 0.92 / 0.95 / 0.95; clone 0.97 / **0.76** / 0.90. **OPEN.**

**Target — every arm within 0.90–1.10.**

The existing safety check catches runaway voices at 3.0x. It cannot see 0.76.

---

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
**Current** — mixed and model-dependent. In the 2026-08-06 run: mushoku16 9/9
clean, grimgar03 and owarimonogatari3 failed chunk 1 outright. **OPEN.**

**Target — ≥ 99% of chunks complete without manual intervention, on the
shipped model.**

**Important caveat.** Those current failures are on qwen2.5-14b, which is *not*
the model the experiments were run against (qwen3-14b). This number is not yet
a fair measure of the shipped path.

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
**Current** — **PARTIAL BASELINE.** Sources: CJK inside non-CJK text in 5 of 8
books (23–779 occurrences); index18 carries 6,662 U+FFFD and is refused by the
existing source gate. **At the TTS boundary: still NO BASELINE.**

**Target — count what reaches the engine, then drive it to 0 with a
verbalization pass.**

The source gate proves the app can refuse bad input. It does not prove nothing
unspeakable survives the journey to the speaker.

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
**Current** — infrastructure exists, lexicon empty. **NO BASELINE.**

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
**Current** — **NO BASELINE.**

**Target — one clean comparison, then wire it in or delete it.**

---

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

**Target — 0 comparisons published from an eval set with an invalid anchor.**

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

1. **Selection (1.2)** — the right answer is on the shortlist 85% of the time
   and gets chosen 29.9% of the time. The largest known headroom in the app.
2. **Chinese anchor (2.2)** — until the ruler is fixed, one third of the voice
   evidence cannot be read at all.
3. **Three-pass baseline (5.3)** — decide the fate of an entire subsystem that
   has never been measured.

## Rules for changing this file

- A current value moves only with an artifact and a date.
- A target moves only with a stated reason.
- Never delete an OPEN goal because it proved difficult. Convert it, or record
  why it was abandoned.
- Do not add a target without evidence that it is reachable. `NO BASELINE` is a
  respectable status; an invented number is not.
