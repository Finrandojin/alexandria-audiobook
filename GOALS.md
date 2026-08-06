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
`app/test_score_anchor.py`.
**Current** — Chinese ceiling 0.691 sits **below** both its arms (0.720, 0.765).
**OPEN.**

**Target — a valid Chinese anchor, or a documented reason the eval set cannot
produce one.**

Clip length is a **hypothesis, not a measurement** — Chinese medians 3.17 s
against English 7.33 s and Japanese 4.71 s. Until the truncation test runs, no
Chinese voice conclusion should be quoted.

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
