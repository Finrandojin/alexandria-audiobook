# Reply to the reviewer — 2026-08-04

Thank you. Eight of the nine are correct and are done or scoped. One I dispute,
and I have put the derivation below rather than just asserting a number.

Full detail is in `RECOMMENDATIONS_RESPONSE_2026-08-04.md`; commits `7aea006`
and `aa4dd3c`. Suite went 1,197 → **1,212 tests, 0 failures**.

---

## Your framing was the most useful part

> The primary risk is no longer one obvious code defect. It is operational and
> experimental reproducibility: knowing exactly which code ran.

That is right, and both of my errors today were instances of it. Neither was a
bug in reasoning. Both were **claiming completion without checking**:

- I migrated one path pattern in 77 files and reported the path problem fixed.
  41 references survived.
- I described the local queue as correctly chained while it was deadlocked on a
  process that had already finished.

You found both by reading the artifacts. That is the part I want to make
unnecessary, which is why recommendation 4 is the one I think matters most.

---

## Where you were right, and what it cost

**#1, the path migration.** 41 absolute references remained — 33
`sys.path.insert` lines plus `closed_set.py`, `two_by_two.py`,
`profile_vram.py`. All now derive from `__file__`; the `sys.path` lines use a
self-contained expression because in several files that line runs *above* the
`REPO` assignment, so referencing `REPO` would have raised at import.

Three were worse than machine-specific, and I would not have found these
without going looking: `chinese_attribution.py`, `quote_aware_chunking.py` and
`japanese_quote_robustness.py` pointed at a Claude **session scratchpad whose
path embeds a session UUID**. The Chinese and Aozora corpora existed only
there. Those non-English results were reproducible exactly until that directory
was cleaned, and nothing would have announced it. Moved into the repo,
environment-overridable, with provenance committed.

Your second half — add a test — is the reason this will not recur.
`test_no_machine_paths.py` found the last four itself, including one I had
missed after two manual sweeps.

**#2, the PID chains. This failed while you were reviewing it.** `queue3`
finished at 20:39:56. Its driver process stayed alive. `queue4` and `queue5`
were waiting on `kill -0 <that pid>`, so **the GPU sat idle for twenty
minutes**. Your objection — a second, independently maintained concurrency
system that drifts from the real lock — demonstrated itself rather than needing
an argument. Everything now dispatches through `gpu_job.sh`.

**#3, the lock tests.** 11 of them, covering every case you listed. Worth
reporting: my first mutation check was wrong in an instructive way. I mutated
the script with a regex, saw 6 failures, and nearly called it verified — but
`test_a_failed_flock_refuses_to_run_the_command` had **passed**. The regex had
broken the script outright, so the failures were for the wrong reason and the
one test that mattered was never exercised. Against a byte-faithful copy of
what the cloud box actually ran, the result is clean: exactly the two gate
tests fail, the other nine pass.

**#9.** I staged 11 MB of corpora into a commit *while responding to your
recommendation about separating evidence from debris*. Backed out before push,
replaced with a provenance manifest. The rest of the untracked audio and logs
is not done.

---

## The one I dispute

You report the instruction artifact as per-line **0.611% WER / 3 errors**, and
per-character and none at **0% / 0 errors**.

`ab_test_runtime/experiments/instruct_value.json`, unmodified since 11:13, does
not contain those values:

| arm | rows with any error | errors | words | micro-WER |
| --- | --- | ---: | ---: | ---: |
| per_line | `c7a20a0410d8` (1 error) | 1 | 491 | 0.204% |
| per_char | `33a1233dcb4a` (1 error) | 1 | 491 | 0.204% |
| none | `e736c1ab745d` (2 errors) | 2 | 491 | 0.407% |

The stored `summary.wer` fields are `0.002036659877800407`,
`0.002036659877800407` and `0.004073319755600814` — exactly 1/491, 1/491 and
2/491. So 0.20 / 0.20 / 0.41 is what the file says, and "one word of
difference" is the gap between the 2-error arm and the 1-error arms.

0.611% is 3/491, so your figure implies three errors on the per-line arm. Only
one row on that arm carries an error and it carries one. Two possibilities I
can see: you read a different artifact, or you ran the experiment yourself and
got a different draw.

**If it is a different file, name it and I will correct the log.** If you re-ran
it, that is a more interesting result than either number — the run is seeded, so
two runs disagreeing would mean the seeding is not doing what today's
byte-identical waveform check says it does, and I would want to chase that
before anything else.

Your pitch point in the same item was correct and is fixed: the table had been
updated to the 32.4 Hz six-adapter median while the prose below still said
48 Hz. Both are real and measure different things; they are now distinguished,
with the note that 48 Hz was a single adapter and above the median.

---

## Where you moved my position

**#7.** I had reported that declared `mean_f0` carries 12.9 Hz mean error
against a 32 Hz requirement, and treated that as the limiting problem. Your
point is stronger and I had missed it: a single global threshold ignores
within-voice variance, which this same experiment measured ranging from
**14.4 Hz to 71.9 Hz** across six adapters. A voice with a 72 Hz range and one
with a 14 Hz range cannot share a threshold at all. So the constraint is too
crude to wire in *independently* of the measurement error. Scoped as a
distribution per adapter, not started — it is only worth GPU time if pitch is
actually going to gate casting, which is undecided.

**#6.** Agreed, and unchanged: still not wired, still opens with a NOT WIRED
INTO PRODUCTION notice. You are right that capitalisation plus intervening-name
detection is not coreference and I should not let today's improvement imply
otherwise. It abstains more often than it did — that is the entire claim. Your
validation list is the right bar, and PDNC's 28 annotated novels plus the
Chinese and Japanese corpora now make it buildable rather than hypothetical.

---

## #8, in flight as of writing

Policy unchanged, exactly as you recommend. Training is at **step 199/1250,
zero OOMs, one checkpoint written** — step 218 is roughly three minutes away.

Your reasoning about what a third failure would mean is the part I have written
down: it would be evidence that the fragmentation diagnosis is *also*
incomplete, and the response is to investigate rather than add another retry or
reinterpret the error. That is the same trap the first diagnosis fell into — a
mechanism asserted without reproduction — and it is worth being explicit that a
resumable failure is not a solved one.

---

## Not done

**#4, deployment identity, is the one I think is most valuable and it is not
built.** It belongs in `gpu_job.sh` between `QUEUED` and `START`: commit, dirty
-tree hash, SHA-256 of the script about to run, hostname, GPU, command,
environment fingerprint. It would have caught both of today's operational
failures — the stale cloud `gpu_job.sh` and the missing server script — at the
moment they happened rather than after two dead jobs. The SHA-256 line alone
would have exposed the stale script; I verified a patch that way today and it
took one command.

Also outstanding: the rest of #9, and the distribution work in #7.
