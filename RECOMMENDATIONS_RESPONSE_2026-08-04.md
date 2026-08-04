# Response to RECOMMENDATIONS_2026-08-04.md

Eight of nine recommendations were correct and are acted on. One is factually
wrong about the artifact it cites, and I show the derivation rather than
asserting it.

Commit: `3d25e6d`. Suite: **1,212 run, 0 failures** (was 1,197).

The reviewer's framing is right and worth repeating: the remaining risk is not a
code defect, it is **knowing which code ran**. Two of today's failures were
exactly that — a cloud box running a superseded script, and a "completed" path
migration that had missed 41 references.

---

## Summary

| # | Recommendation | Verdict | Action |
| --- | --- | --- | --- |
| 1 | Finish the path migration | **correct, I was wrong** | done + enforced by test |
| 2 | Replace PID chains with one dispatcher | **correct, and it deadlocked while being reviewed** | done |
| 3 | Committed tests for `gpu_job.sh` | correct | 11 tests, mutation-checked |
| 4 | Record deployment identity before START | correct | accepted, not yet built |
| 5 | Correct two statements in the work log | **half right** | pitch fixed; instruct numbers disputed with evidence |
| 6 | Don't wire gender inference into production | correct | unchanged, still not wired |
| 7 | Store a seeded distribution, not one `mean_f0` | correct | accepted, scoped |
| 8 | Keep the retry policy unchanged through 218 | correct | unchanged |
| 9 | Separate durable evidence from debris | correct | partly done |

---

## 1 — The path migration was incomplete, and I said it was finished

**Correct, and this is the one that matters most**, because the error was not
the missing paths — it was reporting the job done without checking.

`f28d61f` replaced the `REPO = "..."` literal in 77 files. **41 absolute
references survived**: 33 `sys.path.insert` lines, plus `closed_set.py`,
`two_by_two.py` and `profile_vram.py`. The reviewer found them the same day.

All now derive from `__file__`. The `sys.path.insert` lines use a self-contained
expression rather than referencing `REPO`, because in several files the insert
runs *above* the assignment — referencing `REPO` there would have raised at
import time.

**Three were worse than machine-specific.** `chinese_attribution.py`,
`quote_aware_chunking.py` and `japanese_quote_robustness.py` pointed at a Claude
session scratchpad whose path embeds a **session UUID**. The Chinese and Aozora
corpora existed only there, so those results were reproducible exactly until
that directory was cleaned. Moved to `ab_test_runtime/corpora`, overridable by
environment variable, with `PROVENANCE.md` committed so the data can be
reconstructed.

**`test_no_machine_paths.py` enforces it now.** A grep is what a person does once
and a test does every run — which is precisely why the first pass was described
as complete when it was not. The test found the last four itself, including one
I would have missed, and it is checked against the literal line that shipped in
77 files so it cannot silently stop matching.

---

## 2 — The PID chain deadlocked while this was being reviewed

**Correct, and it failed in the strongest possible way: during the review.**

`queue3` finished at **20:39:56**. Its tmux driver process stayed alive. `queue4`
and `queue5` were waiting on `kill -0 <that pid>`, so **the GPU sat idle for
twenty minutes** while two queues waited on a process that was never going to
exit.

That is the reviewer's exact objection — a second, independently maintained
concurrency system drifting from the real lock — demonstrated live rather than
argued about.

The remaining jobs now dispatch through `gpu_job.sh` with a real `flock`, the
same mechanism the instance uses. One dispatcher, one lock, both machines.

---

## 3 — Committed tests for the lock

**11 tests**, covering every case named plus two more: failed `flock`,
unopenable lock file, two-process serialisation, blocking rather than dropping a
queued job, interruption while waiting, exit-code propagation, queue-log
ordering, and misuse.

**Mutation-checked, and my first attempt at that check was itself wrong.** I
mutated the script with a regex, ran the suite, and got 6 failures — but
`test_a_failed_flock_refuses_to_run_the_command` **passed**. That is the tell:
the regex had broken the script outright rather than removing the gate, so the
tests were failing for the wrong reason and the one test that mattered was not
exercised.

Rewritten as a byte-faithful copy of what the cloud box actually ran, the result
is clean: **exactly the two gate tests fail, the other nine pass.**

---

## 4 — Record deployment identity before START

**Correct and accepted.** This would have caught both of today's operational
failures immediately — the stale cloud `gpu_job.sh` and the missing server
script — instead of after two dead jobs.

Not yet built. It belongs in `gpu_job.sh` between `QUEUED` and `START`: commit,
dirty-tree hash, SHA-256 of the script about to run, hostname, GPU, command,
environment fingerprint. The SHA-256 line alone would have exposed the stale
script, since I verified the `distill_train.py` patch that way today and it took
one command.

---

## 5 — Two statements in the work log

**The pitch point is correct and is fixed.** The table had been updated to the
32.4 Hz six-adapter median while the prose below still called the remaining
spread 48 Hz. Both numbers are real and measure different things — 48 Hz is one
adapter (`husky_tenor_30s_m_fantasy`, the only seeded measurement available when
that paragraph was written), 32.4 Hz is the median across six spanning the
pool's range. They are now distinguished explicitly, with a note that 48 was
above the median and so on the high side.

**The instruct numbers are disputed.** The review reports per-line 0.611% WER
with 3 errors, and per-character and none at 0% with 0 errors. The artifact does
not contain those values. Derived from
`ab_test_runtime/experiments/instruct_value.json`, unmodified since 11:13:

| arm | rows with errors | total errors | total words | micro-WER |
| --- | --- | ---: | ---: | ---: |
| per_line | `c7a20a0410d8` | 1 | 491 | **0.204%** |
| per_char | `33a1233dcb4a` | 1 | 491 | **0.204%** |
| none | `e736c1ab745d` | 2 | 491 | **0.407%** |

The stored `summary.wer` fields are 0.002036659877800407, 0.002036659877800407
and 0.004073319755600814 — exactly 1/491, 1/491 and 2/491. So 0.20 / 0.20 / 0.41
is what the artifact says, and "one word of difference" is the gap between the
2-error arm and the 1-error arms.

0.611% is 3/491, so the review's figure implies 3 errors on the per-line arm.
Only one row on that arm has any error and it has one. If the review is reading
a different file, naming it would settle this immediately — I would rather be
shown wrong than leave two numbers standing.

---

## 6 — Do not wire gender inference into production

**Correct, and unchanged.** `character_evidence.py` still opens with **NOT WIRED
INTO PRODUCTION** and no caller uses it.

The reviewer is right that capitalisation plus intervening-name detection is not
coreference, and I should not let today's improvement suggest otherwise. It
abstains more often than it did; that is the whole of the claim. The validation
list — titles, places, sentence-initial names, aliases, subordinate clauses,
coordination, ambiguous and non-human characters, across several books and
languages — is the right bar, and PDNC's 28 annotated novels plus the Chinese
and Japanese corpora now in `ab_test_runtime/corpora` make it buildable rather
than hypothetical.

Abstention over an unsupported assignment remains the policy.

---

## 7 — Store a distribution, not one number

**Correct, and it sharpens something I had only half-stated.** I reported that
declared `mean_f0` carries 12.9 Hz mean error against a 32 Hz requirement. The
reviewer's point goes further: a single global threshold ignores within-voice
variance, which this experiment itself measured as ranging from **14.4 Hz to
71.9 Hz** across six adapters. A voice with a 72 Hz range and one with a 14 Hz
range cannot share a threshold.

So the constraint as currently stated is too crude to wire in, independent of
the measurement error. If pitch is used, each adapter needs several standardised
lines retained as median, spread and confidence bounds, validated by paired
listening first.

Scoped, not started — it is only worth the GPU time if pitch is actually going
to gate casting, which is not decided.

---

## 8 — Keep the retry policy through step 218

**Correct, and unchanged.** The policy is one decision applied identically on
every attempt: on non-zero exit, resume from the latest checkpoint if one
exists, otherwise stop.

The reviewer's reasoning is the important half: a third failure at 218 under
`expandable_segments` would be evidence that the fragmentation diagnosis is
**also** incomplete, and the response is to investigate rather than to add
another retry or reinterpret the error. That is the same trap the first
diagnosis fell into — a mechanism asserted without reproduction.

---

## 9 — Separate durable evidence from debris

**Correct, and I demonstrated the problem while responding to it.** The corpora
above were staged into a commit by accident: 11 MB of downloadable public-domain
text into a repository whose history is already 6.4 GB. Backed out before push,
replaced with a `PROVENANCE.md` recording what the files are, where they came
from, their licences, and how to reconstruct them.

That is the shape of the rule: **commit the manifest, not the payload**, when the
payload is reproducible from a named source. Not yet applied to the rest of the
untracked audio directories and logs, which is the larger part of this
recommendation and is not done.

---

## What I take from this

The two errors worth naming are the same error twice: **claiming completion
without checking.** The path migration was reported as finished after I had
migrated one pattern and not looked for others; the queue was described as
correctly chained while it was deadlocked on a dead PID.

Both were caught by someone else reading the artifacts. Recommendation 4 —
recording deployment identity automatically — is the one that makes this class
of mistake self-revealing rather than dependent on a reviewer, and it is the
next thing to build.
