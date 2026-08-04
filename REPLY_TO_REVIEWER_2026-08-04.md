# Reply to the reviewer — 2026-08-04

**Updated after your follow-up: all nine were correct.** I disputed the
instruction figures and I was wrong — there were two artifacts and I derived
from the unseeded one. Details below, along with the four follow-up items.

Full detail in `RECOMMENDATIONS_RESPONSE_2026-08-04.md`; commits `7aea006`,
`aa4dd3c`, `2142a32`. Suite 1,197 → **1,216 tests, 0 failures**.

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

## The one I disputed — I was wrong, you were right

I disputed your instruction figures and showed a derivation to back it. **There
are two artifacts and I derived from the wrong one.**

| file | written | per_line | per_char | none |
| --- | --- | ---: | ---: | ---: |
| `instruct_value.json` | 11:13 | 1 error | 1 | 2 |
| `instruct_value_seeded.json` | 14:56 | **3 errors, 0.611%** | **0, 0%** | **0, 0%** |

Your numbers are exactly the seeded file. Mine were the **unseeded** run —
quoted in the *seeded* column of a table whose entire subject is what changed
once generation was seeded. That is the same class of error as the two you
already caught: confident, derived, reproducible, and about the wrong object.

Worse than a wrong number, the direction reverses. Unseeded reads as three
near-identical arms, which is what I wrote. Seeded, **per-line instructions are
the only arm with any error and the other two are perfect**. The magnitude is
trivial either way — 3 words in 491, no failures anywhere — but "all three arms
are indistinguishable" was read off the contaminated file, and the seeded data
weakly suggests per-line is the *worst* arm for content fidelity.

Corrected in the work log and in `instruct_listening.py`'s docstring, which now
names both artifacts and warns which one looks plausible and is not.

I asked you to name the file if I was wrong. You did better by naming the
distinction, and the request I made — "if you re-ran it, that matters more than
either number" — was me constructing an interesting story to occupy the space
where I should have just looked for a second file. There was no third
possibility to chase. There were two files in one directory.

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

## #8 — the boundary is passed, and here is the evaluation you asked for

Retry policy untouched throughout, and your reasoning for why is right: changing
it mid-run would have made the before and after incomparable. Step **218 is
behind us**.

| | |
| --- | --- |
| step | 247 / 1250, 11.06 s/it |
| OOMs | **0** |
| tracebacks / retry attempts | 0 / 0 |
| checkpoints | 100 and 200 on disk, 748 MB each, `save_total_limit=2` holding |
| loss | 0.335 → **0.042** over 200 steps, monotone after the warmup |
| grad_norm | 1.89 → 0.34–0.61, stable, no NaN |
| lr | 9.57e-5, on the cosine schedule |
| epoch | 0.32 of 2 |
| VRAM / util / temp | 34.8 GB, 66%, 78 °C |
| cost | $51.20 accrued this period against $70 credit; $0 due |

So: no corruption, no repeated hard failures, no runaway cost, nothing unsafe.
**The fragmentation diagnosis holds** — same data, same step, same batch order,
and the only change was `expandable_segments:True` plus mid-epoch checkpoints.
Continuing.

One caveat on my own reading. Passing 218 once is consistent with the
fragmentation account but does not prove it; allocator behaviour is
history-dependent, and a different arena layout could have moved the failure
rather than removed it. What would settle it is completing the epoch without an
OOM anywhere, not clearing one step. I am recording that distinction now rather
than declaring victory at the boundary.

Also worth flagging against your "evaluate output samples" point: **there is no
output sample to evaluate yet.** Training loss is not a result — the adapter is
scored cross-book by `distill_eval.py` against four gold sets afterwards, and
0.042 training loss on 25 PDNC novels tells us nothing about transfer. I would
rather say that than offer the loss curve as evidence of quality.

---

## Your four follow-up items

**1. Correct the instruction section and distinguish the artifacts.** Done, and
the correction is above — you were right, I was wrong. Both files are now named
wherever the numbers appear, including in the script docstring, so the
plausible-looking wrong one carries a warning.

**2. Extend the machine-path guard to relevant tracked experiment scripts.**
Done, and you were right that it was too narrow. It listed three directories and
therefore could not see `ab_test_runtime/pipeline_repeats/score_repeats.py`,
which carried the same hard-coded root. "Directories I thought of" is the same
failure mode as the manual sweep the test was meant to replace. It now
enumerates every tracked `.py` via `git ls-files` — **283 files, up from 265** —
with a directory-walk fallback for an exported tree.

**3. Deployment identity.** Built. `gpu_job.sh` writes an `IDENT` line between
`QUEUED` and `START`:

```
IDENT  nonprose_mechanism commit=5a6ab6b tree=dirty:59f53551734e
       gpu_job_sha=dd2d055c1923 host=mitch-linux gpu=AMD Radeon RX 9070 XT
       cmd=...
```

`gpu_job_sha` is the field that would have caught the cloud box running a
superseded copy for hours. Identity is evidence, not a gate: with `git`,
`nvidia-smi`, `rocm-smi`, `sha256sum` and `hostname` all absent it degrades to
`unknown` and the job still runs, which is tested — a provenance record that can
refuse to start work would be worse than none.

Four tests cover it: ordering before `START`, the required fields, that the
logged hash equals the actual hash of the running script, and the degradation
path.

Staged to the instance as `gpu_job.next.sh` rather than swapped in. `gpu_job.sh`
is mid-execution there running the training chain, and bash reads a script
incrementally — overwriting a running script corrupts it. That is not caution
for its own sake: I did exactly that to `tts.py` earlier today and left it
broken for two minutes.

**4. Continue through the retry boundary.** Done, evaluation above. Passed 218,
zero OOMs, policy untouched.

---

## Not done

- **Swapping the staged `gpu_job.next.sh`** on the instance once the training
  chain drains. Until then the box still runs a version without identity
  logging, which is precisely the gap this closes — worth stating rather than
  counting #3 as fully deployed.
- **The rest of #9.** Untracked audio directories and logs are still
  undifferentiated from durable evidence.
- **The per-adapter pitch distribution in #7**, scoped and not started.
- **`clone_vs_lora` and `voice_data_saturation` have not been re-run** since
  being fixed to regenerate rather than reuse. Their published numbers came from
  the reusing version and should not be quoted until they are.
