# Recommendations after the 2026-08-04 work log

## Overall assessment

The work is substantially stronger after the review. Claude responded well to
contradictory evidence, corrected earlier explanations, centralized stale-audio
protection, and reran contaminated experiments instead of defending them.

The primary risk is no longer one obvious code defect. It is operational and
experimental reproducibility: knowing exactly which code ran, ensuring only one
GPU job ran, and preserving enough evidence to reproduce each conclusion.

## Recommendations, in priority order

### 1. Finish the path migration before creating another cloud instance

Commit `f28d61f` replaced the `REPO` assignments, but many scripts still contain
hard-coded `/home/fakemitch/.../app` paths in `sys.path.insert`. Absolute project
paths also remain in `closed_set.py`, `two_by_two.py`, and runtime scripts.

The work log's statement that 77 scripts were fixed is therefore incomplete.
Use the derived `REPO` and `APP` paths consistently, then add a test that rejects
machine-specific repository paths in executable source.

### 2. Replace PID-chain queues with one local GPU dispatcher

Several tmux queue-driver shells remained alive during inspection. The work log
also records a collision caused by waiting on the wrong process.

Use `gpu_job.sh` as the single dispatch mechanism locally and remotely. PID
chains and `pgrep` waits create a second, independently maintained concurrency
system and can drift from the real lock.

### 3. Add committed behavioral tests for `gpu_job.sh`

Lock failure was reproduced on both machines, but the safety behavior still
lacks a repository test. Cover:

- failed lock-file creation;
- failed `flock` acquisition;
- two-process contention and serialization;
- wrapped-command failure and exit-code propagation;
- queue-log ordering;
- interruption while waiting for the lock.

This safety mechanism is too important to depend on manual verification.

### 4. Record deployment identity automatically

Before every queued job starts, record:

- Git commit;
- dirty-tree status or patch hash;
- executable-script SHA-256;
- hostname and GPU;
- command and relevant environment fingerprint.

This would have exposed the old cloud `gpu_job.sh` and missing server script
immediately. The record should be written before `START`, not reconstructed
after a failure.

### 5. Correct two statements in `WORK_LOG_2026-08-04.md`

The current instruction artifact reports:

- per-line: **0.611% WER, 3 errors**;
- per-character: **0% WER, 0 errors**;
- none: **0% WER, 0 errors**.

It does not support the logged 0.20% / 0.20% / 0.41% values or the statement
that there was one word of difference.

The pitch section also updates the six-adapter median spread to 32.4 Hz but
later calls the remaining spread 48 Hz. The document should distinguish the
earlier single-adapter measurement from the six-adapter median.

### 6. Do not wire narration-based gender inference into production yet

The revised heuristic is more honest, but capitalization and intervening-name
detection are still not coreference.

Before it changes casting, validate it against labeled characters across
several books and languages. Include titles, places, sentence-initial names,
aliases, subordinate clauses, coordination, ambiguous characters, and
non-human characters. Continue treating abstention as preferable to an
unsupported gender assignment.

### 7. Re-profile pitch only if it will become a casting constraint

If pitch is used, store a seeded distribution per adapter rather than one
`mean_f0`. A single global 32 Hz threshold ignores within-voice variance,
register, utterance content, and measurement error.

Use several standardized lines per adapter and retain the median, spread, and
confidence bounds. Validate the constraint with paired listening before wiring
it into allocation.

### 8. Keep the current cloud retry policy unchanged through step 218

At inspection time, training had written checkpoint 100 and was around step
120. Step 218 is the repeated failure point.

If it fails there again, resume once from the checkpoint under the already
selected allocator policy. A repeated failure under the same conditions is
evidence that the fragmentation diagnosis is incomplete; investigate it rather
than changing error interpretation or adding another retry policy.

### 9. Separate durable evidence from runtime debris

The worktree contains many untracked JSONL files, audio directories, and logs.
Commit the small manifests and results needed to reproduce conclusions. Ignore
or deliberately archive large transient audio and logs.

Without that separation, a future reviewer cannot reliably distinguish
intentional evidence from stale outputs or incomplete runs.

## Closing view

The strongest part of this work is the willingness to retract mechanism claims
when evidence changes. The next maturity step is to make deployment identity,
queue ownership, and artifact provenance automatic so that this rigor does not
depend on someone noticing discrepancies manually.
