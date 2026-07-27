# Addendum: cloud runs, and a correction to §7.2

Date: 2026-07-27
Status: **untracked on purpose — merge into `BRIEF_2026-07-26_ATTRIBUTION.md`
once the run queue drains.**

This is a separate file rather than an edit to the brief because
`closed_set.py`'s contract sets `require_clean_tree: True`, and the validator
tests `git status --porcelain` across the **whole repository** at artifact-write
time. Editing any tracked file — the brief included — makes every artifact
written during that window refuse.

That is not hypothetical. The grimgar03 @98304 run below completed all three
arms and then refused to write, because a tracked file was edited mid-run
(§3). An untracked `.md` is explicitly exempt: `_git_state` passes
`--untracked-files=no`, and only counts untracked `.py` inside the harness
directory.

---

## 1. Correction: the context confound in §7.2 does not exist

§7.2 states that every "bigger model wins" result is confounded with "bigger
model was also given more context", and that separating them was the reason to
rent an A6000.

**That claim is wrong, and it was checkable from data already on disk.** The
`prompt_chars` field recorded in every closed-set artifact gives prompt size
directly:

| fixture | median prompt | max prompt |
|---|---:|---:|
| Grimgar 03 (1200 rows) | 706 chars ≈ **176 tokens** | 6051 chars ≈ 1513 tokens |
| Mushoku 16 (441 rows) | 896 chars ≈ **224 tokens** | 5322 chars ≈ 1330 tokens |

The local context window is 16384. The longest prompt this harness has ever
produced is roughly a tenth of it. **Context never binds**, so it cannot
confound any §6.1 comparison, and no result in the ledger needs re-examination
on those grounds.

The error is the same shape as the four in the memory note on stating results
no wider than measured: a plausible mechanism asserted without checking whether
its precondition held. The precondition was one field away.

## 2. What the A6000 actually measured, and why it still mattered

Because context does not bind, the 98304 run is not a context test. What it
turned out to be is a **cross-hardware comparability check**: identical weights,
identical prompts, temperature 0, on ROCm/RDNA4 (RX 9070 XT, 15.92 GB, 16384
ctx) versus CUDA/Ampere (A6000, 48 GB, 98304 ctx).

Mushoku 16, paired by gold ID, exact McNemar — artifact
`closed_set__mushoku16__qwen__qwen3-14b.json`, `validation: ok`, `dirty: false`,
commit `249d0d20`:

| arm | cloud @98304 | local @16384 | discordant | p |
|---|---:|---:|---|---:|
| open | 50.4% | 48.3% | 4 cloud / 2 local | 0.6875 |
| closed-6 | 38.8% | 36.7% | 1 cloud / 0 local | 1.0000 |
| closed-oracle | 66.9% | 66.0% | 2 cloud / 4 local | 0.6875 |

Grimgar 03, same comparison. The replacement artifact has now landed as
`closed_set__grimgar03__qwen__qwen3-14b.CLOUD-a6000.json`
(`validation: ok`, `dirty: false`). The preserved local artifact is
`closed_set__grimgar03__qwen__qwen3-14b.LOCAL-9070xt.json`:

| arm | cloud @98304 | local @16384 | cloud-only / local-only |
|---|---:|---:|---:|
| open | 61.3% | 60.8% | 8 / 6 |
| closed-6 | 60.8% | 60.5% | 3 / 2 |
| closed-oracle | 72.8% | 72.5% | 6 / 5 |

**Conclusion: no practically concerning environment difference has been
detected for this task.** The largest apparent gap, +2.1 points on Mushoku's
open arm, is a net two lines out of 139 and p=0.69. Across 400 Grimgar lines the
gap is under half a point.

This is strong operational evidence for comparability, not a formal
equivalence proof. A nonsignificant difference does not establish equivalence
unless an acceptable margin and an equivalence test were specified. The
observed paired differences are small enough to proceed with cloud-only model
screening, provided the claim remains “no material difference detected” rather
than “the backends are identical.”

The reviewer is right, and the error is one the brief's own handoff checklist
already forbids: *“statistical non-significance is not described as
equivalence.”* The original wording here said the environments “are
equivalent.”

A p-value alone still leaves the useful question unanswered — how large a real
difference could this data be hiding? Exact Clopper-Pearson bounds on the
discordant pairs, transformed to the paired difference (Mushoku 16, N=139):

| arm | observed diff | discordant | exact 95% CI |
|---|---:|---|---:|
| open | +1.4 pt | 4 cloud / 2 local | **[−2.4, +3.9]** |
| closed-6 | +0.7 pt | 1 cloud / 0 local | [−0.7, +0.7] |
| closed-oracle | −1.4 pt | 2 cloud / 4 local | **[−3.9, +2.4]** |

So the defensible statement is bounded rather than binary: **on the Mushoku
fixture a
true cross-environment difference larger than about 4 points in either
direction is excluded at 95%; anything smaller is not.** That is sufficient for
screening models whose differences of interest are larger than 4 points, and
insufficient for adjudicating the many results in this project that sit inside
a 2-point band.

The reviewer is right that Grimgar's bound must be computed rather than
borrowed. Computed from its own artifacts — both `dirty: false`, cloud endpoint
`…thundercompute.net/v1` at 98304, local `localhost:1234/v1` at 16384 — over
N=400:

| arm | observed diff | discordant | p | exact 95% CI |
|---|---:|---|---:|---:|
| open | +0.5 pt | 8 cloud / 6 local | 0.7905 | **[−1.5, +2.3]** |
| closed-6 | +0.2 pt | 3 cloud / 2 local | 1.0000 | [−0.9, +1.1] |
| closed-oracle | +0.2 pt | 6 cloud / 5 local | 1.0000 | [−1.5, +1.8] |

Grimgar's larger row count roughly halves the interval: the widest bound is
**±2.3 points**, against Mushoku's ±3.9. The two books agree in direction (all
six arms nominally favour cloud, by 0.2–1.4 points) and neither approaches
significance.

Reporting both bounds rather than the tighter one: the honest screening
threshold is the **wider** of the two, ~4 points, until a fixture-specific
bound is computed for whichever fixture a given claim rests on. A model
difference under 4 points on Mushoku, or under ~2.3 on Grimgar, cannot be
attributed to the model rather than the environment on this evidence.

Note the observed diff here (+1.4 on open) differs from the +2.1 quoted from
the summary lines, because the paired computation uses only the 139 gold IDs
present in both runs while each run's own summary uses its own denominator.
The paired number is the correct one for a comparison.

This retires two hypotheses at once — the context confound (§1) and the
"hardware numerics floor" I proposed when the +2.1 first appeared. Neither
survives contact with the paired data.

It also produces something the cloud programme needs: **cloud-only model results
can be screened against the existing ledger without applying a correction
offset, so long as the differences being claimed are larger than the ±4-point
bound above.** That is what makes §5 worth running at all.

## 3. Two run failures, both instructive

**grimgar03 @98304 — complete results, refused artifact.** All three arms
finished. `manifest.write` then raised
`EnvironmentCaptureError: refusing to write an unverifiable artifact: tree had
modified tracked files: ['M app/experiments/closed_set.py']` — because the
retry patch (§4) was committed to that file *while the run was in flight*.
Execution was unaffected (Python had already loaded the module); only the write
failed. The validator behaved correctly. Re-running.

Operational rule this establishes: **no tracked file may be modified while any
closed-set run is in flight**, because the clean-tree check happens at write
time, not at start. A run can succeed completely and still produce nothing.

**mushoku16 @98304 — killed mid-run by a dropped endpoint.** The Thunder
forwarded port returned its "Nothing running here" page during `closed-oracle`;
the harness raised `NotFoundError` and exited after four minutes and two
completed arms, writing nothing. Fixed by a retry policy (§4). The recovered
run recorded `retries: max=0`, i.e. no drops recurred — the mechanism has been
unit-tested against simulated outages but has not yet fired in production.

## 4. Retry policy added to the closed-set harness

One policy, fixed before the first failure, per the repository's "decide before
you retry" rule:

- retry **availability** errors only — connection, timeout, 404, 429, 5xx —
  with capped exponential backoff, six attempts;
- never retry `BadRequestError`; a malformed request fails identically every
  time;
- 404 counts as availability, because a dropped tunnel and a wrong model name
  are indistinguishable at the call site, and the model is verified loaded
  before a run starts;
- the attempt count is recorded per row in the existing `retries` field.

Retrying is not intended as a quality resampling mechanism: decoding is
temperature 0 and the request is byte-identical on every attempt. In the usual
availability failure there is no answer to retain, so the retry recovers a lost
request. “Cannot change an answer” would be too absolute because backend
inference need not be bitwise deterministic, especially if a response was
computed but lost in transit.

Verified by behaviour against simulated failures: clean call → 0 retries / 1
call; single 404 → 1 retry / 2 calls; three drops → 3 retries / 4 calls;
persistent outage → raises after 6; `BadRequestError` → raises immediately.

## 5. Plan: what is queued

The selection rule changed once context was ruled out. The A6000 buys exactly
one thing: **weights that do not fit in 15.92 GB locally.** Nothing else.

| order | model | size | question it answers |
|---|---|---:|---|
| 1 | `qwen/qwen3-32b` | ~20 GB | scale within a family already measured on both books |
| 2 | `mistralai/magistral-small` | 14.33 GB | closes an existing hole — §6 records it as untested *solely* because 13.51 GiB would not fit 15.92 GiB safely |
| 3 | `google/gemma-3-27b` | 16.43 GB | is gemma-4-e4b's best-in-project end-to-end result the family, or that particular 7.5B model? |
| 4 | `llama-3.3-70b` | 42.52 GB | ceiling probe: does any runnable model reach usable quality |

Each runs the closed-set decomposition on both books **and the five reasoning
arms on Grimgar 03**, the arms having been added on the owner's instruction
after the §6a result. That is eight decomposition runs plus four arm runs.

Two of the four were already on disk before the queue started: the resolution
probes used to verify model ids completed `magistral-small` and `gemma-3-27b`
in full. The queue is therefore ordered by **availability** rather than by
interest, and treats presence on disk as equivalent to a download-complete
record — the probe-completed models never wrote a `DONE` line, and waiting for
one would have blocked the queue indefinitely behind the 32B while two ready
models sat idle on a billing GPU.

Estimated cost: **10–16 hours** of A6000 time, roughly **$5–8** at $0.44/hr.
The arms dominate it — `thinking` alone was 47 minutes on a 14B, and three of
the four queued models are larger. Cheaper scopings, if the budget matters more
than completeness: drop the `scaffold` arm (weakest signal, ~15% of arm time),
or run only baseline and `thinking` on the 70B.

Context is 16384 for the first three, matching local exactly. The 70B gets 8192
because 42.5 GB of weights plus 5 GB of KV at 16384 exceeds 48 GB. Since
prompts top out at 1513 tokens (§1), this is a VRAM accommodation and not an
experimental variable — but it is recorded in each manifest regardless.

Model ids were verified by resolution probe before queueing.
`llama-3.3-70b-instruct` and `meta-llama-3.3-70b-instruct` — the name carried in
the original plan — **do not exist**, and would have failed silently after a
four-hour wait.

## 6. Assessment

The honest summary of the cloud programme so far is that it has produced one
solid negative result and one useful piece of infrastructure, and that its
original justification was wrong.

The justification was wrong: context never bound, so the confound the instance
was rented to remove did not exist. The negative result is worth having anyway
— no material local/cloud difference was detected, bounded at roughly ±4 points
on this fixture, which is a precondition for screening any cloud-only model
number against the existing ledger and was not previously established.

**The larger reservation stands and should be recorded plainly.** All eight
queued runs use the closed-set decomposition, and §6.3 showed that instrument's
model ranking disagreeing with end-to-end output on the same book and fixture.
Adding four models adds four rows to an instrument under active suspicion. The
owner has been told this and elected to proceed on the grounds that verified
data is worth having regardless — a defensible call, since the runs are cheap,
the artifacts are validated, and a decomposition result is still evidence about
pass 2 under its declared inputs.

But the ordering in §10 of the brief should not change because of it: the
segmentation crossover that determines whether the decomposition means anything
remains worth more than any of these four models, and it runs on hardware
already owned.

## 5a. First cloud model: magistral-small, and a scoring defect it exposed

`closed_set__mushoku16__mistralai__magistral-small__thunder-a6000.json`,
`validation: ok`, `dirty: false`, A6000 @16384.

| model | open | closed-6 | oracle |
|---|---:|---:|---:|
| **magistral-small** | **52.5%** | 45.3% | 56.8% |
| qwen3-14b | 48.3% | 36.7% | **66.0%** |
| ministral-14b | 47.6% | 41.5% | 61.2% |
| phi-4 | 45.6% | 32.7% | 59.2% |
| gemma-4-e4b | 39.5% | 38.8% | 49.7% |
| qwen3.5-9b (shipped) | 35.4% | 34.7% | 49.0% |

Paired exact McNemar on the open arm: significantly ahead of the shipped 9B
(p=0.0031) and of gemma-4-e4b (p=0.0213); **unresolved against qwen3-14b
(p=0.5327), phi-4 (p=0.2559) and ministral-14b (p=0.5114)**. Its +4.2 over
qwen3-14b also sits inside the ±4-point cross-environment bound from §2, so the
lead should not be reported as a lead. The defensible statement is that
magistral-small joins the group of 14B-class models that beat the shipped
model and are not separable from each other on this fixture.

That closes the gap §6 of the main brief recorded: magistral-small was untested
solely because 13.51 GiB would not fit 15.92 GiB safely.

### 5a.1 The open/oracle inversion is a scorer defect, not a model property

magistral posts the best open arm and the second-worst oracle — the only model
that gains little from being handed the answer plus four distractors. Every
other model gains 10–18 points from the oracle set; magistral gains 4.3.

Diagnosis: **19 of its 60 oracle errors are answers that are not in the
five-name list at all**, against 0 off-list errors in its open arm. Inspecting
them, it is selecting the right character and spelling the name differently:

| expected | magistral produced |
|---|---|
| RUDEUS | RUDIUS, RUDIEUS, RUDUEUS |
| ALMANFI | ARUMANFI |

The fixture declares a `RUDEUS`/`RUDI` alias group; these variants are not in
it, so alias-aware scoring counts them wrong. Scoring near-miss spellings as
correct (SequenceMatcher ratio ≥ 0.75) changes the picture:

| model | arm | scored | spelling-corrected |
|---|---|---:|---:|
| **magistral-small** | oracle | 56.8% | **64.7%** |
| magistral-small | open | 52.5% | 52.5% |
| gemma-4-e4b | oracle | 49.7% | 51.7% |
| qwen3-14b | oracle | 66.0% | 66.0% |
| ministral-14b | oracle | 61.2% | 61.2% |
| phi-4 | oracle | 59.2% | 59.2% |
| qwen3.5-9b | oracle | 49.0% | 49.0% |

**The defect is model-specific.** magistral loses 7.9 points of oracle accuracy
to spelling; no other model loses more than 2.0, and three lose nothing. This
is the same class of error that cost 9.5 points project-wide before the
`RUDEUS`/`RUDI` alias fix, resurfacing as a per-model bias rather than a
global one.

Consequences that need deciding, not just noting:

1. Every cross-model comparison in §4 and §6.1 is affected to an unknown
   degree, because the penalty is not uniform across models. Models were
   compared on a metric that partly measures orthographic conformity to the
   fixture's chosen spelling.
2. It is not obvious this should be "fixed". In production a misspelled
   speaker name is a real failure — it fragments the cast list and breaks voice
   assignment — so 56.8% may be the correct *product* number while 64.7% is the
   correct *attribution-ability* number. These are different questions and the
   brief currently conflates them.
3. Why the effect appears only in the oracle arm is unexplained. The open arm
   presents a 17-name roster and magistral makes zero spelling errors; the
   oracle arm presents 5 names and it makes 19. A shorter list should make
   copying easier, not harder.

### 5a.2 Questions I cannot answer from this data

Recorded for the reviewer rather than resolved:

**ANSWERED — see §5a.3.** The owner's suggestion ("how would they sound if
spelled phonetically") identified the mechanism: these are not typos.

- **Should the scorer accept near-miss spellings?** If model choice is the
  decision, orthographic penalties are noise and should be normalised out. If
  shipping quality is the decision, they are real defects. My inclination is to
  report both numbers permanently rather than pick, but that doubles every
  table and may be worse than choosing.
- **Is a fuzzy threshold safe?** Ratio ≥ 0.75 merges `RUDIUS`→`RUDEUS`, but
  this corpus contains genuinely distinct short names, and a threshold that
  merges two real characters would silently manufacture accuracy. A
  roster-anchored rule (snap to the nearest attested name, only if unique
  within the candidate set) is probably safer than a global ratio, but I have
  not tested whether it introduces collisions on either fixture.
- **Why only the oracle arm?** I have no mechanism. If it is prompt-shape
  dependent it would affect production, where the roster is long — which is
  the open-arm condition where magistral spells correctly. That would be
  reassuring but it is an inference, not a measurement.
- **Does this change the model ranking?** Spelling-corrected, magistral's
  oracle goes 56.8 → 64.7, still below qwen3-14b's 66.0. So the headline
  ranking survives; what does not survive is the claim that magistral is
  unusually bad at conditional selection.

### 5a.3 Resolved for scoring: use a phonetic diagnostic, not fuzzy matching

The question in §5a.2 was whether a fuzzy string threshold is safe. It is the
wrong tool, and the right one falls out of noticing what the variants *are*.

These corpora are translated from Japanese. `ALMANFI` / `ARUMANFI` is
consistent with alternate romanization because Japanese does not distinguish
L/R in the same way as English and consonant clusters may acquire vowels.
`RUDEUS` / `RUDIUS` / `RUDIEUS` / `RUDUEUS` also has systematic phonetic
structure. That supports a rule-based diagnostic rather than an arbitrary
edit-distance threshold; it does not prove that every generated spelling is an
accepted canonical translation.

`romaji_key` (in `attribution_accuracy.py`, commit `dc87eed`) is **first vowel
+ consonant skeleton, with L merged to R**:

| pair | key | result |
|---|---|---|
| RUDEUS / RUDIUS / RUDIEUS / RUDUEUS | `U\|RDS` | merge |
| ALMANFI / ARUMANFI | `A\|RMNF` | merge |
| REIDA / RUDI | `E\|RD` vs `U\|RD` | distinct |
| ROXY / ROXIE | — | distinct |

The first vowel is load-bearing and was found by testing, not reasoning:
**dropping all vowels also merges every observed variant, but collides `REIDA`
with `RUDI`** — two distinct mushoku16 characters. That is precisely the
"silently manufacture accuracy" failure §5a.2 warned about, and the naive
version walked into it. The first vowel is stable under romanization while
medial vowels are not, so retaining it discriminates the real names and still
merges the variants. **Verified: zero collisions between distinct characters
across both fixtures' full name sets** (mushoku16 21 names, grimgar03 27).

Rescored, oracle arm:

| model | exact | phonetic | penalty |
|---|---:|---:|---:|
| **magistral-small** | 56.8% | **64.7%** | **+7.9** |
| gemma-4-e4b | 49.7% | 51.7% | +2.0 |
| qwen3-14b | 66.0% | 66.0% | 0.0 |
| ministral-14b | 61.2% | 61.2% | 0.0 |
| phi-4 | 59.2% | 59.2% | 0.0 |
| qwen3.5-9b | 49.0% | 49.0% | 0.0 |

**Both numbers are now reported, and `same_person` is unchanged.** That
separation is important: `romaji_key` is fixture-scoped analytical evidence,
not a production identity resolver. Zero collisions among the current 48
fixture names does not establish safety on future books or larger rosters.
The two
answer different questions and the brief should stop conflating them:

- **exact match is the product number.** A misspelled speaker fragments the
  cast list and breaks voice assignment, so `RUDIUS` really is a failure in a
  shipped audiobook.
- **phonetic match is the attribution-ability number.** It measures whether the
  model identified the right character, which is what a *model comparison*
  should be measuring.

`score_run` and `summarize` now carry `correct`, `correct_phonetic` and
`spelling_penalty`; the CLI prints the second line only when it differs, so the
common case stays a one-line answer but a model paying a romanization penalty
cannot be silently compared against one that is not.

Still unexplained (§5a.2 question 3): magistral produces **zero** variants
against a 17-name open roster and 19 against a 5-name oracle list. A shorter
list should make copying easier. Until that has a mechanism, the phonetic score
should be read as a correction of known size, not as evidence the underlying
behaviour is understood.

## 5b. Second cloud model: gemma-3-27b joins the leading group

`closed_set__mushoku16__google__gemma-3-27b__thunder-a6000.json`,
`validation: ok`, `dirty: false`.

| model | open | closed-6 | oracle |
|---|---:|---:|---:|
| **gemma-3-27b** | **55.4%** | 44.6% | 59.0% |
| magistral-small | 52.5% | 45.3% | 56.8% |
| qwen3-14b | 48.3% | 36.7% | 66.0% |
| ministral-14b | 47.6% | 41.5% | 61.2% |
| phi-4 | 45.6% | 32.7% | 59.2% |
| gemma-4-e4b | 39.5% | 38.8% | 49.7% |
| qwen3.5-9b (shipped) | 35.4% | 34.7% | 49.0% |

It was queued to test whether a larger Gemma candidate improves on
gemma-4-e4b. It does: **gemma-3-27b beats gemma-4-e4b by 15.9 points, 36
repairs to 15, p=0.0046.** This supports gemma-3-27b as the stronger tested
Gemma candidate. It does not isolate parameter scaling as the cause because
model generation, training data, tuning, and architecture also differ.

Against everything else, though, the pattern from §5a repeats exactly:

| gemma-3-27b vs | discordant | p |
|---|---|---:|
| gemma-4-e4b | 36 / 15 | **0.0046** |
| qwen3.5-9b | 38 / 13 | **0.0006** |
| phi-4 | 25 / 13 | 0.0730 |
| ministral-14b | 27 / 18 | 0.2327 |
| qwen3-14b | 28 / 19 | 0.2430 |
| magistral-small | 18 / 14 | 0.5966 |

**Five models are now bunched between 45.6% and 55.4% with no significant
separation between any of them.** Every one beats the shipped 9B; none beats
another. A 10-point spread on points that is statistically one group.

That is worth stating plainly because it is the third time today the same shape
has appeared: the ledger keeps producing ORDERINGS that do not survive a paired
test. The actionable content of the whole model programme remains what it was
this morning — *move off qwen3.5-9b* — and the choice among the rest is not
determined by this fixture.

On grimgar03, magistral-small scores open 61.3% / closed-6 58.0% / oracle
70.8%, against qwen3-14b's local 60.8% open. Same story, second book.

Note also gemma-3-27b's oracle (59.0%) sits below qwen3-14b's (66.0%) despite a
better open arm — the same open/oracle inversion documented in §5a.1 for
magistral, now in a second model. That strengthens the case that the inversion
is a property of the measurement rather than of one model, and it has not been
re-checked with `romaji_key` (§5a.3) for these two runs.

## 5c. What is running, and one harness change in flight

- **gemma-3-27b**: grimgar03 decomposition, then its five reasoning arms.
- **Then**: qwen3-32b (downloading), llama-3.3-70b (downloading), and
  magistral-small's arms, re-queued at the end (§5c.1).
- **Local card**: free. Crossover (§5, pre-registered) is next.

### 5c.1 magistral's arms were lost to a dropped tunnel

The Thunder endpoint dropped a second time, three minutes into
magistral-small's five-arm grimgar03 run, and the harness exited because the
retry policy from §4 had been added to `closed_set.py` **only**. That was
backwards: closed-set runs take 4–15 minutes, the arms take two hours, so the
arms are far the more exposed. Fixed in `13e9a77` with the identical policy,
deliberately not a variation. The arms are re-queued at the end of the sweep;
magistral's two closed-set artifacts already exist and validated, so the runner
skips them rather than spending A6000 time on work already done.

### 5c.2 TEMPORARY row-level checkpointing (in progress)

Two endpoint drops in ninety minutes is enough to say retry is not sufficient:
it covers a blip, not a tunnel that is gone for minutes, and the arms are
hour-scale runs. `ExperimentRecord` is being given `enable_checkpoint()` /
`done()` so any harness can resume mid-run — one mechanism in `manifest.py`
rather than three variants.

**The hazard is the design constraint.** Resume can silently merge rows
produced under different configurations, which is worse than losing a run
because the artifact still validates. A checkpoint is therefore adopted only if
experiment, model, endpoint, gold hash, **harness source hash** and decoding all
match; otherwise the stale file is moved aside, the run starts clean, and the
differing fields are printed. Including the harness hash means editing a
harness automatically invalidates its checkpoints. A completed `write()`
deletes its checkpoint so a later run cannot resume from finished work.

It is marked TEMPORARY with an explicit removal condition — when experiments no
longer run against an endpoint that can vanish — because a resumed artifact is
weaker evidence than a single-process one and should not quietly become the
norm.

## 5d. RDNA4 GPU patch: a null result, for an unexpected reason

Not attribution work, recorded here so it is not repeated. `tts.py` corrects
ROCm's under-reported compute-unit count for consumer RDNA cards; its lookup
table stopped at RDNA3, so on the RX 9070 XT (RDNA4) torch reported
`multi_processor_count=32` against `rocminfo`'s 64. An entry was added
(`aeeefed`) and A/B'd off/on/on/off on an idle card, five timed repeats each.

Medians: 8.58 / 8.60 / 8.09 / 8.52 s — on/off differences smaller than the
within-arm spread. No effect.

The reason is the useful part. The patch's `[RDNA fix]` line prints from inside
the patched function on first call, and **it never printed once across 24
generations**. Nothing in the TTS path calls
`torch.cuda.get_device_properties`. The patch changes what Python *reports*,
not what the HIP kernel launcher uses, so on this workload it cannot do
anything. Applied directly it does work (verified 32 → 64), so the entry's
values are right; it is simply inert here.

This says nothing about whether the RDNA2/3 rows help on their paths, and those
were left untouched. It does suggest the upstream idea targets code that
queries device properties, which this pipeline does not.

## 6a. Thinking reverses on the second book — CONFIRMED, paired

The run completed. Artifact
`reasoning_arms__grimgar03__qwen__qwen3-14b.json`, `validation: ok`, qwen3-14b
@16384. Paired exact McNemar against the same harness's own baseline, N=400:

| arm | Grimgar 03 | vs baseline | repairs/regressions | p | 95% CI |
|---|---:|---:|---|---:|---:|
| baseline | 56.5% | — | — | — | — |
| `because` | 54.8% | −1.8 | 41/48 | 0.525 | [−6.5, +3.1] |
| scaffold | 52.5% | −4.0 | 47/63 | 0.152 | [−9.2, +1.4] |
| **thinking** | **66.2%** | **+9.8** | **59/20** | **0.00001** | **[+5.4, +13.4]** |
| scaffold_thinking | 56.5% | +0.0 | 55/55 | 1.000 | [−5.3, +5.3] |

**`thinking` is the largest and most significant positive effect this project
has measured.** 59 repairs against 20 regressions; the interval excludes zero
by a wide margin.

### The factorial result was pre-declared, and it is the fourth outcome

The harness docstring listed four possible readings before the run. This is the
one it called "scaffold_thinking is worse — the questions constrain a model
that reasons better unprompted":

- thinking alone **+9.8** (p=0.00001); scaffold alone −4.0; both together
  **+0.0**
- thinking-only beats scaffold_thinking by **+9.8, p=0.00013**

**Scaffold cancels the entire benefit of thinking.** Asking the model the
judge's questions actively prevents it from reasoning well unprompted. That is
a specific, actionable negative result about prompt design, and it would have
been invisible without crossing the two factors.

### The two books reverse completely

| arm | Mushoku 16 | Grimgar 03 |
|---|---:|---:|
| `because` | **+10.8** (p=0.004) | −1.8 (n.s.) |
| thinking | +2.2 (n.s.) | **+9.8** (p=0.00001) |

The two arms swap places. Every conclusion in §5 came from Mushoku 16 alone.

One reason to weight the Grimgar result more than the Mushoku one: `because`
won on Mushoku against a **39.6%** baseline and then reversed to −7.2 on the
production path — the weak-baseline pattern. `thinking` wins here against a
**56.5%** baseline, i.e. with *less* headroom, which is the opposite pattern.

### Caveats

**The artifact records `dirty: true`** — `closed_set.py` and
`reasoning_arms.py` were modified at write time, because the retry fix was
being committed while this ran. The harness SHA-256 is recorded and execution
was unaffected, but it is not a clean-commit experiment and carries the same
caveat as the reexamine artifact in §4.

**Cost:** 2844 s against baseline's 454 s — 6.3× wall time, 140 260 reasoning
tokens. A confirmed +9.8 is a throughput decision, not a free improvement.

**Still the exploratory harness.** Per the standing rule, this nominates a
production-path `thinking` A/B on Grimgar 03; it does not rewrite the shipping
ledger. The paired exploratory effect is established; production transfer is
the remaining question.

One qualification, because it changes what that A/B should look for. The
`because` confound was an **asymmetry**: `because` altered the prompt and the
response schema, so it was compared against a baseline weakened by a different
prompt. `thinking` alters neither — both arms send the identical `BASE_SYS`
prompt and differ only in whether `reasoning_effort` is set to none — so it is
not exposed to that specific failure.

The exposure is different and still real: **headroom**. The exploratory
baseline is 56.5% where the production baseline on the other book was ten
points higher than its exploratory counterpart, and an intervention has more
room to help a weaker starting point than a stronger one. So the prediction to
test is not "the effect was an artefact of an unfair comparison" but "the
effect shrinks as the baseline rises." That is a quantitative question the
production-path A/B answers directly, and it is the reason the A/B must run on
the production prompt rather than a repeat of this harness.

### The pass-1 counter-observation

The cloud end-to-end run (§7) failed twice before running, and the second
failure is evidence on the same question. Omitting `--reasoning-effort none`
let qwen3-14b think during **pass 1**, and its reasoning tokens consumed the
completion budget until segmentation truncated at ~1476 tokens and failed on
chunk 24 of 99.

The provisional evidence is therefore that thinking **may help exploratory
pass 2 on Grimgar and can break pass 1 under the current completion budget**.
If the pass-2 result survives paired analysis and production-path testing, the
shipping question is not “thinking on or off” but per-pass configuration.

## 7. Run status at time of writing

- **Local reasoning arms**: complete; paired result in §6a.
- **Local RDNA4 CU-count A/B**: complete; null result in §5d.
- **Cloud whole-pipeline repeat**: stopped after 82% of pass 2; partial paired
  result in §7.0.
- **Cloud, model queue**: four models, each running the closed-set
  decomposition on both books *and* the five reasoning arms on Grimgar 03.

### 7.0 RESULT: one repeat exposes substantial per-line churn

The repeat completed 82% of pass 2 before pass 2 hard-failed on a one-entry
batch under `--pass2-on-exhaustion fail` (the same flag, and the same class of
failure, as the local run, which failed at 99.96%). Scored on the **291 gold
lines both runs attributed**:

| | local run | cloud repeat |
|---|---:|---:|
| accuracy on the shared 291 | 57.4% | 54.6% |
| net paired difference | — | **−2.7 pt**, p=0.3317 |
| exact 95% CI on the difference | — | **[−7.6, +2.4]** |
| **lines that changed answer** | — | **52 / 291 = 17.9%** |

Two things follow, and the second is the important one.

**Aggregate stability is moderate.** The net difference is −2.7 points and not
significant, so a pipeline score is roughly repeatable at the whole-book level.

**Per-line stability is poor.** Nearly one line in five received a different
speaker on a second run of the *same model on the same book*. The aggregate
looks stable only because the changes largely cancel.

Set the observed repeat difference against the two numbers the main brief leans
on, both from single pipeline runs on this book:

- §6.3's decomposition-versus-pipeline ranking gap: **5.5 points**
- the gemma-versus-qwen3-14b end-to-end gap: **5.5 points**

The observed churn makes the earlier single-run conclusions less secure, but
one pair of runs cannot estimate a run-level standard deviation or define a
95% “noise band.” The 291 lines are also not independent experimental
replicates: they share one book, prompts, and run state. Therefore the claims
that a single run cannot resolve differences below about five points, or that
the §6.3 disagreement is explained by sampling noise, are not yet established.

What is established is narrower: **one nominal repeat changed 17.9% of speaker
answers while its net accuracy moved only 2.7 points.** Aggregate stability can
hide substantial line-level instability. The §6.3 instrument disagreement and
end-to-end ranking should remain unresolved until genuine run-level replication
is available.

Caveats, stated rather than buried: this is one repeat, the cloud run covers
82% of pass 2 rather than 100%, and the two runs differ in environment as well
as sampling trajectory. Section 2 bounds the *net accuracy difference* between
environments under the temperature-zero decomposition; it does not bound
per-line prediction churn at temperature 0.6 and cannot show that most of the
17.9% churn came from sampling.

### 7.1 Why the cloud end-to-end run is a repeat, not a new model

The pipeline runs at **temperature 0.6**. Unlike the closed-set decomposition
at temperature 0, it is not deterministic, and **its run-to-run variance has
never been measured.** §6.3 of the main brief rests on a 5.5-point gap between
the decomposition ranking and the pipeline ranking on this exact book and
fixture. If repeating the same model on the same book moves the result by that
much on its own, the disagreement is not interpretable and the crossover in §5
is chasing noise. Local end-to-end for this pair scored 215/399 = 53.9%.

### 7.2 Two configuration defects found by running it

Both were mine, and both would have silently produced wrong or absent results.

1. **No SSH alias.** `ensure_ideal_settings` returns early on a remote endpoint
   with no `llm_remote_ssh`, so the pipeline could not read the remote context
   length, fell back to a conservative assumption, and truncated pass 1 at
   650–1500 tokens. `tnr-0` already resolved to this instance in
   `~/.ssh/config`; the isolated config now sets it. Verified:
   `get_remote_lmstudio_status('tnr-0', ...)` returns
   `context_length: 98304, parallel: 1, server_reachable: True`.
2. **Missing `--reasoning-effort none`.** The overnight local run used it
   (`run.sh:89`); the cloud invocation did not. Beyond causing the pass-1
   failure in §6a, matching flags is what makes this run comparable to the
   local result it is being compared against. An unmatched flag would have
   produced a difference attributable to the flag, not the environment.

The run uses an isolated `ALEXANDRIA_DATA_DIR` so the application's own
`app/config.json` is untouched.

### 7.3 Artifact-naming defect, and a near-miss

`closed_set.py` named artifacts by book and model only. The cloud grimgar03
run therefore wrote to the same path as the local one and **did overwrite it**;
the local artifact survived only because it had been copied aside minutes
earlier on suspicion. Both are now kept as
`closed_set__grimgar03__qwen__qwen3-14b.LOCAL-9070xt.json` and
`.CLOUD-a6000.json`, and both harnesses take `EXPERIMENT_TAG` (commit
`ae74558`). Any comparison drawn from artifacts written before that commit
should confirm which environment produced the file it read.

## 9. THE GATE PASSED: thinking survives the production path

Recommendation 6 asked for a production-path baseline-versus-thinking check
before paying to replicate across four larger models. It ran, on grimgar03,
qwen3-14b, through `attribute_batch` with the shipping prompt.
`because_production__grimgar03__qwen__qwen3-14b__local.json`, **`validation:
ok`, `dirty: false`**:

| arm | accuracy | vs baseline | repairs / regressions | p | 95% CI |
|---|---:|---:|---|---:|---:|
| baseline | 55.5% | — | — | — | — |
| **thinking** | **63.7%** | **+8.2** | **72 / 39** | **0.0022** | **[+2.9, +13.1]** |

**This is the first intervention in the investigation to clear the gate that
killed `because`.** The comparison is exact: identical production prompt,
differing only in `reasoning_effort`, so the effect is attributable to enabling
the model's reasoning mode under this configuration. That identifies the
intervention, not a detailed cognitive mechanism.

It also behaved as §6a predicted. `because` collapsed +10.8 → −7.2 when moved to
production; `thinking` went +9.8 → +8.2. The mechanism is visible in the
baselines: on grimgar03 the exploratory and production baselines are **56.5% vs
55.5%**, nearly identical, so there was no weak-baseline gap to inflate the
result. On mushoku16 those two differed by ten points, which is exactly what
manufactured the false `because` positive.

**Cost remains the decision.** 3939 s against baseline's 826 s — 4.8× wall time.
A shipped thinking pass roughly quintuples pass 2.

### 9.1 The second book does not confirm it

The gate ran on mushoku16. Both artifacts `validation: ok`, `dirty: false`:

| book | baseline | thinking | diff | repairs / regressions | p | 95% CI | N |
|---|---:|---:|---:|---|---:|---:|---:|
| grimgar03 | 55.5% | 63.7% | **+8.2** | 72 / 39 | **0.0022** | [+2.9, +13.1] | 400 |
| mushoku16 | 49.6% | 52.5% | +2.9 | 21 / 17 | 0.627 | [-6.4, +11.7] | 139 |

**This is a failure to confirm, not a contradiction.** mushoku16's interval
contains grimgar03's +8.2, so the two results are statistically compatible; with
139 lines and 38 discordant pairs this run cannot resolve an effect of that
size. The point estimate is positive on both books. The data are equally
compatible with a real +5 to +8 point effect and with zero.

That differs from the other reversals in this document, where a significant
result in one direction became significant or clearly null in the other. Here
the second book is simply underpowered.

Two things do sharpen:

**Headroom is refuted, not confirmed** (corrected 2026-07-27 after review; the
original text here claimed the opposite). Section 6a predicted the effect would
SHRINK as the baseline rises. The data run the other way:

| book | production baseline | thinking effect |
|---|---:|---:|
| mushoku16 | 49.6% (lower) | +2.9 (smaller) |
| grimgar03 | 55.5% (higher) | **+8.2 (larger)** |

Higher baseline, larger gain. Headroom does not explain the cross-book
difference and should not be carried forward as a mechanism. What separates the
books is unexplained; sample size (139 vs 400) accounts for the difference in
*resolution* but not for the direction of the point estimates.

**The cost is worse on the second book.** 2345 s against a 323 s baseline -
**7.3x**, versus 4.8x on grimgar03. An unconfirmed +2.9 points for seven times
the wall time is not a shipping case.

What would settle it is more power on mushoku16 - 139 gold lines against
grimgar03's 400 - AND a third large book. Those answer different questions:
more mushoku16 labels improve precision on a book already measured, while a
third book at grimgar03's scale adds generalization evidence. An earlier draft
here dismissed the third book as inferior; that was wrong. Until then `thinking` stands as one significant
production-path result plus one underpowered positive - more than any other
intervention here has achieved, and less than a shipping decision requires.

## 10. gemma-3-27b's thinking arms measured nothing

| arm | accuracy | elapsed | reasoning tokens |
|---|---:|---:|---:|
| baseline | 71.5% | 1223 s | 0 |
| `because` | 72.2% | 2290 s | 0 |
| scaffold | 68.2% | 2354 s | 0 |
| **thinking** | **71.2%** | **1212 s** | **0** |

**gemma-3-27b emits no reasoning tokens at all**, so unsetting
`reasoning_effort` changes nothing. Its `thinking` arm is `baseline` under
another name — same accuracy, same wall time — where qwen3-14b emitted 140 260
tokens on the identical arm and took 4.8× longer.

Three consequences:

1. **The +8.2 result cannot be replicated on models without a reasoning mode.**
   It is a qwen3-family finding until shown otherwise, and a cross-model
   replication programme has to check `reasoning_tokens` before spending GPU
   time on arms that are structurally no-ops.
2. The queue was about to spend roughly four hours per model on those arms.
3. It supplies a near-repeat, not a strict same-configuration repeat:
   `reasoning_effort` differs even though the server reports zero reasoning
   tokens. The 71.5% versus 71.2% result is reassuringly close, but should not
   be used as an independent estimate of ordinary run variance.

gemma-3-27b's baseline of **71.5%** is nonetheless the highest exploratory
reasoning-harness number posted on grimgar03 — above qwen3-14b's thinking arm
(66.2%). It nominates Gemma for a production-path comparison. It does not yet
show that Gemma beats production Qwen thinking: the models ran on different
hardware and neither number for Gemma came from the shipping prompt. The
production comparison, including latency and reliability, is the one that
matters.

## 11. The local/cloud stack bundle differs on the second book

Section 2 found no material local/cloud difference for Qwen on Mushoku.
Magistral-small on **grimgar03** differs between llama.cpp/ROCm with q8_0 KV on
the 9070 XT and LM Studio/f16 on the A6000:

| arm | local | cloud | diff | discordant | p |
|---|---:|---:|---:|---|---:|
| open | 59.8% | 61.3% | −1.5 | 3 / 9 | 0.146 |
| closed-6 | 61.5% | 58.0% | **+3.5** | 23 / 9 | **0.020** |
| closed-oracle | 74.8% | 70.8% | **+4.0** | 20 / 4 | **0.0015** |

On Mushoku the same model across those two stack bundles agreed to within 0.7
points with every p=1.000. Here two of three arms separate, and the local
q8_0-KV bundle is better on both.

This is not a clean environment A/B. Hardware, serving frontend/backend, KV
precision, context setting, and recorded harness commit differ together.
Candidate causes include backend numerics, KV precision, or another stack
difference; context capacity does not bind. The result shows that the whole
bundle is not interchangeable for close comparisons. It does not identify
hardware or KV quantization as the cause. Section 2's Qwen/Mushoku screening
bound should not be generalized to another model, book, and stack bundle.

Also from that run: the romanization penalty is **Mushoku-specific**. On
grimgar03 it is +0.0 (open) and +0.2 (oracle), against +7.9 on mushoku16's
oracle arm — those are the `RUDEUS`/`ALMANFI` names, and Grimgar's cast does not
trigger it.

## 12. Serving stack: measured, and mostly irrelevant to accuracy

Three stacks, same 9070 XT, same qwen3-14b Q4_K_M, same 139 lines, loopback,
ctx 16384, f16 KV, `parallel 1`, idle card, all within 15 minutes:

| stack | s/call | relative |
|---|---:|---:|
| llama.cpp ROCm (b10121 HIP) | 0.1959 | 1.000 |
| llama.cpp Vulkan (b10142 RADV) | 0.2101 | 1.072 |
| LM Studio (ROCm) | 0.2173 | 1.109 |

No material accuracy difference was detected across the three stacks (1–4
discordant lines, all p≥0.63). The measured HIP configuration was 7% faster
than the measured Vulkan configuration and 11% faster than LM Studio. Those
figures describe complete serving configurations; they do not isolate frontend
overhead or prove a general ROCm-over-Vulkan backend advantage.

**The capability, not the speed, is the reason llama.cpp matters here.** With
`-ctk q8_0 -ctv q8_0` magistral-small loads at 15 GiB on a 15.92 GiB card that
refused it under LM Studio's f16 + 2 GiB reserve. That converts a model the
ledger records as *untestable locally* into a tested one.

**Correction to an earlier reading in this document.** The Thunder forwarded
port costs **~0.374 s median round trip** against ~0.24 s of compute, so roughly
60% of a short-call cloud timing is network. An earlier note implying the A6000
was much slower than the local card was measuring the tunnel. On compute the two
are comparable; **the cloud buys VRAM capacity, not speed.** Any future speed
comparison must run on the instance over loopback.

Operational detail worth keeping: `lms get` stalled on the 70B with a live
process and **zero bytes written for 113 minutes**, twice, with no error. It was
recovered with a resumable `curl -C -` against HuggingFace's range support.
LM Studio cannot see the resulting file — its `gguf-metadata-cache.json` never
learned about it — so the 70B must be served by llama.cpp by path, or refetched
with `-hf`. Full details in `INFERENCE_STACK_NOTES.md`.

## 13. Crossover: complete, and it does NOT resolve section 6.3

The pre-registered crossover has now run twice. The first attempt completed all
20 cells and its artifact was refused — `environment is missing context_length`,
a field the validator requires that my hand-supplied environment dict omitted —
so 35 minutes of GPU produced no file. Its rows survived in the checkpoint,
which the resume guard preserved as `.stale` when the corrected harness changed
`harness_sha256` and it refused to resume: the safety rule working exactly as
designed.

The re-run completed and validated:
`crossover__grimgar03__local.json`, **`validation: ok`, `dirty: false`**,
commit `5457339`. All numbers below are from that artifact.

### 13.1 A correction to what I first reported

I initially reported both main effects as **RESOLVED** — "gemma segments
better, qwen attributes better" — and called it the answer to section 6.3.
**That was wrong.**

The pre-registered rule was "an effect counts as resolved only if it exceeds
2× the within-cell SD." At temperature 0 the SD is exactly **0.00**, so the
threshold is 0.00 and *any* nonzero difference passes. The rule degenerates
where variance vanishes. I noted the degeneracy in the same message and
reported the effects as resolved regardless.

Paired exact McNemar on the deterministic t=0 cells of the VALIDATED artifact,
N=399:

| comparison | effect | discordant | p | 95% CI |
|---|---:|---|---:|---:|
| segmentation, attr=gemma | +2.01 | 22 / 14 | 0.243 | [−1.2, +4.8] |
| segmentation, attr=qwen | +2.51 | 24 / 14 | 0.143 | [−0.8, +5.4] |
| attribution, seg=gemma | +2.26 | 55 / 46 | 0.426 | [−2.9, +7.3] |
| attribution, seg=qwen | +1.75 | 52 / 45 | 0.543 | [−3.3, +6.7] |
| **diagonal**: qwen/qwen vs gemma/gemma | **−0.25** | 54 / 55 | **1.000** | — |

**Every interval spans zero.** Nothing is significant.

Pooling the two held-fixed conditions moves segmentation to the edge, and is
reported here only to be dismissed:

| effect | pooled | discordant | p |
|---|---:|---|---:|
| segmentation | +2.26 | 46 / 28 | 0.047 |
| attribution | +2.01 | 107 / 91 | 0.286 |

**That pooled p is not valid.** The two contrasts run over the same 399 gold
lines, so treating them as 798 independent paired observations counts every
line twice and inflates significance. The defensible reading is that
segmentation is the more promising of the two effects and neither is
established — not that segmentation clears 0.05.

The two tests disagree because they ask different questions. 2×SD asks *is this
effect larger than run-to-run noise?* — yardstick 0.80 points. McNemar asks *is
this effect distinguishable from zero given 399 lines?* — yardstick roughly ±5
points. For any claim that generalizes beyond these particular lines, the second
is the correct one, and it says no.

### 13.2 What the crossover does establish

**Temperature-0 determinism, with one qualification.** Within a run, zero
differing rows across repeats in all four t=0 cells. **Across runs it is not
perfect.** Comparing the refused first run against the validated re-run — same
harness logic, same lines, model unloaded and reloaded in between:

| cell | run 1 | run 2 | differing rows |
|---|---:|---:|---:|
| seg=gemma, attr=gemma (both reps) | 234/399 | 234/399 | 0 |
| seg=qwen, attr=gemma (both reps) | 227/399 | 226/399 | **1** |

One line in 399 — **0.25%** — flipped, and it is the same line both times
(`grimgar03-02476`, `ABAEL` → `RANTA`): a borderline case resolving differently
after a model reload, plausibly GPU reduction ordering or memory layout.

So the accurate statement is: **temperature 0 is deterministic within a loaded
model instance, with a ~0.25% floor across reloads.** That floor is far below
any effect size in the ledger and does not disturb any conclusion, but two
temperature-0 runs should not be described as bit-identical evidence.

**Run-level variance at temperature 0.6.** Within-cell SD **0.80 pt** across
three repeats; `seg=qwen,attr=gemma` spanned 54.4 / 56.1 / 58.9. This is the
run-level replication section 7.0 correctly said one pipeline repeat could not
provide. Three repeats do not characterize a distribution, but the figure is
small enough that ~2-point effects are not obviously noise and large enough to
matter for ledger comparisons sitting inside 2 points.

**The end-to-end diagonal is flat.** Each model on its OWN segmentation:
**−0.25 points, p=1.000, 54/55 discordant.** Within this harness qwen3-14b and
gemma-4-e4b are indistinguishable end to end. That sits against section 6.3's
5.5-point end-to-end gap on the same book, and is consistent with that gap being
the pipeline's own run-to-run churn (17.9% of lines changed answer between two
runs) rather than a model difference.

### 13.3 Consequence for section 6.3

**Section 6.3 remains unresolved**, and the experiment built to resolve it says
so explicitly. The pre-registered rules covered this outcome: *"the honest
conclusion is that this design cannot resolve the question at this n — the
answer is more repeats, not a narrative."*

Effects of roughly 2 points against a line-level interval of about ±5 points
need substantially more data: more lines, both books, or a paired design across
repeats rather than single cells. What can be said is narrower than either
earlier claim — neither the decomposition nor the pipeline has been shown to
mismeasure the other, and the apparent disagreement between them is not
distinguishable from noise at this sample size.

### 13.4 A cheap probe that should be standard: check reasoning_tokens first

gemma-3-27b's thinking arms cost roughly four hours of A6000 time and measured
nothing, because the model emits no reasoning tokens at all (§10). That was only
discovered after the fact, by reading `reasoning_tokens` in the finished
artifact.

A single API call answers it in advance. Probing qwen/qwen3-32b while its
baseline arm was running:

| condition | completion tokens | reasoning tokens |
|---|---:|---:|
| `reasoning_effort="none"` (baseline arm) | 52 | **0** |
| `reasoning_effort` unset (thinking arm) | 377 | **340** |

**qwen3-32b's thinking arms will genuinely fire**, so its result will be
interpretable — the first real test of whether the +8.2 production gain (§9)
holds on a larger model of the same family, or shrinks as the baseline rises.

Two practical consequences:

1. **Probe before committing.** One call, a few seconds, versus four hours of
   GPU per model. gemma's waste was avoidable and llama-3.3-70b is likely the
   same case, having no reasoning mode.
2. **Revise the time estimate upward.** 340 reasoning tokens on a single-entry
   batch, against production batches of 25. qwen3-14b's thinking arm took 6.3x
   its baseline; a 32B doing the same puts its five arms nearer **10-11 hours**
   than the 7 estimated in §5, which pushes the remaining cloud queue past 30
   hours.

### 13.5 qwen3-32b: scale within a family buys nothing measurable

`closed_set__mushoku16__qwen__qwen3-32b__thunder-a6000.json`, `validation: ok`,
`dirty: false`. Open arm **52.5%**.

| model | open arm (mushoku16) |
|---|---:|
| gemma-3-27b | 55.4% |
| qwen3-32b | 52.5% |
| magistral-small | 52.5% |
| qwen3-14b | 49.6% |
| ministral-14b | 47.6% |
| phi-4 | 45.6% |
| qwen3.5-9b (shipped) | 35.4% |

Paired against every other model, qwen3-32b gives **p = 0.30 to 1.00 — no
separation from any of them.** The only significant comparison is against the
shipped 9B (30/9 discordant, p = 0.0011).

**Six models between 45.6% and 55.4% with no significant separation between any
pair** — a 32B, a 27B, a 24B, two 14B-class and a 7.5B, spanning 4× in parameter
count and four architectures.

Within the qwen family specifically, 14B → 32B is 49.6% → 52.5%, p = 0.64,
22/18 discordant: doubling the parameters moved four net lines out of 139. If
model scale were the lever, this is the comparison where it should have
appeared.

This is the fourth ordering today to fail its paired test, after `closed-6`, the
cross-book model ranking, and the crossover effects in §13.1.

**Scope, narrowed after review.** An earlier draft said accuracy "plateaus above
about 7B" and therefore points at task representation. That is wider than the
evidence. This measurement uses one fixture and a component harness, has limited
power among the clustered models, and mixes architectures, training recipes,
quantizations and serving stacks. It does not show that all larger models, or
the production path, share a plateau.

The defensible statement: **within this Mushoku closed-set measurement,
increasing Qwen from 14B to 32B produced no detectable gain, and model size
alone has not explained the remaining error.** The stable finding of the model
programme remains the one it started with - move off qwen3.5-9b.

## 13.6 Grammar-constrained decoding: repairs, but not where production lives

`grammar_constraint__mushoku16__mistralai__magistral-small__local-llamacpp.json`,
`validation: ok`, `dirty: false`, llama.cpp-hip b10121. Prompts identical in
both arms - the candidate list is stated either way - so only the sampler
differs.

| arm | free | grammar | delta | repaired / broke | p |
|---|---:|---:|---:|---|---:|
| oracle | 58.3% | **66.2%** | **+7.9** | 12 / 1 | **0.0034** |
| open | 53.2% | 51.8% | -1.4 | 0 / 2 | 0.500 |

Off-list answers went to zero in both grammar arms, which was the built-in
check that the constraint actually applied.

**It repairs rather than relabels.** Tracking what happened to the free arm's
off-list answers: on the oracle arm **15 off-list answers became 11 correct**
under the grammar. When the model wrote `RUDIUS` meaning `RUDEUS`, forcing it
onto the roster recovers the right character about three times in four. That
confirms the mechanism the phonetic scorer (§5a.3) could only infer.

**But it does nothing on the arm that resembles production.** The open arm is
-1.4 points on two discordant lines - noise, not harm. That is exactly what the
sizing predicted: off-list answers are 13.7% of oracle rows and about 1% of open
rows, so with the full roster in front of it the model rarely leaves the list.

Consequence: **not a shipping win as tested**, because production uses the full
roster. The place it might matter is the `is_attested_name` gate, which
currently *rejects* invented names after decoding - once 279 per book. This
result says constraining at decode time recovers the right answer in most such
cases instead of discarding it. Testing that requires the production pass-2
prompt, not this diagnostic arm.

## 13.7 Where the remaining points might be, and a request

Ten interventions have now been measured. One survives (`thinking`, §9), and it
is one significant book plus one underpowered book at 4.8-7.3x wall time. Six
models across four architectures and 4x parameter count sit in a single
indistinguishable band on both books. Scale within a family moved four net lines
out of 139.

That pattern - flat across model, flat across prompt engineering - is more
consistent with an **input** limit than a capability limit. The strongest
untested lever follows directly:

**Context width (running now).** Production pass 2 supplies **one segment
either side** of the line being attributed. Every diagnostic in this ledger used
four. Measured prompts are 176 tokens median against a 16384-token window -
**1.1% utilisation** - while **62.1% of errors have no character name anywhere
nearby**. `context_width.py` sweeps w1/w4/w15/w40 on grimgar03. If accuracy
rises, production is leaving points on the table for nothing; if it is flat,
context is excluded and the plateau needs another explanation.

Three further candidates, each tied to a measured number rather than a hunch:

1. **Candidate-set RECALL, not size.** `closed-6` (recall 73-87%) was null;
   `closed-oracle` (recall 100%) is +17. The entire oracle advantage is that the
   answer is in the set. So the lever is a higher-recall scene-cast pass
   optimised on recall, not a smaller set. Getting recall to ~97% with sets of
   ~8 should capture much of that +17.
2. **Sequential attribution with committed history.** Pass 2 attributes batches
   of 25 entries *independently*; the model never sees its own prior decisions,
   yet dialogue alternates and **12.6% of errors name the addressee rather than
   the speaker**. Note the `scaffold` arm asked the model to *infer*
   `previous_speaker` and lost 4.0 points - supplying the actually-resolved
   speaker is a different mechanism (state, not introspection).
3. **Selective thinking.** `thinking` is the only surviving intervention and its
   blocker is cost, not effect. Routing it to lines a cheap pass flags uncertain
   needs only correlation with difficulty, a much lower bar than the
   17%-coverage-at-76% confidence signal that failed as an *acceptance*
   criterion.

**Request to the reviewer.** The four above are what the error analysis
supports. What is missing? In particular:

- Is there a reading of the 62.1%/12.6%/6.8% error breakdown that points
  somewhere none of these do?
- The oracle arm caps at 66-76% even with the true speaker among five
  candidates. That ceiling is not explained by recall, roster, or model size.
  What would explain a model failing a five-way choice a quarter of the time
  when the answer is present?
- Is the ~50% plateau better read as a property of the **fixture** - gold sets
  built from hard or disputed lines - than of the task? If so, the ceiling is
  an artefact and the real question is what accuracy on *representative* lines
  looks like.

## 14. Infrastructure added today

- **Row-level checkpointing** (TEMPORARY, `manifest.py`) — resume for any
  harness, adopted only when experiment, model, endpoint, gold hash, harness
  hash and decoding all match; otherwise the stale file is moved aside and the
  run starts clean, naming the field that differed. Verified against a model
  change and a decoding change.
- **`romaji_key` / `same_person_phonetic`** in the canonical scorer, reported
  alongside exact match rather than replacing it (§5a.3).
- **`EXPERIMENT_TAG`** in artifact filenames, after a cloud run overwrote a
  local one.
- **`RESULTS_INDEX.md` / `results_index.csv`** — every artifact flattened into
  one table with provenance (validation, dirty, endpoint, backend, context)
  beside each number. 31 artifacts, 98 arms. It immediately surfaces that **25
  of 98 rows are `dirty: true`** and 7 predate the validator.
- **`INFERENCE_STACK_NOTES.md`** — the serving/model-management reference.

## 15. Queue paused

The cloud sweep was stopped by request after gemma-3-27b's arms completed.
Pending: qwen3-32b, llama-3.3-70b, magistral's re-queued arms, and gpt-oss-20b
(which arrived accidentally via a name-resolution probe and is the only
non-Qwen/Gemma/Mistral/Llama architecture in the set). All four models are
downloaded; nothing is lost by the pause.

**The instance bills while it exists**, at roughly $0.44/hr, whether the GPU is
busy or idle — stopping is not sufficient, only deletion ends billing. If the
pause is longer than about an hour, snapshot and delete rather than idle.

## 16. Reviewer recommendations (from the 2026-07-27 review)

Recommendation 6 has now been acted on and passed - see §9.
Recommendation 7 is partly answered by §13: the crossover's
temperature-0.6 repeats give the run-level replication the review
correctly said one pipeline repeat could not provide.

The addendum is a useful correction and should remain separate until the active
clean-tree runs finish. When it is merged, §7.2 of the main brief should be
replaced rather than followed by another correction; the handoff should expose
one current conclusion, with Git history preserving the mistake.

Recommended decision rules:

1. Treat the cloud queue as **pass-2 screening**, not model selection for the
   shipping pipeline. A strong closed-set result nominates a model for
   end-to-end testing; it does not select a winner.
2. Keep the segmentation × attribution crossover ahead of any production
   model decision. It is the experiment that can separate pass-1 quality,
   pass-2 quality, and their interaction. Because the shipping pipeline is
   stochastic, use run-level replication rather than treating gold lines as
   independent repeats. A small pilot of at least three runs per cell can
   estimate whether more replication is needed; predeclare the decision margin
   and expand if its uncertainty overlaps that margin.
3. Do not rank close cloud results from percentages alone. Preserve paired
   discordance, exact tests, retries, latency, failures, and memory settings.
4. Predeclare what would justify an end-to-end run. A sensible gate is a
   meaningful paired improvement on both books, or a large improvement on one
   with no material regression on the other.
5. Delete the rented instance immediately when the declared queue and artifact
   verification finish. Do not extend a continuously billed session merely
   because spare experiments are available.
6. Stage-gate the newly expanded reasoning queue. The Grimgar artifact is now
   complete and strongly positive for exploratory thinking; run the
   production-path baseline-versus-thinking check before paying to run all five
   arms on four larger models. Cross-model replication of an effect that does
   not survive the production prompt would repeat the `because` mistake at
   greater cost.
7. Treat the cloud end-to-end run as one repeat, not an estimate of variance.
   Compare per-line transitions, segmentation identities, omissions, retries,
   and failures with the local run. The observed churn justifies replication;
   it does not quantify the run-level distribution or isolate a hardware,
   sampling, truncation, or state effect.
8. Keep exact and phonetic attribution scores separate. Use `romaji_key` only
   as an evaluation diagnostic until it is collision-tested on a substantially
   larger roster set or replaced by explicitly adjudicated aliases. Production
   cast identity should continue to require canonical names.
9. The next shipping comparison should be **Gemma-3-27B baseline versus
   Qwen3-14B thinking through the same production path**, on the same book,
   fixture, serving environment, and paired IDs. Report latency, reasoning
   tokens, parse failures, retries, and VRAM alongside accuracy. Comparing
   Gemma's exploratory 71.5% directly with Qwen's production 63.7% is not
   decision-grade.
10. Treat the Magistral local/cloud result as a bundled-stack warning. If a
    close decision depends on it, vary one factor at a time—first KV precision
    on the same llama.cpp/hardware stack, then backend or hardware. Do not label
    the current contrast a hardware effect.
11. Rename or generalize `because_production.py` and future artifact names
    before it becomes the permanent production-intervention harness. An
    artifact named `because_production` whose only experimental arm is
    `thinking` is mechanically valid but easy to misread in an index or later
    audit.
12. The cloud queue is paused and the instance bills while it exists. Unless
    work will resume within the declared short window, snapshot what is needed,
    verify the downloaded-model inventory, and delete the instance now. A
    paused experimental plan is not a reason to keep an idle billed resource.

The context investigation still paid for itself: it corrected an unsupported
confound claim, established that the local results can be compared
operationally with cloud screening results, and exposed missing retry behavior.
Its value should be described that way, not as validation of a context effect.
