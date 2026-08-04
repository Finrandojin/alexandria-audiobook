# Local test plan — 2026-08-04

## Objective

Test the assumptions behind the current TTS, casting, pitch, adapter-health,
and experiment-infrastructure claims without treating an assumption as a
result. GPU work runs serially through `gpu_job.sh`. Existing artifacts are
preserved, new artifacts record provenance, and perceptual conclusions remain
pending until people listen to blinded audio.

## Non-negotiable execution rules

- Run every GPU experiment through `gpu_job.sh` with
  `$HOME/.alexandria_gpu.lock`.
- Use `app/env/bin/python` with `app/` as the experiment working directory.
- Give every GPU job a timeout.
- Do not edit a shell script while a process is executing it.
- Do not overwrite an old artifact without retaining enough identity to
  distinguish the runs.
- Check generation return values and require a fresh, nonempty, decodable WAV.
- Record commit, dirty-tree identity, script, arguments, seed, host, and inputs.
- Do not call a run complete unless its JSON artifact was written and validated.
- Do not convert automated acoustic or transcript metrics into claims about
  listener preference.

## Stage 1 — Experimental controls

Estimated runtime: **20–40 minutes**.

Run `seed_instruction_controls` across three configured LoRA adapters, with
every render performed in a fresh Python process.

Tests:

1. Same text, adapter, instruction, and seed produce identical WAV hashes.
2. Changing only the seed changes the WAV hash.
3. An extreme slow instruction produces longer audio than an extreme fast
   instruction when every other input is held fixed.
4. The final artifact records exact instructions, seeds, waveform hashes,
   durations, arguments, commit, and environment identity.

Gate:

- If fresh-process determinism fails, stop paired TTS experiments until the
  uncontrolled state is identified.
- If the extreme instruction control fails, do not interpret subtle
  instruction comparisons as evidence that instructions work.

Current state:

- Harness and tests committed in `8788c7c`.
- Provenance-publication fix committed in `2c652ff`.
- Full suite before the first run: 1,217 tests passed.
- First attempt completed all 18 renders and printed passing controls, but
  failed before publishing JSON. It is not accepted as completed evidence.
- A full provenance-bearing rerun is in progress.

## Stage 2 — Existing-evidence audit

Estimated runtime: **2–4 CPU hours**. This should not occupy the GPU.

For each decision-bearing experiment artifact, record:

- provenance present or absent;
- commit and dirty-tree identity;
- seeded, unseeded, or unknown;
- current-run audio versus possible stale-file reuse;
- whether generation success was checked;
- corpus/configuration identity;
- whether arms differ by exactly one intended variable;
- whether the metric can support the written conclusion;
- whether sample selection was declared before results were observed.

Classify each artifact as:

- **supported** — reproducible inputs and a conclusion within the metric;
- **provisional** — useful evidence with a stated unresolved limitation;
- **exploratory** — insufficient provenance or uncontrolled comparison;
- **invalid** — known stale input, failed generation, contaminated comparison,
  or a conclusion contradicted by the artifact.

Only rerun unreliable artifacts that influence a current product decision.

## Stage 3 — Known-unreliable TTS experiments

Estimated runtime: **2–5 GPU hours**.

Rerun:

- `clone_vs_lora`;
- `voice_data_saturation`.

The existing artifacts are not quotable because their earlier harness could
reuse old audio. New runs must use fixed seeds, fresh output paths, checked
generation results, decodable-WAV validation, and provenance.

Gate:

Compare old and new conclusions. Explicitly report whether stale audio changed
the magnitude or direction of either finding.

## Stage 4 — Non-prose replication

Initial estimate: **5–9 GPU hours**.

Replicate the non-prose finding with:

- three contrasting LoRA adapters;
- three fixed seeds;
- the same eight passages used by the mechanism experiment;
- prose controls matched as closely as practical for length, token count,
  digit density, punctuation, capitalization, and expected duration;
- results reported separately by adapter and seed, not only as a pooled total.

This tests whether the current finding is a general limitation or an effect of
one adapter, one seed, or unmatched surface features.

If the initial result survives, expand the corpus by category:

- ISBNs and identifiers;
- URLs;
- copyright notices;
- lists and tables;
- dates and numbers;
- headings and sentence fragments.

Expanded-category estimate: **6–12 additional GPU hours**.

Gate:

Do not recommend routing non-prose away from TTS as a general policy unless the
effect survives adapters, seeds, categories, and matched controls.

## Stage 5 — Non-prose remedy comparison

Estimated runtime: **4–8 GPU hours**. Run only if Stage 4 confirms a general
problem.

Compare paired outputs for:

1. current production behavior;
2. normalization or rewrite;
3. splitting into individual items;
4. deterministic pronunciation of identifiers, dates, and URLs;
5. omission or summarization only where the product policy permits it.

Automated outcomes:

- transcription errors;
- missing content;
- invented content;
- generation failures;
- non-speech;
- duration and throughput.

Naturalness and preference remain human-listening questions.

## Stage 6 — Blinded listening materials

Estimated generation runtime: **2–5 GPU hours**, depending on how much existing
audio passes the provenance audit.

Prepare randomized, unlabeled comparisons for:

- no/per-character/per-line instruction;
- current versus scene-aware casting;
- competing non-prose remedies;
- obvious positive controls such as extreme slow versus extreme fast.

Record the concealed key separately. Ask listeners to rate delivery, emotional
fit, voice distinction, intelligibility, defects, and preference. Automated
systems may build these artifacts but must not supply the human verdict.

## Stage 7 — Seeded pitch profiling

Estimated runtime: **12–24 GPU hours**. Run only if pitch will affect production
casting.

For every usable adapter, measure:

- several standardized passages and text types;
- multiple fixed seeds;
- median pitch and within-voice dispersion;
- voiced-frame coverage;
- pitch-tracker failures and likely octave errors.

Then recompute voice-pair separation from the new measurements. Do not use the
current declared `mean_f0` values as a numerical casting constraint: their
observed error is comparable to the proposed separation threshold.

Gate:

Before adopting any threshold, verify with blinded listening that differences
near that threshold are perceptually useful.

## Stage 8 — Adapter-health validation

Estimated runtime: **6–12 GPU hours**, conditional on a functioning local
speaker-embedding dependency.

Test whether LoRA weight magnitude predicts voice identity by comparing:

- low-sample or low-norm adapters;
- normally trained controls;
- held-out reference recordings;
- standardized seeded generations;
- speaker-embedding similarity and human identity judgments.

Until this correlation is demonstrated, weight norm is a diagnostic lead, not
proof that an adapter is undertrained.

## Stage 9 — Operational failure tests

Estimated runtime: **2–5 hours**, mostly CPU or short controlled GPU work.

Verify:

- GPU lock behavior on success, wrapped-command failure, timeout, and lock
  acquisition failure;
- queue logging and exit-code propagation;
- checkpoint/resume preservation of model, optimizer, scheduler, RNG, and
  sample order where locally testable;
- stale output removal;
- false returns, missing files, empty files, and invalid WAV rejection;
- deployment identity recording before job start;
- results-index behavior for unreadable and unsupported artifact shapes.

Do not intentionally disrupt the active Thunder job.

## Stage 10 — Validation and index regeneration

After every completed stage:

1. Validate the JSON shape, output files, and provenance.
2. Confirm artifacts belong to the current run.
3. Run `python3 collect_results.py` from the repository root.
4. Regenerate `RESULTS_INDEX.md` and `results_index.csv`.
5. Confirm each new artifact appears either as indexed arm rows or explicitly
   under **Not indexed**.
6. Run full unit-test discovery.
7. Record confirmed, disproved, provisional, invalid, and human-pending claims.

`collect_results.py` currently flattens per-arm attribution accuracy. TTS,
acoustic, and listening artifacts may correctly appear under **Not indexed**.
Changing that schema is separate design work and is not part of this plan.

## Schedule and stopping policy

- Stages 1–4: approximately **8–16 local GPU hours**, plus the CPU audit.
- Complete conditional program: approximately **2–4 local GPU days**.
- Human listening time is separate.

Later stages are not automatic merely because they are listed. A stage runs
only when earlier evidence leaves its underlying product decision open. A
failed control, invalid artifact, unavailable dependency, or conclusion already
settled by stronger evidence is reported rather than worked around silently.
