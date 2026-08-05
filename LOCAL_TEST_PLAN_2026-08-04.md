# Local test plan — 2026-08-04

## Objective

Test the assumptions behind the current TTS, casting, pitch, adapter-health,
and experiment-infrastructure claims without treating an assumption as a
result. GPU work runs serially through `gpu_job.sh`. Existing artifacts are
preserved, new artifacts record provenance, and perceptual conclusions remain
pending until people listen to blinded audio.

## Progress snapshot — updated 2026-08-05 07:17 UTC

| stage | state | evidence |
|---|---|---|
| 1 — controls | **complete** | Fresh-process determinism and instruction positive controls passed for three adapters. |
| 2 — evidence audit | **complete** | All 241 artifacts have structural classifications; all 112 legacy attribution artifacts have current-gold integrity and family-level semantic review. |
| 3 — unreliable TTS reruns | **complete** | Seeded clone-vs-LoRA and saturation generation plus ECAPA scoring completed with provenance. |
| 4 — non-prose replication | **complete** | Fixed 144-row matrix and six-category 432-row expansion both passed strict validation. The effect is category-specific, not a general non-prose failure. |
| 5 — non-prose remedies | **stopped at gate** | Stage 4 did not justify a general non-prose routing policy, so the general remedy comparison is not eligible. |
| 6 — blinded materials | **generation complete; human verdict pending** | Eight randomized sets contain 20 validated WAVs and a separately hashed concealed key. |
| 7 — pitch profiling | **measurement complete; listening gate pending** | The pilot passed and the full 1,350-row matrix passed independent pitch/WAV/provenance validation. No numerical casting threshold is adopted without blinded listening. |
| 8 — adapter health | **stopped at gate** | Weight norm is not production-driving and existing seeded ECAPA samples do not demonstrate a positive relationship; no adapter action is justified. |
| 9 — operational tests | **complete for implemented resume paths** | Lock, timeout release, queue propagation, generation guards, invalid/truncated WAVs, identity, row resume, and distillation training-state resume are covered. |
| 10 — validation/index | **complete** | Indexes and both audits are current; all new artifacts are explicit; final discovery passed all 1,289 tests. |

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
- Full provenance-bearing rerun completed successfully at commit `2c652ff`.
- Same-seed hashes matched across fresh processes for all three adapters;
  different seeds changed all three outputs.
- Slow > neutral > fast duration held for all three adapters. This establishes
  plumbing, not perceptual quality.

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

Current state:

- Structural audit refreshed for 241 artifacts and written to
  `ab_test_runtime/audit/artifact_structural_audit.json`.
- Current counts: 12 reproducible structural candidates, 112 older-metadata
  artifacts requiring semantic review, and 117 without sufficient
  embedded identity.
- `audit_experiment_artifacts.py` now regenerates this inventory and preserves
  the original 232 classifications exactly; `--check` fails loudly when the
  checked artifact set or any file hash changes.
- Manual decision-bearing classifications are committed in
  `ARTIFACT_AUDIT_2026-08-04.md` as part of `8321f16`.
- The broader legacy-attribution audit is complete and reproducible in
  `LEGACY_ATTRIBUTION_AUDIT_2026-08-05.md`. All 112 files appear exactly once:
  39 supported measurements, 29 provisional measurements, 42 historical-only
  artifacts whose rows no longer score identically against current gold, and
  two exploratory artifacts without the required environment/harness identity.
  Family-level interpretation limits prevent integrity from being mistaken for
  a product conclusion.

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

Current state: **complete**.

- Generation/scoring phase split, strict ECAPA dependency, correct
  `app/config.json`, provenance, and fail-loud tests committed in `fb5d55a`.
- Sibling `torchaudio.load()` was proven to segfault on a valid WAV. SoundFile
  decoding plus SciPy resampling fixed it in `ac99417`; no acoustic fallback
  was used. Full suite: 1,232 tests passed at that checkpoint.
- `clone_vs_lora_seeded.json`: 9/9 voices scored. One-sentence/one-seed mean
  LoRA-minus-clone ECAPA similarity was approximately -0.0236; LoRA led 4/9.
  This is provisional, not a Voice Lab policy.
- `voice_data_saturation_seeded.json`: 14/14 voices scored. The selected set
  did not show the assumed positive sample-count relationship; the comparison
  remains observational and cannot establish causality.
- The old `voice_data_saturation.json` has zero results and is invalid evidence.
- No old `clone_vs_lora.json` existed, so that corrected artifact was a first
  retained result rather than literally a rerun.

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

Current state: **complete; general-policy gate not cleared**.

- `word_error_breakdown` now reports substitutions, deletions, and insertions
  separately so over-generation is not hidden inside pooled WER.
- Harness and audit committed in `8321f16`; full suite passed 1,237 tests.
- All eight original passages were recovered. Their prose controls match exact
  character counts: 417, 222, 213, 131, 90, 78, 65, and 64. Digit, uppercase,
  punctuation, and word-count gaps are recorded rather than assumed away.
- Full matrix is fixed before results: 3 adapters × 3 seeds × 8 pairs × 2
  classes = 144 renders, reported per adapter, seed, pair, class, and error kind.
- The 2-pair × 1-adapter × 1-seed pilot published 4/4 valid rows and passed its
  gate. The full run then published all 144 rows and the queue logged `OK`.
- Independent validation confirmed 144 unique matrix keys, 144 fully decodable
  and RIFF-complete WAVs, exact error-component arithmetic, eight input-pair
  identities, reproducible harness provenance, and all 18 recomputed summaries.
- Across the selected samples, non-prose failed 62/72 renders versus 0/72 prose
  controls. Aggregate WER was approximately 49.3% versus 1.0%, with 619 versus
  zero insertions. The direction held in all nine adapter×seed cells.
- This does **not** clear the general-policy gate. The eight passages were
  selected from earlier failures; seven reproduced broadly, while one failed
  in zero of nine cells. The predeclared ISBN/URL/copyright/list/date/heading
  category expansion is therefore the next Stage 4 experiment.
- The harness now writes an atomic checkpoint after every row and resumes only
  when source code, config, source artifact, adapter weights, seeds, and pair
  identities match. The tracked Stage 4 checkpoint runner strictly validates
  provenance, matrix identity, WAVs, summaries, indexes, and the full suite.
- The category fixture is now locked and tracked: four pre-render probes in
  each of identifiers, URLs, copyright, lists/tables, dates/numbers, and
  headings/fragments. Each probe retains its saved-library citation, exact text,
  and hash. Distinct ordinary-prose controls are chosen deterministically by a
  declared surface-feature cost, with all residual gaps recorded.
- The all-category pilot is fixed at 12 renders (six categories × probe/control
  × one adapter × one seed). A strictly valid pilot gates the full expansion of
  3 adapters × 3 seeds × 24 pairs × 2 classes = 432 renders.
- The pilot passed strict validation at 12/12 rows. The full expansion then
  completed 432/432 distinct matrix rows with 432 fully decoded, RIFF-complete
  WAVs, reproducible provenance, exact error arithmetic, and 108 summaries
  equal to an independent recomputation.
- Probe versus prose WER by category was: identifiers 31.6% versus 0.0%; URLs
  11.1% versus 0.4%; copyright 17.4% versus 11.1%; lists/tables 3.6% versus
  1.3%; dates/numbers 6.8% versus 2.2%; and headings/fragments 1.1% versus
  0.0%. Probe failure counts were respectively 34, 4, 19, 2, 7, and 0 out of
  36 renders per category. These are transcript metrics, not naturalness or
  listener-preference results.
- The probe had higher WER in all nine adapter×seed cells for identifiers and
  dates/numbers, but only 7/9 for URLs, 7/9 for copyright, 6/9 for
  lists/tables, and 2/9 for headings/fragments. One copyright prose control
  failed in 8/9 cells. The effect therefore does not survive categories and
  matched controls uniformly enough to support routing non-prose away from TTS
  as a general policy.
- Stage 5 is stopped at its gate. The evidence supports category-specific
  follow-up for identifiers and possibly dates/numbers, not the plan's general
  non-prose remedy comparison. All selected probes and controls came from one
  saved-book library, so cross-book transfer remains untested rather than
  assumed.

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

Current state: **generation complete; human verdict pending**.

- A committed, resumable runner generated four instruction comparisons, one
  current-versus-scene-aware casting comparison, and three extreme slow/fast
  positive controls. Stage 5 remedies are absent because that stage did not
  clear its gate; the public manifest says so explicitly.
- The public package contains eight randomized sets and 20 generic WAV names.
  The concealed key is separate and its exact SHA-256 is recorded publicly.
- Independent validation fully decoded every WAV, checked RIFF completeness,
  verified every source/package/key hash, confirmed the 4+1+3 composition, and
  found no arm-label leakage in the public JSON.
- Both GPU jobs logged `OK`; the indexes were regenerated and the full suite
  passed 1,267 tests. No automated delivery, preference, or casting verdict is
  claimed.

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

Current state: **measurement complete; blinded-listening gate pending**.

- `app/routers/voices.py::_infer_lora_gender` still uses declared `mean_f0`
  with a 165 Hz threshold when explicit/name/description evidence is absent,
  so pitch affects production auto-selection.
- All 75 manifest adapters have complete weights/configuration and declared
  pitch. The old artifact covers six adapters, one seed, and one text set; it
  lacks voiced-frame coverage and octave-error evidence and cannot satisfy this
  stage.
- Six passages were locked before generation across narration, plain dialogue,
  question dialogue, exclamation dialogue, short dialogue, and long dialogue.
  The full matrix is fixed at 75 adapters × 3 seeds × 6 passages = 1,350 rows.
- The replacement harness checkpoints every row against exact code, input,
  adapter-weight, seed, and passage identities. Resume validation fully decodes
  each WAV and recomputes pYIN measurements; completed artifacts also require a
  reproducible harness, current input/weight hashes, exact matrix coverage,
  recomputed summaries, and recomputed likely-octave flags.
- The pilot is fixed at one adapter × two seeds × two passage types = four
  rows. At least three must produce valid pitch tracks before the full matrix
  may start. Tracker failures remain explicit rows rather than being dropped.
- The shared GPU-job helper is now the single source for the approved lock,
  repository queue log, timeout wrapper, checked exit status, and per-stage
  log. Focused verification passed 18 tests; full discovery passed all 1,286
  tests before any Stage 7 generation.
- The four-row pilot passed its gate with four valid pitch tracks. The full GPU
  job then logged `OK` and published all 1,350 predeclared matrix rows.
- Independent post-run validation decoded every WAV, checked RIFF completeness,
  remeasured every pitch track with pYIN, and exactly reproduced the saved
  summaries and octave flags. One row remains an explicit tracker failure:
  `husky_baritone_20s_m_anime`, seed 9012, question dialogue produced only five
  voiced frames (2.55% coverage); it was not silently dropped.
- All 75 adapters have measured summaries. Typical within-adapter P90-minus-P10
  dispersion is approximately 70.81 Hz. Only 924/2,775 adapter pairs (33.30%)
  are farther apart than that measured dispersion, so pitch alone does not
  establish broad voice separability.
- The declared and measured values fall on opposite sides of production's
  165 Hz fallback threshold for 18/75 adapters. The heuristic marked 523/1,350
  rows as possible octave errors; because this diagnostic is itself broad, it
  is a warning to inspect/listen, not proof that those rows are wrong.
- No production metadata, fallback threshold, or casting behavior was changed.
  The plan's final gate still requires blinded listening near any proposed
  threshold before such a threshold can be adopted.

## Stage 8 — Adapter-health validation

Estimated runtime: **6–12 GPU hours**, conditional on the earlier decision
gate. The speaker-embedding dependency is now verified.

Test whether LoRA weight magnitude predicts voice identity by comparing:

- low-sample or low-norm adapters;
- normally trained controls;
- held-out reference recordings;
- standardized seeded generations;
- speaker-embedding similarity and human identity judgments.

Until this correlation is demonstrated, weight norm is a diagnostic lead, not
proof that an adapter is undertrained.

Current state: **stopped at gate; no production action**.

- Weight norm is not consumed by production code. The two available seeded
  ECAPA samples join 9 and 14 adapters respectively and show no supported
  positive norm-similarity relationship (permutation p=0.265 and p=0.220;
  uncertainty intervals cross zero).
- Those selected, observational samples also cannot establish a negative or
  causal sample-count effect. No adapter will be removed or retrained from
  them; human identity judgments remain required if such an action is later
  proposed.

Dependency update: SpeechBrain ECAPA loads in the sibling interpreter. Its
`torchaudio.load()` path segfaulted and was replaced with verified SoundFile +
SciPy decoding/resampling in `ac99417`. Real ECAPA scoring now completes.

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

Current state: **complete for implemented resume paths**.

- GPU lock behavior and deployment identity: covered by 11 committed tests.
- Stale, false-return, missing, empty, undecodable, zero-frame, and truncated
  WAV handling: covered by 17 generation-guard tests.
- A real `timeout` wrapper test now verifies exit code 124, `FAILED` queue
  logging, lock release, and successful execution of the next queued job.
- Stage 4 row-level checkpoint tests cover exact fingerprint matching,
  incompatible-checkpoint archival without overwrite, corrupted/foreign/
  duplicate rows, error arithmetic, repository-local paths, full decoding,
  and RIFF completeness.
- A CPU-only integration test now interrupts the installed Transformers
  `Trainer` at step 3 of a fixed six-step run and resumes through the same
  `get_resume_checkpoint` dispatch used by `distill_train.py`. The checkpoint
  is required to contain model, optimizer, scheduler, RNG, trainer state, and
  training arguments. The resumed run must reproduce the uninterrupted
  forward-pass sample order and match final model weights, optimizer state,
  and scheduler state exactly; dropout makes RNG restoration behavior-bearing
  rather than a file-presence check.
- `train_lora.py` does not implement training resume. The completed claim is
  therefore limited to the distillation `Trainer` path; no resume guarantee is
  asserted for the standalone Voice Lab LoRA trainer.

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

Current checkpoint:

- `nonprose_replication.json` and its pilot appear explicitly under **Not
  indexed** in both `RESULTS_INDEX.md` and `results_index.csv`.
- The hardened Stage 4 runner validated both real artifacts, regenerated both
  indexes, and passed the complete unit-test discovery. This checkpoint must be
  repeated after the category expansion rather than treated as final for all
  later stages.
- The category pilot and full expansion now also appear explicitly under **Not
  indexed** in both indexes. The full expansion passed strict validation at
  432/432 rows. The first post-run suite correctly failed on one missing test-
  inventory entry; after adding that entry, full discovery passed all 1,256
  tests. No test was skipped silently in that pass.
- `blinded_listening.json` now appears explicitly under **Not indexed** in both
  indexes. Its independent audit verified eight sets and 20 WAVs; the Stage 6
  post-run discovery passed all 1,267 tests.
- `pitch_profile_matrix_pilot.json` and `pitch_profile_matrix.json` now appear
  explicitly under **Not indexed** in both indexes. Independent validation
  remeasured all 1,350 full-matrix WAVs and reproduced the artifact exactly;
  the Stage 7 post-run discovery passed all 1,286 tests with no skipped suite.
- The final CPU audit accounts for all 112 legacy attribution artifacts exactly
  once and passes its deterministic `--check`. After adding its three
  behavior-bearing tests and refreshing the committed inventory, final full
  discovery passed all 1,289 tests. The local GPU queue is empty, its lock is
  acquirable, and no Stage 7 checkpoint remains.

## Schedule and stopping policy

- Stages 1–4: approximately **8–16 local GPU hours**, plus the CPU audit.
- Complete conditional program: approximately **2–4 local GPU days**.
- Human listening time is separate.

Later stages are not automatic merely because they are listed. A stage runs
only when earlier evidence leaves its underlying product decision open. A
failed control, invalid artifact, unavailable dependency, or conclusion already
settled by stronger evidence is reported rather than worked around silently.
