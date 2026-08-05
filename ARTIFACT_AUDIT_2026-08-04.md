# Experiment artifact audit — 2026-08-04

## Scope

This audit separates artifact integrity from scientific support. Valid JSON or
a recorded commit does not prove a conclusion. The structural inventory covers
232 JSON files under `ab_test_runtime/experiments/`; manual review begins with
the artifacts currently used to justify TTS, instruction, non-prose, pitch,
and adapter decisions.

Structural inventory refreshed after Stage 6: 10 reproducible structural
candidates, 112 artifacts on the older metadata contract, and 117 artifacts
without sufficient embedded identity (239 total).
The machine-readable snapshot is
`ab_test_runtime/audit/artifact_structural_audit.json`.

## Classification meanings

- **Supported:** identifiable harness and inputs; controlled measurement; the
  written conclusion stays within what the metric establishes.
- **Provisional:** useful, identifiable evidence with a material unresolved
  limitation.
- **Exploratory:** missing identity or uncontrolled comparison; may motivate a
  test but should not be quoted as settled evidence.
- **Invalid:** known contamination, empty result, failed publication, or a
  conclusion contradicted by the artifact.
- **Human-pending:** generation succeeded, but the question is perceptual and
  cannot be answered by the automated metric.

## Decision-bearing artifacts

| artifact | classification | verified scope | unresolved limit |
|---|---|---|---|
| `seed_instruction_controls.json` | Supported + human-pending | Same seed produced byte-identical WAVs across fresh processes for three adapters; a different seed changed each hash; extreme slow/neutral/fast instructions produced monotonic durations for all three adapters. | Establishes determinism and instruction plumbing, not naturalness, emotional fit, or subtle instruction quality. |
| `clone_vs_lora_seeded.json` | Provisional | Nine voices, fresh clone and LoRA renders, one fixed sentence and seed, ECAPA scoring, generation and scoring provenance. | One sentence and one seed; selected voices are not a randomized population. Mean LoRA-minus-clone similarity of approximately -0.0236 is descriptive, not a general pipeline verdict. |
| `voice_data_saturation_seeded.json` | Provisional | Fourteen voices, fresh seeded renders, ECAPA scoring, generation and scoring provenance. | Observational groups differ in voice, references, training data, and adapter history. The lower mean for the larger-sample group does not establish that more data harms fidelity. |
| `voice_data_saturation.json` | Invalid | None. | Contains zero results, uses acoustic-feature distance, and has no provenance. |
| `instruct_value.json` | Exploratory | Internally reports 1/491, 1/491, and 2/491 errors across its three arms. | Predates the seed fix and records no provenance or seed. It cannot support seeded comparisons. |
| `instruct_value_seeded.json` | Provisional | Reports 3/491 errors for per-line and 0/491 for per-character and none; WER measures content only. | The file itself records neither seed nor commit. Its seeded identity depends on external timestamp/log history. Delivery remains unmeasured. |
| `instruct_listening.json` | Human-pending, structurally exploratory | Four fixed-order comparison files exist and record seed 1234. | No embedded commit/harness provenance; spoken labels and fixed ordering make it a listening aid, not a blind preference test. |
| `blinded_listening.json` | Supported structure + human-pending | Eight randomized sets, 20 fully validated WAVs, separately hashed concealed key, exact source identities, and no public arm-label leakage. | Contains no human ratings and supplies no automated perceptual verdict. Non-prose remedy arms are absent because Stage 5 did not clear its gate. |
| `prose_vs_nonprose_v3.json` | Provisional | Seeded rerun reports 11/25 non-prose failures versus 0/25 prose failures at equal mean character length. | No embedded seed or commit; matching character length does not control digits, punctuation, capitalization, syntax, or category. |
| `nonprose_mechanism.json` | Provisional measurement; unsupported policy | For eight selected failures on one adapter and seed, every ablation still failed 7/8. | One adapter, one seed, selected failures, uncontrolled length/category, and no commit provenance. It does not establish that all non-prose should be routed away from TTS. |
| `nonprose_split_v2.json` | Provisional | Paired whole-versus-split results exist for eight recovered segments. | No embedded provenance and tests only one proposed repair on a selected set. |
| `pitch_stability.json` | Provisional | Records seeded same-text and across-line pitch measurements for one adapter. | One adapter is not the pool; pitch-tracker validity and voiced-frame coverage are not carried in the artifact. |
| `pitch_separation.json` | Provisional | Records six-adapter seeded measurements and the discrepancy between declared and measured pitch. | Pair-separation percentage relies on declared pool values whose observed error is comparable to the proposed threshold. It cannot yet drive casting. |

## Restated claims requiring care

- `WORK_LOG_2026-08-04.md` correctly distinguishes the seeded and unseeded
  instruction artifacts after correction, but readers must not infer delivery
  quality from WER.
- The work log's non-prose routing sentence is wider than the eight-segment,
  one-adapter mechanism run. Stage 4 is required before a general policy.
- The pitch separation percentage is approximate until the pool is re-profiled
  with standardized seeded inputs; it must not be converted directly into a
  production threshold.
- The old claim that weight norm or sample count establishes adapter fidelity
  is unsupported. The new ECAPA saturation result is observational and, in
  this selected set, does not show the assumed positive relationship.

## Required next evidence

1. Stage 4: non-prose replication across adapters, seeds, categories, and
   surface-feature-matched controls, reporting insertions separately.
2. Human ratings of the completed Stage 6 randomized package; the fixed-order
   legacy audio remains only a listening aid.
3. Additional sentences and seeds before turning clone-versus-LoRA ECAPA scores
   into a Voice Lab policy.
4. Controlled or matched analysis before attributing saturation scores to
   training sample count.
5. Seeded standardized pitch profiling is now eligible because production
   auto-selection still falls back to declared pitch.

## Audit integrity note

The structural inventory is an automated screen, not a scientific judge. Its
`supported_structure` label means the artifact can identify how it was made;
only this manual review determines whether the claimed conclusion fits the
measurement.
