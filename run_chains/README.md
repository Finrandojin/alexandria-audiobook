# Run chains

The scripts that actually produced the 2026-08-07 voice-library results. Kept
because a result whose invocation is lost is a result nobody can re-run, and
several of these encode ordering that is not obvious from the experiment
scripts alone.

Each waits for the GPU via `gpu_job.sh`, so two of them running at once queue
rather than fight over VRAM.

| script | what it produced |
|---|---|
| `run_fidelity.sh` | `library_voice_fidelity.json` — every adapter scored against its own narrator |
| `run_consistency.sh` | `dataset_speaker_consistency.json` — is each dataset one speaker |
| `ref_audit_chain.sh` | `dataset_ref_audit.json` + the n=10 reclassification |
| `run_rebuild_retrain.sh` | `retrain_rebuild_group.json` — do mixed-speaker datasets recover |
| `determinism_chain.sh` | `training_determinism.json` — three runs at one seed |
| `verify_gate_test.sh` | `gate_known_good/bad.json` — proves the identity gate refuses a bad adapter |
| `intervention_chain.sh` | the two-arm reference intervention |
| `sharp_intervention_chain.sh` | the three-arm version with a foreign narrator |
| `rescore_anchor.sh` | re-scored all three language sets after the anchor fix |

## Two things that will bite

**Do not switch git branches while one of these is running.** They read the
working tree. `sharp_intervention_chain.sh` died with `ModuleNotFoundError`
because `voice_reference.py` existed only on an unmerged branch and a
`checkout` pulled it out from under the running job. See CLAUDE.md Rule 20 —
a working tree is shared mutable state in the same way a local ref is.

**Waiting on a PID beats waiting on a pattern.** Several of these take a PID to
wait for. `pgrep -f <pattern>` also matches the shell that ran it, which is one
of the three mistakes `gpu_job.sh`'s own header calls out.

## Paths

`ALEXANDRIA_VOICE_ZIPS` overrides the dataset-zip location,
`ALEXANDRIA_SIBLING_PYTHON` the interpreter holding speechbrain. The scripts
themselves still carry an absolute `REPO=` line, since they were written to be
launched by hand rather than to be portable.
