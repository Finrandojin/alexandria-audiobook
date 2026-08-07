#!/bin/bash
set -uo pipefail
REPO=/home/fakemitch/pinokio/api/alexandria-audiobook2.git
export GPU_LOCK="${GPU_LOCK:-$HOME/.alexandria_gpu.lock}"
export GPU_QLOG="$REPO/ab_test_runtime/logs/gpu_jobq.log"
while pgrep -f 'training_determinis[m].py|verify_adapter_identit[y].py|retrain_hones[t].py' >/dev/null; do sleep 30; done
cd "$REPO/app"

# 1. REF AUDIT — would a medoid reference beat the sample-0 default?
"$REPO/gpu_job.sh" dataset_ref_audit \
  timeout 10800 "$REPO/app/env/bin/python" -u experiments/repair_dataset_ref.py \
  > "$REPO/ab_test_runtime/logs/dataset_ref_audit.log" 2>&1
echo "ref_audit rc=$?"
tail -4 "$REPO/ab_test_runtime/logs/dataset_ref_audit.log"

# 2. RECLASSIFY — the RETRAIN/REBUILD verdicts were computed from 4-clip
#    adapter scores, and three adapters moved by more than 0.15 at ten clips
#    (warm_tenor_20s_m 0.725 -> 0.090). Verdicts resting on those numbers need
#    recomputing before anyone acts on them.
"$REPO/gpu_job.sh" consistency_n10 \
  timeout 10800 "$REPO/app/env/bin/python" -u experiments/dataset_speaker_consistency.py \
    --fidelity "$REPO/ab_test_runtime/experiments/library_voice_fidelity_n10.json" \
    --out "$REPO/ab_test_runtime/experiments/dataset_speaker_consistency_n10.json" \
  > "$REPO/ab_test_runtime/logs/consistency_n10.log" 2>&1
echo "reclassify rc=$?"
sed -n '/SUMMARY/,$p' "$REPO/ab_test_runtime/logs/consistency_n10.log" | head -8
echo "REF CHAIN DONE $(date -u +%FT%TZ)"
