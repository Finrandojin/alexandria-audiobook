#!/bin/bash
# Stage 1: the counter-example, with an EXPLICIT medoid reference.
# Stage 2: every remaining recoverable adapter, same treatment.
#
# Why stage 1 first. husky_baritone_40s_m_military has a mismatched reference
# (0.133) and a CLEAN dataset (0.772) - the same profile as
# husky_baritone_20s_m_anime, which recovered 0.004 -> 0.691. It did not
# recover: 0.141 -> 0.149. The difference is that those retrains used
# train_lora's FALLBACK reference (the first training clip), which is a
# lottery. --use-medoid removes that variable.
#
#   recovers  -> the counter-example dissolves and the reference theory is
#                complete; run stage 2 with confidence
#   does not  -> there is a second failure mode, and stage 2 would waste hours
#                retraining adapters against something not yet understood
set -uo pipefail
REPO=/home/fakemitch/pinokio/api/alexandria-audiobook2.git
L="$REPO/ab_test_runtime/logs"
export GPU_LOCK="${GPU_LOCK:-$HOME/.alexandria_gpu.lock}"
export GPU_QLOG="$L/gpu_jobq.log"
cd "$REPO/app"

"$REPO/gpu_job.sh" medoid_counterexample \
  timeout 7200 "$REPO/app/env/bin/python" -u experiments/retrain_honest.py \
    --adapters husky_baritone_40s_m_military \
    --controls husky_baritone_20s_m_anime \
    --use-medoid \
    --out "$REPO/ab_test_runtime/experiments/medoid_counterexample.json" \
  > "$L/medoid_counterexample.log" 2>&1
echo "counterexample rc=$?"
grep -E 'reference:|held-out ecapa|was|now' "$L/medoid_counterexample.log" | tail -6

echo ""
echo "=== stage 2: the rest of the recoverable library ==="
"$REPO/gpu_job.sh" medoid_library_retrain \
  timeout 28800 "$REPO/app/env/bin/python" -u experiments/retrain_honest.py \
    --adapters silky_baritone_30s_m breathy_tenor_18s_m_supernatural \
               warm_tenor_25s_m_military husky_tenor_30s_m \
               warm_baritone_30s_m_3 warm_tenor_20s_m_scifi \
               warm_baritone_40s_m_1 warm_baritone_20s_m_postapoc \
               breathy_baritone_30s_m_fantasy warm_mezzo_30s_f_fantasy_2 \
    --use-medoid \
    --out "$REPO/ab_test_runtime/experiments/medoid_library_retrain.json" \
  > "$L/medoid_library_retrain.log" 2>&1
echo "library rc=$?"
grep -E 'held-out ecapa' "$L/medoid_library_retrain.log" | tail -12
echo "MEDOID RETRAIN CHAIN DONE $(date -u +%FT%TZ)"
