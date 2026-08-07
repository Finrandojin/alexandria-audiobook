#!/bin/bash
# Retrain the five adapters diagnosed as REBUILD-DATASET, plus a repeat of the
# one control that collapsed.
#
# WHY, given the diagnosis said retraining them was pointless: that reasoning
# assumed training was deterministic. It is not. On 2026-08-07 five clean-data
# failures all improved on a rerun with identical settings, two of them by
# +0.66 and +0.57 - 0.027 became 0.685. So "the dataset is mixed, a retrain
# reproduces the same average" is an inference that has not been tested since
# the determinism assumption fell.
#
# warm_tenor_20s_m repeats because it dropped 0.725 -> 0.382 as a control while
# the other two controls moved -0.012 and -0.017. One reading is contamination
# it was flattered by; another is the same training lottery. A repeat separates
# them.
set -uo pipefail
REPO=/home/fakemitch/pinokio/api/alexandria-audiobook2.git
export GPU_LOCK="${GPU_LOCK:-$HOME/.alexandria_gpu.lock}"
export GPU_QLOG="$REPO/ab_test_runtime/logs/gpu_jobq.log"
cd "$REPO/app"
"$REPO/gpu_job.sh" retrain_rebuild_group \
  timeout 14400 "$REPO/app/env/bin/python" -u experiments/retrain_honest.py \
    --adapters warm_alto_50s_f_gothic husky_baritone_20s_m_supernatural \
               breathy_alto_50s_f_fantasy silky_baritone_45s_m silky_baritone_30s_m \
               velvety_mezzo_30s_f_gothic \
    --controls warm_tenor_20s_m \
    --out "$REPO/ab_test_runtime/experiments/retrain_rebuild_group.json" \
  > "$REPO/ab_test_runtime/logs/retrain_rebuild_group.log" 2>&1
echo "rc=$? $(date -u +%FT%TZ)"
