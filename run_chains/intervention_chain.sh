#!/bin/bash
set -uo pipefail
REPO=/home/fakemitch/pinokio/api/alexandria-audiobook2.git
export GPU_LOCK="${GPU_LOCK:-$HOME/.alexandria_gpu.lock}"
export GPU_QLOG="$REPO/ab_test_runtime/logs/gpu_jobq.log"
cd "$REPO/app"
# Two adapters so a result is not one dataset's quirk: the most extreme case,
# and one whose reference was fine, where the prediction is a SMALL gap.
for a in husky_baritone_20s_m_anime husky_tenor_30s_m_literary; do
  "$REPO/gpu_job.sh" "refintervene_$a" \
    timeout 10800 "$REPO/app/env/bin/python" -u experiments/reference_intervention.py \
      --adapter "$a" \
      --out "$REPO/ab_test_runtime/experiments/reference_intervention__$a.json" \
    > "$REPO/ab_test_runtime/logs/refintervene_$a.log" 2>&1
  echo "$a rc=$?"
  grep -E 'medoid|worst|VERDICT' "$REPO/ab_test_runtime/logs/refintervene_$a.log" | tail -4
done
echo "INTERVENTION DONE $(date -u +%FT%TZ)"
