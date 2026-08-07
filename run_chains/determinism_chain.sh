#!/bin/bash
set -uo pipefail
REPO=/home/fakemitch/pinokio/api/alexandria-audiobook2.git
export GPU_LOCK="${GPU_LOCK:-$HOME/.alexandria_gpu.lock}"
export GPU_QLOG="$REPO/ab_test_runtime/logs/gpu_jobq.log"
# wait for the gate verification to finish first
while pgrep -f 'verify_adapter_identit[y].py|retrain_hones[t].py' >/dev/null; do sleep 30; done
cd "$REPO/app"
"$REPO/gpu_job.sh" training_determinism \
  timeout 10800 "$REPO/app/env/bin/python" -u experiments/training_determinism.py --runs 3 \
  > "$REPO/ab_test_runtime/logs/training_determinism.log" 2>&1
echo "rc=$? $(date -u +%FT%TZ)"
tail -8 "$REPO/ab_test_runtime/logs/training_determinism.log"
