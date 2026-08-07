#!/bin/bash
set -uo pipefail
REPO=/home/fakemitch/pinokio/api/alexandria-audiobook2.git
export GPU_LOCK="${GPU_LOCK:-$HOME/.alexandria_gpu.lock}"
export GPU_QLOG="$REPO/ab_test_runtime/logs/gpu_jobq.log"
cd "$REPO/app"
"$REPO/gpu_job.sh" library_voice_fidelity \
  timeout 14400 "$REPO/app/env/bin/python" -u experiments/library_voice_fidelity.py \
    --lines 4 > "$REPO/ab_test_runtime/logs/library_voice_fidelity.log" 2>&1
echo "rc=$? $(date -u +%FT%TZ)"
