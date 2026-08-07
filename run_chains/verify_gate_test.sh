#!/bin/bash
# Verify the identity gate catches the real thing: it must PASS the retrained
# adapter (0.685) and FAIL the original one (0.027). Same adapter name, same
# dataset - only the weights differ. Rule 12: a gate that has not been shown to
# refuse a known-bad input is not a gate.
set -uo pipefail
REPO=/home/fakemitch/pinokio/api/alexandria-audiobook2.git
export GPU_LOCK="${GPU_LOCK:-$HOME/.alexandria_gpu.lock}"
export GPU_QLOG="$REPO/ab_test_runtime/logs/gpu_jobq.log"
A=husky_baritone_20s_m_anime
DATA="$REPO/ab_test_runtime/retrain_honest/$A/data"
cd "$REPO/app"

echo "=== KNOWN GOOD: the retrained adapter (scored 0.685) ==="
"$REPO/gpu_job.sh" gate_known_good \
  "$REPO/app/env/bin/python" -u experiments/verify_adapter_identity.py \
    --adapter "$REPO/ab_test_runtime/retrain_honest/$A/adapter" \
    --dataset "$DATA" --lines 5 \
    --out "$REPO/ab_test_runtime/experiments/gate_known_good.json"
echo "  exit=$?  (expect 0)"

echo "=== KNOWN BAD: the original shipped adapter (scored 0.027) ==="
"$REPO/gpu_job.sh" gate_known_bad \
  "$REPO/app/env/bin/python" -u experiments/verify_adapter_identity.py \
    --adapter "$REPO/lora_models/$A" \
    --dataset "$DATA" --lines 5 \
    --out "$REPO/ab_test_runtime/experiments/gate_known_bad.json"
echo "  exit=$?  (expect 3)"
